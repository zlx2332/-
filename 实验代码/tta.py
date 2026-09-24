"""Multi-label test-time adaptation: teacher-student with consistency,
conditional-distribution matching and covariance-structure matching.

Pipeline per test batch (all losses computed on the STUDENT, only student
parameters receive gradients):

    teacher(weak aug)  ->  p_t, h_t
    student(strong aug) ->  p_s, h_s
    L_cons    : dual-threshold filtered BCE with adaptive class weight w_c
    L_cond    : align class-conditional mean of p_s with dynamic prior pi^c
    L_struct  : align student batch covariance with teacher EMA covariance
    update student (train_mode BN, frozen backbone), then EMA -> teacher
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def set_bn_train_mode(module):
    """Enable batch statistics in all (non-frozen) BN layers."""
    for m in module.modules():
        if isinstance(m, _BN_TYPES):
            m.train()


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)
_AFFINE_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)


def set_bn_tent_mode(module):
    """TENT-style BN: train-mode forward with batch statistics, running buffer
    disabled (never read/used) — the pure parameter-update paradigm. BN stats
    are neither modified nor relied upon."""
    module.train()
    for m in module.modules():
        if isinstance(m, _BN_TYPES):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


def _freeze_non_bn(model, train_head=False, head_prefixes=("head.",)):
    """Freeze everything except norm-layer affine parameters (BN / LN) and,
    optionally, the classifier head."""
    norm_ids = set()
    for m in model.modules():
        if isinstance(m, _AFFINE_NORM_TYPES):
            for p in m.parameters():
                norm_ids.add(id(p))
    for name, p in model.named_parameters():
        head = train_head and any(name.startswith(pre) for pre in head_prefixes)
        p.requires_grad_(id(p) in norm_ids or head)
    return [p for p in model.parameters() if p.requires_grad]


def _ema_update(student, teacher, m, buffer_mode="copy"):
    """teacher <- m * teacher + (1-m) * student for float params.
    Running stats: 'copy' mode takes the student's batch-updated stats,
    'ema' mode smooths them with the same momentum. Under TENT-style BN the
    running buffers are None — state_dict() simply omits/None-matches, and we
    skip None-valued entries."""
    with torch.no_grad():
        msd = student.state_dict()
        for k, tv in teacher.state_dict().items():
            if k not in msd or tv is None or msd[k] is None:
                continue
            sv = msd[k]
            is_stat = ("running_" in k) or ("num_batches" in k)
            if is_stat and buffer_mode == "ema" and sv.dtype.is_floating_point:
                tv.copy_(m * tv + (1.0 - m) * sv)
            else:  # params EMA / copy running stats / integer buffers
                tv.copy_(m * tv + (1.0 - m) * sv if sv.dtype.is_floating_point and not is_stat
                         else sv)


# --------------------------------------------------------------------------- #
# main algorithm
# --------------------------------------------------------------------------- #
class MultiLabelTTA:
    def __init__(self, model_fn, cfg, source_stats=None, head_prefixes=("head.",)):
        """
        model_fn      : zero-arg callable returning a fresh net (same init for both)
        cfg           : config.Config
        source_stats  : dict from source-domain statistics --
                        {"prototypes": Tensor[C,D], "class_freq": Tensor[C],
                         "cov": Tensor[C,C], "cov_mean": Tensor[C]}
        head_prefixes : parameter-name prefixes of the classifier head
                        (real backbones use e.g. ("fc.",) for torchvision resnet)
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.C = cfg.num_classes

        # identical initialisation for teacher and student
        self.student = model_fn().to(self.device)
        self.teacher = model_fn().to(self.device)
        self.teacher.load_state_dict(self.student.state_dict())

        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self.params = _freeze_non_bn(self.student, train_head=cfg.train_head,
                                     head_prefixes=head_prefixes)
        if not self.params:
            raise RuntimeError("no trainable parameters (no BN/LN affine found)")
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.tta_lr)

        # ---------- source statistics ----------
        source_stats = source_stats or {}
        protos = source_stats.get("prototypes")
        if protos is None:
            protos = torch.eye(self.C, self.student.feature_dim)
        self.prototypes = protos.clone().float().to(self.device)          # C_c
        freq = source_stats.get("class_freq")
        if freq is None:
            freq = torch.full((self.C,), 1.0 / self.C)
        self.pi = freq.clone().float().to(self.device)                    # pi^c(t)
        self.pi_src = freq.clone().float().to(self.device)                # source prior (drift log)
        self.pi_floor = freq.clone().float().to(self.device) * cfg.pi_floor_ratio
        cov = source_stats.get("cov")
        if cov is None:
            cov = torch.eye(self.C)
        self.cov = cov.clone().float().to(self.device)                    # Sigma^t
        cov_mean = source_stats.get("cov_mean")
        if cov_mean is None:
            cov_mean = self.pi.clone()
        self.cov_mean = cov_mean.clone().float().to(self.device)

        # normalise prototypes for cosine similarity
        self.proto_norm = self.prototypes / self.prototypes.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.protos0 = self.prototypes.clone()   # arrival state (drift log)

        self.num_batches = 0
        self.dec_ema = None
        self.dec_init = None            # arrival-state decision rates (drift gate)
        self._dec_acc = None
        self._dec_n = 0
        self.fallback = False           # latched predict-only mode (drift gate)
        self.fallback_batch = -1
        self.last_drift = 0.0
        self.last_conc = 0.0
        if cfg.drift_gate > 0:
            # arrival-state snapshot (CPU) for the revert-on-trigger fallback
            self._init_state = OrderedDict(
                (k, v.detach().cpu().clone())
                for k, v in self.teacher.state_dict().items() if v is not None)
        else:
            self._init_state = None
        self.pi_trace = []            # per-batch dynamic pi (for tracking curves)
        self.log = {k: [] for k in
                    ("cons", "cond", "struct", "pos_ratio", "neg_ratio",
                     "pi_drift", "drift", "conc", "proto_drift")}

    @torch.no_grad()
    def _restore_initial(self):
        """Latch the safe fallback: rewind BOTH nets to the arrival state so
        post-trigger predictions equal zero-shot exactly. Single-pass streams
        cannot retract earlier predictions — the latch stops further damage."""
        if self._init_state is None:
            return
        for sd in (self.student.state_dict(), self.teacher.state_dict()):
            for k, v in self._init_state.items():
                if k in sd and sd[k] is not None:
                    sd[k].copy_(v.to(sd[k].device))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_stats(self, p_t, h_t):
        """Prototype EMA, dynamic prior pi^c, teacher covariance EMA."""
        cfg = self.cfg

        # --- per-class adaptive positive threshold (slow EMA of batch quantiles) ---
        # computed first so prototype updates below can reuse the same threshold
        if cfg.tau_adaptive:
            if cfg.rho_from_pi:
                # per-class budget rho_c = pi^c (the scheme's own dynamic frequency:
                # head classes get a larger positive budget than tail classes)
                rho = self.pi.clamp(0.02, 0.30)
                p_sorted = torch.sort(p_t, dim=0).values              # B x C
                Bq = p_t.shape[0]
                q = torch.empty_like(self.pi)
                for c in range(p_t.shape[1]):
                    k = max(1, int(round(rho[c].item() * Bq)))
                    q[c] = p_sorted[Bq - k, c]                        # top-k value
            else:
                q = torch.quantile(p_t, 1.0 - cfg.tau_adaptive_rho, dim=0)   # C
            if not hasattr(self, "tau_pos_c"):
                self.tau_pos_c = q
            else:
                self.tau_pos_c.mul_(cfg.tau_adaptive_ema).add_(
                    q, alpha=1 - cfg.tau_adaptive_ema)

        # --- prototypes: slow EMA with high-confidence positive features ---
        thr = self.tau_pos_c if hasattr(self, "tau_pos_c") else \
            torch.full_like(self.pi, cfg.tau_pos)
        pos = p_t > thr.unsqueeze(0)                               # B x C
        if cfg.sc_confirm or cfg.sc_feat_only:
            # S_c admission design space (advisor Q1 + new comment):
            #   sc_feat_only — PURE feature gate: S_c = A_c^{(t)}, the same
            #     criterion that defines L_cond's subset (advisor: build S_c
            #     from A_c/C_c); prototypes follow the feature neighbourhood
            #     regardless of miscalibrated probabilities.
            #   sc_confirm — DOUBLE gate: probability AND feature evidence.
            sim_sc = Fn.normalize(h_t, dim=1) @ self.proto_norm.T
            feat_gate = sim_sc > cfg.tau_proto
            pos = feat_gate if cfg.sc_feat_only else pos * feat_gate
        if pos.any():
            h_n = Fn.normalize(h_t, dim=1)                        # B x D
            protos = self.prototypes
            pn = protos / protos.norm(dim=1, keepdim=True).clamp_min(1e-8)
            new_p = torch.einsum("bc,bd->cd", pos.float(), h_n)   # sum of feats per class
            cnt = pos.float().sum(0)                              # C
            sel = cnt > 0
            upd = new_p[sel] / cnt[sel].unsqueeze(1)
            protos[sel] = cfg.proto_alpha * protos[sel] + (1 - cfg.proto_alpha) * upd
            self.proto_norm[sel] = protos[sel] / protos[sel].norm(dim=1, keepdim=True).clamp_min(1e-8)

        # --- dynamic class frequency ---
        if cfg.pi_freeze:                     # no prior shift: pi stays at source
            batch_freq = None
        elif cfg.pi_hard_count:               # decision-level rate, robust to calibration
            batch_freq = (p_t > 0.5).float().mean(0)              # C
        else:
            batch_freq = p_t.mean(0)                             # C
        if batch_freq is not None:
            self.pi.mul_(cfg.beta_pi).add_(batch_freq, alpha=1 - cfg.beta_pi)
            self.pi.clamp_(min=0.0, max=1.0)
            if cfg.pi_floor_ratio > 0:
                self.pi.clamp_(min=self.pi_floor)

        # --- teacher covariance EMA ---
        B = p_t.shape[0]
        if B >= 2:
            mu = p_t.mean(0, keepdim=True)
            cov_b = (p_t - mu).T @ (p_t - mu) / B
            self.cov.mul_(cfg.eta_sigma).add_(cov_b, alpha=1 - cfg.eta_sigma)
            self.cov_mean.mul_(cfg.eta_sigma).add_(mu.squeeze(0), alpha=1 - cfg.eta_sigma)

    # ------------------------------------------------------------------ #
    def _loss_cons(self, p_t, p_s, p_sw=None, h_t=None):
        """Dual-threshold filtered BCE with adaptive class weight (optional focal).

        cons_rebalance: per-class mask-normalised variant -- positive and negative
        terms are each divided by their own mask count, so the (rare) positive
        pseudo-labels are not drowned by the (ubiquitous) negatives. This changes
        only the gradient scale, never its direction.

        agree_gate (spec 挑战1): a pseudo-label counts only when teacher AND
        student-on-weak-view agree; disagreement falls into the uncertain zone
        and contributes no gradient.

        proto_confirm: positive pseudo-labels additionally require the teacher
        feature to be close to the class prototype (independent feature-space
        evidence, using the scheme's own prototypes).

        cons_form="soft": non-committed soft-target consistency — plain MSE
        between the student's strong-view predictions and the teacher's
        weak-view predictions. No thresholds, no committed labels: the
        gradient is proportional to disagreement, so the update volume
        shrinks as the student matches the teacher AND as the teacher gets
        quieter — the SLEB-style self-limiting property, inside the
        teacher-student loop. tau/proto/agree gates are irrelevant in this
        form (no selection)."""
        cfg = self.cfg
        if cfg.cons_form in ("soft", "mse"):   # "mse" = legacy alias
            loss = (p_s - p_t).pow(2).mean()
            pos_ratio = (p_t > 0.5).float().mean().item()   # diagnostic only
            return loss, pos_ratio, 1.0 - pos_ratio
        if cfg.tau_adaptive and hasattr(self, "tau_pos_c"):
            tau_c = self.tau_pos_c.unsqueeze(0)                    # 1 x C
            m_pos = (p_t > tau_c).float()                          # B x C, per-class tau
        else:
            tau_c = cfg.tau_pos
            m_pos = (p_t > cfg.tau_pos).float()                    # B x C
        m_neg = (p_t < cfg.tau_neg).float()

        if cfg.agree_gate and p_sw is not None:
            m_pos = m_pos * (p_sw > tau_c).float()
            m_neg = m_neg * (p_sw < cfg.tau_neg).float()

        if cfg.proto_confirm and h_t is not None:
            h_n = Fn.normalize(h_t.detach(), dim=1)                # B x D
            sim = h_n @ self.proto_norm.T                          # B x C
            m_pos = m_pos * (sim > cfg.tau_proto).float()
        w = (1.0 - p_t.mean(0)).pow(cfg.gamma_w)                  # C  (head -> small)
        eps = 1e-6
        p_s_c = p_s.clamp(eps, 1 - eps)

        if cfg.focal:
            fp = (1 - p_s_c).pow(cfg.focal_gamma)
            fn = p_s_c.pow(cfg.focal_gamma)
            per_pos = w * m_pos * fp * p_s_c.log()
            per_neg = w * m_neg * fn * (1 - p_s_c).log()
        else:
            per_pos = w * m_pos * p_s_c.log()
            per_neg = w * m_neg * (1 - p_s_c).log()

        if cfg.cons_rebalance:
            n_pos = m_pos.sum(0)                                  # C
            n_neg = m_neg.sum(0)                                  # C
            loss = -0.5 * ((per_pos.sum(0) / (n_pos + 1)).mean()
                           + (per_neg.sum(0) / (n_neg + 1)).mean())
        else:
            loss = -(per_pos + per_neg).sum() / m_pos.numel()     # 1/(B*C)
        pos_ratio = m_pos.mean().item()
        neg_ratio = m_neg.mean().item()
        return loss, pos_ratio, neg_ratio

    def _loss_cond(self, h_s, p_s, h_t=None):
        """Conditional distribution matching on prototype-related subsets."""
        cfg = self.cfg
        # A_c membership features: student (default) or teacher (unified
        # feature evidence — advisor Q2: one distribution, one tau)
        h_feat = h_t.detach() if (cfg.cond_teacher_feat and h_t is not None) else h_s
        h_n = Fn.normalize(h_feat, dim=1)                         # B x D
        sim = h_n @ self.proto_norm.T                             # B x C
        A = sim > cfg.proto_tau_cos                               # B x C membership
        Acol = A.any(0)
        if not Acol.any():
            return p_s.new_zeros(())

        mean_s = (p_s * A).sum(0) / A.sum(0).clamp_min(1)         # C  masked mean
        err = (mean_s[Acol] - self.pi[Acol]).pow(2)
        return err.sum() / self.C                                 # 1/C sum incl. empty sets

    def _corr(self, S):
        """Scale-free co-occurrence structure (correlation matrix).

        corr_eps > 0 (advisor): corr_ij = S_ij / sqrt((S_ii+eps)(S_jj+eps)).
        Beyond preventing 0/0, eps SHRINKS the correlation rows of near-dead
        classes (S_ii << eps) toward zero by the factor sqrt(S_ii/(S_ii+eps)) —
        exactly the estimation-noise rows quantified in exp33's floor probe.
        eps = 0 keeps the legacy clamp_min(1e-8) guard bit-for-bit (report
        snippets omitted it, but literal 0-division was already prevented —
        eps is a regulariser, not just a guard)."""
        if self.cfg.corr_eps > 0:
            d = torch.sqrt(S.diagonal() + self.cfg.corr_eps)
        else:
            d = torch.sqrt(S.diagonal().clamp_min(1e-8))
        return S / torch.outer(d, d)

    def _loss_struct(self, p_s):
        """Student batch covariance vs teacher online covariance.

        struct_form:
          "frob"    — original ||Σs−Σt||²_F / C² (scales with probability magnitude;
                      on ASL-sparse models this is ~1e-6 and contributes nothing);
          "relfrob" — relative Frobenius (scale-free magnitude);
          "corr"    — correlation-matrix alignment: matches the scheme's stated
                      intent ("类别共现关系与源域一致") — co-occurrence structure
                      independent of per-class confidence levels. struct_offdiag
                      restricts it to the co-occurrence (off-diagonal) block."""
        cfg = self.cfg
        B = p_s.shape[0]
        if B < cfg.min_batch_for_struct:
            return p_s.new_zeros(())
        mu = p_s.mean(0, keepdim=True)
        cov_s = (p_s - mu).T @ (p_s - mu) / B
        if cfg.struct_form == "corr":
            Cs, Ct = self._corr(cov_s), self._corr(self.cov)
            D = Cs - Ct
            if cfg.struct_offdiag:
                D = D.clone()
                D.fill_diagonal_(0.0)
                denom = Ct.clone()
                denom.fill_diagonal_(0.0)
                denom = denom.pow(2).sum().clamp_min(1e-8)
            else:
                denom = Ct.pow(2).sum().clamp_min(1e-8)
            return D.pow(2).sum() / denom
        if cfg.struct_form in ("relfrob",) or cfg.struct_norm:
            return (cov_s - self.cov).pow(2).sum() / self.cov.pow(2).sum().clamp_min(1e-8)
        return (cov_s - self.cov).pow(2).sum() / (self.C * self.C)

    # ------------------------------------------------------------------ #
    def _pi_correct(self, p):
        """Prediction-time prior correction (label-shift odds adjustment):
        p'/(1-p') = [p/(1-p)] · (pi_hat_c / pi_src_c). Per-class monotone =>
        mAP invariant by construction; moves decisions relative to a fixed
        threshold => visible in F1@0.5. The dynamic pi estimate finally gets a
        direct actuator at the output end."""
        cfg = self.cfg
        if not cfg.pi_pred_correct:
            return p
        r = (self.pi / self.pi_src.clamp_min(1e-6)).clamp(0.2, 5.0)
        odds = p / (1.0 - p).clamp_min(1e-6)
        return (odds * r.unsqueeze(0)).clamp(1e-6, 1e6) / \
            (1.0 + (odds * r.unsqueeze(0)).clamp(1e-6, 1e6))

    @torch.enable_grad()
    def adapt_batch(self, x_w, x_s):
        """One TTA step. Returns (p_t, loss_dict)."""
        cfg = self.cfg

        # ---- latched safe fallback (drift gate): predict-only, both nets at
        # the arrival snapshot — post-trigger predictions equal zero-shot ----
        if self.fallback:
            with torch.no_grad():
                p_t = self.teacher(x_w).sigmoid()
            d = {"cons": 0.0, "cond": 0.0, "struct": 0.0,
                 "pos_ratio": 0.0, "neg_ratio": 0.0, "pi_drift": 0.0,
                 "drift": self.last_drift, "conc": self.last_conc}
            for k in d:
                self.log[k].append(d[k])
            self.num_batches += 1
            return self._pi_correct(p_t), d

        if cfg.bn_tent:
            # TENT-style: batch-statistics forward for BOTH nets; the running
            # buffer is disabled (never read) — no statistics are touched.
            set_bn_tent_mode(self.student)
            set_bn_tent_mode(self.teacher)
            for m in self.student.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()
        else:
            self.student.train()
            set_bn_train_mode(self.student)      # BN uses batch stats, dropout off
            for m in self.student.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()

        # ---- optional warm-up: refresh BN stats only, skip the gradient step ----
        # (skipped under bn_tent: there are no running stats to warm up)
        if not cfg.bn_tent and cfg.warmup_batches and \
                self.num_batches < cfg.warmup_batches:
            with torch.no_grad():
                self.student(x_w)                # running stats <- test view
            _ema_update(self.student, self.teacher, cfg.ema_momentum, cfg.buffer_mode)
            self.num_batches += 1
            with torch.no_grad():
                p_t = self.teacher(x_w).sigmoid()
            d = {"cons": 0.0, "cond": 0.0, "struct": 0.0,
                 "pos_ratio": 0.0, "neg_ratio": 0.0, "pi_drift": 0.0,
                 "drift": 0.0, "conc": 0.0}
            for k in d:
                self.log[k].append(d[k])
            return self._pi_correct(p_t), d

        # ---- forward ----
        p_t = None
        with torch.no_grad():
            t_logits, h_t = self.teacher(x_w, return_features=True)
            p_t = t_logits.sigmoid()

            # ---- health signals on the teacher decision distribution ----
            # dec_ema: ~10-batch EMA of per-class decision RATES (shared by the
            # collapse gate and the drift gate). dec_init: the arrival-state
            # decision rates, averaged over the first drift_warm batches.
            conc_v, drift_v = 0.0, 0.0
            tot = 0.0
            if cfg.conc_gate > 0 or cfg.drift_gate > 0:
                # GLOBAL 0.5 threshold (no per-class tau_c — the quantile
                # budget flattens the distribution, and the monopoly class's own
                # high tau_c would cut exactly the counts that reveal the monopoly)
                dec = (p_t > 0.5).float().mean(0)
                self.dec_ema = dec.clone() if self.dec_ema is None else \
                    0.9 * self.dec_ema + 0.1 * dec
                if self.dec_init is None:
                    if self._dec_acc is None:
                        self._dec_acc = dec.clone().float()
                        self._dec_n = 1
                    else:
                        self._dec_acc += dec
                        self._dec_n += 1
                    if self._dec_n >= max(1, cfg.drift_warm):
                        self.dec_init = self._dec_acc / self._dec_n
                tot = self.dec_ema.sum().item()
                conc_v = (self.dec_ema.max().item() / tot) if tot > 1e-6 else 0.0
                if self.dec_init is not None:
                    drift_v = (self.dec_ema - self.dec_init).abs().mean().item()
                self.last_conc, self.last_drift = conc_v, drift_v

            # ---- collapse gate: EMA decision-distribution concentration.
            # A collapsed model is still confidently wrong — mask coverage stays
            # normal but positive pseudo-labels monopolize one/few classes
            # (health: concentration ~ the top class's pi share; collapse: 0.5+).
            # Gated batches fall back to stats-only (BN refresh + EMA, no gradient,
            # no stat contamination) — safe silence, mirroring the warmup branch.----
            if cfg.conc_gate > 0 and (tot < 1e-6 or conc_v > cfg.conc_gate):
                with torch.no_grad():
                    self.student(x_w)              # BN stats <- test view
                _ema_update(self.student, self.teacher, cfg.ema_momentum,
                            cfg.buffer_mode)
                self.num_batches += 1
                d = {"cons": 0.0, "cond": 0.0, "struct": 0.0,
                     "pos_ratio": 0.0, "neg_ratio": 0.0, "pi_drift": 0.0,
                     "drift": drift_v, "conc": conc_v}
                for k in d:
                    self.log[k].append(d[k])
                return self._pi_correct(p_t), d

            # ---- drift gate (safe fallback): the teacher's decision rates
            # departing from their arrival state is the measured signature of
            # the long-stream poisoning loop (student degrades -> EMA teacher
            # absorbs it -> worse pseudo-labels -> ...; NUS pi_drift 0.37 vs
            # VOC 0.01-0.03). Latch: revert BOTH nets to the arrival snapshot;
            # from here on predictions equal zero-shot exactly. ----
            if cfg.drift_gate > 0 and self.dec_init is not None \
                    and drift_v > cfg.drift_gate and not self.fallback:
                self.fallback = True
                self.fallback_batch = self.num_batches
                self._restore_initial()
                p_t = self.teacher(x_w).sigmoid()
                d = {"cons": 0.0, "cond": 0.0, "struct": 0.0,
                     "pos_ratio": 0.0, "neg_ratio": 0.0, "pi_drift": 0.0,
                     "drift": drift_v, "conc": conc_v}
                for k in d:
                    self.log[k].append(d[k])
                self.num_batches += 1
                return self._pi_correct(p_t), d

            p_sw = None
            if cfg.agree_gate:                     # student's own view on x_w (no grad)
                p_sw = self.student(x_w).sigmoid() # dual-network agreement signal
        s_logits, h_s = self.student(x_s, return_features=True)
        p_s = s_logits.sigmoid()

        # ---- update running statistics with teacher outputs ----
        self._update_stats(p_t, h_t)

        # ---- losses ----
        l_cons, pos_r, neg_r = self._loss_cons(p_t, p_s, p_sw=p_sw, h_t=h_t)
        l_cond = self._loss_cond(h_s, p_s, h_t=h_t)
        l_struct = self._loss_struct(p_s)
        loss = l_cons + cfg.lambda_cond * l_cond + cfg.lambda_struct * l_struct

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # ---- refresh BN running stats with the weak/test view, then EMA -> teacher ----
        # (under bn_tent the running buffer is disabled — nothing to refresh;
        #  the EMA still propagates the updated affine PARAMETERS)
        if cfg.refresh_weak and not cfg.bn_tent:
            with torch.no_grad():
                self.student(x_w)                # stats <- true test distribution
        _ema_update(self.student, self.teacher, cfg.ema_momentum, cfg.buffer_mode)

        self.num_batches += 1
        d = {"cons": l_cons.item(), "cond": l_cond.item(), "struct": l_struct.item(),
             "pos_ratio": pos_r, "neg_ratio": neg_r,
             "pi_drift": (self.pi - self.pi_src).abs().mean().item(),
             "drift": drift_v, "conc": conc_v,
             "proto_drift": (self.prototypes - self.protos0).norm().item() /
                            self.protos0.norm().item()}
        for k, v in d.items():
            self.log[k].append(v)
        self.pi_trace.append(self.pi.detach().cpu().clone())
        return self._pi_correct(p_t), d

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, x_w):
        """Test-time prediction uses the teacher on the weak (test) view.
        (bn_tent: the teacher stays in tent mode — batch statistics.)"""
        return self.teacher(x_w).sigmoid()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def final_report(self):
        out = {}
        for k, v in self.log.items():
            v = np.asarray(v, dtype=np.float64)
            q = max(1, int(0.1 * len(v)))
            out[k] = float(v[-q:].mean())  # last 10% average
        return out
