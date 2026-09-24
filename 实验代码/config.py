"""Central configuration for multi-label test-time adaptation (teacher-student)."""
from dataclasses import dataclass, asdict

import torch


@dataclass
class Config:
    # ---------------- data ----------------
    dataset: str = "synthetic"      # "synthetic" | "csv"
    data_dir: str = ""              # csv mode: root dir of images
    train_csv: str = ""             # csv mode: first column `image`, remaining columns = classes (0/1)
    test_csv: str = ""
    image_size: int = 64            # synthetic image size
    csv_image_size: int = 128       # csv mode input size
    num_classes: int = 20
    num_train: int = 2500
    num_test: int = 3000
    # ---------------- model ----------------
    width: int = 16                 # base width of the small resnet (feature dim = width*8)
    dropout: float = 0.2
    # ---------------- source pre-training ----------------
    source_epochs: int = 15
    source_batch_size: int = 64
    source_lr: float = 3e-3
    pos_weight_cap: float = 20.0    # cap for neg/pos pos_weight in source BCE
    # ---------------- TTA ----------------
    tta_batch_size: int = 32
    tta_lr: float = 1e-3
    ema_momentum: float = 0.999     # m: teacher parameter EMA
    buffer_mode: str = "copy"       # BN running stats: "copy" from student, or "ema" with m
    train_head: bool = False        # additionally unfreeze the classifier head (severe shift)
    refresh_weak: bool = True       # refresh student BN running stats with the weak/test view
    warmup_batches: int = 0         # first N batches: stats-only, no gradient step
    bn_tent: bool = False           # TENT-style BN: batch-stats forward, running buffer
                                    # disabled (never read) — pure parameter-update
                                    # paradigm, supersedes refresh_weak/buffer_mode
    # dual thresholds for pseudo labels
    tau_pos: float = 0.8
    tau_neg: float = 0.2
    tau_adaptive: bool = False     # per-class tau_pos = slow EMA of batch quantiles
    tau_adaptive_rho: float = 0.10  # target positive rate per class (top rho quantile)
    tau_adaptive_ema: float = 0.9   # EMA of the per-class quantile
    # adaptive class weight gamma
    gamma_w: float = 2.0
    # class prototypes
    proto_alpha: float = 0.99       # alpha: prototype EMA
    proto_tau_cos: float = 0.4      # tau: cosine threshold defining subset A_c
    # dynamic class frequency
    beta_pi: float = 0.9
    pi_floor_ratio: float = 0.0     # optional floor: pi >= ratio * source frequency (0 = off)
    pi_hard_count: bool = False     # update pi with decision-level rate mean(1[p_t>0.5])
    pi_freeze: bool = False         # keep pi at the source frequency (no prior shift)
    cons_rebalance: bool = False    # per-class mask-normalised L_cons (pos/neg balanced)
    cons_form: str = "hard"          # L_cons supervision FORM (named by
                                    # supervision kind — unrelated to the
                                    # source model's training loss):
                                    # "hard" = committed thresholded
                                    #   pseudo-labels, BCE against 0/1 targets
                                    #   (legacy value "bce" still accepted);
                                    # "soft" = non-committed soft-target
                                    #   consistency MSE(p_s, p_t), no
                                    #   thresholds, update volume shrinks with
                                    #   teacher quality (legacy "mse")
    drift_gate: float = 0.0         # >0: safe-fallback gate — teacher decision
                                    # rates departing from the arrival state
                                    # beyond this latch to predict-only and both
                                    # nets are reverted to the initial teacher
                                    # (post-trigger predictions = zero-shot)
    drift_warm: int = 10            # batches used to snapshot the arrival
                                    # decision distribution (drift_gate)
    agree_gate: bool = False        # dual-network gate: pseudo-label valid only when
                                    # teacher AND student(weak view) agree (spec 挑战1)
    proto_confirm: bool = False     # positive pseudo-labels additionally require
                                    # cos(h_t, C_c) > tau_proto (prototype confirmation)
    tau_proto: float = 0.3          # loose threshold for prototype confirmation
    sc_confirm: bool = False        # S_c admission upgrade, DOUBLE gate: prototype-update
                                    # samples additionally require cos(h_t, C_c) >
                                    # tau_proto (same gate/evidence as
                                    # proto_confirm; cuts the pollution loop
                                    # false-positive -> proto -> confirm gate)
    sc_feat_only: bool = False      # S_c admission = PURE feature gate
                                    # cos(h_t, C_c) > tau_proto — identical to
                                    # A_c's criterion (advisor: build S_c from
                                    # A_c/C_c); drops the probability gate.
                                    # Overrides sc_confirm when both set.
    cond_teacher_feat: bool = False # L_cond subset A_c membership on TEACHER
                                    # features (default: student features) —
                                    # aligns all feature evidence on one
                                    # feature distribution for tau unification
    rho_from_pi: bool = False       # per-class positive budget rho_c = pi^c (not fixed)
    struct_norm: bool = False       # relative Frobenius normalisation for L_struct
    struct_form: str = "frob"       # "frob" | "relfrob" | "corr" (correlation matrix)
    struct_offdiag: bool = False    # corr form: align only co-occurrence (off-diagonal)
    corr_eps: float = 0.001         # corr form: additive diagonal stabiliser —
                                    # corr_ij = S_ij/sqrt((S_ii+eps)(S_jj+eps)).
                                    # 0.001 per advisor (2026-09-16): guards the
                                    # denominator AND shrinks near-dead-class
                                    # noise rows (Sigma_ii << eps). Verified
                                    # inert on VOC/COCO (min teacher-cov diag
                                    # 1.6e-3 > eps on COCO; exp34) — kept on as
                                    # a zero-cost stability fuse; pass 0 to
                                    # restore the legacy clamp_min(1e-8) form
    conc_gate: float = 0.0          # >0: collapse gate — EMA decision-distribution
                                    # concentration above this -> stats-only batch
                                    # (safe silence; healthy ~0.15-0.26, collapsed 0.5+)
    pi_pred_correct: bool = False   # prediction-time prior correction (Saerens-style):
                                    # logit(p) += log(pi_hat/pi_src) — per-class monotone,
                                    # mAP-neutral by construction, shifts F1@0.5
    # online covariance EMA
    eta_sigma: float = 0.9
    min_batch_for_struct: int = 8   # skip L_struct if batch smaller than this
    # loss weights
    lambda_cond: float = 0.1
    lambda_struct: float = 0.05
    # focal option for L_cons
    focal: bool = False
    focal_gamma: float = 2.0
    # ---------------- misc ----------------
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    log_every: int = 10

    def to_dict(self) -> dict:
        return asdict(self)
