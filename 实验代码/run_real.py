"""Run the teacher-student multi-label TTA on the real project in E:/mttta.

Uses your trained checkpoints (torchvision resnet50 / vit with a C-way head)
and your dataset loaders (ML-TTA/data), evaluating on the corrupted VOC-C
test streams, e.g.:

    E:/anaa/envs/pytorch/python.exe run_real.py \
        --arch resnet50 --num_classes 20 \
        --checkpoint E:/mttta/resnet50_voc_best.pth \
        --source_dir E:/mttta/DATASETS2/VOCtrainval_06-Nov-2007 \
        --test_root  E:/mttta/DATASETS2 \
        --corruptions gaussian_noise glass_blur snow brightness jpeg_compression

Per corruption it reports zero-shot / BN-stats-only / full teacher-student
(11-point interpolated mAP, identical to your tta_tent_eata.py protocol), and
reuses the cached source statistics (prototypes, pi^0, Sigma^0) across runs.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get("ML_TTA_PATH", "E:/mttta/ML-TTA"))  # VOC loader (nuswide/coco paths are self-contained)

from config import Config                       # noqa: E402
from mldata import StrongAugmentNormalized      # noqa: E402
from real_model import build_real_model, clone_model, HEAD_PREFIXES  # noqa: E402
from tta import MultiLabelTTA                   # noqa: E402

ZERO_SHOT_VOC = {  # your reference numbers (tta_tent_eata.py grid script)
    "gaussian_noise": 75.236, "glass_blur": 66.137, "snow": 68.241,
    "brightness": 84.025, "jpeg_compression": 82.313,
}


def f1_at_threshold(Y, P, thr=0.5):
    """Macro F1 with a FIXED decision threshold — the metric through which
    calibration-level mechanisms (dynamic pi -> L_cond marginal alignment)
    become visible (mAP is rank-based and blind to them). Returns (macro,
    per-class vector)."""
    pred = (P > thr).astype(np.float64)
    tp = (pred * Y).sum(0)
    f1c = 2 * tp / (pred.sum(0) + Y.sum(0) + 1e-9)
    return float(f1c.mean() * 100), f1c


def f1_best_threshold(Y, P):
    """Per-class best-threshold F1 (calibration-free control): if dynamic-pi
    equals frozen-pi here but wins at 0.5, the gain is pure threshold
    calibration."""
    f1c = np.zeros(P.shape[1])
    for c in range(P.shape[1]):
        y, s = Y[:, c], P[:, c]
        if y.sum() == 0:
            continue
        order = np.argsort(-s)
        ys = y[order]
        tp = np.cumsum(ys)
        k = np.arange(1, len(ys) + 1)
        prec = tp / k
        rec = tp / tp[-1]
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        f1c[c] = f1.max()
    return float(f1c.mean() * 100)


def mAP_official(targets, preds):
    """Identical to E:/mttta/ML-TTA/utils/tools.py::mAP (continuous AP:
    mean over classes of sum-of-precision-at-positives / #positives), so the
    numbers are directly comparable with your logged results."""
    epsilon = 1e-8
    aps = []
    for k in range(preds.shape[1]):
        scores, target = preds[:, k], targets[:, k]
        indices = scores.argsort()[::-1]
        total_count_ = np.cumsum(np.ones((len(scores), 1)))
        target_ = target[indices]
        ind = target_ == 1
        pos_count_ = np.cumsum(ind)
        total = pos_count_[-1]
        pos_count_[np.logical_not(ind)] = 0
        pp = pos_count_ / total_count_
        aps.append(np.sum(pp) / (total + epsilon))
    return float(100 * np.mean(aps))


def get_test_dataset(dataset, data_dir, transform, limit=None):
    if dataset == "voc":
        from data.voc_custom import VOCCustom
        return VOCCustom("test", data_dir, transform)
    if dataset == "nuswide":
        return NuswideCsv(data_dir, transform=transform, limit=limit)
    if dataset == "coco":
        return CocoMultiHot(data_dir, transform=transform)
    raise ValueError(dataset)


class NuswideCsv(torch.utils.data.Dataset):
    """NUS-WIDE corrupted stream: flat corrupted Flickr dir + csv
    (first column path, remaining 81 columns 0/1 labels).

    limit > 0 takes a fixed-seed random subsample (the csv is sorted by
    concept folder, so head-slicing would be biased)."""

    def __init__(self, nuswide_dir, transform=None, limit=None, split="test"):
        import csv as _csv
        from PIL import Image
        self.Image = Image
        rows = []
        csv_path = os.path.join(nuswide_dir, f"{split}.csv")
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in _csv.reader(f):
                if r:
                    rows.append(r)
        if limit and limit < len(rows):
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(rows))[:limit]
            rows = [rows[i] for i in sorted(idx)]
        self.root = os.path.join(nuswide_dir, "Flickr")
        self.transform = transform
        # the csv's first column carries the ORIGINAL machine's absolute paths
        # (Windows backslashes) — on Linux os.path.basename won't split them,
        # so normalise slashes first, then take the flat corrupted filename
        self.items = [(os.path.basename(r[0].replace("\\", "/")),
                       torch.tensor([float(x) for x in r[1:]], dtype=torch.float32))
                      for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, y = self.items[i]
        img = self.Image.open(os.path.join(self.root, fname)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, y


class ResampledStream(torch.utils.data.Dataset):
    """Prior-shift wrapper: resample (with replacement, seeded) the underlying
    dataset so that tail classes (lowest tail_frac by SOURCE frequency) appear
    upscaled by factor K — a real label-prior shift with covariate corruption
    held fixed. Exposes the true shifted pi for analysis."""

    def __init__(self, base, pi_src, K, seed=0, tail_frac=0.25):
        Y = torch.stack([y for _, _, y in base]).float()
        C = Y.shape[1]
        k = max(1, int(round(tail_frac * C)))
        tail = torch.argsort(torch.as_tensor(pi_src, dtype=torch.float32))[:k]
        is_tail = (Y[:, tail].sum(1) > 0.5)
        w = torch.where(is_tail, torch.full((len(Y),), float(K)), torch.ones(len(Y)))
        probs = (w / w.sum()).numpy()
        rng = np.random.default_rng(seed)
        self.idx = rng.choice(len(base), size=len(base), replace=True, p=probs)
        self.base = base
        self.true_pi = Y[self.idx].mean(0)
        self.source_pi = torch.as_tensor(pi_src, dtype=torch.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


class CocoMultiHot(torch.utils.data.Dataset):
    """COCO2017 val multi-hot dataset — replicates ML-TTA/data/coco2014.py's
    category order (cats sorted by id -> 0..79) but returns a fixed-width
    multi-hot vector so batch>1 collation works. No tiny-box filter, matching
    their val path (filter_tiny=False)."""

    def __init__(self, coco_dir, transform=None):
        from pycocotools.coco import COCO
        from PIL import Image
        self.Image = Image
        self.transform = transform
        self.coco = COCO(os.path.join(coco_dir, "annotations/instances_val2017.json"))
        cats = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda x: x["id"])
        self.inv = {c["id"]: i for i, c in enumerate(cats)}
        self.C = len(cats)
        self.items = []
        for imgid in self.coco.getImgIds():
            info = self.coco.loadImgs(imgid)[0]
            y = torch.zeros(self.C)
            for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=imgid,
                                                            iscrowd=False)):
                y[self.inv[a["category_id"]]] = 1.0
            self.items.append((os.path.join(coco_dir, "val2017", info["file_name"]), y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = self.Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, path, y


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------ #
# source statistics with on-disk cache
# ------------------------------------------------------------------ #
@torch.no_grad()
def compute_source_stats(model, train_ds, device, batch_size=64):
    model.eval()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=True)
    feats, probs, labels = [], [], []
    for img, _, tgt in loader:
        img = img.to(device, non_blocking=True)
        logits, h = model(img, return_features=True)
        feats.append(h.cpu())
        probs.append(logits.sigmoid().cpu())
        labels.append(tgt)
    if not feats:
        raise RuntimeError(
            "source-statistics loader produced 0 batches — the source dataset "
            "is empty (check NUS_CLEAN_ROOT / source_dir layout)")
    H, P = torch.cat(feats), torch.cat(probs)
    Y = torch.cat(labels).float()
    C = P.shape[1]

    protos = torch.zeros(C, H.shape[1])
    freq = torch.zeros(C)
    for c in range(C):
        m = Y[:, c] > 0.5
        if m.any():
            protos[c] = H[m].mean(0)
            freq[c] = m.float().mean()
        else:
            top = P[:, c].topk(min(16, len(P))).indices
            protos[c] = H[top].mean(0)
            freq[c] = P[:, c].mean()
    mu = P.mean(0, keepdim=True)
    cov = (P - mu).T @ (P - mu) / P.shape[0]
    return {"prototypes": protos, "class_freq": freq,
            "cov": cov, "cov_mean": mu.squeeze(0)}


def get_source_stats(cfg, model, cache_path):
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        if blob.get("checkpoint") == os.path.abspath(cfg.checkpoint):
            print(f"  source stats: loaded from cache {cache_path}")
            return blob["stats"]
    if cfg.dataset_name == "voc":
        from data.voc_custom import VOCCustom
        train_ds = VOCCustom(cfg.source_split, cfg.source_dir, build_transform())
    elif cfg.dataset_name == "nuswide":
        # train.csv carries the original machine's absolute paths — rebuild
        # each path under the LOCAL clean root (env var or conventional layout)
        train_ds = NuswideCsvClean(cfg.source_dir, transform=build_transform(),
                                   limit=cfg.source_limit,
                                   clean_root=os.environ.get("NUS_CLEAN_ROOT"))
    elif cfg.dataset_name == "coco":
        # source stats from the clean val2017 (train2017 too large for iteration;
        # prototypes/pi/cov need only a representative sample)
        train_ds = CocoMultiHot(cfg.source_dir, transform=build_transform())
    else:
        raise ValueError(f"source stats unsupported for {cfg.dataset_name}")
    print(f"  source stats: computing over {len(train_ds)} images ...")
    t0 = time.time()
    stats = compute_source_stats(model, train_ds, cfg.device)
    print(f"  source stats: done in {time.time() - t0:.1f}s")
    torch.save({"checkpoint": os.path.abspath(cfg.checkpoint), "stats": stats},
               cache_path)
    return stats


class NuswideCsvClean(torch.utils.data.Dataset):
    """Clean NUS-WIDE train split via train.csv.

    train.csv's first column carries the ORIGINAL training machine's absolute
    paths (e.g. E:\\mttta\\DATASETS5\\nuswide\\<concept>\\<file>) — meaningless on
    another machine. We therefore rebuild each path from its basename under a
    local clean root that must contain <concept>/<file> (or Flickr/<concept>/
    <file>): pass --clean_root (defaults to the conventional local layout).
    Subsamples 30k by default (source stats only need a representative sample).
    """

    def __init__(self, nuswide_dir, transform=None, limit=30000, split="train",
                 clean_root=None):
        import csv as _csv
        from PIL import Image
        self.Image = Image
        rows = []
        with open(os.path.join(nuswide_dir, f"{split}.csv"), newline="",
                  encoding="utf-8") as f:
            for r in _csv.reader(f):
                if r:
                    rows.append(r)
        if limit and limit < len(rows):
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(rows))[:limit]
            rows = [rows[i] for i in sorted(idx)]
        self.transform = transform

        # locate the local clean image root
        root = clean_root
        if root is None:
            for cand in (r"E:/mttta/DATASETS5/nuswide/Flickr",
                         os.path.join(nuswide_dir, "..", "Flickr"),
                         os.path.join(nuswide_dir, "Flickr")):
                if os.path.isdir(cand):
                    root = cand
                    break
        if root is None:
            raise FileNotFoundError(
                "NUS-WIDE clean image root not found — pass --clean_root "
                "<dir with concept subfolders or concept dirs under Flickr>")
        self.items = []
        missing = 0
        for r in rows:
            base = os.path.basename(r[0].replace("\\", "/"))
            concept = os.path.basename(os.path.dirname(r[0].replace("\\", "/")))
            for cand in (os.path.join(root, concept, base),
                         os.path.join(root, "Flickr", concept, base)):
                if os.path.exists(cand):
                    self.items.append((cand,
                                       torch.tensor([float(x) for x in r[1:]],
                                                    dtype=torch.float32)))
                    break
            else:
                missing += 1
        if missing:
            print(f"  [NuswideCsvClean] WARNING: {missing}/{len(rows)} images "
                  f"not found under {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = self.Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, path, y


# ------------------------------------------------------------------ #
# the three evaluation modes
# ------------------------------------------------------------------ #
@torch.no_grad()
def eval_zero_shot(model, loader, device):
    model.eval()
    preds, labels = [], []
    for img, _, tgt in loader:
        probs = model(img.to(device, non_blocking=True)).sigmoid().cpu()
        preds.append(probs)
        labels.append(tgt)
    P = torch.cat(preds).numpy()
    Y = torch.cat(labels).numpy()
    return mAP_official(Y, P), P, Y


@torch.no_grad()
def eval_bn_only(model, loader, device, momentum_reset=True):
    """BN-stats-only adaptation while streaming, predicting online.

    Labels are collected FROM THE LOADER (stream order) — with --shuffle the
    stream order differs from dataset order, so stacking targets from the
    dataset would misalign P and Y."""
    model = clone_model(model).to(device)
    model.eval()
    if momentum_reset:  # faster re-estimation on a short stream
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = None  # cumulative average
                m.reset_running_stats()
    preds, targs = [], []
    for img, _, tgt in loader:
        model.train()  # only BN layers are affected (whole net is frozen)
        logits = model(img.to(device, non_blocking=True))
        model.eval()
        preds.append(logits.sigmoid().cpu())
        targs.append(tgt)
    P = torch.cat(preds).numpy()
    Y = torch.cat(targs).numpy()
    return model, P, Y


def eval_full_tta(cfg, model, loader, stats, strong, head_prefixes):
    tta = MultiLabelTTA(lambda: clone_model(model), cfg, source_stats=stats,
                        head_prefixes=head_prefixes)
    n = len(loader.dataset)
    P = torch.zeros(n, cfg.num_classes)
    ptr = 0
    targs = []
    t0 = time.time()
    for i, (img, _, tgt) in enumerate(loader):
        x = img.to(cfg.device, non_blocking=True)
        p_t, d = tta.adapt_batch(x, strong(x))
        P[ptr:ptr + img.shape[0]] = p_t.detach().cpu()
        ptr += img.shape[0]
        targs.append(tgt)
        if (i + 1) % 50 == 0:
            print(f"    batch {i + 1}: cons={d['cons']:.4f} cond={d['cond']:.5f} "
                  f"struct={d['struct']:.5f} pos%={d['pos_ratio']:.2f} "
                  f"neg%={d['neg_ratio']:.2f} drift={d.get('drift', 0.0):.4f} "
                  f"conc={d.get('conc', 0.0):.3f} "
                  f"proto_drift={d.get('proto_drift', 0.0):.4f}")
    assert ptr == n
    print(f"    TTA stream finished in {time.time() - t0:.1f}s")
    if getattr(tta, "fallback", False):
        print(f"    [safe-fallback] latched at batch {tta.fallback_batch} "
              f"(nets reverted to the arrival snapshot)")
    else:
        print("    [safe-fallback] never triggered")
    # labels in STREAM order (shuffle-safe; dataset order would misalign P/Y)
    Y = torch.cat(targs).numpy()
    return tta, P.numpy(), Y


# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet50", choices=["resnet50", "vit"])
    p.add_argument("--checkpoint", default="E:/mttta/resnet50_voc_best.pth")
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--dataset_name", default="voc", choices=["voc", "nuswide", "coco"])
    p.add_argument("--source_dir", default=None,
                   help="voc: clean trainval root; nuswide: any corrupted NUSWIDE "
                        "dir (train.csv lives there, identical across corruptions)")
    p.add_argument("--source_split", default="trainval")
    p.add_argument("--source_limit", type=int, default=30000,
                   help="nuswide: subsample size for source statistics")
    p.add_argument("--test_root", default=None)
    p.add_argument("--test_pattern", default=None,
                   help="voc: VOCtest_{corr}_severity_5; nuswide: {corr}_5/NUSWIDE")
    p.add_argument("--limit", type=int, default=None,
                   help="test-stream subsample (fixed seed), for iteration")
    p.add_argument("--corruptions", nargs="+",
                   default=["gaussian_noise", "glass_blur", "snow",
                            "brightness", "jpeg_compression"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--tau_pos", type=float, default=0.8)
    p.add_argument("--tau_neg", type=float, default=0.2)
    p.add_argument("--lambda_cond", type=float, default=0.1)
    p.add_argument("--lambda_struct", type=float, default=0.05)
    p.add_argument("--gamma_w", type=float, default=2.0)
    p.add_argument("--proto_tau_cos", type=float, default=0.4)
    p.add_argument("--focal", action="store_true")
    p.add_argument("--train_head", action="store_true")
    p.add_argument("--bn_warmup", type=int, default=3,
                   help="first N batches: refresh BN stats only (no gradient step)")
    p.add_argument("--no_refresh_weak", action="store_true",
                   help="disable BN running-stat refresh with the weak/test view")
    p.add_argument("--bn_tent", action="store_true",
                   help="TENT-style BN: batch-stats forward, running buffer "
                        "disabled (pure parameter paradigm; supersedes refresh)")
    p.add_argument("--erase_prob", type=float, default=0.5)
    p.add_argument("--aug_mild", action="store_true",
                   help="mild strong-view augmentation (for models trained WITHOUT "
                        "any augmentation): tiny jitter, no erasing")
    p.add_argument("--rebalance", action="store_true",
                   help="per-class mask-normalised L_cons (pos/neg balanced)")
    p.add_argument("--cons_form", default="hard",
                   choices=["hard", "soft", "bce", "mse"],
                   help="L_cons supervision form (NOT the source model's "
                        "training loss): hard = committed thresholded "
                        "pseudo-labels ('bce' legacy alias); soft = "
                        "non-committed soft-target consistency, MSE ('mse' "
                        "legacy alias) — update volume shrinks with teacher "
                        "quality")
    p.add_argument("--drift_gate", type=float, default=0.0,
                   help=">0: safe-fallback gate — teacher decision rates "
                        "departing from the arrival state beyond this latch "
                        "to predict-only with both nets reverted to the "
                        "initial teacher (post-trigger = zero-shot)")
    p.add_argument("--drift_warm", type=int, default=10,
                   help="batches to snapshot the arrival decision distribution")
    p.add_argument("--pi_hard", action="store_true",
                   help="update pi with decision-level rate mean(1[p_t>0.5])")
    p.add_argument("--pi_floor", type=float, default=0.0,
                   help="floor pi at this ratio of the source frequency")
    p.add_argument("--pi_freeze", action="store_true",
                   help="freeze pi at the source frequency (no prior shift)")
    p.add_argument("--tau_adaptive", action="store_true",
                   help="per-class tau_pos = slow EMA of teacher-prob quantiles")
    p.add_argument("--rho", type=float, default=0.10,
                   help="target positive rate per class for adaptive tau")
    p.add_argument("--gamma_w_flag", type=float, default=None,
                   help="override adaptive class-weight gamma (0 disables w_c)")
    p.add_argument("--agree_gate", action="store_true",
                   help="pseudo-labels need teacher AND student(weak) agreement")
    p.add_argument("--proto_confirm", action="store_true",
                   help="positive pseudo-labels need prototype similarity")
    p.add_argument("--sc_confirm", action="store_true",
                   help="S_c admission upgrade, DOUBLE gate: prototype-update "
                        "samples also need cos(h_t, C_c) > tau_proto")
    p.add_argument("--sc_feat_only", action="store_true",
                   help="S_c admission = PURE feature gate (= A_c's criterion; "
                        "advisor: build S_c from A_c/C_c); drops the "
                        "probability gate; overrides --sc_confirm")
    p.add_argument("--cond_teacher_feat", action="store_true",
                   help="L_cond subset A_c membership on teacher features "
                        "(default student) — unified feature evidence for "
                        "tau unification")
    p.add_argument("--tau_proto", type=float, default=0.3)
    p.add_argument("--rho_from_pi", action="store_true",
                   help="per-class positive budget rho_c = pi^c")
    p.add_argument("--struct_norm", action="store_true",
                   help="relative Frobenius normalisation of L_struct")
    p.add_argument("--struct_form", default="frob", choices=["frob", "relfrob", "corr"],
                   help="L_struct form; corr = correlation (co-occurrence) alignment")
    p.add_argument("--struct_offdiag", action="store_true",
                   help="corr form: align only the off-diagonal co-occurrence block")
    p.add_argument("--corr_eps", type=float, default=0.001,
                   help="corr form: additive diagonal stabiliser "
                        "corr_ij = S_ij/sqrt((S_ii+eps)(S_jj+eps)); 0.001 "
                        "(advisor, default) = zero-cost stability fuse, "
                        "verified inert on VOC/COCO (exp34); 0 = legacy "
                        "clamp(1e-8)")
    p.add_argument("--conc_gate", type=float, default=0.0,
                   help="collapse gate: EMA decision concentration above this -> "
                        "stats-only batch (0.35 recommended)")
    p.add_argument("--pi_pred_correct", action="store_true",
                   help="prediction-time prior correction logit(p)+=log(pi_hat/pi_src)")
    p.add_argument("--prior_shift", type=float, default=0.0,
                   help=">0: resample test stream upsampling tail classes (lowest "
                        "25%% by source freq) x this factor (seeded, with "
                        "replacement) — real prior-shift evaluation")
    p.add_argument("--shuffle", action="store_true",
                   help="shuffle the test stream (seeded) — diagnostic control "
                        "arm: same images/labels, only arrival order changes. "
                        "NUS official stream is concept-blocked (62%% single-"
                        "concept batches); shuffling restores unbiased batch "
                        "statistics. NOT a deployment method (real streams "
                        "arrive as given) — attribution control only")
    p.add_argument("--methods", nargs="+", default=["zero", "bn", "full"])
    p.add_argument("--cache", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    # dataset-dependent defaults
    if a.dataset_name == "nuswide":
        a.num_classes = 81 if a.num_classes == 20 else a.num_classes
        a.source_dir = a.source_dir or \
            "E:/mttta/DATASETS5/nuswide_corrupted/gaussian_noise_5/NUSWIDE"
        a.test_root = a.test_root or "E:/mttta/DATASETS5/nuswide_corrupted"
        a.test_pattern = a.test_pattern or "{corr}_5/NUSWIDE"
        a.cache = a.cache or "E:/mttta/cache_tsml_nuswide_src_stats.pt"
    elif a.dataset_name == "coco":
        a.num_classes = 80 if a.num_classes == 20 else a.num_classes
        a.source_dir = a.source_dir or "E:/mttta/DATASETS4/coco2017/COCO"
        a.test_root = a.test_root or "E:/mttta/DATASETS4"
        a.test_pattern = a.test_pattern or "COCO_{corr}_severity_5/COCO"
        a.cache = a.cache or "E:/mttta/cache_tsml_coco_src_stats.pt"
    else:
        a.source_dir = a.source_dir or "E:/mttta/DATASETS2/VOCtrainval_06-Nov-2007"
        a.test_root = a.test_root or "E:/mttta/DATASETS2"
        a.test_pattern = a.test_pattern or "VOCtest_{corr}_severity_5"
        a.cache = a.cache or "E:/mttta/cache_tsml_src_stats.pt"

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    cfg = Config()
    cfg.device = a.device
    cfg.num_classes = a.num_classes
    cfg.checkpoint = a.checkpoint
    cfg.dataset_name = a.dataset_name
    cfg.source_dir, cfg.source_split = a.source_dir, a.source_split
    cfg.source_limit = a.source_limit
    cfg.test_limit = a.limit
    cfg.tta_lr = a.lr
    cfg.ema_momentum = a.ema
    cfg.tau_pos, cfg.tau_neg = a.tau_pos, a.tau_neg
    cfg.lambda_cond, cfg.lambda_struct = a.lambda_cond, a.lambda_struct
    cfg.gamma_w = a.gamma_w
    cfg.proto_tau_cos = a.proto_tau_cos
    cfg.focal, cfg.train_head = a.focal, a.train_head
    cfg.tta_batch_size = a.batch_size
    cfg.warmup_batches = a.bn_warmup
    cfg.refresh_weak = not a.no_refresh_weak
    cfg.bn_tent = a.bn_tent
    cfg.cons_rebalance = a.rebalance
    cfg.cons_form = a.cons_form
    cfg.drift_gate = a.drift_gate
    cfg.drift_warm = a.drift_warm
    cfg.pi_hard_count = a.pi_hard
    cfg.pi_floor_ratio = a.pi_floor
    cfg.pi_freeze = a.pi_freeze
    cfg.tau_adaptive = a.tau_adaptive
    cfg.tau_adaptive_rho = a.rho
    if a.gamma_w_flag is not None:
        cfg.gamma_w = a.gamma_w_flag
    cfg.agree_gate = a.agree_gate
    cfg.proto_confirm = a.proto_confirm
    cfg.sc_confirm = a.sc_confirm
    cfg.sc_feat_only = a.sc_feat_only
    cfg.cond_teacher_feat = a.cond_teacher_feat
    cfg.tau_proto = a.tau_proto
    cfg.rho_from_pi = a.rho_from_pi
    cfg.struct_norm = a.struct_norm
    cfg.struct_form = a.struct_form
    cfg.struct_offdiag = a.struct_offdiag
    cfg.conc_gate = a.conc_gate
    cfg.pi_pred_correct = a.pi_pred_correct

    print(f"device={cfg.device} arch={a.arch} ckpt={os.path.basename(a.checkpoint)}")
    model = build_real_model(a.arch, a.num_classes, a.checkpoint).to(cfg.device)

    stats = None
    if "full" in a.methods or a.prior_shift > 0:
        stats = get_source_stats(cfg, model, a.cache)

    transform = build_transform()
    if a.aug_mild:
        strong = StrongAugmentNormalized(p_erase=0.0, p_gray=0.1,
                                         scale=(0.92, 1.08), offset=0.05,
                                         channel=0.08, noise=0.02)
    else:
        strong = StrongAugmentNormalized(p_erase=a.erase_prob)
    header = f"{'corruption':<20}{'zero-shot':>10}{'bn-only':>10}{'full TTA':>10}"
    print("\n" + header + "\n" + "-" * len(header))
    rows = []
    for corr in a.corruptions:
        data_dir = os.path.join(a.test_root, a.test_pattern.format(corr=corr))
        ds = get_test_dataset(a.dataset_name, data_dir, transform, limit=a.limit)
        if a.prior_shift > 0:
            ds = ResampledStream(ds, stats["class_freq"], a.prior_shift, seed=a.seed)
            drift = (ds.true_pi - ds.source_pi).abs().mean().item()
            print(f"  [prior-shift x{a.prior_shift:g}] mean |pi_true - pi_src| = {drift:.4f}")
        gen = torch.Generator().manual_seed(a.seed) if a.shuffle else None
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=a.shuffle,
                            generator=gen, num_workers=0, pin_memory=True)
        row = {"corruption": corr}

        if "zero" in a.methods:
            z_map, zP, zY = eval_zero_shot(model, loader, cfg.device)
            row["zero"] = z_map
            ref = ZERO_SHOT_VOC.get(corr) if a.dataset_name == "voc" else None
            note = f" (ref {ref:.2f})" if ref else ""
            print(f"{corr:<20}{row['zero']:>10.3f}{note}  n={len(ds)}")
            zf1, _ = f1_at_threshold(zY, zP)
            print(f"    [zero] F1@0.5: macro={zf1:.2f}")

        if "bn" in a.methods:
            _, P, Y = eval_bn_only(model, loader, cfg.device)
            row["bn"] = mAP_official(Y, P)
            print(f"{'':<20}{'':>10}{row['bn']:>10.3f}  [BN-only]")

        if "full" in a.methods:
            tta, P, Y = eval_full_tta(cfg, model, loader, stats, strong,
                                      HEAD_PREFIXES[a.arch])
            row["full"] = mAP_official(Y, P)
            rep = tta.final_report()
            print(f"{'':<20}{'':>10}{'':>10}{row['full']:>10.3f}  [full TTA]")
            print(f"    tail-run means: " +
                  " ".join(f"{k}={v:.4f}" for k, v in rep.items()))
            # ---- decision-level metrics (visible to calibration mechanisms) ----
            f1m, f1c = f1_at_threshold(Y, P)
            f1b = f1_best_threshold(Y, P)
            tail_idx = np.argsort(stats["class_freq"].numpy())[:max(1, P.shape[1] // 4)]
            f1_tail = float(f1c[tail_idx].mean() * 100)
            f1_head = float(np.delete(f1c, tail_idx).mean() * 100)
            print(f"    F1@0.5: macro={f1m:.2f} tail={f1_tail:.2f} head={f1_head:.2f} "
                  f"| F1@best: {f1b:.2f}")
            # ---- pi tracking accuracy (needs a shifted stream) ----
            if hasattr(ds, "true_pi"):
                pi_err = (tta.pi.cpu() - ds.true_pi).abs().mean().item()
                pi_err_tail = (tta.pi.cpu() - ds.true_pi).abs()[tail_idx].mean().item()
                base_err = (ds.source_pi - ds.true_pi).abs().mean().item()
                print(f"    pi-tracking: |pi_hat-pi_true|={pi_err:.4f} "
                      f"(tail {pi_err_tail:.4f}) vs frozen baseline {base_err:.4f}")
        rows.append(row)

    # summary
    print("\n===== summary (mAP %) =====")
    print(f"{'corruption':<20}" + "".join(f"{m:>10}" for m in a.methods))
    for row in rows:
        print(f"{row['corruption']:<20}" +
              "".join(f"{row.get(m, float('nan')):>10.3f}" for m in a.methods))
    means = {m: np.nanmean([r.get(m, np.nan) for r in rows]) for m in a.methods}
    print(f"{'MEAN':<20}" + "".join(f"{means[m]:>10.3f}" for m in a.methods))


if __name__ == "__main__":
    main()
