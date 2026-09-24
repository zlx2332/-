"""Datasets and (weak / strong) augmentations.

Synthetic mode is fully self-contained (no downloads): 20 classes grouped into 5
co-occurring "scenes"; each group holds one head / two mid / one tail class.
The test split applies a covariate shift (photometric) plus a prior shift
(tail classes become much more frequent), which is exactly the regime the
TTA method targets.

All datasets return raw [0,1] float tensors (C x H x W); normalisation is done
by the Weak/Strong augmenters so that teacher and student views share code.
"""
import csv as _csv
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

_GROUP = 4
# role probability inside a group: [head, mid, mid, tail]
_SRC_P = (0.90, 0.55, 0.30, 0.12)
_TGT_P = (0.72, 0.50, 0.45, 0.42)  # prior shift: tail boosted, head slightly down


# --------------------------------------------------------------------------- #
# synthetic multi-label data
# --------------------------------------------------------------------------- #
class SyntheticMultiLabel(Dataset):
    def __init__(self, num_samples, num_classes=20, shift=False, seed=0,
                 image_size=64, tpl_size=12):
        assert num_classes % _GROUP == 0
        self.n = num_samples
        self.C = num_classes
        self.shift = shift
        self.seed = seed
        self.S = image_size
        self.T = tpl_size

        g = num_classes // _GROUP
        self.group_p_src = np.full(g, 1.0 / g)
        w = np.linspace(1.6, 0.7, g)
        self.group_p_tgt = w / w.sum()  # mild scene-prior skew on target

        role_p = _TGT_P if shift else _SRC_P
        self.cls_p = np.array([role_p[c % _GROUP] for c in range(num_classes)],
                              dtype=np.float32)

        # fixed grid layout: one slot per class
        self.cols = int(math.ceil(math.sqrt(num_classes)))
        self.rows = int(math.ceil(num_classes / self.cols))
        self.cell = image_size // max(self.rows, self.cols)
        self.off = max(0, (self.cell - tpl_size) // 2)

        # per-class fixed template: dominant channel + texture -> easily learnable
        tpl = []
        for c in range(num_classes):
            rng = np.random.default_rng(1234 + c)
            base = np.zeros(3, dtype=np.float32)
            base[c % 3] = 1.0
            base[(c + 1) % 3] = 0.25
            tex = rng.uniform(0.6, 1.0, (tpl_size, tpl_size, 3)).astype(np.float32)
            tpl.append(np.clip(base[None, None, :] * tex * 0.9 + 0.1, 0.0, 1.0))
        self.tpl = np.stack(tpl).astype(np.float32)                   # C x T x T x 3
        self._pregenerate()

    def _slot(self, c):
        r, col = divmod(c, self.cols)
        return r * self.cell + self.off, col * self.cell + self.off

    def _pregenerate(self):
        """Generate the whole split up front with numpy RNG.

        Iteration-time RNG (numpy or torch, per item) deadlocks nondeterministically
        in this torch/Windows build once a model/optimizer has been created; doing
        all sampling here (before any torch module exists) and making __getitem__ a
        pure index lookup sidesteps that entirely."""
        S, T, C = self.S, self.T, self.C
        imgs = np.empty((self.n, 3, S, S), dtype=np.float32)
        labels = np.empty((self.n, C), dtype=np.float32)
        gp = self.group_p_tgt if self.shift else self.group_p_src
        for idx in range(self.n):
            rng = np.random.default_rng((self.seed + 1) * 1_000_003 + idx)
            img = 0.08 + 0.06 * rng.standard_normal((S, S, 3))
            gi = int(rng.choice(len(gp), p=gp))
            lab = np.zeros(C, dtype=np.float32)
            for c in range(gi * _GROUP, min(gi * _GROUP + _GROUP, C)):
                if rng.random() < self.cls_p[c]:
                    lab[c] = 1.0
                    y0, x0 = self._slot(c)
                    a = rng.uniform(0.75, 1.0)
                    img[y0:y0 + T, x0:x0 + T] = a * self.tpl[c] + (1 - a) * img[y0:y0 + T, x0:x0 + T]
            if self.shift:  # covariate shift: photometric + colour cast + noise
                img = img * 0.55 + 0.22 + np.array([0.10, -0.04, 0.10], dtype=np.float32)
                img = img + 0.06 * rng.standard_normal(img.shape)
            imgs[idx] = np.clip(img, 0.0, 1.0).astype(np.float32).transpose(2, 0, 1)
            labels[idx] = lab
        self.images = torch.from_numpy(imgs)
        self.labels = torch.from_numpy(labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.n


# --------------------------------------------------------------------------- #
# augmentations (tensor-based, [0,1] input -> normalised output)
# --------------------------------------------------------------------------- #
class WeakAugment:
    """Teacher view: deterministic normalisation only (the test transform)."""

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, x):
        return (x - self.mean) / self.std


class StrongAugment:
    """Student view: label-preserving photometric jitter, colour cast, noise,
    optional grayscale blending and random erasing."""

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 p_gray=0.2, p_erase=0.5):
        self.weak = WeakAugment(mean, std)
        self.p_gray = p_gray
        self.p_erase = p_erase

    def __call__(self, x):
        """Accepts B x 3 x H x W batches (per-sample params) or a single 3 x H x W."""
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        B, _, H, W = x.shape
        a = torch.empty(B, 1, 1, 1).uniform_(0.6, 1.4)
        b = torch.empty(B, 1, 1, 1).uniform_(-0.15, 0.15)
        x = x * a + b
        x = x * (1.0 + 0.25 * torch.randn(B, 3, 1, 1))
        gray = torch.rand(B, 1, 1, 1) < self.p_gray
        g = x.mean(dim=1, keepdim=True)
        x = torch.where(gray, 0.5 * x + 0.5 * g, x)
        x = x + 0.05 * torch.randn_like(x)
        if self.p_erase > 0:
            erase = torch.rand(B) < self.p_erase
            for i in range(B):
                if not erase[i]:
                    continue
                eh = max(1, int(H * float(torch.empty(()).uniform_(0.08, 0.25))))
                ew = max(1, int(W * float(torch.empty(()).uniform_(0.08, 0.25))))
                y0 = int(torch.randint(0, H - eh + 1, (1,)))
                x0 = int(torch.randint(0, W - ew + 1, (1,)))
                x[i, :, y0:y0 + eh, x0:x0 + ew] = 0.0
        x = x.clamp(0.0, 1.0)
        x = self.weak(x)
        return x.squeeze(0) if squeeze else x


class StrongAugmentNormalized:
    """Strong view for real-data pipelines whose dataset transform already
    applies Resize/ToTensor/Normalize. Operates directly on the normalised
    tensor (B x 3 x H x W, GPU ok): photometric jitter, colour cast, noise,
    per-sample random erasing. Single 3-D images are also accepted."""

    def __init__(self, p_gray=0.2, p_erase=0.5, scale=(0.7, 1.3), offset=0.15,
                 channel=0.2, noise=0.04):
        self.p_gray = p_gray
        self.p_erase = p_erase
        self.scale = scale
        self.offset = offset
        self.channel = channel
        self.noise = noise

    def __call__(self, x):
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        B, _, H, W = x.shape
        dev = x.device
        a = torch.empty(B, 1, 1, 1, device=dev).uniform_(*self.scale)
        b = torch.empty(B, 1, 1, 1, device=dev).uniform_(-self.offset, self.offset)
        x = x * a + b
        x = x * (1.0 + self.channel * torch.randn(B, 3, 1, 1, device=dev))
        gray = torch.rand(B, 1, 1, 1, device=dev) < self.p_gray
        g = x.mean(dim=1, keepdim=True)
        x = torch.where(gray, 0.5 * x + 0.5 * g, x)
        x = x + self.noise * torch.randn_like(x)
        if self.p_erase > 0:
            skip = torch.rand(B, device=x.device) < self.p_erase
            for i in range(B):
                if not skip[i]:
                    continue
                eh = max(1, int(H * float(torch.empty(()).uniform_(0.08, 0.25))))
                ew = max(1, int(W * float(torch.empty(()).uniform_(0.08, 0.25))))
                y0 = int(torch.randint(0, H - eh + 1, (1,)))
                x0 = int(torch.randint(0, W - ew + 1, (1,)))
                x[i, :, y0:y0 + eh, x0:x0 + ew] = x[i].mean()
        return x.squeeze(0) if squeeze else x


# --------------------------------------------------------------------------- #
# generic real-data loader (csv: image filename + 0/1 columns)
# --------------------------------------------------------------------------- #
class ImageCsvMultiLabel(Dataset):
    """CSV header: image,name_1,...,name_C. Cells 0/1. Paths relative to img_dir."""

    def __init__(self, csv_path, img_dir, transform=None):
        from PIL import Image  # lazy: only needed in csv mode
        self.Image = Image
        self.img_dir = img_dir
        self.transform = transform
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [r for r in _csv.reader(f) if r]
        self.names = rows[0][1:]
        self.items = [(r[0], np.array([float(v) for v in r[1:]], dtype=np.float32))
                      for r in rows[1:]]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = self.Image.open(os.path.join(self.img_dir, path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(
                np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return img, torch.from_numpy(y)


def build_csv_transform(size=128, train=True):
    import torchvision.transforms as T  # lazy
    if train:
        return T.Compose([T.Resize(size + 16), T.RandomCrop(size),
                          T.RandomHorizontalFlip(), T.ToTensor()])
    return T.Compose([T.Resize(size + 16), T.CenterCrop(size), T.ToTensor()])
