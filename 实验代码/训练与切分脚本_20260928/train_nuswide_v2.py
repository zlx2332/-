# 方案A第2步：NUS-WIDE 2x2 实验统一训练脚本
# 维度：分辨率(224/448) x 数据增强(标准增强/无增强)，其余配方与旧协议逐字一致
#   - 损失: ASL(gamma_neg=4, gamma_pos=1, clip=0.05)
#   - 优化: AdamW lr=5e-4, wd=1e-4, batch 32, AMP
#   - 标准增强: RandomResizedCrop + RandomHorizontalFlip(0.5) + ColorJitter(0.2,0.2,0.2)
#   - 无增强:   Resize((size,size), BICUBIC)（与旧 train_nuswide.py 完全一致）
# 协议改动（方案A）：
#   - 在 train_select.csv（约15.4万）上训练，每轮在 val_select.csv（8000张）上算 mAP
#   - 逐轮保存 checkpoint，并单独保存验证 mAP 最优的 {tag}_bestval.pth
#   - 官方测试集只在最后用 eval_nuswide_v2.py 评一次，不参与任何选择
# 运行示例（224 无增强组，本地 3060 可跑）：
#   E:/anaa/envs/pytorch/python.exe train_nuswide_v2.py --arch resnet50 --size 224
# 运行示例（448 组，需 AutoDL 4090，24GB 下 batch 32 可容纳）：
#   E:/anaa/envs/pytorch/python.exe train_nuswide_v2.py --arch resnet50 --size 448
import os, time, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# -------------------- ASL Loss（与旧 train_nuswide.py 逐字一致） --------------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        xs_pos = x
        xs_neg = x - self.clip
        xs_pos = torch.clamp(xs_pos, min=-50, max=50)
        loss_pos = F.binary_cross_entropy_with_logits(xs_pos, y, reduction='none')
        p_pos = torch.sigmoid(xs_pos)
        xs_neg = torch.clamp(xs_neg, min=-50, max=50)
        loss_neg = F.binary_cross_entropy_with_logits(xs_neg, y, reduction='none')
        p_neg = torch.sigmoid(xs_neg)
        pos_weight = (1 - p_pos) ** self.gamma_pos
        neg_weight = (p_neg) ** self.gamma_neg
        loss = y * pos_weight * loss_pos + (1 - y) * neg_weight * loss_neg
        if self.disable_torch_grad_focal_loss:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        return loss.mean()

# -------------------- 数据集（与旧脚本一致：按 basename 索引 Flickr） --------------------
class NUSWIDEDataset(Dataset):
    def __init__(self, csv_file, transform=None, flickr_root=r'E:\mttta\DATASETS5\nuswide\Flickr'):
        self.transform = transform
        self.samples = []
        with open(csv_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                img_path = parts[0]
                # 平台无关取文件名：CSV 里可能是 Windows 反斜杠路径，Linux 的 basename 不认
                fname = img_path.replace('\\', '/').split('/')[-1]
                label = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                self.samples.append((fname, label))
        print("正在构建文件索引...")
        self.file_index = {}
        for root, _, files in os.walk(flickr_root):
            for f in files:
                self.file_index[f] = os.path.join(root, f)
        print(f"索引建立完成，共 {len(self.file_index)} 个文件。", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        real_path = self.file_index.get(fname)
        if real_path is None:
            raise FileNotFoundError(f"在 Flickr 目录中未找到文件: {fname}")
        image = Image.open(real_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, label

# -------------------- 手动 mAP（与 eval_nuswide_full.py 一致） --------------------
def manual_mAP(targs, preds):
    aps = []
    for c in range(targs.shape[1]):
        if np.sum(targs[:, c]) == 0:
            continue
        scores = preds[:, c]
        labels = targs[:, c]
        idx = np.argsort(-scores)
        labels = labels[idx]
        tp = np.cumsum(labels)
        fp = np.cumsum(1 - labels)
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (np.sum(labels) + 1e-12)
        rec = np.concatenate(([0.], rec, [1.]))
        prec = np.concatenate(([1.], prec, [0.]))
        for i in range(len(prec)-2, -1, -1):
            prec[i] = max(prec[i], prec[i+1])
        idx = np.where(rec[1:] != rec[:-1])[0]
        ap = np.sum((rec[idx+1] - rec[idx]) * prec[idx+1])
        aps.append(ap)
    return np.mean(aps) * 100.0 if aps else 0.0

@torch.no_grad()
def evaluate(model, loader, device='cuda'):
    model.eval()
    preds, labels = [], []
    for img, lbl, _ in loader:
        out = model(img.to(device, non_blocking=True))
        preds.append(torch.sigmoid(out).float().cpu().numpy())
        labels.append(lbl.numpy())
    model.train()
    return manual_mAP(np.concatenate(labels), np.concatenate(preds))

def worker_init(worker_id):
    # Windows 下 DataLoader 使用 spawn，初始化函数必须在模块顶层
    np.random.seed(42 + worker_id)

def main():
    p = argparse.ArgumentParser(description='NUS-WIDE 2x2 (分辨率x增强) 统一训练，方案A协议')
    p.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'vit'])
    p.add_argument('--size', type=int, default=224, choices=[224, 448])
    p.add_argument('--aug', action='store_true', help='启用标准增强（不传即无增强，与旧协议一致）')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    # Windows 下多进程加载要把 26.9 万条文件索引序列化给每个子进程，启动极慢（冒烟实测卡 10 分钟），
    # 默认 0 与旧协议一致（本地 3060 实测约 148 张/秒，一轮约 18 分钟）；AutoDL/Linux 上建议 --workers 8
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--train_csv', type=str, default=r'E:\mttta\DATASETS5\nuswide\train_select.csv')
    p.add_argument('--val_csv', type=str, default=r'E:\mttta\DATASETS5\nuswide\val_select.csv')
    p.add_argument('--flickr_root', type=str, default=r'E:\mttta\DATASETS5\nuswide\Flickr')
    p.add_argument('--out_dir', type=str, default=r'E:\mttta\2x2_ckpt')
    args = p.parse_args()

    if args.arch == 'vit' and args.size != 224:
        raise SystemExit('ViT-B/16 位置编码固定 14x14，本实验仅 ResNet50 支持 448；ViT 分支只在 224 下可用')

    tag = f"{'rn' if args.arch == 'resnet50' else 'vit'}_{args.size}_{'aug' if args.aug else 'noaug'}"
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_epoch = os.path.join(args.out_dir, f'{tag}_epoch{{n}}.pth')
    ckpt_best  = os.path.join(args.out_dir, f'{tag}_bestval.pth')
    log_csv    = os.path.join(args.out_dir, f'{tag}_log.csv')

    # ---------- 复现性 ----------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = 'cuda'

    # ---------- 模型（与旧脚本一致） ----------
    if args.arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 81)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, 81)
        model = model.to(device)
        param_groups = [
            {'params': [q for n, q in model.named_parameters() if 'heads.head' not in n], 'lr': args.lr * 0.05},
            {'params': model.heads.head.parameters(), 'lr': args.lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.wd)

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    scaler = torch.amp.GradScaler('cuda')

    # ---------- 数据 ----------
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.aug:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(args.size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((args.size, args.size), interpolation=BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
    val_tf = transforms.Compose([
        transforms.Resize((args.size, args.size), interpolation=BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = NUSWIDEDataset(args.train_csv, transform=train_tf, flickr_root=args.flickr_root)
    val_ds   = NUSWIDEDataset(args.val_csv, transform=val_tf, flickr_root=args.flickr_root)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              generator=g, worker_init_fn=worker_init)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    print(f'[{tag}] 训练 {len(train_ds)} 张 / 选优验证 {len(val_ds)} 张 / '
          f'{"标准增强" if args.aug else "无增强"} / {args.size}x{args.size} / '
          f'{args.epochs} 轮 / batch {args.batch_size}', flush=True)

    best_mAP, best_ep = -1.0, -1
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_mAP', 'is_best'])

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for it, (images, _, targets) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                output = model(images)
                loss = criterion(output, targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            seen += images.size(0)
            if it % 500 == 0:
                print(f'  epoch {epoch} batch {it}/{len(train_loader)} '
                      f'loss {running/seen:.4f} 已用时 {time.time()-t0:.0f}s', flush=True)
        train_loss = running / max(seen, 1)

        val_mAP = evaluate(model, val_loader, device)
        is_best = val_mAP > best_mAP
        if is_best:
            best_mAP, best_ep = val_mAP, epoch
            torch.save(model.state_dict(), ckpt_best)
        torch.save(model.state_dict(), ckpt_epoch.format(n=epoch))
        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{train_loss:.6f}', f'{val_mAP:.2f}', int(is_best)])
        print(f'[epoch {epoch}/{args.epochs}] loss {train_loss:.4f} | 选优验证 mAP {val_mAP:.2f} | '
              f'当前最优 {best_mAP:.2f} @ epoch {best_ep} | {time.time()-t0:.0f}s', flush=True)

    print(f'\n[{tag}] 完成。最优选优验证 mAP {best_mAP:.2f}（epoch {best_ep}），'
          f'最优模型: {ckpt_best}\n逐轮轨迹: {log_csv}', flush=True)

if __name__ == '__main__':
    main()
