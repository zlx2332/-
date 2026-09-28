import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

from data.voc_custom import VOCCustom

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

# ================== ASL Loss ==================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # x: logits [B, C], y: multi-hot labels [B, C]
        xs_pos = x
        xs_neg = x - self.clip

        xs_pos = torch.clamp(xs_pos, min=-50, max=50)
        xs_neg = torch.clamp(xs_neg, min=-50, max=50)

        loss_pos = F.binary_cross_entropy_with_logits(xs_pos, y, reduction='none')
        p_pos = torch.sigmoid(xs_pos)
        pos_weight = (1 - p_pos) ** self.gamma_pos

        loss_neg = F.binary_cross_entropy_with_logits(xs_neg, y, reduction='none')
        p_neg = torch.sigmoid(xs_neg)
        neg_weight = p_neg ** self.gamma_neg

        loss = y * pos_weight * loss_pos + (1 - y) * neg_weight * loss_neg
        return loss.mean()
# ==============================================

def main():
    parser = argparse.ArgumentParser(description="Train VOC 2007 model with ASL")
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'vit'])
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--data', type=str, default='./DATASETS2/VOCtrainval_06-Nov-2007/VOCdevkit',
                        help='VOCdevkit 根目录')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    # ---------- 模型 ----------
    if args.arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 20)
        model = model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, 20)
        model = model.cuda()
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'heads.head' not in n],
             'lr': args.lr * 0.05},
            {'params': model.heads.head.parameters(), 'lr': args.lr}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    scaler = torch.amp.GradScaler('cuda')   # 使用新版 API

    # ---------- 数据 ----------
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=BICUBIC),
        transforms.ToTensor(),
        normalize
    ])

    # VOCCustom(split, root, transform)
    # root 应指向 VOCdevkit 目录，如 ./DATASETS2/VOCtrainval_06-Nov-2007/VOCdevkit
    train_dataset = VOCCustom('trainval', args.data, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    print(f"训练集: {len(train_dataset)} 张 (VOC2007 trainval)")

    # ---------- 训练 ----------
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, _, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            with torch.amp.autocast('cuda'):
                output = model(images)
                loss = criterion(output, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | 训练 Loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), f"{args.arch}_voc_best.pth")
    print(f"训练完成，模型已保存为 {args.arch}_voc_best.pth")

if __name__ == '__main__':
    main()