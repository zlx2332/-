"""Wrap torchvision backbones (resnet50 / vit_b_16) for the TTA framework.

Subclasses the torchvision modules directly so parameter names stay unchanged
('conv1.*' ... 'fc.*' / 'heads.head.*') and the checkpoints in E:/mttta load
with strict=True. Checkpoints may be raw state_dicts or wrapped in
{'model': ...} / {'state_dict': ...} with an optional 'module.' prefix.

forward(x, return_features=True) -> (logits, h); h is the GAP embedding
(resnet) / class-token embedding (ViT), used for the class prototypes.
"""
import copy

import torch
from torchvision.models import resnet as tvr
from torchvision.models import vision_transformer as tvv

HEAD_PREFIXES = {"resnet50": ("fc.",), "vit": ("heads.",)}


def load_checkpoint(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model


class ResNet50MultiLabel(tvr.ResNet):
    def __init__(self, num_classes):
        super().__init__(block=tvr.Bottleneck, layers=[3, 4, 6, 3],
                         num_classes=num_classes)
        self.feature_dim = 2048
        self.num_classes = num_classes

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

    def forward(self, x, return_features=False):
        h = self.extract_features(x)
        logits = self.fc(h)
        return (logits, h) if return_features else logits


class ViTB16MultiLabel(tvv.VisionTransformer):
    def __init__(self, num_classes):
        super().__init__(image_size=224, patch_size=16, num_layers=12, num_heads=12,
                         hidden_dim=768, mlp_dim=3072, num_classes=num_classes)
        self.feature_dim = 768
        self.num_classes = num_classes

    def extract_features(self, x):
        x = self._process_input(x)
        cls = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.encoder.pos_embedding
        x = self.encoder.layers(x)
        x = self.encoder.ln(x)
        return x[:, 0]                                # class token

    def forward(self, x, return_features=False):
        h = self.extract_features(x)
        logits = self.heads.head(h)
        return (logits, h) if return_features else logits


def build_real_model(arch, num_classes, checkpoint=None):
    if arch == "resnet50":
        model = ResNet50MultiLabel(num_classes)
    elif arch == "vit":
        model = ViTB16MultiLabel(num_classes)
    else:
        raise ValueError(f"unknown arch: {arch}")
    if checkpoint:
        load_checkpoint(model, checkpoint)
    return model


def clone_model(model):
    return copy.deepcopy(model)
