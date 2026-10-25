# -*- coding: utf-8 -*-
"""Model factory for every method compared in the benchmark.

Convention: every model returns logits of shape (B, num_classes, H, W) with
num_classes = 1 for the binary task. The CropRWKV family routes its token
mixing through the pure-PyTorch WKV scan in ``backbone_a0``.

Pretraining: where a convolutional or Transformer baseline loads ImageNet
weights, the source is stated in the factory, and the training entry point
records per run whether pretrained weights were used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# ---------------------------------------------------------------------------
# U-Net (basic semantic-segmentation baseline, classic two-path design)
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Following Ronneberger 2015. windows=[64,128,256,512], no pretraining (trained from scratch)."""
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.d1 = nn.MaxPool2d(2); self.c1 = DoubleConv(64, 128)
        self.d2 = nn.MaxPool2d(2); self.c2 = DoubleConv(128, 256)
        self.d3 = nn.MaxPool2d(2); self.c3 = DoubleConv(256, 512)
        self.d4 = nn.MaxPool2d(2); self.c4 = DoubleConv(512, 512)
        self.u1 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.c5 = DoubleConv(768, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.c6 = DoubleConv(384, 128)
        self.u3 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.c7 = DoubleConv(192, 64)
        self.u4 = nn.ConvTranspose2d(64, 64, 2, stride=2); self.c8 = DoubleConv(128, 64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e0 = self.inc(x)
        e1 = self.c1(self.d1(e0))
        e2 = self.c2(self.d2(e1))
        e3 = self.c3(self.d3(e2))
        e4 = self.c4(self.d4(e3))
        u = torch.cat([self.u1(e4), e3], dim=1); u = self.c5(u)
        u = torch.cat([self.u2(u), e2], dim=1); u = self.c6(u)
        u = torch.cat([self.u3(u), e1], dim=1); u = self.c7(u)
        u = torch.cat([self.u4(u), e0], dim=1); u = self.c8(u)
        return self.head(u)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(name: str, in_channels=3, num_classes=1, img_size=512, pretrained=False, **kw):
    name = name.lower()
    if name in ("a0", "crw-rwkv-s"):
        from backbone_a0 import create_a0
        return create_a0(img_size=img_size, in_channels=in_channels, num_classes=num_classes, **kw)
    if name in ("full", "crw-rwkv-full"):
        from backbone_full import create_full
        return create_full(img_size=img_size, in_channels=in_channels, num_classes=num_classes, **kw)
    if name == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes)
    if name in ("unet_pretrained", "deeplabv3plus", "segformer"):
        enc = {"unet_pretrained": "resnet50", "deeplabv3plus": "resnet50", "segformer": "mit_b2"}[name]
        w = "imagenet" if pretrained else None
        import segmentation_models_pytorch as smp
        arch = {"unet_pretrained": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus, "segformer": smp.Segformer}[name]
        return arch(encoder_name=enc, encoder_weights=w, classes=num_classes, in_channels=in_channels)
    if name == "swin_unet":
        from swin_unet_model import SwinUNet
        return SwinUNet(in_channels=in_channels, num_classes=num_classes, img_size=512, pretrained=pretrained)
    raise KeyError(f"unknown model '{name}'. implemented: a0, unet, full, unet_pretrained, deeplabv3plus, segformer, swin_unet")


if __name__ == "__main__":
    for n in ["a0", "unet"]:
        m = build_model(n, img_size=128)
        x = torch.randn(1, 3, 128, 128)
        print(n, "out", tuple(m(x).shape), "params_M", round(sum(p.numel() for p in m.parameters()) / 1e6, 2))