# -*- coding: utf-8 -*-
"""Swin-UNet -- transformer baseline (Swin Transformer as a U-shaped encoder-decoder).
Encoder: timm Swin Transformer (ImageNet-pretrained weights from timm), fixed input size 512
(identical training/evaluation protocol).
features_only outputs (N,H,W,C), permuted and progressively upsampled with U-Net-style skip
connections. Decoder follows the registry U-Net style (ConvTranspose + BN + ReLU + concat)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, mid_ch, up_factor=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, mid_ch, up_factor, stride=up_factor)
        self.conv = nn.Sequential(
            nn.Conv2d(mid_ch + skip_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SwinUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, img_size=512, pretrained=False,
                 model_name="swin_small_patch4_window7_224"):
        super().__init__()
        if img_size != 512:
            raise ValueError("Swin-UNet uses a fixed-resolution timm backbone; this protocol fixes the input at 512x512 (training/evaluation consistent)")
        import timm
        self.encoder = timm.create_model(model_name, pretrained=pretrained,
                                         features_only=True, out_indices=(1, 2, 3), img_size=512,
                                         pretrained_cfg_overlay={'hf_hub_id': None})
        chs = self.encoder.feature_info.channels()  # [c1,c2,c3] from shallow to deep
        c1, c2, c3 = chs[0], chs[1], chs[2]
        m2 = max(128, min(384, c2))
        m1 = max(96, min(192, c1))
        self.up3 = _DecBlock(c3, c2, m2, up_factor=2)                 # 16->32
        self.up2 = _DecBlock(m2, c1, m1, up_factor=2)                 # 32->64
        self.final = nn.Sequential(
            nn.ConvTranspose2d(m1, m1, 4, stride=2, padding=1, bias=False),  # 64->128
            nn.BatchNorm2d(m1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m1, m1 // 2, 4, stride=2, padding=1, bias=False),  # 128->256
            nn.BatchNorm2d(m1 // 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m1 // 2, m1 // 2, 4, stride=2, padding=1, bias=False),  # 256->512
            nn.BatchNorm2d(m1 // 2), nn.ReLU(inplace=True),
            nn.Conv2d(m1 // 2, num_classes, 1))
        self.num_classes = num_classes

    def forward(self, x):
        feats = self.encoder(x)               # list len=3, each (N,H,W,C)
        f0, f1, f2 = [f.permute(0, 3, 1, 2) for f in feats]  # (N,C,H,W), shallow -> deep
        u = self.up3(f2, f1)
        u = self.up2(u, f0)
        return self.final(u)