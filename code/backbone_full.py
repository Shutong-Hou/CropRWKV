# -*- coding: utf-8 -*-
"""CropRWKV FULL: the A0 backbone plus VDAS (encoder stages 3 and 4),
FEEM (skip paths of encoder stages 1 and 2) and MPSA (bottleneck).

Pure PyTorch; reuses the encoder, decoder and CCM band of ``backbone_a0``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
REPO = PROJECT_ROOT / "third_party" / "RWKV-UNet"
for _p in (str(REPO), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch.utils.cpp_extension as _cppe
import wkv_pure
_cppe.load = lambda *a, **k: None

import rwkv_unet as _ru
from ccm.ccm import CCMix
from modules import VDAS, FEEM, MPSA


def _run_pure(B, T, C, w, u, k, v):
    return wkv_pure.run(w, u, k, v)


_ru.RUN_CUDA = _run_pure


class CropRWKV_FULL(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, img_size=512,
                 depths=(2, 2, 2, 2), embed_dims=(64, 128, 256, 512),
                 exp_ratios=(1.5, 1.5, 1.5, 2.0),
                 norm_layers=("bn_2d", "bn_2d", "ln_2d", "ln_2d"),
                 act_layers=("silu", "silu", "gelu", "gelu"),
                 dw_kss=(5, 5, 5, 5), attn_ss=(False, False, True, True),
                 drop_path=0.05, enable_mpsa=True, enable_vdas=True, enable_feem=True,
                 vdas_residual_weight=1.0, vdas_stop_grad=False, vdas_mode="gated", feem_all=False):
        super().__init__()
        self.enable_mpsa = enable_mpsa
        self.enable_vdas = enable_vdas
        self.enable_feem = enable_feem
        self.vdas_rw = float(vdas_residual_weight)
        self.vdas_sg = bool(vdas_stop_grad)
        self.embed_dims = list(embed_dims)
        self.encoder = _ru.RWKV_UNet_encoder(
            dim_in=in_channels, num_classes=1000, img_size=img_size,
            depths=list(depths), stem_dim=24, embed_dims=list(embed_dims),
            exp_ratios=list(exp_ratios), norm_layers=list(norm_layers),
            act_layers=list(act_layers), dw_kss=list(dw_kss), attn_ss=list(attn_ss),
            drop=0.0, drop_path=drop_path)
        self.vdas3 = VDAS(embed_dims[2], scan_mode=vdas_mode)
        self.vdas4 = VDAS(embed_dims[3], scan_mode=vdas_mode)
        self.feem2 = FEEM(embed_dims[1])   # stage2 skip
        self.feem1 = FEEM(embed_dims[0])   # stage3 skip
        self.feem_all = bool(feem_all)
        if self.feem_all:                  # ablation: FEEM at all four stages
            self.feem3 = FEEM(embed_dims[2])   # stage4 (enc3)
            self.feem4 = FEEM(embed_dims[3])   # bottleneck (enc4)
        self.mpsa = MPSA(embed_dims[3])
        self.ccm = CCMix([embed_dims[2], embed_dims[1], embed_dims[0]], embed_dims[0], img_size // 2)
        Up = _ru.UpBlock
        self.decoder1 = Up(embed_dims[3], embed_dims[2], norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0)
        self.decoder2 = Up(embed_dims[2] * 2, embed_dims[1], norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0)
        self.decoder3 = Up(embed_dims[1] * 2, embed_dims[0], norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0)
        self.decoder4 = Up(embed_dims[0] * 2, 24, norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0)
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        for blk in self.encoder.stage0:
            x = blk(x)
        for blk in self.encoder.stage1:
            x = blk(x)
        enc1 = x
        for blk in self.encoder.stage2:
            x = blk(x)
        enc2 = x
        for blk in self.encoder.stage3:
            x = blk(x)
        enc3 = x
        for blk in self.encoder.stage4:
            x = blk(x)
        # Module placement: VDAS (enc3, enc4), FEEM (enc1, enc2), MPSA (bottleneck = x)
        # The enable_* flags allow individual modules to be disabled (all enabled by default)
        if self.enable_vdas:
            if self.vdas_sg:   # stop-gradient on the VDAS input, removing its gradient amplification into the encoder
                v3 = self.vdas3(enc3.detach()); v4 = self.vdas4(x.detach())
            else:
                v3 = self.vdas3(enc3); v4 = self.vdas4(x)
            enc3 = enc3 + self.vdas_rw * v3
            x = x + self.vdas_rw * v4
        if self.enable_mpsa:
            x = self.mpsa(x)
        if self.enable_feem:
            enc2 = self.feem2(enc2)
            enc1 = self.feem1(enc1)
            if self.feem_all:
                enc3 = self.feem3(enc3)
                x = self.feem4(x)
        enc3, enc2, enc1 = self.ccm([enc3, enc2, enc1])
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        return self.final_conv(dec0)


def create_full(img_size=512, num_classes=1, in_channels=3, **kw) -> CropRWKV_FULL:
    return CropRWKV_FULL(in_channels=in_channels, num_classes=num_classes, img_size=img_size, **kw)