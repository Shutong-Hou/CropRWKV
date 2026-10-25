# -*- coding: utf-8 -*-
"""A0 baseline: the bare RWKV-UNet backbone with the pure-PyTorch WKV scan.

Notes:
  1. The encoder, up-blocks and channel cross-mixing band are reused from the
     public RWKV-UNet implementation in ``third_party/RWKV-UNet``.
  2. ``rwkv_unet.py`` compiles a CUDA WKV extension at import time. A shim
     replaces ``torch.utils.cpp_extension.load`` with a no-op so that nothing is
     compiled, and ``RUN_CUDA`` is bound to ``wkv_pure.run``. The public CUDA
     kernel implements a different operator (exponential key weights) and is
     therefore not used here; see the manuscript's limitations section.
  3. Channel widths [64, 128, 256, 512], two IR-RWKV blocks per stage, three
     input channels and a single binary output channel.
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

# ---- shim: intercept the CUDA WKV compilation ----
_ORIG_LOAD = _cppe.load
_cppe.load = lambda *a, **k: None          # no-op, nothing is compiled

import rwkv_unet as _ru       # noqa: E402  (WKV is redirected via RUN_CUDA)
from ccm.ccm import CCMix     # noqa: E402


def _run_pure(B, T, C, w, u, k, v):
    """Replaces rwkv_unet.RUN_CUDA with the pure-PyTorch scan."""
    return wkv_pure.run(w, u, k, v)


_ru.RUN_CUDA = _run_pure
UpBlock = _ru.UpBlock


class RWKVUNetA0(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 512,
        depths=(2, 2, 2, 2),
        embed_dims=(64, 128, 256, 512),
        exp_ratios=(1.5, 1.5, 1.5, 2.0),
        norm_layers=("bn_2d", "bn_2d", "ln_2d", "ln_2d"),
        act_layers=("silu", "silu", "gelu", "gelu"),
        dw_kss=(5, 5, 5, 5),
        attn_ss=(False, False, True, True),
        drop_path: float = 0.05,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dims = list(embed_dims)
        self.encoder = _ru.RWKV_UNet_encoder(
            dim_in=in_channels, num_classes=1000, img_size=img_size,
            depths=list(depths), stem_dim=24, embed_dims=list(embed_dims),
            exp_ratios=list(exp_ratios), norm_layers=list(norm_layers),
            act_layers=list(act_layers), dw_kss=list(dw_kss),
            attn_ss=list(attn_ss), drop=0.0, drop_path=drop_path)
        self.ccm = CCMix([self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]],
                         self.embed_dims[0], img_size // 2)
        self.decoder1 = UpBlock(self.embed_dims[3], self.embed_dims[2], norm_in=False, has_skip=False,
                                exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0)
        self.decoder2 = UpBlock(self.embed_dims[2] * 2, self.embed_dims[1], norm_in=False, has_skip=False,
                                exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0)
        self.decoder3 = UpBlock(self.embed_dims[1] * 2, self.embed_dims[0], norm_in=False, has_skip=False,
                                exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0)
        self.decoder4 = UpBlock(self.embed_dims[0] * 2, 24, norm_in=False, has_skip=False,
                                exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0)
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
        enc3, enc2, enc1 = self.ccm([enc3, enc2, enc1])
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        out = self.final_conv(dec0)
        return out


def create_a0(img_size=512, num_classes=1, in_channels=3, **kw) -> RWKVUNetA0:
    return RWKVUNetA0(in_channels=in_channels, num_classes=num_classes,
                      img_size=img_size, **kw)


if __name__ == "__main__":
    m = create_a0(img_size=128, num_classes=1, in_channels=3)
    x = torch.randn(1, 3, 128, 128)
    torch.manual_seed(42)
    y = m(x)
    print("A0 forward ok:", tuple(y.shape))
    n = sum(p.numel() for p in m.parameters())
    print("params:", n)