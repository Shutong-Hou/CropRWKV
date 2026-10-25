# -*- coding: utf-8 -*-
"""The three purpose-built modules of the CropRWKV segmentation network.

M1  VDAS  Vertical-directional adaptive scan. The feature map is scanned in
          row-major and column-major raster order; each raster sequence is
          processed by one WKV scan and the two outputs are fused by a
          global-pooled two-way gate.
M2  FEEM  Feature edge-enhancement module. A fixed Sobel edge map is computed
          from the feature itself and used to gate a 1x1 refinement of that
          feature. Ground truth is never touched.
M3  MPSA  Multi-scale pooled spatial attention. Dilated depthwise
          convolutions at rates {1,2,3} are concatenated, fused by a 1x1
          convolution and re-weighted by a squeeze-and-excitation gate.

All three depend only on torch and on the pure-PyTorch WKV scan in
``wkv_pure.py``; everything is differentiable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# M1  VDAS
# ---------------------------------------------------------------------------
class VDAS(nn.Module):
    """Dual-raster WKV scan over a (B, C, H, W) feature map, gated fusion.

    Each instance owns its own decay, first-token bonus, key/value projections
    and gate, so the two encoder positions that carry a VDAS block are
    parameterised independently.
    """

    def __init__(self, C, normalize=0.05, decay_init=1.0, scan_mode="gated"):
        super().__init__()
        self.C = C
        # Per-channel log-decay. Initialised away from zero so that the
        # effective decay w = decay / T is non-zero: with w = 0 the scan
        # accumulates an undamped prefix sum, which produces heavy-tailed
        # activations and inflated gradients.
        self.decay = nn.Parameter(torch.full((C,), float(decay_init)))
        self.first = nn.Parameter(torch.zeros(C))   # first-token bonus
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.norm = nn.LayerNorm(C)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(C, 2), nn.Softmax(dim=1))
        self.scan_mode = str(scan_mode)   # gated | h_only | v_only | equal

    def _scan(self, x, orient):
        B, C, H, W = x.shape
        if orient == "h":          # row-major raster
            z = x.reshape(B, C, H * W).transpose(1, 2)      # (B, T, C)
            shape_back = (B, C, H, W)
        else:                      # column-major raster
            zv = x.transpose(2, 3)                         # (B, C, W, H)
            z = zv.reshape(B, C, W * H).transpose(1, 2)
            shape_back = (B, C, H, W)
        k = self.key(z)
        v = self.value(z)
        T = z.size(1)
        w = (self.decay / T).float()
        u = (self.first / T).float()
        y = _wkv(w, u, k, v).transpose(1, 2).reshape(shape_back)
        return y.view(B, C, H, W)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.scan_mode == "h_only":                      # horizontal raster only
            out = self._scan(x, "h")
        elif self.scan_mode == "v_only":                    # vertical raster only
            out = self._scan(x, "v")
        elif self.scan_mode == "equal":                     # H + V, equal weights
            yh = self._scan(x, "h")
            yv = self._scan(x, "v")
            out = 0.5 * yh + 0.5 * yv
        else:                                               # H + V, adaptive gate
            yh = self._scan(x, "h")
            yv = self._scan(x, "v")
            al = self.gate(x)
            out = al[:, 0:1, None, None] * yh + al[:, 1:2, None, None] * yv
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out).permute(0, 3, 1, 2)
        return out


def _wkv(w, u, k, v):
    from wkv_pure import run as _wrun
    return _wrun(w, u, k, v)


# ---------------------------------------------------------------------------
# M2  FEEM
# ---------------------------------------------------------------------------
class FEEM(nn.Module):
    """Sobel edge enhancement: F' = F + sigmoid(conv1x1([F;E])) * tanh(conv1x1([F;E]))."""

    def __init__(self, C):
        super().__init__()
        gx = torch.tensor([[-1., 0, 1], [-2., 0, 2], [-1., 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        gy = torch.tensor([[1., 2, 1], [0., 0, 0], [-1., -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobelx", gx)
        self.register_buffer("sobely", gy)
        self.conv_fuse = nn.Conv2d(C * 2, C, 1)   # [F; E] -> C
        self.conv_gate = nn.Conv2d(C * 2, C, 1)   # [F; E] -> C, gate
        self.act = nn.Sigmoid()

    def forward(self, f):
        ex = F.conv2d(f, torch.cat([self.sobelx] * f.size(1), dim=0).abs(),
                      padding=1, groups=f.size(1))
        ey = F.conv2d(f, torch.cat([self.sobely] * f.size(1), dim=0),
                      padding=1, groups=f.size(1))
        e = torch.sqrt(ex ** 2 + ey ** 2 + 1e-8)
        fe = torch.cat([f, e], dim=1)
        g = self.act(self.conv_gate(fe))
        return f + g * torch.tanh(self.conv_fuse(fe))


# ---------------------------------------------------------------------------
# M3  MPSA
# ---------------------------------------------------------------------------
class MPSA(nn.Module):
    """Dilated depthwise convs at rates {1,2,3} -> concat -> 1x1 -> SE -> residual."""

    def __init__(self, C, r=4):
        super().__init__()
        self.dws = nn.ModuleList([nn.Conv2d(C, C, 3, padding=d, dilation=d, groups=C, bias=False)
                                  for d in (1, 2, 3)])
        self.fuse = nn.Conv2d(C * 3, C, 1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(C, C // r, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(C // r, C, 1), nn.Sigmoid())

    def forward(self, x):
        m = torch.cat([dw(x) for dw in self.dws], dim=1)
        m = self.fuse(m)
        g = self.se(m)
        return x + m * g
