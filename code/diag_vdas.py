# -*- coding: utf-8 -*-
"""Locate the mechanism by which VDAS prevents FULL from converging.

Runs one forward pass of FULL on real training data with forward hooks that
observe:
  1) mean / std / abs-max of the VDAS3 and VDAS4 outputs (to separate
     explosion, collapse and normal behaviour);
  2) the gate coefficients alpha;
  3) after one backward pass, the gradient norms of the VDAS sub-parameters
     against those of the early convolutions (to detect vanishing gradients).
A single batch, no training; the hooks return None and do not alter outputs.
"""
from __future__ import annotations

import sys
import random
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: reuse data/loss (main is guarded by __name__)
from models_registry import build_model

CACHE = train.CACHE
IMG = 512
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_crop():
    ids = [l.strip() for l in (train.SPLITS / "train.txt").read_text().splitlines()]
    cid = random.choice(ids)
    img = Image.open(CACHE / "train" / "images" / f"{cid}.jpg").convert("RGB")
    msk = Image.open(CACHE / "train" / "masks" / f"{cid}.png").convert("L")
    w, h = img.size
    sz = IMG
    scale = (0.5, 1.0)
    m_arr = np.asarray(msk) > 127
    les = np.argwhere(m_arr)
    r = random.uniform(*scale)
    cw = min(w, max(sz, int(w * math.sqrt(r))))
    ch = min(h, max(sz, int(h * math.sqrt(r))))
    bi = 0.9
    if random.random() < bi and len(les) > 0:
        ly, lx = les[random.randint(0, len(les) - 1)]
        x = random.randint(max(0, min(lx, w - cw)), min(max(0, lx), w - cw))
        y = random.randint(max(0, min(ly, h - ch)), min(max(0, ly), h - ch))
    else:
        x = random.randint(0, max(0, w - cw)); y = random.randint(0, max(0, h - ch))
    img = img.crop((x, y, x + cw, y + ch)).resize((sz, sz), Image.BILINEAR)
    msk = msk.crop((x, y, x + cw, y + ch)).resize((sz, sz), Image.NEAREST)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT); msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
    t = np.asarray(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    m = ((np.asarray(msk).astype(np.float32) / 255.0) > 0.5).astype(np.float32)
    return torch.tensor(t).unsqueeze(0).to(DEV), torch.tensor(m).unsqueeze(0).unsqueeze(0).to(DEV)


def stats(name, t):
    if not isinstance(t, torch.Tensor):
        print(f"  {name}: (non-tensor)"); return
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(f"  {name}: NaN/Inf!"); return
    print(f"  {name}: shape={tuple(t.shape)} mean={t.mean().item():.4f} std={t.std().item():.4f} absmax={t.abs().max().item():.5f}")


def grad_stats(name, p):
    g = p.grad
    if g is None:
        print(f"  grad[{name}]: None (not involved) "); return
    print(f"  grad[{name}]: absmax={g.abs().max().item():.2e} l2={g.norm().item():.3e} nonzero={g.count_nonzero().item()} NaN={bool(torch.isnan(g).any())}")


def main():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = build_model("full", img_size=IMG, in_channels=3, num_classes=1).to(DEV)
    model.train()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"FULL(default all on) params={nparams} dev={DEV}")

    def _mk_hook(nm):
        def _hook(mod, i, o):
            stats(f"{nm}.OUT", o)
            a = mod.gate(i[0]).detach().cpu().numpy().round(3)
            print(f"  {nm}.alpha(h/v)={a}")
        return _hook

    hooks = []
    for nm in ["vdas3", "vdas4"]:
        hooks.append(getattr(model, nm).register_forward_hook(_mk_hook(nm)))

    x, y = one_crop()
    out = model(x)
    for h in hooks:
        h.remove()
    loss = train.focal_tversky(out, y)
    print(f"loss={loss.item():.4f}")

    # feature magnitudes before and after the residual
    with torch.no_grad():
        # re-run the forward pass to collect values around enc3 -- the hooks already print OUT;
        # record the raw encoder magnitudes here
        pass

    loss.backward()

    print("--- VDAS sub-parameter gradients ---")
    for nm in ["vdas3", "vdas4"]:
        m = getattr(model, nm)
        grad_stats(f"{nm}.decay", m.decay)
        grad_stats(f"{nm}.first", m.first)
        grad_stats(f"{nm}.key.w", m.key.weight)
        grad_stats(f"{nm}.value.w", m.value.weight)
        grad_stats(f"{nm}.norm.w", m.norm.weight)
        grad_stats(f"{nm}.gate.2.w", m.gate[2].weight)
    print("--- reference: early/late-layer parameter gradients ---")

    def first_conv(mods):
        for m in mods:
            for sub in m.modules():
                if isinstance(sub, torch.nn.Conv2d) and sub.weight.requires_grad:
                    return sub.weight
        return None

    f1 = first_conv(model.encoder.stage0)
    if f1 is not None:
        grad_stats("stage0.conv(w)", f1)
    grad_stats("final_conv(w)", model.final_conv.weight)


if __name__ == "__main__":
    main()