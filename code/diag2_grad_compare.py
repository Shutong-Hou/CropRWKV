# -*- coding: utf-8 -*-
"""Gradient comparison with VDAS enabled and disabled, same network and input.

Builds FULL once and runs one forward+backward with enable_vdas=True and once
with enable_vdas=False, comparing the weight-gradient L2 norms of the earliest
convolution (stage0), of a deep encoder block and of the final convolution.
The input (a 512 lesion-biased crop) and the random seed are identical.
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
import train  # noqa
from models_registry import build_model

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_crop(seed):
    rng = random.Random(seed)
    cap = np.random.RandomState(seed)
    ids = [l.strip() for l in (train.SPLITS / "train.txt").read_text().splitlines()]
    cid = rng.choice(ids)
    img = Image.open(train.CACHE / "train" / "images" / f"{cid}.jpg").convert("RGB")
    msk = Image.open(train.CACHE / "train" / "masks" / f"{cid}.png").convert("L")
    w, h = img.size
    sz = 512
    m = np.asarray(msk) > 127
    les = np.argwhere(m)
    r = rng.uniform(0.5, 1.0)
    cw = min(w, max(sz, int(w * math.sqrt(r))))
    ch = min(h, max(sz, int(h * math.sqrt(r))))
    if rng.random() < 0.9 and len(les) > 0:
        ly, lx = les[rng.randint(0, len(les) - 1)]
        x = rng.randint(max(0, min(lx, w - cw)), min(max(0, lx), w - cw))
        y = rng.randint(max(0, min(ly, h - ch)), min(max(0, ly), h - ch))
    else:
        x = rng.randint(0, max(0, w - cw)); y = rng.randint(0, max(0, h - ch))
    img = img.crop((x, y, x + cw, y + ch)).resize((sz, sz), Image.BILINEAR)
    msk = msk.crop((x, y, x + cw, y + ch)).resize((sz, sz), Image.NEAREST)
    t = np.asarray(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    mm = ((np.asarray(msk).astype(np.float32) / 255.0) > 0.5).astype(np.float32)
    return torch.tensor(t).unsqueeze(0).to(DEV), torch.tensor(mm).unsqueeze(0).unsqueeze(0).to(DEV)


def first_conv_grad(model, stage):
    for sub in getattr(model.encoder, stage).modules():
        if isinstance(sub, torch.nn.Conv2d) and sub.weight.requires_grad:
            return sub.weight.grad
    return None


def collect(model, x, y, vdas_on):
    model.enable_vdas = vdas_on
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = train.focal_tversky(out, y)
    loss.backward()
    g_st0 = first_conv_grad(model, "stage0")
    g_st4 = None
    for sub in getattr(model.encoder, "stage4").modules():
        if isinstance(sub, torch.nn.Conv2d) and sub.weight.requires_grad:
            g_st4 = sub.weight.grad; break
    g_fin = model.final_conv.weight.grad
    r = {"loss": loss.item()}
    r["stage0_grad_l2"] = g_st0.norm().item() if g_st0 is not None else float("nan")
    r["stage0_grad_absmax"] = g_st0.abs().max().item() if g_st0 is not None else float("nan")
    r["stage4_grad_l2"] = g_st4.norm().item() if g_st4 is not None else float("nan")
    r["finalconv_grad_l2"] = g_fin.norm().item()
    if vdas_on:
        r["vdas3_key_grad_l2"] = model.vdas3.key.weight.grad.norm().item()
        r["vdas3_value_grad_l2"] = model.vdas3.value.weight.grad.norm().item()
    return r


def main():
    torch.manual_seed(7); np.random.seed(7); random.seed(7)
    model = build_model("full", img_size=512, in_channels=3, num_classes=1).to(DEV)
    model.train()
    x, y = one_crop(100)
    print(f"C-2 grad compare | FULL params={sum(p.numel() for p in model.parameters())} | dev={DEV}")
    on = collect(model, x, y, True)
    print("--- VDAS ON ---")
    for k, v in on.items():
        print(f"  {k}: {v if isinstance(v, int) else round(v, 4)}")
    off = collect(model, x, y, False)
    print("--- VDAS OFF ---")
    for k, v in off.items():
        print(f"  {k}: {v if isinstance(v, int) else round(v, 4)}")
    # ratio
    ratio = (on["stage0_grad_l2"] / max(off["stage0_grad_l2"], 1e-30))
    print(f">>> stage0_grad_l2 ON/OFF ratio = {ratio:.2f}x")


if __name__ == "__main__":
    main()