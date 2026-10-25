# -*- coding: utf-8 -*-
"""Test-time sliding-window evaluation and per-image metrics.

Protocol: sliding window of 512 px with stride 256 px, overlapping logits
averaged before the argmax; no test-time augmentation and no threshold tuning.
Per-image outputs (soft-Dice, soft-IoU, hard IoU/Dice, best-Dice) feed the
bootstrap confidence intervals and the paired Wilcoxon tests. Boundary metrics
are computed afterwards from the same stored per-image predictions.

Usage:
  python eval.py --ex EX-006 --ckpt results/EX-005/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from exp_harness import ExSnapshot
from models_registry import build_model

SPLITS = PROJECT_ROOT / "results" / "EX-020" / "splits"
SRC = PROJECT_ROOT / "data" / "kaggle_leafseg" / "extracted" / "data" / "data"
CACHE = PROJECT_ROOT / "data" / "kaggle_leafseg" / "cache"
CACHE_SIDE = 1024
WIN = 512
STRIDE = WIN // 2

# Dataset switch; the plantdoc entry reproduces the benchmark configuration and
# the plantseg entry is the external-validation dataset.
DATASETS = {
    "plantdoc": dict(src=PROJECT_ROOT / "data" / "kaggle_leafseg" / "extracted" / "data" / "data",
                     cache=PROJECT_ROOT / "data" / "kaggle_leafseg" / "cache",
                     splits=PROJECT_ROOT / "results" / "EX-020" / "splits"),
    "plantseg": dict(src=PROJECT_ROOT / "data" / "plantseg" / "raw" / "data",
                     cache=PROJECT_ROOT / "data" / "plantseg" / "cache",
                     splits=PROJECT_ROOT / "data" / "plantseg" / "splits"),
}


def build_cache(ids):
    (CACHE / "test" / "images").mkdir(parents=True, exist_ok=True)
    (CACHE / "test" / "masks").mkdir(parents=True, exist_ok=True)
    for imid in ids:
        ip = CACHE / "test" / "images" / f"{imid}.jpg"
        mp = CACHE / "test" / "masks" / f"{imid}.png"
        if ip.exists() and mp.exists():
            continue
        im = Image.open(SRC / "images" / f"{imid}.jpg")
        mk = Image.open(SRC / "masks" / f"{imid}.png").convert("L")
        w, h = im.size
        s = CACHE_SIDE / max(w, h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        im.resize((nw, nh), Image.BILINEAR).save(ip, "JPEG", quality=95)
        mk.resize((nw, nh), Image.NEAREST).save(mp, "PNG")


def slide_predict(model, img_t, dev):
    """img_t: (C,H,W) normalised float image. Returns the (H,W) averaged logit map from sliding-window inference."""
    H, W = img_t.shape[1], img_t.shape[2]
    acc = torch.zeros(H, W, device=dev)
    cnt = torch.zeros(H, W, device=dev)
    model.eval()
    with torch.no_grad():
        ys = range(0, max(1, H - STRIDE), STRIDE)
        xs = range(0, max(1, W - STRIDE), STRIDE)
        if (H - WIN) % STRIDE != 0 or (H <= WIN):
            pass
        # cover the bottom-right corner
        ys = list(ys) + ([max(0, H - WIN)] if max(0, H - WIN) not in ys else [])
        xs = list(xs) + ([max(0, W - WIN)] if max(0, W - WIN) not in xs else [])
        for y in ys:
            for x in xs:
                crop = img_t[:, y:y + WIN, x:x + WIN].unsqueeze(0)
                if crop.shape[2] != WIN or crop.shape[3] != WIN:
                    continue
                logit = model(crop)[0, 0]  # (WIN,WIN)
                acc[y:y + WIN, x:x + WIN] += logit
                cnt[y:y + WIN, x:x + WIN] += 1.0
    avg = acc / cnt.clamp(min=1)
    return avg


def metrics(pred, gt):
    """pred: continuous prediction map (logits); gt: 0/1. Returns hard-threshold metrics at 0.5 and threshold-free soft metrics."""
    pred = torch.sigmoid(pred)  # logits -> probabilities in [0,1]; soft and hard metrics both use probabilities
    p = (pred > 0.5).float()
    tp = (p * gt).sum().float(); fp = (p * (1 - gt)).sum().float(); fn = ((1 - p) * gt).sum().float()
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    # threshold-free soft metrics (primary): continuous p in [0,1]
    spa = (pred * gt).sum().float()
    sdice = (2 * spa + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
    siou = (spa + 1e-6) / (pred.sum() + gt.sum() - spa + 1e-6)
    # best-threshold Dice (secondary, reported explicitly)
    best_dice, best_th = 0.0, 0.5
    for th in [x / 20 for x in range(1, 20)]:
        pt = (pred > th).float()
        i = (pt * gt).sum(); u = (pt + gt).clamp(max=1).sum(); d = (2 * i + 1e-6) / (2 * i + (pt * (1 - gt)).sum() + ((1 - pt) * gt).sum() + 1e-6)
        if float(d) > best_dice:
            best_dice = float(d); best_th = float(th)
    return {"iou05": float(iou), "dice05": float(dice),
            "soft_iou": float(siou), "soft_dice": float(sdice),
            "best_dice": best_dice, "best_th": best_th}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", default="EX-006")
    ap.add_argument("--tag", default="a0_test_eval")
    ap.add_argument("--model", default="a0")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--res_side", type=int, default=0, help="unseen test resolution: rescale the image to this long side before the sliding window; 0 = CACHE_SIDE")
    ap.add_argument("--embed_dims", default="", help="capacity scaling: comma-separated stage widths; empty = default")
    ap.add_argument("--vdas_mode", default="gated", help="scan ablation: gated / h_only / v_only / equal")
    ap.add_argument("--no_vdas", action="store_true", help="evaluate with VDAS removed")
    ap.add_argument("--no_mpsa", action="store_true", help="evaluate with MPSA removed")
    ap.add_argument("--no_feem", action="store_true", help="evaluate with FEEM removed")
    ap.add_argument("--feem_all", action="store_true", help="evaluate with FEEM at all stages")
    ap.add_argument("--dataset", default="plantdoc", choices=sorted(DATASETS),
                    help="plantdoc (benchmark) or plantseg (external validation)")
    args = ap.parse_args()
    global SRC, CACHE, SPLITS
    _d = DATASETS[args.dataset]
    SRC, CACHE, SPLITS = _d["src"], _d["cache"], _d["splits"]
    print(f"[{args.ex}] dataset={args.dataset} cache={CACHE}", flush=True)
    snap = ExSnapshot(args.ex, tag=args.tag, root=PROJECT_ROOT / "results")
    snap.set(ckpt=str(args.ckpt), img_size=args.img_size, postprocess="sliding_window_stride_256_argmax")

    test_ids = [l.strip() for l in (SPLITS / "test.txt").read_text().splitlines() if l.strip()]
    print(f"[{args.ex}] test ids = {len(test_ids)}", flush=True)
    build_cache(test_ids)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _mk_eval = dict(img_size=args.img_size, in_channels=3, num_classes=1)
    if args.model in ("full", "crw-rwkv-full"):
        _mk_eval["vdas_stop_grad"] = True
        _mk_eval["vdas_mode"] = args.vdas_mode
        _mk_eval["enable_vdas"] = not args.no_vdas
        _mk_eval["enable_mpsa"] = not args.no_mpsa
        _mk_eval["enable_feem"] = not args.no_feem
        _mk_eval["feem_all"] = args.feem_all
        if args.embed_dims:
            _mk_eval["embed_dims"] = tuple(int(x) for x in args.embed_dims.split(",") if x.strip())
    model = build_model(args.model, **_mk_eval).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"])
    model.eval()

    rows = ["image_id,iou05,dice05,soft_iou,soft_dice,best_dice,best_th"]
    agg = {"iou05": [], "dice05": [], "soft_iou": [], "soft_dice": [], "best_dice": []}
    for imid in test_ids:
        _imPil = Image.open(CACHE / "test" / "images" / f"{imid}.jpg").convert("RGB")
        _mkPil = Image.open(CACHE / "test" / "masks" / f"{imid}.png").convert("L")
        if args.res_side and args.res_side > 0:  # unseen test resolution
            _s = args.res_side / max(_imPil.size)
            _imPil = _imPil.resize((max(1, int(round(_imPil.size[0] * _s))), max(1, int(round(_imPil.size[1] * _s)))), Image.BILINEAR)
            _mkPil = _mkPil.resize(_imPil.size, Image.NEAREST)
        img = np.asarray(_imPil)
        gt = ((np.asarray(_mkPil) > 127).astype(np.float32))
        t = torch.from_numpy(img.transpose(2, 0, 1) / 255.0).float().to(dev)
        logit = slide_predict(model, t, dev)
        gg = torch.from_numpy(gt).float().to(dev)
        m = metrics(logit, gg)
        rows.append(f"{imid},{m['iou05']:.6f},{m['dice05']:.6f},{m['soft_iou']:.6f},{m['soft_dice']:.6f},{m['best_dice']:.6f},{m['best_th']:.3f}")
        for k in agg:
            agg[k].append(m[k])
        print(f"  {imid}: IoU@.5={m['iou05']:.3f} softDice={m['soft_dice']:.3f} softIoU={m['soft_iou']:.3f} bestDice={m['best_dice']:.3f}", flush=True)
    (snap.dir / "per-image_metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("aggregate:", {k: f"{np.mean(v):.4f}±{np.std(v)/math.sqrt(len(v)):.4f}" for k, v in agg.items()})
    for k, v in agg.items():
        snap.set(**{f"test_{k}_mean": float(np.mean(v)), f"test_{k}_se": float(np.std(v) / math.sqrt(len(v)))})
    snap.set(n_test=len(test_ids))
    snap.write_meta("SUCCESS")
    snap.write_manifest(title=f"{args.ex} {args.tag}")
    print(f"[{args.ex}] done. outputs={snap.dir}")


if __name__ == "__main__":
    main()