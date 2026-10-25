# -*- coding: utf-8 -*-
"""Field-degradation robustness and boundary metrics.

Design (kept identical across models so that the comparison is fair):
  * reuses slide_predict / metrics from eval.py, i.e. exactly the inference
    protocol of the main experiments (sliding window 512/256, mean of
    overlapping logits, no test-time augmentation);
  * degradations are applied to the inference input only, the ground truth is
    unchanged, and every condition runs over all 60 test images;
  * both overlap metrics and boundary metrics are written out (Boundary IoU,
    BF-score, boundary band 2% of the image diagonal);
  * per-condition and per-image values are stored for verification.

Usage:
  python experiments/exp_degradation_boundary.py --ex EX-163 --tag degbnd_full \\
      --model full --ckpt results/EX-032/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE = PROJECT_ROOT / "code"
for _p in (str(PROJECT_ROOT), str(CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from PIL import Image
from scipy import ndimage

from exp_harness import ExSnapshot
from models_registry import build_model
from eval import slide_predict, metrics, build_cache, SPLITS, CACHE


# ---------------- degradation conditions (simulating field-acquisition variation) ----------------
def deg_clean(a):
    return a


def deg_blur2(a):
    return np.clip(ndimage.gaussian_filter(a.astype(np.float32), (2.0, 2.0, 0)), 0, 255)


def deg_blur4(a):
    return np.clip(ndimage.gaussian_filter(a.astype(np.float32), (4.0, 4.0, 0)), 0, 255)


def deg_motion(a):
    k = np.zeros((1, 9), np.float32); k[0, :] = 1.0 / 9.0
    out = np.stack([ndimage.convolve(a[..., c].astype(np.float32), k, mode="nearest") for c in range(3)], -1)
    return np.clip(out, 0, 255)


def deg_jpeg30(a):
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(buf, "JPEG", quality=30)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def deg_dark(a):
    return np.clip(a.astype(np.float32) * 0.5, 0, 255)


def deg_bright(a):
    return np.clip(a.astype(np.float32) * 1.4 + 15, 0, 255)


def deg_lowcontrast(a):
    m = a.astype(np.float32).mean()
    return np.clip((a.astype(np.float32) - m) * 0.5 + m, 0, 255)


def deg_colorshift(a):
    f = a.astype(np.float32).copy()
    f[..., 0] *= 1.12; f[..., 2] *= 0.88
    return np.clip(f, 0, 255)


def deg_illum_gradient(a):
    h, w = a.shape[:2]
    ramp = np.linspace(1.25, 0.6, w, dtype=np.float32)[None, :, None]
    return np.clip(a.astype(np.float32) * ramp, 0, 255)


def deg_noise(a, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(a.astype(np.float32) + rng.normal(0, 10.0, a.shape), 0, 255)


def deg_occlusion(a, seed=0, frac=0.10):
    out = a.copy()
    h, w = out.shape[:2]
    rng = np.random.default_rng(seed)
    target, done = int(frac * h * w), 0
    while done < target:
        s = int(0.06 * max(h, w))
        y = int(rng.integers(0, max(1, h - s))); x = int(rng.integers(0, max(1, w - s)))
        out[y:y + s, x:x + s] = 128
        done += s * s
    return out


CONDITIONS = [
    ("clean", deg_clean), ("blur_sigma2", deg_blur2), ("blur_sigma4", deg_blur4),
    ("motion_blur_9", deg_motion), ("jpeg_q30", deg_jpeg30), ("low_light_0.5x", deg_dark),
    ("overexposed_1.4x", deg_bright), ("low_contrast_0.5x", deg_lowcontrast),
    ("color_shift", deg_colorshift), ("illum_gradient", deg_illum_gradient),
    ("gauss_noise_s10", deg_noise), ("occlusion_10pct", deg_occlusion),
]
SEEDED = {"gauss_noise_s10", "occlusion_10pct"}


# ---------------- boundary metrics (boundary band as in Cheng et al. 2021) ----------------
def boundary_band(mask, ratio):
    d = max(1, int(round(ratio * math.hypot(*mask.shape))))
    er = ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=d)
    return mask & ~er


def boundary_iou(pred, gt, ratio=0.02):
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    pb, gb = boundary_band(pred, ratio), boundary_band(gt, ratio)
    u = np.logical_or(pb, gb).sum()
    return float(np.logical_and(pb, gb).sum()) / float(u) if u > 0 else 0.0


def bf_score(pred, gt, ratio=0.02):
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    theta = max(1.0, ratio * math.hypot(*gt.shape))
    pb, gb = boundary_band(pred, ratio), boundary_band(gt, ratio)
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    dt_g = ndimage.distance_transform_edt(~gb)
    dt_p = ndimage.distance_transform_edt(~pb)
    prec = float((dt_g[pb] <= theta).mean()); rec = float((dt_p[gb] <= theta).mean())
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", required=True)
    ap.add_argument("--tag", default="degrad_boundary")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--vdas_mode", default="gated")
    ap.add_argument("--lim_n", type=int, default=0)
    ap.add_argument("--conds", default="")
    args = ap.parse_args()

    snap = ExSnapshot(args.ex, tag=args.tag, root=PROJECT_ROOT / "results")
    (snap.dir / "command.txt").write_text(
        f'{sys.executable} experiments/{Path(__file__).name} '
        + " ".join(sys.argv[1:]) + "\n", encoding="utf-8")
    snap.set(ckpt=str(args.ckpt), model=args.model,
             protocol="sliding_window_512_stride256_overlap_mean_argmax",
             boundary_band_ratio=0.02, bf_theta_ratio=0.02,
             note="degradation applied to inference input only; GT unchanged; same protocol as eval.py")

    test_ids = [l.strip() for l in (SPLITS / "test.txt").read_text().splitlines() if l.strip()]
    if args.lim_n > 0:
        test_ids = test_ids[:args.lim_n]
    build_cache(test_ids)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _mk = dict(img_size=args.img_size, in_channels=3, num_classes=1)
    if args.model in ("full", "crw-rwkv-full"):
        _mk["vdas_stop_grad"] = True
        _mk["vdas_mode"] = args.vdas_mode
    model = build_model(args.model, **_mk).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    model.eval()
    print(f"[{args.ex}] model={args.model} ckpt={args.ckpt} n_test={len(test_ids)} dev={dev}", flush=True)

    conds = CONDITIONS if not args.conds else [c for c in CONDITIONS if c[0] in args.conds.split(",")]
    rows = ["condition,image_id,soft_dice,soft_iou,iou05,dice05,boundary_iou,bf_score"]
    agg, persister = {}, {}
    for cname, cfun in conds:
        vals = {k: [] for k in ("soft_dice", "soft_iou", "iou05", "dice05", "boundary_iou", "bf_score")}
        pv = []
        for j, imid in enumerate(test_ids):
            im = np.asarray(Image.open(CACHE / "test" / "images" / f"{imid}.jpg").convert("RGB"))
            gt = (np.asarray(Image.open(CACHE / "test" / "masks" / f"{imid}.png").convert("L")) > 127)
            a = cfun(im, seed=j) if cname in SEEDED else cfun(im)
            t = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)) / 255.0).float().to(dev)
            logit = slide_predict(model, t, dev)
            m = metrics(logit, torch.from_numpy(gt.astype(np.float32)).to(dev))
            prob = torch.sigmoid(logit).detach().cpu().numpy()
            pred = prob > 0.5
            bi, bf = boundary_iou(pred, gt), bf_score(pred, gt)
            vals["soft_dice"].append(m["soft_dice"]); vals["soft_iou"].append(m["soft_iou"])
            vals["iou05"].append(m["iou05"]); vals["dice05"].append(m["dice05"])
            vals["boundary_iou"].append(bi); vals["bf_score"].append(bf)
            pv.append((imid, m["soft_dice"], bi, bf))
            rows.append(f"{cname},{imid},{m['soft_dice']:.6f},{m['soft_iou']:.6f},"
                        f"{m['iou05']:.6f},{m['dice05']:.6f},{bi:.6f},{bf:.6f}")
        persister[cname] = pv
        agg[cname] = {k: (float(np.mean(v)), float(np.std(v) / math.sqrt(len(v)))) for k, v in vals.items()}
        print(f"  {cname:18s} soft_dice={agg[cname]['soft_dice'][0]:.4f}  "
              f"iou05={agg[cname]['iou05'][0]:.4f}  boundary_iou={agg[cname]['boundary_iou'][0]:.4f}  "
              f"bf={agg[cname]['bf_score'][0]:.4f}", flush=True)

    (snap.dir / "degrad_boundary_per_image.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    base = agg["clean"]["soft_dice"][0]
    summ = ["condition,soft_dice,dSoftDice_vs_clean,soft_iou,iou05,dice05,boundary_iou,bf_score"]
    for c, _ in conds:
        d = agg[c]
        summ.append(f"{c},{d['soft_dice'][0]:.4f},{d['soft_dice'][0]-base:+.4f},{d['soft_iou'][0]:.4f},"
                    f"{d['iou05'][0]:.4f},{d['dice05'][0]:.4f},{d['boundary_iou'][0]:.4f},{d['bf_score'][0]:.4f}")
    (snap.dir / "degrad_boundary_summary.csv").write_text("\n".join(summ) + "\n", encoding="utf-8")
    for c, d in agg.items():
        for k, (mu, se) in d.items():
            snap.set(**{f"{c}__{k}_mean": mu, f"{c}__{k}_se": se})
    snap.set(clean_soft_dice=base)
    snap.write_env_snapshot()
    snap.write_meta("SUCCESS")
    snap.write_manifest(title=f"{args.ex} {args.tag}")
    print(f"[{args.ex}] done -> {snap.dir}")


if __name__ == "__main__":
    main()
