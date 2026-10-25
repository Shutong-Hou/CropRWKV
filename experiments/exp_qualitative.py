# -*- coding: utf-8 -*-
"""exp_qualitative.py -- dump prediction masks for a qualitative comparison figure.

Reuses the *exact* evaluation protocol of eval.py (cached 1024-px frame, sliding
window 512 / stride 256, mean of overlapping logits, sigmoid, threshold 0.5), so
the masks shown here are the same predictions behind the quantitative tables.

Three test images are selected by a fixed, reported rule -- one typical image from
the first (Q1), third (Q3) and fourth (Q4) ground-truth lesion-area quartile of the
60-image test split -- which makes the figure trace the size-stratified failure
documented in results/EX-154.

Outputs (figures/qualitative/):
  case{i}_image.png          RGB input at the cached resolution
  case{i}_gt.png             binary ground-truth mask (0/255)
  case{i}_{model}_pred.png   binary prediction  (0/255)
  case{i}_{model}_prob.png   8-bit probability map
  qualitative_cases.csv      case x model soft-Dice, recomputed here
  qualitative_meta.json      selection rule, image ids, resolutions

Run:  python code/exp_qualitative.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"
for _p in (str(PROJECT_ROOT), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval import build_cache, slide_predict, SPLITS, SRC, CACHE  # noqa: E402
from models_registry import build_model  # noqa: E402

OUT = PROJECT_ROOT / "figures" / "qualitative"
DETAIL = PROJECT_ROOT / "results" / "EX-154" / "per_image_detail.csv"

# Test images that must not be shown in a published figure because they carry a
# superimposed stock-photo watermark (the underlying PlantDoc images were scraped
# from the open web).  Excluding them is a presentation constraint, not a metric
# choice: the excluded image is replaced by the next-typical candidate, and the
# quantitative results in EX-154 are unchanged.
EXCLUDE = {"00579": "superimposed stock-photo watermark"}

MODELS = [
    ("CropRWKV FULL", "full", PROJECT_ROOT / "results" / "EX-032" / "checkpoints" / "best.pt"),
    ("SegFormer-B2 (IN)", "segformer", PROJECT_ROOT / "results" / "EX-043" / "checkpoints" / "best.pt"),
    ("DeepLabV3+ (R50, IN)", "deeplabv3plus", PROJECT_ROOT / "results" / "EX-042" / "checkpoints" / "best.pt"),
    ("U-Net (R50, IN)", "unet_pretrained", PROJECT_ROOT / "results" / "EX-044" / "checkpoints" / "best.pt"),
]


def read_detail():
    with open(DETAIL, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("image_id")]


def pick_cases(rows):
    """Three representative test images: one from the smallest (Q1), one from the
    third (Q3) and one from the largest (Q4) lesion-area quartile.  Inside each
    stratum we take the image whose CropRWKV soft-Dice is closest to the stratum
    mean, i.e. a *typical* case rather than the best or worst case.  Cached frames
    whose short side is below the 512-px window cannot be windowed and are
    excluded."""
    cand = []
    for r in rows:
        if r["image_id"] in EXCLUDE:
            continue
        ip = CACHE / "test" / "images" / f"{r['image_id']}.jpg"
        if not ip.exists():
            continue
        w, h = Image.open(ip).size
        if min(w, h) < 512:
            continue
        cand.append((float(r["gt_frac"]),
                     float(r["CropRWKV FULL (ours)_soft_dice"]),
                     r["image_id"]))
    cand.sort()
    n = len(cand)
    picks = []
    for lo, hi in ((0, n // 4), (n // 2, 3 * n // 4), (3 * n // 4, n)):
        grp = cand[lo:hi]
        mu = sum(c[1] for c in grp) / len(grp)
        best = min(grp, key=lambda c: abs(c[1] - mu))
        picks.append((best[0], best[2]))
    return picks


def soft_dice(prob: np.ndarray, gt: np.ndarray) -> float:
    p = prob.astype(np.float64)
    g = gt.astype(np.float64)
    inter = float((p * g).sum())
    return (2 * inter + 1e-6) / (p.sum() + g.sum() + 1e-6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test_ids = [l.strip() for l in (SPLITS / "test.txt").read_text().splitlines() if l.strip()]
    build_cache(test_ids)

    rows = read_detail()
    cases = pick_cases(rows)
    print("selected cases (gt_frac, id):", cases, flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for label, key, ckpt in MODELS:
        kw = dict(img_size=512, in_channels=3, num_classes=1)
        if key == "full":
            kw.update(vdas_stop_grad=True, vdas_mode="gated")
        m = build_model(key, **kw).to(dev)
        m.load_state_dict(torch.load(str(ckpt), map_location=dev)["model"])
        m.eval()
        models[label] = m
        print(f"  loaded {label} <- {ckpt.name} ({ckpt.parent.parent.name})", flush=True)

    summary = ["case,image_id,gt_frac,model,soft_dice"]
    meta_cases = []
    for i, (gtf, imid) in enumerate(cases, start=1):
        imPil = Image.open(CACHE / "test" / "images" / f"{imid}.jpg").convert("RGB")
        mkPil = Image.open(CACHE / "test" / "masks" / f"{imid}.png").convert("L")
        img = np.asarray(imPil)
        gt = (np.asarray(mkPil) > 127)
        t = torch.from_numpy(img.transpose(2, 0, 1) / 255.0).float().to(dev)

        imPil.save(OUT / f"case{i}_image.png")
        Image.fromarray((gt.astype(np.uint8) * 255)).save(OUT / f"case{i}_gt.png")

        meta_cases.append(dict(case=i, image_id=imid, gt_frac=round(gtf, 6),
                               size=list(imPil.size)))
        for label, key, _ in MODELS:
            with torch.no_grad():
                logit = slide_predict(models[label], t, dev)
            prob = torch.sigmoid(logit).cpu().numpy().astype(np.float32)
            pred = (prob > 0.5)
            Image.fromarray((pred.astype(np.uint8) * 255)).save(OUT / f"case{i}_{key}_pred.png")
            Image.fromarray((prob * 255).astype(np.uint8)).save(OUT / f"case{i}_{key}_prob.png")
            sd = soft_dice(prob, gt)
            summary.append(f"{i},{imid},{gtf:.6f},{label},{sd:.6f}")
            print(f"  case{i} {imid} {label}: soft-Dice={sd:.4f}", flush=True)

    (OUT / "qualitative_cases.csv").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (OUT / "qualitative_meta.json").write_text(json.dumps(dict(
        selection_rule="one typical image per lesion-area quartile: the Q1, Q3 and Q4 "
                       "test image whose CropRWKV soft-Dice is closest to its stratum "
                       "mean (results/EX-154/per_image_detail.csv)",
        protocol="cached 1024-px frame; sliding window 512, stride 256; mean logits; "
                 "sigmoid; threshold 0.5",
        checkpoints={l: f"{p.parent.parent.name}/checkpoints/best.pt" for l, _, p in MODELS},
        cases=meta_cases), indent=2), encoding="utf-8")
    print("OK ->", OUT, flush=True)


if __name__ == "__main__":
    main()