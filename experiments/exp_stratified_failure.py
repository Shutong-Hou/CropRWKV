# -*- coding: utf-8 -*-
"""Size-stratified failure analysis under two independent measures of difficulty.

Question: is the error driven by lesion size? Two independent measures are used
so that they can corroborate each other:
  measure A: ground-truth lesion-area fraction of the test image (quartiles);
  measure B: equivalent diameter of the largest connected lesion component,
             computed in the cached 1024-px coordinate frame that the models
             actually see (quartiles).
Per-image soft-Dice is also correlated (Spearman) with both difficulty measures.
All eight methods are covered.

Output (results/EX-154/):
  stratified_by_area.csv / stratified_by_diameter.csv / spearman_correlations.csv
  per_image_detail.csv / report.txt / command.txt / meta.yaml / manifest.sha256
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from exp_harness import ExSnapshot

R = PROJECT_ROOT / "results"
CACHE_TEST = PROJECT_ROOT / "data" / "kaggle_leafseg" / "cache" / "test" / "masks"
SPLITS = R / "EX-020" / "splits"

MODELS = {
    "CropRWKV FULL (ours)": ["EX-033", "EX-036", "EX-037"],
    "DeepLabV3+ (R50, IN)": ["EX-045", "EX-075", "EX-076"],
    "SegFormer-B2 (IN)": ["EX-046", "EX-077", "EX-078"],
    "U-Net (R50, IN)": ["EX-047", "EX-098", "EX-099"],
    "Swin-UNet (IN)": ["EX-081", "EX-197", "EX-198"],
    "U-Net (scratch)": ["EX-039", "EX-160", "EX-161"],
    "SegFormer-B2 (scratch)": ["EX-083", "EX-162", "EX-163"],
    "CropRWKV A0": ["EX-023", "EX-272", "EX-273"],
}


def load(exs):
    """Per-image metrics averaged over the seeds listed for one method."""
    if isinstance(exs, str):
        exs = [exs]
    acc: dict = {}
    for ex in exs:
        with open(R / ex / "per-image_metrics.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = acc.setdefault(r["image_id"], {k: [] for k in ("soft_dice", "iou05", "dice05", "soft_iou")})
                for k in d:
                    d[k].append(float(r[k]))
    return {i: {k: float(np.mean(v)) for k, v in d.items()} for i, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", default="EX-154")
    ap.add_argument("--tag", default="stratified_failure_dual_criteria")
    args = ap.parse_args()
    snap = ExSnapshot(args.ex, tag=args.tag, root=R)
    (snap.dir / "command.txt").write_text(
        f'{sys.executable} experiments/{Path(__file__).name} --ex {args.ex}\n',
        encoding="utf-8")
    snap.set(criteria=["GT lesion area fraction", "max lesion component equivalent diameter (cached 1024 frame)"],
             metric="per-image soft_dice", frame_note="cached frame = model input resolution (max side 1024)")

    test_ids = [l.strip() for l in (SPLITS / "test.txt").read_text().splitlines() if l.strip()]
    gt_frac, max_eqd, n_comp = {}, {}, {}
    for i in test_ids:
        a = np.asarray(Image.open(CACHE_TEST / f"{i}.png").convert("L")) > 127
        gt_frac[i] = float(a.mean())
        lab, n = ndimage.label(a)
        n_comp[i] = int(n)
        if n > 0:
            areas = np.bincount(lab.ravel())[1:]
            max_eqd[i] = float((2.0 * np.sqrt(areas / np.pi)).max())
        else:
            max_eqd[i] = 0.0

    met = {m: load(e) for m, e in MODELS.items()}
    lines = [f"test n={len(test_ids)}; methods={len(MODELS)}",
             f"GT lesion frac: mean={np.mean(list(gt_frac.values())):.4f} "
             f"min={min(gt_frac.values()):.4f} max={max(gt_frac.values()):.4f}",
             f"max component eqd (cached px): median={np.median(list(max_eqd.values())):.1f}"]

    def stratify(name, key, unit, out_name):
        s = sorted(test_ids, key=lambda i: key(i))
        q = len(s) // 4
        groups = {"Q1": s[:q], "Q2": s[q:2 * q], "Q3": s[2 * q:3 * q], "Q4": s[3 * q:]}
        bounds = [f"Q1<={key(s[q-1]):.4g}", f"Q2={key(s[q]):.4g}-{key(s[2*q-1]):.4g}",
                  f"Q3={key(s[2*q]):.4g}-{key(s[3*q-1]):.4g}", f"Q4>={key(s[3*q]):.4g}"]
        lines.append(f"\n=== stratification by {name} ({unit}) ===")
        lines.append("boundaries: " + "  ".join(bounds))
        rows = ["method," + ",".join(f"{k}_soft_dice" for k in groups) + ",Q4_minus_Q1,missing"]
        lines.append(f"{'method':26s}" + "".join(f"{k:>9s}" for k in groups) + "   Q4-Q1")
        for m, ex in MODELS.items():
            vals, miss = [], 0
            for k in groups:
                v = [met[m][i]["soft_dice"] for i in groups[k] if i in met[m]]
                miss += (len(groups[k]) - len(v))
                vals.append(float(np.mean(v)) if v else float("nan"))
            rows.append(f"{m}," + ",".join(f"{v:.4f}" for v in vals) + f",{vals[3]-vals[0]:+.4f},{miss}")
            lines.append(f"{m:26s}" + "".join(f"{v:9.3f}" for v in vals) + f"  {vals[3]-vals[0]:+.3f}"
                         + (f"  [missing {miss}]" if miss else ""))
        (snap.dir / out_name).write_text("\n".join(rows) + "\n", encoding="utf-8")
        return groups

    stratify("GT lesion area fraction", lambda i: gt_frac[i], "fraction of image", "stratified_by_area.csv")
    stratify("max lesion component equivalent diameter", lambda i: max_eqd[i], "px (cached 1024 frame)",
             "stratified_by_diameter.csv")

    lines.append("\n=== Spearman correlation of per-image soft-Dice with difficulty ===")
    lines.append(f"{'method':26s}{'vs lesion area':>18s}{'vs max comp diam':>20s}"
                 f"{'vs n components':>18s}{'mid-50% (area) mean':>22s}")
    rows = ["method,rho_area,p_area,rho_diam,p_diam,rho_ncomp,p_ncomp,mid50_mean"]
    mid = sorted(test_ids, key=lambda i: gt_frac[i])[int(0.25 * len(test_ids)):int(0.75 * len(test_ids))]
    for m, ex in MODELS.items():
        ids = [i for i in test_ids if i in met[m]]
        sd = [met[m][i]["soft_dice"] for i in ids]
        r1, p1 = spearmanr(sd, [gt_frac[i] for i in ids])
        r2, p2 = spearmanr(sd, [max_eqd[i] for i in ids])
        r3, p3 = spearmanr(sd, [n_comp[i] for i in ids])
        vmid = float(np.mean([met[m][i]["soft_dice"] for i in mid if i in met[m]]))
        rows.append(f"{m},{r1:.4f},{p1:.3e},{r2:.4f},{p2:.3e},{r3:.4f},{p3:.3e},{vmid:.4f}")
        lines.append(f"{m:26s}{r1:>11.3f}(p{p1:.1e}){r2:>13.3f}(p{p2:.1e})"
                     f"{r3:>11.3f}(p{p3:.1e}){vmid:>22.3f}")
    (snap.dir / "spearman_correlations.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    det = ["image_id,gt_frac,max_eqd_cached,n_comp_cached," + ",".join(f"{m}_soft_dice" for m in MODELS)]
    for i in test_ids:
        det.append(f"{i},{gt_frac[i]:.6f},{max_eqd[i]:.3f},{n_comp[i]}," +
                   ",".join(f"{met[m][i]['soft_dice']:.6f}" if i in met[m] else "nan" for m in MODELS))
    (snap.dir / "per_image_detail.csv").write_text("\n".join(det) + "\n", encoding="utf-8")

    (snap.dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    snap.write_meta("SUCCESS")
    snap.write_manifest(title=f"{args.ex} {args.tag}")
    print(f"[{args.ex}] done -> {snap.dir}")


if __name__ == "__main__":
    main()
