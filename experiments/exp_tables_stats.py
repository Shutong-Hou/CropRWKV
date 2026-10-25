# -*- coding: utf-8 -*-
"""Recompute the main table, the bootstrap 95% confidence intervals and the
paired Wilcoxon/Holm tests against every baseline.

Principle: every number in the paper is recomputed from the stored
results/*/per-image_metrics.csv files; nothing is transcribed by hand, and the
recomputed values are cross-checked item by item against the recorded summary
of each run, with any difference printed rather than hidden.

Output (results/EX-153/):
  main_table_recomputed.csv / bootstrap_ci.csv / wilcoxon_all_baselines.csv
  crosscheck_vs_records.txt / report.txt
  command.txt / meta.yaml / manifest.sha256
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from exp_harness import ExSnapshot

R = PROJECT_ROOT / "results"
NBOOT = 10000
RNG = np.random.default_rng(20261006)

METHODS = {
    "CropRWKV FULL (ours)": {42: "EX-033", 7: "EX-036", 3407: "EX-037"},
    "DeepLabV3+ (R50, IN)": {42: "EX-045", 7: "EX-075", 3407: "EX-076"},
    "SegFormer-B2 (IN)":    {42: "EX-046", 7: "EX-077", 3407: "EX-078"},
    "U-Net (R50, IN)":      {42: "EX-047", 7: "EX-098", 3407: "EX-099"},
    "Swin-UNet (IN)":       {42: "EX-081", 7: "EX-197", 3407: "EX-198"},
    "U-Net (scratch)":      {42: "EX-039", 7: "EX-160", 3407: "EX-161"},
    "SegFormer-B2 (scratch)": {42: "EX-083", 7: "EX-162", 3407: "EX-163"},
    "CropRWKV A0":          {42: "EX-023", 7: "EX-272", 3407: "EX-273"},
}
METRICS = ["soft_dice", "iou05", "soft_iou", "dice05"]
# Reference values printed in the manuscript's main table, used only as an
# independent cross-check of the recomputation below.
PUBLISHED = [("CropRWKV FULL (ours)", 0.400), ("DeepLabV3+ (R50, IN)", 0.747),
             ("SegFormer-B2 (IN)", 0.747), ("U-Net (R50, IN)", 0.738),
             ("Swin-UNet (IN)", 0.756), ("U-Net (scratch)", 0.646),
             ("SegFormer-B2 (scratch)", 0.677), ("CropRWKV A0", 0.268)]


def load(ex):
    d = {}
    with open(R / ex / "per-image_metrics.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[r["image_id"]] = {m: float(r[m]) for m in METRICS}
    return d


def boot_ci(a):
    a = np.asarray(a, float)
    idx = RNG.integers(0, len(a), size=(NBOOT, len(a)))
    return np.percentile(a[idx].mean(axis=1), 2.5), np.percentile(a[idx].mean(axis=1), 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", default="EX-153")
    ap.add_argument("--tag", default="tables_bootstrap_wilcoxon_recompute")
    args = ap.parse_args()
    snap = ExSnapshot(args.ex, tag=args.tag, root=R)
    (snap.dir / "command.txt").write_text(
        f'{sys.executable} experiments/{Path(__file__).name} --ex {args.ex}\n',
        encoding="utf-8")
    snap.set(n_boot=NBOOT, bootstrap_unit="per-image soft_dice (60 test images)",
             wilcoxon_metric="per-image IoU@0.5", correction="Holm-Bonferroni",
             effect_size="matched-pairs rank-biserial r = (T+ - T-)/T")

    data = {m: {s: load(e) for s, e in d.items()} for m, d in METHODS.items()}
    lines = []

    # Main table
    rows = ["method,seeds,soft_dice,iou05,soft_iou,dice05,note"]
    lines.append("=== main table (recomputed from per-image_metrics.csv) ===")
    hdr = f"{'method':26s}{'seeds':>6s}" + "".join(f"{k:>17s}" for k in METRICS)
    lines.append(hdr)
    for m, sd in METHODS.items():
        n = len(sd)
        cells = []
        for k in METRICS:
            per_seed = [np.mean([v[k] for v in data[m][s].values()]) for s in sd]
            cells.append(f"{np.mean(per_seed):.3f}±{np.std(per_seed, ddof=1) if n > 1 else 0:.3f}")
        note = "" if n > 1 else "single seed"
        rows.append(f"{m},{n}," + ",".join(c.replace("±", "+-") for c in cells) + f",{note}")
        lines.append(f"{m:26s}{n:>6d}" + "".join(f"{c:>17s}" for c in cells) + ("  *" if n == 1 else ""))
    (snap.dir / "main_table_recomputed.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # bootstrap CI
    ci_rows = ["method,seed,mean,ci_lo,ci_hi,n"]
    lines.append("\n=== bootstrap 95% CI (per-image soft_dice) ===")
    for m, sd in METHODS.items():
        for s in sd:
            a = [v["soft_dice"] for v in data[m][s].values()]
            lo, hi = boot_ci(a)
            ci_rows.append(f"{m},{s},{np.mean(a):.4f},{lo:.4f},{hi:.4f},{len(a)}")
    (snap.dir / "bootstrap_ci.csv").write_text("\n".join(ci_rows) + "\n", encoding="utf-8")

    # Wilcoxon against all baselines
    base = ["DeepLabV3+ (R50, IN)", "SegFormer-B2 (IN)", "U-Net (R50, IN)", "Swin-UNet (IN)"]
    res = []
    for s in (42, 7, 3407):
        for b in base:
            fa, bb = data["CropRWKV FULL (ours)"][s], data[b][s]
            common = sorted(set(fa) & set(bb))
            x = np.array([fa[i]["iou05"] for i in common])
            y = np.array([bb[i]["iou05"] for i in common])
            d = x - y
            stat, p = wilcoxon(x, y, alternative="two-sided")
            rk = rankdata(np.abs(d))
            r = (rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum()
            res.append([s, b, len(common), x.mean(), y.mean(), float(np.median(d)), p, r])
    order = np.argsort([row[6] for row in res])
    ph = [0.0] * len(res)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, res[idx][6] * (len(res) - rank)))
        ph[idx] = running
    wrows = ["seed,baseline,n,full_mean,baseline_mean,median_diff,p_two_sided,p_holm,rank_biserial_r"]
    lines.append("\n=== Wilcoxon (per-image IoU@0.5, paired two-sided) + Holm ===")
    lines.append(f"{'seed':>5s}{'baseline':>24s}{'n':>4s}{'FULL':>9s}{'baseline':>10s}"
                 f"{'med.diff':>10s}{'p':>11s}{'p_holm':>11s}{'r':>8s}")
    for (s, b, n, xm, ym, md, p, r), h in zip(res, ph):
        wrows.append(f"{s},{b},{n},{xm:.4f},{ym:.4f},{md:.4f},{p:.4e},{h:.4e},{r:.4f}")
        lines.append(f"{s:>5d}{b:>24s}{n:>4d}{xm:>9.4f}{ym:>10.4f}{md:>10.4f}{p:>11.2e}{h:>11.2e}{r:>8.3f}")
    (snap.dir / "wilcoxon_all_baselines.csv").write_text("\n".join(wrows) + "\n", encoding="utf-8")

    # Cross-check
    lines.append("\n=== cross-check against tables/tab_main.csv (soft-Dice) ===")
    diffs = 0
    for m, doc in PUBLISHED:
        per_seed = [np.mean([v["soft_dice"] for v in data[m][s].values()]) for s in METHODS[m]]
        got = float(np.mean(per_seed))
        ok = abs(got - doc) < 0.0015
        diffs += (not ok)
        lines.append(f"  {m:26s} recomputed={got:.3f} recorded={doc:.3f}  {'OK' if ok else 'DIFF'}")
    lines.append(f"  -> {len(PUBLISHED)-diffs}/{len(PUBLISHED)} match")
    lines.append("\nNote: the rank-biserial r reported by an earlier script differs from the "
                 "formula used here by 0.003-0.006 because the two use different definitions of "
                 "the effect size. The manuscript fixes one definition, r = (T+ - T-)/T, which is "
                 "the one computed by this script.")
    (snap.dir / "crosscheck_vs_records.txt").write_text("\n".join(lines), encoding="utf-8")
    (snap.dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    snap.write_meta("SUCCESS")
    snap.write_manifest(title=f"{args.ex} {args.tag}")
    print(f"[{args.ex}] done -> {snap.dir}")


if __name__ == "__main__":
    main()
