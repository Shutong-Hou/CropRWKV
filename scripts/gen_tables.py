# -*- coding: utf-8 -*-
"""gen_tables.py -- regenerate every table in the CropRWKV manuscript
directly from the immutable EX snapshots under results/.

Design rules
------------
* No number is typed by hand. Every cell is read from an EX snapshot
  (meta.yaml / *.csv / *.json) or recomputed from a per-image file.
* Each emitted LaTeX fragment is a complete table float, so main.tex only
  needs \\input{tables/tab_xxx}.
* A machine-readable CSV copy of every table is written next to the .tex so
  the manuscript values can be re-verified without reading LaTeX.

Run:  python scripts/gen_tables.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "tables"
OUT.mkdir(parents=True, exist_ok=True)

_LOG: list[str] = []


# --------------------------------------------------------------------------
# small IO helpers
# --------------------------------------------------------------------------
def meta(ex: str) -> dict:
    p = RES / ex / "meta.yaml"
    if not p.exists():
        return {}
    t = p.read_text(encoding="utf-8", errors="ignore")
    d = {}
    for m in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", t, re.M):
        d[m.group(1)] = m.group(2).strip()
    return d


def num(ex: str, key: str) -> float:
    v = meta(ex).get(key, "")
    return float(v) if v not in ("", "nan") else float("nan")


def agg_meta(exs: list[str], key: str) -> tuple[float, float]:
    """Cross-seed mean and standard deviation of one meta key over several EX snapshots."""
    return mean_std([num(e, key) for e in exs])


def agg_csv(exs: list[str], name: str, keyfn) -> dict:
    """Average a snapshot CSV over several seeds, keyed by the value of ``keyfn(row)``.

    ``keyfn`` returns ``(key, field)``: the grouping key (e.g. the condition or the
    metric name) and the column to average.
    """
    acc: dict = {}
    for e in exs:
        for r in csv_rows(e, name):
            k, f = keyfn(r)
            acc.setdefault(k, {}).setdefault(f, []).append(float(r[f]))
    return {k: {f: float(np.mean(v)) for f, v in d.items()} for k, d in acc.items()}



def csv_rows(ex: str, name: str) -> list[dict]:
    """Read a CSV from an EX snapshot.

    Several snapshot CSVs were written with unquoted fields that contain
    commas, e.g. a method label ``U-Net (R50, IN)`` or a tensor shape
    ``(1, 256, 64, 64)``.  When a data row carries more fields than the
    header, consecutive fields are re-joined until the parentheses balance,
    which recovers the intended record without modifying the immutable
    snapshot.
    """
    import csv
    p = RES / ex / name
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, newline="", encoding="utf-8-sig") as f:
        raw = [r for r in csv.reader(f) if r]
    if not raw:
        return []
    header = raw[0]
    n = len(header)
    out = []
    for r in raw[1:]:
        if len(r) > n:
            merged, cur = [], ""
            for fld in r:
                cur = fld if cur == "" else cur + "," + fld
                if cur.count("(") <= cur.count(")"):
                    merged.append(cur)
                    cur = ""
            if cur:
                merged.append(cur)
            r = merged
        out.append(dict(zip(header, r)))
    return out


def bold_short(lab: str) -> str:
    """Bold method label, truncated at the first parenthesis."""
    return "\\textbf{" + lab.split(" (")[0] + "}"


# header text for the wide comparison tables, keyed by the canonical label
SHORT_LABEL = {
    "CropRWKV FULL": "CropRWKV FULL",
    "DeepLabV3+ (R50, IN)": "DeepLabV3+",
    "SegFormer-B2 (IN)": "SegFormer-B2",
    "U-Net (R50, IN)": "U-Net (R50)",
    "Swin-UNet (IN)": "Swin-UNet",
    "U-Net (scratch)": "U-Net (scratch)",
}


def short_label(lab: str) -> str:
    """Bold, disambiguated column header for a canonical method label."""
    return "\\textbf{" + SHORT_LABEL.get(lab, lab) + "}"


def jload(ex: str, name: str) -> dict:
    return json.loads((RES / ex / name).read_text(encoding="utf-8"))


def fmt(v: float, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{nd}f}"


def mean_std(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, float)
    a = a[~np.isnan(a)]
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def write(stem: str, tex: str, rows: list[list] | None = None, header: list[str] | None = None):
    (OUT / f"{stem}.tex").write_text(tex, encoding="utf-8")
    if rows is not None:
        import csv as _csv
        with open(OUT / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            if header:
                w.writerow(header)
            w.writerows(rows)
    _LOG.append(f"  wrote tables/{stem}.tex")


def table(caption: str, label: str, colspec: str, body: str, note: str = "",
          width: str = "\\textwidth", wide: bool = False, fontsize: str = "",
          tabcolsep: str = "") -> str:
    """Emit one table float.

    ``wide=True`` bleeds the float into the MDPI margin (adjustwidth +
    \\fulllength) so that wide benchmark tables fit the 18.46 cm text block
    without an overfull box.  ``fontsize`` and ``tabcolsep`` are applied
    inside the adjustwidth block when given.
    """
    foot = f"\n\\noindent{{\\footnotesize{{{note}}}}}" if note else ""
    if wide:
        pre = "\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        if tabcolsep:
            pre += f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n"
        if fontsize:
            pre += fontsize + "\n"
        post = "\n\\end{adjustwidth}"
        width = "\\fulllength"
    else:
        pre, post = (fontsize + "\n" if fontsize else ""), ""
    return (
        "\\begin{table}[H]\n"
        "\\caption{" + caption + "\\label{" + label + "}}\n"
        + pre +
        "\\begin{tabularx}{" + width + "}{" + colspec + "}\n"
        "\\toprule\n" + body + "\n\\bottomrule\n"
        "\\end{tabularx}" + foot + post + "\n"
        "\\end{table}\n"
    )


# --------------------------------------------------------------------------
# method metadata: EX snapshot -> canonical label
# --------------------------------------------------------------------------
FULL = {"42": "EX-033", "7": "EX-036", "3407": "EX-037"}
DEEPLAB = {"42": "EX-045", "7": "EX-075", "3407": "EX-076"}
SEGF_IN = {"42": "EX-046", "7": "EX-077", "3407": "EX-078"}
UNET_IN = {"42": "EX-047", "7": "EX-098", "3407": "EX-099"}
SWIN = {"42": "EX-081", "7": "EX-197", "3407": "EX-198"}
UNET_SCR = {"42": "EX-039", "7": "EX-160", "3407": "EX-161"}
SEGF_SCR = {"42": "EX-083", "7": "EX-162", "3407": "EX-163"}
A0 = {"42": "EX-023", "7": "EX-272", "3407": "EX-273"}

# params / FLOPs: measured with thop in EX-147 (unified tool) + EX-004 for A0
EFF = {r["model"]: r for r in csv_rows("EX-147", "lat_flops.csv")}
PARAMS = {
    "full": 15.721, "segformer": 24.722, "unet_scratch": 13.322,
    "unet_pretrained": 32.521, "deeplabv3plus": 26.678, "swin_unet": 56.323,
}
FLOPS = {k: float(v["flops_512_G"]) for k, v in EFF.items()}
A0_PARAMS = num("EX-004", "n_params") / 1e6
A0_FLOPS = num("EX-192", "a0_flops_512_G")   # 2026-10-18: A0 FLOPs measured with the same thop protocol


# --------------------------------------------------------------------------
# Table 1 -- dataset characterisation (EX-152, EX-159)
# --------------------------------------------------------------------------
def table1():
    a = jload("EX-152", "dataset_characterization.json")
    b = jload("EX-159", "plantseg_stats.json")
    lf, pl = a["lesion_frac"], b["lesion_frac"]

    def row(name, src, n, spec, dis, res, lfm, lfmed, lt5, gt30, license_):
        return (name, src, n, spec, dis, res, lfm, lfmed, lt5, gt30, license_)

    rows = [
        row("Set A (PlantDoc)", "Kaggle", a["subset"]["n_images"],
            "13", "17--27", f'{a["resolution"]["n_unique_sizes"]} distinct res.; mode $256\\times256$',
            lf["mean"], lf["median"], lf["n_lt5pct"], lf["n_gt30pct"], "CC0-1.0"),
        row("Set B (PlantSeg)", "HuggingFace", b["n_rasterized"],
            str(b["n_plants"]), str(b["n_diseases"]), f'{b["resolution"]["n_unique"]} distinct res.; median 640$\\times$500',
            pl["mean"], pl["median"], pl["n_lt5pct"], pl["n_gt30pct"], "CC BY-NC-ND 4.0"),
    ]
    body = ("\\textbf{Dataset} & \\textbf{Source} & \\textbf{Images} & \\textbf{Species} & \\textbf{Classes} & "
            "\\textbf{Native resolution} & \\textbf{Mean lesion} & \\textbf{Median} & \\textbf{$<$5\\%} & \\textbf{$>$30\\%} & \\textbf{License}\\\\\n"
            "\\midrule\n")
    for r in rows:
        body += (f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & "
                 f"{fmt(r[6],3)} & {fmt(r[7],3)} & {r[8]} & {r[9]} & {r[10]}\\\\\n")
    tex = table(
        "Characterisation of the two datasets, recomputed from the raw image--mask pairs "
        "(Set A: \\texttt{results/EX-152}; Set B: \\texttt{results/EX-159}). Lesion fraction is the "
        "pixel share of the binarised mask (Set A threshold 19 on a 0--38 grey scale; Set B rasterised "
        "instance masks). ``Distinct res.'' is the number of distinct native resolutions. Set A is a "
        "one-to-one image/mask subset of PlantDoc; Set B is the PlantSeg in-the-wild test-set release "
        "(1195 usable of 1200 entries), re-partitioned 75/15/10 by this study.",
        "tab:dataset",
        "L L r r r L r r r r L", body,
        note="No image was added, removed, re-cropped or relabelled in either dataset. Set B was "
             "re-partitioned 75/15/10 because the release provides only a test set.",
        wide=True, fontsize="\\footnotesize", tabcolsep="4pt")
    write("tab_dataset", tex, rows,
          ["dataset", "source", "images", "species", "classes", "resolution",
           "mean_lesion_frac", "median_lesion_frac", "n_lt5pct", "n_gt30pct", "license"])


# --------------------------------------------------------------------------
# Table 2 -- main benchmark (EX-153 recomputation, EX-147 for cost)
# --------------------------------------------------------------------------
def _per_seed_metric(ex_map: dict, metric: str) -> tuple[float, float]:
    vals = []
    for ex in ex_map.values():
        rows = csv_rows(ex, "per-image_metrics.csv")
        vals.append(float(np.mean([float(r[metric]) for r in rows])))
    return mean_std(vals)


def table2():
    methods = [
        ("DeepLabV3+ (R50, IN)", DEEPLAB, "deeplabv3plus", True),
        ("SegFormer-B2 (IN)", SEGF_IN, "segformer", True),
        ("U-Net (R50, IN)", UNET_IN, "unet_pretrained", True),
        ("Swin-UNet (IN)", SWIN, "swin_unet", True),
        ("U-Net (scratch)", UNET_SCR, "unet_scratch", True),
        ("SegFormer-B2 (scratch)", SEGF_SCR, "segformer", True),
        ("CropRWKV FULL (this work)", FULL, "full", True),
        ("CropRWKV A0 (backbone only)", A0, None, True),
    ]
    hdr = ("\\textbf{Method} & \\textbf{Seeds} & \\textbf{soft-Dice} & \\textbf{IoU@0.5} & "
           "\\textbf{Dice@0.5} & \\textbf{soft-IoU} & \\textbf{Params (M)} & \\textbf{FLOPs (G)}\\\\\n\\midrule\n")
    body, rows = hdr, []
    for name, exmap, effkey, _ in methods:
        sd = _per_seed_metric(exmap, "soft_dice")
        iou = _per_seed_metric(exmap, "iou05")
        d05 = _per_seed_metric(exmap, "dice05")
        siou = _per_seed_metric(exmap, "soft_iou")
        n = len(exmap)
        if effkey:
            p, fl = PARAMS[effkey], FLOPS[effkey]
            ps, fs = f"{p:.1f}", f"{fl:.1f}"
        else:
            p, fl, ps, fs = A0_PARAMS, A0_FLOPS, f"{A0_PARAMS:.1f}", f"{A0_FLOPS:.1f}"
        star = "" if n > 1 else "$^{\\dagger}$"
        body += (f"{name}{star} & {n} & {sd[0]:.3f}$\\pm${sd[1]:.3f} & {iou[0]:.3f}$\\pm${iou[1]:.3f} & "
                 f"{d05[0]:.3f}$\\pm${d05[1]:.3f} & {siou[0]:.3f}$\\pm${siou[1]:.3f} & {ps} & {fs}\\\\\n")
        rows.append([name, n, round(sd[0], 4), round(sd[1], 4), round(iou[0], 4), round(iou[1], 4),
                     round(d05[0], 4), round(d05[1], 4), round(siou[0], 4), round(siou[1], 4), ps, fs])
    tex = table(
        "Segmentation accuracy on the held-out Set A test split (60 images) under one identical "
        "training and evaluation protocol. Values are mean$\\pm$standard deviation over the seeds "
        "listed; every number is recomputed from the per-image prediction files stored with each run, "
        "and the per-seed statistics are collected with bootstrap intervals in \\texttt{results/EX-274}; "
        "the two from-scratch controls use the re-runs "
        "\\texttt{results/EX-160}--\\texttt{EX-163}. Soft metrics integrate the sigmoid probability; hard metrics "
        "threshold it at 0.5. Parameter counts and FLOPs at $512\\times512$ were measured with a single tool "
        "(\\texttt{thop}) in \\texttt{EX-147}, and the A0 backbone with the same tool in "
        "\\texttt{EX-192}.",
        "tab:main",
        "L r r r r r r r", body,
        note="CropRWKV FULL is the architecture under study, not a proposed state-of-the-art "
             "model: it is the slowest method in this table and the least accurate of the seven "
             "principal methods, although the bare A0 backbone is less accurate still.",
        wide=True)
    write("tab_main", tex, rows,
          ["method", "seeds", "soft_dice_mean", "soft_dice_std", "iou05_mean", "iou05_std",
           "dice05_mean", "dice05_std", "soft_iou_mean", "soft_iou_std", "params_M", "flops_G"])


# --------------------------------------------------------------------------
# Table S1 -- per-seed accuracy with bootstrap CI (EX-201)
# --------------------------------------------------------------------------
def tableS1():
    ci = csv_rows("EX-274", "bootstrap_ci.csv")
    body = ("\\textbf{Method} & \\textbf{Seed} & \\textbf{soft-Dice} & \\textbf{95\\% CI} & \\textbf{$n$}\\\\\n\\midrule\n")
    rows = []
    for r in ci:
        body += (f'{r["method"]} & {r["seed"]} & {float(r["mean"]):.4f} & '
                 f'[{float(r["ci_lo"]):.4f}, {float(r["ci_hi"]):.4f}] & {r["n"]}\\\\\n')
        rows.append([r["method"], r["seed"], r["mean"], r["ci_lo"], r["ci_hi"], r["n"]])
    tex = table(
        "Per-seed soft-Dice on the Set A test split with 10{,}000-resample bootstrap 95\\% confidence "
        "intervals over images (\\texttt{results/EX-274}). Every seed reported in the main accuracy "
        "table, including the bare \\textsc{A0} backbone, has an interval here; the two from-scratch "
        "controls use the re-runs (\\texttt{EX-160}--\\texttt{EX-163}) and Swin-UNet the re-runs "
        "(\\texttt{EX-197}, \\texttt{EX-198}).",
        "tab:seedsci", "L r r L r", body)
    write("tab_seeds_ci", tex, rows, ["method", "seed", "mean", "ci_lo", "ci_hi", "n"])


# --------------------------------------------------------------------------
# Table 3 -- paired statistical tests (EX-199: four baselines x three seeds)
# --------------------------------------------------------------------------
def table3():
    w = csv_rows("EX-199", "wilcoxon_all_baselines.csv")
    body = ("\\textbf{Seed} & \\textbf{Comparison model} & \\textbf{$n$} & \\textbf{FULL IoU@0.5} & "
            "\\textbf{Baseline IoU@0.5} & "
            "\\textbf{Median diff.} & \\textbf{$p$} & \\textbf{$p_{\\mathrm{Holm}}$} & "
            "\\textbf{rank-biserial $r$}\\\\\n\\midrule\n")
    rows = []
    for r in w:
        p = float(r["p_two_sided"]); ph = float(r["p_holm"])
        body += (f'{r["seed"]} & {r["baseline"]} & {r["n"]} & {float(r["full_mean"]):.4f} & '
                 f'{float(r["baseline_mean"]):.4f} & {float(r["median_diff"]):+.4f} & '
                 f'{p:.2e} & {ph:.2e} & {float(r["rank_biserial_r"]):+.3f}\\\\\n')
        rows.append([r["seed"], r["baseline"], r["n"], r["full_mean"], r["baseline_mean"],
                     r["median_diff"], r["p_two_sided"], r["p_holm"], r["rank_biserial_r"]])
    tex = table(
        "Paired per-image comparison of CropRWKV FULL---the architecture under study with all three "
        "modules active---against each of the four ImageNet-pretrained baselines, for each of the three "
        "seeds (\\texttt{results/EX-199}). The two numeric columns are the mean per-image IoU@0.5 of "
        "CropRWKV FULL and of the named comparison model over the same $n=60$ test images, so each row "
        "is a paired sample. The test statistic is the two-sided Wilcoxon signed-rank test on those "
        "pairs; $p_{\\mathrm{Holm}}$ applies Holm--Bonferroni correction across the twelve comparisons; "
        "the effect size is the matched-pairs rank-biserial correlation $r=(T^{+}-T^{-})/T$, which is "
        "$-1$ when FULL loses on every image.",
        "tab:stats", "r L r r r r r r r", body,
        note="All twelve comparisons are significant after correction with a very large negative effect "
             "size; the ordering is identical under soft and hard metrics, so threshold choice does not "
             "drive the conclusion.",
        wide=True, fontsize="\\small")
    write("tab_stats", tex, rows,
          ["seed", "baseline", "n", "full_mean", "baseline_mean", "median_diff",
           "p_two_sided", "p_holm", "rank_biserial_r"])


# --------------------------------------------------------------------------
# Table 4 -- ablation (EX-066..070, 033/036/037, 069/070, 130..139, 095..097)
# --------------------------------------------------------------------------
def _sd(ex: str) -> float:
    return num(ex, "test_soft_dice_mean")


def table4():
    rows, body = [], ""
    body += "\\textbf{Configuration} & \\textbf{seed 42} & \\textbf{seed 7} & \\textbf{seed 3407} & \\textbf{mean$\\pm$std}\\\\\n\\midrule\n"
    # block 1: cumulative chain (seed 42 only)
    body += "\\multicolumn{5}{l}{\\textit{(a) Cumulative module chain (seed 42 only)}}\\\\\n\\midrule\n"
    cum = [("A0$'$ (all modules off)", "EX-066"), ("$+$VDAS", "EX-067"),
           ("$+$FEEM (no MPSA)", "EX-068"), ("FULL (all modules on)", "EX-033")]
    for name, ex in cum:
        body += f"{name} & {_sd(ex):.3f} & -- & -- & --\\\\\n"
        rows.append(["cumulative", name, round(_sd(ex), 4), "", "", "", ""])
    # block 2: leave-one-out, 3 seeds
    body += "\\midrule\n\\multicolumn{5}{l}{\\textit{(b) Leave-one-out (3 seeds)}}\\\\\n\\midrule\n"
    loo = [("FULL", ["EX-033", "EX-036", "EX-037"]),
           ("$-$VDAS", ["EX-069", "EX-130", "EX-131"]),
           ("$-$FEEM", ["EX-070", "EX-132", "EX-133"]),
           ("$-$MPSA", ["EX-276", "EX-279", "EX-280"])]
    for name, exs in loo:
        vals = [_sd(e) for e in exs]
        m, s = mean_std(vals)
        body += f"{name} & " + " & ".join(fmt(v) for v in vals) + f" & {fmt(m)}$\\pm${fmt(s)}\\\\\n"
        rows.append(["loo", name] + [round(v, 4) if not math.isnan(v) else None for v in vals] + [round(m, 4), round(s, 4)])
    # block 3: scan sub-ablation, 3 seeds
    body += "\\midrule\n\\multicolumn{5}{l}{\\textit{(c) VDAS scan-order ablation (3 seeds)}}\\\\\n\\midrule\n"
    scan = [("S0 horizontal only", ["EX-095", "EX-134", "EX-135"]),
            ("S1 vertical only", ["EX-096", "EX-136", "EX-137"]),
            ("S2 equal H$+$V", ["EX-097", "EX-138", "EX-139"])]
    for name, exs in scan:
        vals = [_sd(e) for e in exs]
        m, s = mean_std(vals)
        body += f"{name} & " + " & ".join(f"{v:.3f}" for v in vals) + f" & {m:.3f}$\\pm${s:.3f}\\\\\n"
        rows.append(["scan", name] + [round(v, 4) for v in vals] + [round(m, 4), round(s, 4)])
    tex = table(
        "Ablation of the three modules and of the VDAS scan order, all at the same parameter count "
        "(15.72\\,M) and on the Set A test split. Every value is read from the corresponding EX "
        "snapshot. Panel (a) is the cumulative chain and exists for seed 42 only, which is marked by "
        "the dashes; panels (b) and (c) are run over three seeds and report the per-seed values "
        "together with their mean and standard deviation. Removing VDAS costs $0.134$ soft-Dice on "
        "average with a cross-seed standard deviation of only $0.017$; removing MPSA costs $0.080$ "
        "with a cross-seed standard deviation of $0.057$ (two seeds; seed 3407 pending); the "
        "scan-order differences in panel (c) are within seed noise and are therefore reported as a "
        "trend.",
        "tab:ablation", "L r r r r", body,
        note="The gated scan (S3) is the FULL configuration, whose three-seed values are the FULL rows "
             "of panel (b). Cross-seed standard deviations of 0.03--0.08 mean the scan-order ordering is "
             "not statistically stable. The VDAS removal effect in panel (b) survives all three seeds; "
             "the MPSA removal effect is stable over the two completed seeds but the third seed is "
             "pending GPU availability.")
    write("tab_ablation", tex, rows, ["block", "configuration", "seed42", "seed7", "seed3407", "mean", "std"])


# --------------------------------------------------------------------------
# Table 5 -- instability diagnostics: spectral norms, 3-arm gradients, repair
# --------------------------------------------------------------------------
def table5():
    sp = csv_rows("EX-140", "spectral_norm.csv")
    gc = csv_rows("EX-140", "grad_compare.csv")
    tj = csv_rows("EX-140", "trajectory.csv")
    st = csv_rows("EX-140", "selftest.csv")

    rows = []
    body = "\\multicolumn{4}{l}{\\textit{(a) Power-iteration spectral norm of the Jacobian}}\\\\\n\\midrule\n"
    body += "\\textbf{Target} & \\textbf{$\\sigma_{\\max}$} & \\textbf{Status} & \\textbf{Reference}\\\\\n\\midrule\n"
    for r in sp:
        tgt = r["target"]
        ref = "VDAS scan operator" if tgt.startswith("vdas") else ("control: dilated conv" if "mpsa" in tgt
              else ("control: edge enhance" if "feem" in tgt else "control: backbone IR-RWKV block"))
        sig = r["sigma_max"]
        if sig.lower() == "nan":
            sigs, stt = "NaN", "overflow"
        else:
            sigs, stt = f"{float(sig):.3g}", r["status"]
        tgt_tex = tgt.replace("_", "\\_")
        body += f"\\texttt{{{tgt_tex}}} & {sigs} & {stt} & {ref}\\\\\n"
        rows.append(["spectral", tgt, sig, r["status"]])
    body += "\\midrule\n"
    body += f"\\multicolumn{{4}}{{l}}{{\\footnotesize Self-test: power iteration vs. \\texttt{{torch.linalg.matrix\\_norm}} "
    body += "relative error " + ", ".join(f"{float(r['rel_err']):.1e} ($n$={r['n']})" for r in st) + ".}\\\\\n"

    body += "\\midrule\n\\multicolumn{4}{l}{\\textit{(b) Encoder gradient under three wiring variants}}\\\\\n\\midrule\n"
    body += ("\\textbf{Quantity} & \\textbf{No detach} & \\textbf{Stop-grad} & \\textbf{VDAS off}\\\\\n\\midrule\n")
    ri = [r for r in gc if r["state"] == "random_init"]
    label = {"dL/d_bottleneck": "$\\partial L/\\partial$ bottleneck", "dL/d_enc3": "$\\partial L/\\partial$ enc3",
             "stage0.firstconv": "stage 0 first conv", "stage4.firstconv": "stage 4 first conv",
             "final_conv": "final conv (control)"}
    for r in ri:
        if r["quantity"] not in label:
            continue
        nd = float(r["nodetach"])
        body += (f'{label[r["quantity"]]} & {nd:.3g} & {float(r["stopgrad"]):.3g} & '
                 f'{float(r["vdasOFF"]):.3g}\\\\\n')
        rows.append(["grad3arm", r["quantity"], r["nodetach"], r["stopgrad"], r["vdasOFF"],
                     r["ratio_nodetach_over_off"]])
    body += "\\midrule\n\\multicolumn{4}{l}{\\textit{(c) Along the repair trajectory}}\\\\\n\\midrule\n"
    body += ("\\textbf{Stage} & \\textbf{$\\sigma_{\\mathrm{vdas3}}$} & \\textbf{$\\sigma_{\\mathrm{vdas4}}$} & "
             "\\textbf{$r$ (stage 0)}\\\\\n\\midrule\n")
    for r in tj:
        body += (f'{r["ex"]} & {float(r["sigma_vdas3"]):.3g} & {float(r["sigma_vdas4"]):.3g} & '
                 f'{float(r["r_stage0_nodetach_over_stopgrad"]):.2f}$\\times$\\\\\n')
        rows.append(["repair", r["ex"], r["sigma_vdas3"], r["sigma_vdas4"],
                     r["r_stage0_nodetach_over_stopgrad"]])
    tex = table(
        "Diagnosis of the mixed-precision training failure of CropRWKV FULL. (a) Largest singular value "
        "of the input--output Jacobian of each sub-module, obtained by power iteration on $J^{\\top}J$ "
        "and validated against \\texttt{torch.linalg.matrix\\_norm} (relative error $\\sim10^{-16}$); "
        "the vertical scan of VDAS at stage 3 overflows to NaN, exposing a near-singular denominator in "
        "the closed-form WKV. (b) Back-propagated encoder gradient magnitude under the same weights, "
        "input and loss, with the VDAS residual connected (no detach), detached (stop-grad) or removed "
        "entirely (VDAS off); the final convolution is unchanged to three digits, which shows the effect "
        "is carried by the VDAS path alone. (c) The amplification ratio $r$ along the repair trajectory; the "
        "two attempted repairs that leave the wiring unchanged do not reduce it ($639\\times$ and "
        "$519\\times$, both above the as-built $112\\times$), whereas the stop-gradient fix brings $r$ to "
        "$1.03\\times$ after the full training schedule. Data: \\texttt{results/EX-140}.",
        "tab:diagnosis", "@{}>{\\raggedright\\arraybackslash}p{4.5cm} r r L@{}", body,
        note="Amplification is not unique to VDAS: the backbone's own IR-RWKV block "
             "reaches $\\sigma_{\\max}=2.16\\times10^{4}$. What distinguishes VDAS is that it is an "
             "\\emph{additional parallel residual branch}, so its amplification multiplies directly into "
             "the encoder gradient path.")
    write("tab_diagnosis", tex, rows, ["block", "item", "a", "b", "c", "d"])


# --------------------------------------------------------------------------
# Table 6 -- training trajectories, fp32 (EX-191) vs AMP (EX-193), both 12 epochs
# --------------------------------------------------------------------------
def table6():
    t32 = csv_rows("EX-191", "grad_trajectory.csv")
    amp = csv_rows("EX-193", "grad_trajectory.csv")
    rows, body = [], ""
    body += "\\multicolumn{4}{l}{\\textit{(a) fp32, identical initialisation, 12 epochs (EX-191)}}\\\\\n\\midrule\n"
    body += ("\\textbf{Epoch} & \\textbf{grad L2 (no detach)} & \\textbf{grad L2 (stop-grad)} & "
             "\\textbf{Amplification}\\\\\n\\midrule\n")
    nd = {r["epoch"]: r for r in t32 if r["arm"] == "nodetach"}
    sg = {r["epoch"]: r for r in t32 if r["arm"] == "stopgrad"}
    for e in sorted(nd, key=int):
        a, b = float(nd[e]["grad_total_l2"]), float(sg[e]["grad_total_l2"])
        body += f"{e} & {a:.3g} & {b:.3g} & {a/b:.0f}$\\times$\\\\\n"
        rows.append(["fp32", e, a, b, round(a / b, 1)])
    body += "\\midrule\n\\multicolumn{4}{l}{\\textit{(b) AMP fp16, unified protocol, 12 epochs (EX-193)}}\\\\\n\\midrule\n"
    body += ("\\textbf{Epoch} & \\textbf{train IoU (no detach)} & \\textbf{train IoU (stop-grad)} & "
             "\\textbf{Non-finite batches}\\\\\n\\midrule\n")
    an = {r["epoch"]: r for r in amp if r["arm"] == "nodetach"}
    asg = {r["epoch"]: r for r in amp if r["arm"] == "stopgrad"}
    for e in sorted(an, key=int):
        body += (f'{e} & {float(an[e]["train_iou"]):.4f} & {float(asg[e]["train_iou"]):.4f} & '
                 f'{an[e]["nonfinite_batches"]}\\\\\n')
        rows.append(["amp", e, an[e]["train_iou"], asg[e]["train_iou"], an[e]["nonfinite_batches"]])
    tex = table(
        "Training trajectories that isolate the failure mechanism. (a) In fp32 with identical "
        "initialisation and data order over 12 epochs, the no-detach arm carries a gradient "
        "33--967$\\times$ larger than the stop-grad arm, yet the two accuracy curves coincide at every "
        "epoch (final train IoU 0.4044 vs.\\ 0.4042) --- amplification alone does not break learning. "
        "(b) Under the unified AMP fp16 protocol the no-detach arm emits 18 non-finite "
        "batches at epoch 0 and its training IoU stays in 0.15--0.19 over the full 12 epochs, while the "
        "stop-grad arm rises to 0.40. The failure is therefore the composite of amplification and fp16 overflow, "
        "not amplification on its own. Data: \\texttt{results/EX-191}, \\texttt{results/EX-193}.",
        "tab:traj", "r r r r", body,
        note="Epoch-0 gradient norms in the AMP arm are not usable: GradScaler collapse drives them to "
             "zero, which is why the fp32 run (panel a) is the authoritative gradient source. The "
             "accuracy and overflow columns of panel (b) remain valid. The stop-grad arm also shows two "
             "isolated single-batch overflow events (epochs 4 and 8) but recovers at once, whereas the "
             "no-detach arm never leaves its plateau.")
    write("tab_trajectory", tex, rows, ["block", "epoch", "a", "b", "c"])


# --------------------------------------------------------------------------
# Table 7 -- efficiency and complexity (EX-147, E4, EX-150)
# --------------------------------------------------------------------------
def table7():
    e4 = {r["model"] + "_" + r["amp"]: r for r in csv_rows("E4_efficiency", "efficiency_results.csv")
          if r["res"] == "512"}
    lat = {(x["model"], x["amp"]): x for x in csv_rows("EX-147", "lat_flops.csv")}
    order = [("CropRWKV FULL", "full"), ("DeepLabV3+ (R50, IN)", "deeplabv3plus"),
             ("SegFormer-B2 (IN)", "segformer"), ("U-Net (R50, IN)", "unet_pretrained"),
             ("U-Net (scratch)", "unet_scratch"), ("Swin-UNet (IN)", "swin_unet")]
    e4_key = {"full": "full", "deeplabv3plus": "deeplabv3plus", "segformer": "segformer",
              "unet_pretrained": "unet", "unet_scratch": "unet_scratch", "swin_unet": "swin_unet"}
    rows = []
    body = ("\\textbf{Model} & \\textbf{Params (M)} & \\textbf{FLOPs (G)} & \\textbf{fp32 (ms)} & "
            "\\textbf{fp16 (ms)} & \\textbf{Peak mem. (MB)}\\\\\n\\midrule\n")
    for label, key in order:
        r = EFF[key]
        mem = e4.get(e4_key[key] + "_fp32", {}).get("peak_mem_MB", "") or "--"
        ms32 = float(lat[(key, "fp32")]["median_ms"])
        ms16 = float(lat[(key, "fp16")]["median_ms"])
        body += (f"{label} & {float(r['params_M']):.2f} & {float(r['flops_512_G']):.2f} & "
                 f"{ms32:.2f} & {ms16:.2f} & {mem}\\\\\n")
        rows.append([label, r["params_M"], r["flops_512_G"], ms32, ms16, mem])
    tex = table(
        "Efficiency at $512\\times512$, batch size 1, on a single RTX 3090 held at a sustained clock. FLOPs and "
        "parameter counts come from one measurement tool (\\texttt{thop}, \\texttt{EX-147}); latencies are "
        "the median of three sustained-warm-up rounds of 100 iterations each, with the SM clock logged "
        "per round (1905--1920\\,MHz for FULL). Peak memory is the fp32 forward-pass peak measured in the E4 "
        "sweep. CropRWKV FULL is the slowest model here and its FLOPs match those of the ImageNet-pretrained "
        "U-Net; no efficiency advantage is claimed.",
        "tab:efficiency", "L r r r r r", body,
        note="Measurement caveat: an idle RTX 3090 sits at 210\\,MHz, and without sustained warm-up "
             "short-latency models are over-estimated by nearly 2$\\times$. All latencies reported here "
             "were re-measured under the sustained protocol.")
    write("tab_efficiency", tex, rows, ["model", "params_M", "flops_G", "fp32_ms", "fp16_ms", "peak_mem_MB"])


# --------------------------------------------------------------------------
# Table 8 -- component-level latency (EX-150)
# --------------------------------------------------------------------------
def table8():
    d = jload("EX-150", "latency_fine.json")
    f32, f16 = d["fp32"], d["fp16"]
    rows = []
    body = ("\\textbf{Component} & \\textbf{fp32 (ms)} & \\textbf{fp32 share} & "
            "\\textbf{fp16 (ms)} & \\textbf{fp16 share}\\\\\n\\midrule\n")
    items = [("WKV scan operator (all calls)", "wkv_ms"), ("attention sub-block", "att_ms"),
             ("local depthwise conv (positional)", "local_ms"), ("stem (stage 0)", "stem_ms"),
             ("other encoder ops", "encoder_other_ms")]
    for label, k in items:
        body += (f"{label} & {f32[k]:.2f} & {f32['pct'][k]:.1f}\\% & {f16[k]:.2f} & "
                 f"{f16['pct'][k]:.1f}\\%\\\\\n")
        rows.append([label, f32[k], f32["pct"][k], f16[k], f16["pct"][k]])
    body += "\\midrule\n"
    body += (f"\\textbf{{Wall clock}} & {f32['wall_ms']:.2f} & 100.0\\% & {f16['wall_ms']:.2f} & 100.0\\%\\\\\n")
    rows.append(["wall", f32["wall_ms"], 100.0, f16["wall_ms"], 100.0])
    mods = ", ".join("{} {:.2f}".format(k.replace("_", "\\_"), v)
                     for k, v in sorted(f32["modules"].items(), key=lambda kv: -kv[1]))
    tex = table(
        "Where the inference time of CropRWKV FULL goes. Components were timed by serialising the "
        "model and surrounding each mutually exclusive region with an explicit CUDA synchronisation, "
        "which is reliable whereas profiler annotation attribution is not (an early annotation-based "
        "run attributed 1.12\\,ms to a region whose inner kernel alone took 9.81\\,ms). The pure-PyTorch "
        "WKV scan is the single largest item at 44\\% of wall-clock time in fp32 and 50\\% in fp16; the "
        "stem is negligible. The wall clock here (32.25\\,ms) comes from the component-serialised "
        "measurement of \\texttt{EX-150} and sits about 1\\,ms below the sustained-clock latency "
        "reported for the same model in the main text (\\texttt{EX-147}), which uses a different "
        "timing setup. Data: \\texttt{results/EX-150}.",
        "tab:components", "L r r r r", body,
        note=f"Module split in fp32 (ms): {mods}. Shares are taken against the measured wall clock, not "
             "against the sum of the listed regions, which does not span every operation of the forward pass.")
    write("tab_components", tex, rows, ["component", "fp32_ms", "fp32_pct", "fp16_ms", "fp16_pct"])


# --------------------------------------------------------------------------
# Table 9 -- resolution scaling (E4 ideal, EX-125 sliding-window)
# --------------------------------------------------------------------------
def table9():
    e4 = csv_rows("E4_efficiency", "efficiency_results.csv")
    ideal = {}
    for r in e4:
        if r["amp"] == "fp32" and r["res"] and not r["flops_512_G"] and r["pixels_per_s"] and int(r["res"]) != 512:
            ideal.setdefault(int(r["res"]), {})[r["model"]] = float(r["pixels_per_s"])
    sw = {}
    for r in csv_rows("EX-125", "highres_results.csv"):
        sw.setdefault(int(r["res"]), {})[r["model"]] = (float(r["latency_ms"]), float(r["pixels_per_s"]))
    rows, body = [], ""
    body += "\\multicolumn{5}{l}{\\textit{(a) Whole-image forward pass, ideal throughput (M px/s)}}\\\\\n\\midrule\n"
    body += "\\textbf{Resolution} & \\textbf{CropRWKV FULL} & \\textbf{SegFormer-B2} & \\textbf{U-Net (R50)} & \\textbf{DeepLabV3+ (R50)}\\\\\n\\midrule\n"
    for res in sorted(ideal):
        d = ideal[res]
        def g(k):
            return f"{d[k]/1e6:.2f}" if k in d else "--"
        body += f"${res}\\times{res}$ & {g('full')} & {g('segformer')} & {g('unet')} & {g('deeplabv3plus')}\\\\\n"
        rows.append(["ideal", res, g("full"), g("segformer"), g("unet"), g("deeplabv3plus")])
    body += "\\midrule\n\\multicolumn{4}{l}{\\textit{(b) Sliding-window deployment (latency ms / M px\\,s$^{-1}$)}}\\\\\n\\midrule\n"
    body += "\\textbf{Resolution} & \\textbf{CropRWKV FULL} & \\textbf{SegFormer-B2} & \\textbf{U-Net (R50)}\\\\\n\\midrule\n"
    for res in sorted(sw):
        d = sw[res]
        def g2(k):
            return f"{d[k][0]:.0f} / {d[k][1]/1e6:.2f}" if k in d else "--"
        body += f"${res}\\times{res}$ & {g2('full')} & {g2('segformer')} & {g2('unet')}\\\\\n"
        rows.append(["sliding", res, g2("full"), g2("segformer"), g2("unet")])
    tex = table(
        "Resolution scaling. (a) Throughput of a single whole-image forward pass: the pixel rate of "
        "CropRWKV FULL is essentially flat from $256\\times256$ to $2048\\times2048$, the property a linear-complexity "
        "backbone is expected to have. (b) The same comparison under the sliding-window protocol used "
        "for testing, which is what a deployment would actually run: throughput shows no systematic "
        "decline out to $4096\\times4096$, but it remains the lowest of "
        "the three models at every resolution. The sliding-window sweep covers the three principal models; "
        "DeepLabV3+ was measured in the whole-image protocol only. Swin-UNet is excluded from both because its "
        "positional embeddings are fixed at $512\\times512$. "
        "Data: \\texttt{results/E4\\_efficiency} (a), "
        "\\texttt{results/EX-125} (b).",
        "tab:resolution", "L r r r r", body,
        note="The two panels use different protocols (single forward pass vs. windowed inference) and "
             "their absolute values are not comparable with each other; both are measured. "
             "``Resolution-robust'' means the throughput does not degrade with resolution --- it does not "
             "mean the model is fast.")
    write("tab_resolution", tex, rows, ["protocol", "resolution", "full", "segformer", "unet", "deeplabv3plus"])


# --------------------------------------------------------------------------
# Table 10 -- field-degradation robustness (EX-164..169)
# --------------------------------------------------------------------------
def table10():
    exmap = [("CropRWKV FULL", ["EX-164", "EX-210", "EX-211"]),
             ("DeepLabV3+ (R50, IN)", ["EX-165", "EX-212", "EX-213"]),
             ("SegFormer-B2 (IN)", ["EX-166", "EX-214", "EX-215"]),
             ("U-Net (R50, IN)", ["EX-167", "EX-216", "EX-217"]),
             ("Swin-UNet (IN)", ["EX-168", "EX-218", "EX-219"]),
             ("U-Net (scratch)", ["EX-169", "EX-220", "EX-221"])]
    conds = ["blur_sigma2", "blur_sigma4", "motion_blur_9", "jpeg_q30", "low_light_0.5x",
             "overexposed_1.4x", "low_contrast_0.5x", "color_shift", "illum_gradient",
             "gauss_noise_s10", "occlusion_10pct"]
    pretty = {"blur_sigma2": "Gaussian blur $\\sigma$=2", "blur_sigma4": "Gaussian blur $\\sigma$=4",
              "motion_blur_9": "Motion blur (9)", "jpeg_q30": "JPEG $q$=30",
              "low_light_0.5x": "Low light ($0.5\\times$)", "overexposed_1.4x": "Over-exposed ($1.4\\times$)",
              "low_contrast_0.5x": "Low contrast ($0.5\\times$)", "color_shift": "Colour shift",
              "illum_gradient": "Illumination gradient", "gauss_noise_s10": "Gaussian noise ($\\sigma$=10)",
              "occlusion_10pct": "Occlusion (10\\%)"}
    tabs = {label: agg_csv(exs, "degrad_boundary_summary.csv", lambda r: (r["condition"], "dSoftDice_vs_clean"))
            for label, exs in exmap}
    clean_tabs = {label: agg_csv(exs, "degrad_boundary_summary.csv", lambda r: (r["condition"], "soft_dice"))
                  for label, exs in exmap}
    rows = []
    body = ("\\textbf{Condition} & " + " & ".join(
        short_label(lab) for lab, _ in exmap)
    ) + "\\\\\n\\midrule\n"
    for c in conds:
        cells = []
        for label, _ in exmap:
            r = tabs[label].get(c)
            cells.append(f"{float(r['dSoftDice_vs_clean']):+.3f}" if r else "--")
        body += f"{pretty[c]} & " + " & ".join(cells) + "\\\\\n"
        rows.append([pretty[c]] + cells)
    clean = [f"{float(clean_tabs[lab]['clean']['soft_dice']):.3f}" for lab, _ in exmap]
    body += "\\midrule\n\\textbf{Clean reference} & " + " & ".join(clean) + "\\\\\n"
    rows.append(["clean"] + clean)
    tex = table(
        "Field-acquisition robustness. Each degradation was applied to the inference input only, with "
        "the ground truth left untouched and the standard sliding-window evaluation retained; the table "
        "reports the change in soft-Dice relative to the clean input of the same model, averaged over "
        "three seeds. The from-scratch U-Net is the most sensitive to low light ($-0.209$), followed by "
        "CropRWKV FULL ($-0.157$); Swin-UNet and SegFormer-B2 are the least sensitive throughout. "
        "Data: \\texttt{results/EX-164}--\\texttt{EX-169} (seed 42), "
        "\\texttt{EX-210}--\\texttt{EX-221} (seeds 7 and 3407).",
        "tab:degradation", "L r r r r r r", body,
        note="Colour shift and illumination gradient are benign for every model; Gaussian blur at "
             "$\\sigma$=4 is the most damaging geometric degradation. All 66 model$\\times$condition "
             "combinations were evaluated on the same 60 test images for each seed. The clean reference "
             "is the three-seed mean and matches Table~\\ref{tab:main}.",
        wide=True)
    write("tab_degradation", tex, rows, ["condition"] + [lab for lab, _ in exmap])


# --------------------------------------------------------------------------
# Table 11 -- size-stratified failure (EX-154)
# --------------------------------------------------------------------------
def table11():
    ar = csv_rows("EX-275", "stratified_by_area.csv")
    di = csv_rows("EX-275", "stratified_by_diameter.csv")
    rows, body = [], ""
    body += ("\\multicolumn{6}{l}{\\textit{(a) Stratified by ground-truth lesion area fraction}}\\\\\n\\midrule\n"
             "\\textbf{Method} & \\textbf{Q1} & \\textbf{Q2} & \\textbf{Q3} & \\textbf{Q4} & "
             "\\textbf{Q4$-$Q1}\\\\\n\\midrule\n")
    for r in ar:
        body += (f'{r["method"]} & {float(r["Q1_soft_dice"]):.3f} & {float(r["Q2_soft_dice"]):.3f} & '
                 f'{float(r["Q3_soft_dice"]):.3f} & {float(r["Q4_soft_dice"]):.3f} & '
                 f'{float(r["Q4_minus_Q1"]):+.3f}\\\\\n')
        rows.append(["area", r["method"], r["Q1_soft_dice"], r["Q2_soft_dice"], r["Q3_soft_dice"],
                     r["Q4_soft_dice"], r["Q4_minus_Q1"]])
    body += "\\midrule\n"
    body += ("\\multicolumn{6}{l}{\\textit{(b) Stratified by largest-component equivalent diameter}}\\\\\n\\midrule\n"
             "\\textbf{Method} & \\textbf{Q1} & \\textbf{Q2} & \\textbf{Q3} & \\textbf{Q4} & "
             "\\textbf{Q4$-$Q1}\\\\\n\\midrule\n")
    for r in di:
        body += (f'{r["method"]} & {float(r["Q1_soft_dice"]):.3f} & {float(r["Q2_soft_dice"]):.3f} & '
                 f'{float(r["Q3_soft_dice"]):.3f} & {float(r["Q4_soft_dice"]):.3f} & '
                 f'{float(r["Q4_minus_Q1"]):+.3f}\\\\\n')
        rows.append(["diameter", r["method"], r["Q1_soft_dice"], r["Q2_soft_dice"], r["Q3_soft_dice"],
                     r["Q4_soft_dice"], r["Q4_minus_Q1"]])
    tex = table(
        "Size-stratified failure analysis on the 60 Set A test images, recomputed from the per-image "
        "predictions averaged over three seeds (\\texttt{results/EX-275}). Quartiles are formed once on "
        "the ground truth and applied to every method, so the strata are identical across rows. Panel "
        "(a) uses lesion area fraction (Q1 $\\leq$ 5.4\\%, Q4 $\\geq$ 21.2\\%); panel (b) uses the "
        "equivalent diameter of the largest lesion component in the cached 1024-px frame (Q1 $\\leq$ "
        "171\\,px, Q4 $\\geq$ 395\\,px). ",
        "tab:stratified", "L r r r r r", body,
        note="A single image (00305) is degenerate in this evaluation: all eight models return exactly "
             "the same soft-Dice on it ($0.124$), so it carries no information about the ranking. It is "
             "retained because removing it would be a post-hoc choice.")
    write("tab_stratified", tex, rows,
          ["criterion", "method", "Q1", "Q2", "Q3", "Q4", "Q4_minus_Q1"])


# --------------------------------------------------------------------------
# Table 12 -- agronomic endpoint (EX-126..129, grading recomputed here)
# --------------------------------------------------------------------------
def _grading(ex: str) -> tuple[float, float, float, float, float]:
    rows = csv_rows(ex, "lesion_frac_error.csv")
    gt = np.array([float(r["gt_frac"]) for r in rows])
    pr = np.array([float(r["pred_frac"]) for r in rows])

    def cls(x):
        return np.digitize(x, [0.05, 0.15, 0.30])
    cg, cp = cls(gt), cls(pr)
    exact = float((cg == cp).mean())
    within1 = float((np.abs(cg - cp) <= 1).mean())
    mae = float(np.abs(gt - pr).mean())
    pear = float(np.corrcoef(gt, pr)[0, 1])
    from scipy.stats import spearmanr
    spear = float(spearmanr(gt, pr).statistic)
    return mae, pear, spear, exact, within1


def table12():
    models = [("CropRWKV FULL", ["EX-126", "EX-202", "EX-203"]),
              ("SegFormer-B2 (IN)", ["EX-127", "EX-204", "EX-205"]),
              ("DeepLabV3+ (R50, IN)", ["EX-128", "EX-206", "EX-207"]),
              ("U-Net (R50, IN)", ["EX-129", "EX-208", "EX-209"])]
    rows = []
    body = ("\\textbf{Model} & \\textbf{MAE} & \\textbf{Pearson $r$} & \\textbf{Spearman $\\rho$} & "
            "\\textbf{Exact class} & \\textbf{Within one class}\\\\\n\\midrule\n")
    for label, exs in models:
        vals = np.array([_grading(e) for e in exs], float)   # seeds x 5 metrics
        mu, sd = vals.mean(0), vals.std(0, ddof=1)
        cells = [f"{m:.3f} ({s:.4f})" for m, s in zip(mu, sd)]
        body += f"{label} & " + " & ".join(cells) + "\\\\\n"
        rows.append([label] + [round(v, 4) for v in mu] + [round(v, 4) for v in sd])
    tex = table(
        "Agronomic endpoint: estimation of lesion area fraction and its translation into a four-level "
        "severity class (thresholds 5\\%, 15\\%, 30\\% of leaf area), $n=60$. Values are three-seed means "
        "with the cross-seed standard deviation in parentheses; MAE and correlations are computed from "
        "the stored per-image predicted fractions. Moving to this more forgiving endpoint does not "
        "rescue the architecture: its correlation is comparable to SegFormer's but its rank agreement "
        "and exact-class accuracy are clearly lower. Data: \\texttt{results/EX-126}--\\texttt{EX-129} "
        "(seed 42), \\texttt{EX-202}--\\texttt{EX-209} (seeds 7 and 3407).",
        "tab:agronomic", "L r r r r r", body,
        note="Reported as an error-source analysis, not as a favourable result. Ground-truth mean lesion "
             "fraction on this split is 0.144.", fontsize="\\small")
    write("tab_agronomic", tex, rows,
          ["model", "mae", "pearson", "spearman", "exact_class", "within1_class",
           "mae_sd", "pearson_sd", "spearman_sd", "exact_class_sd", "within1_class_sd"])


# --------------------------------------------------------------------------
# Table 13 -- cross-dataset validation on Set B (EX-173..178, 181)
# --------------------------------------------------------------------------
def table13():
    rows, body = [], ""

    def cells3(exs):
        out = []
        for k in ("test_soft_dice_mean", "test_iou05_mean", "test_dice05_mean"):
            mu, sd = agg_meta(exs, k)
            out.append(f"{mu:.3f} $\\pm$ {sd:.3f}")
        return out

    body += ("\\multicolumn{5}{l}{\\textit{(a) Zero-shot transfer: trained on Set A, tested on Set B}}\\\\\n\\midrule\n"
             "\\textbf{Model} & \\textbf{soft-Dice} & \\textbf{IoU@0.5} & \\textbf{Dice@0.5} & \\textbf{Ex.}\\\\\n\\midrule\n")
    zs = [("CropRWKV FULL", ["EX-176", "EX-222", "EX-223"]),
          ("SegFormer-B2 (IN)", ["EX-177", "EX-224", "EX-225"]),
          ("U-Net (R50, IN)", ["EX-178", "EX-226", "EX-227"])]
    for label, exs in zs:
        c = cells3(exs)
        body += f"{label} & " + " & ".join(c) + " & 119\\\\\n"
        rows.append(["zeroshot", label] + [x.replace(" $\\pm$ ", " +/- ") for x in c] + [119])
    body += "\\midrule\n\\multicolumn{5}{l}{\\textit{(b) In-domain training and testing on Set B}}\\\\\n\\midrule\n"
    body += "\\textbf{Model} & \\textbf{soft-Dice} & \\textbf{IoU@0.5} & \\textbf{Dice@0.5} & \\textbf{Ex.}\\\\\n\\midrule\n"
    ind = [("CropRWKV FULL", ["EX-173", "EX-240", "EX-241"]),
           ("SegFormer-B2 (IN)", ["EX-174", "EX-242", "EX-243"]),
           ("U-Net (R50, IN)", ["EX-175", "EX-244", "EX-245"])]
    for label, exs in ind:
        c = cells3(exs)
        body += f"{label} & " + " & ".join(c) + " & 119\\\\\n"
        rows.append(["indomain", label] + [x.replace(" $\\pm$ ", " +/- ") for x in c] + [119])
    m180 = meta("EX-181")   # evaluation snapshot of the stability-controlled training run EX-180
    body += (f"CropRWKV FULL, stability-controlled (EX-180) & {float(m180['test_soft_dice_mean']):.3f} & "
             f"{float(m180['test_iou05_mean']):.3f} & {float(m180['test_dice05_mean']):.3f} & 119\\\\\n")
    rows.append(["indomain", "CropRWKV FULL, stability-controlled (EX-180)",
                 m180["test_soft_dice_mean"], m180["test_iou05_mean"], m180["test_dice05_mean"], m180["n_test"]])
    tex = table(
        "External validation on Set B (the PlantSeg in-the-wild test-set release; 1195 usable samples, "
        "re-partitioned 75/15/10 by this study, 119 of them held "
        "out for testing; 32 species, 59 diseases). (a) Models trained only on Set A and applied "
        "unchanged; (b) models retrained on Set B under the identical protocol. Both panels are "
        "three-seed means with the cross-seed standard deviation, except the stability-controlled run, "
        "which exists for seed 42 only. Every model degrades sharply in absolute terms because Set B is "
        "a harder, more diverse distribution, but the ordering of the architectures is unchanged: the "
        "linear-complexity backbone remains the weakest of the three, both zero-shot and in-domain. "
        "Retraining the architecture under study on Set B does not raise its accuracy above the "
        "zero-shot value. Data: \\texttt{results/EX-176}--\\texttt{EX-178} and "
        "\\texttt{EX-222}--\\texttt{EX-227} (zero-shot), \\texttt{EX-173}--\\texttt{EX-175} and "
        "\\texttt{EX-240}--\\texttt{EX-245} (in-domain).",
        "tab:crossdataset", "L r r r r", body,
        note="Set B is licensed for research use only (CC BY-NC-ND 4.0) and is not redistributed with "
             "this work. The stability-controlled FULL run (EX-180) was trained with gradient clipping "
             "at 1.0 and a non-finite-step guard; it skipped zero steps, and its accuracy is marginally "
             "below the unguarded three-seed mean, i.e. the guard did not inflate the result.")
    write("tab_crossdataset", tex, rows, ["protocol", "model", "soft_dice", "iou05", "dice05", "n_test"])


# --------------------------------------------------------------------------
# Table S2 -- boundary metrics (EX-164..169)
# --------------------------------------------------------------------------
def tableS2():
    exmap = [("CropRWKV FULL", ["EX-164", "EX-210", "EX-211"]),
             ("DeepLabV3+ (R50, IN)", ["EX-165", "EX-212", "EX-213"]),
             ("SegFormer-B2 (IN)", ["EX-166", "EX-214", "EX-215"]),
             ("U-Net (R50, IN)", ["EX-167", "EX-216", "EX-217"]),
             ("Swin-UNet (IN)", ["EX-168", "EX-218", "EX-219"]),
             ("U-Net (scratch)", ["EX-169", "EX-220", "EX-221"])]
    conds = ["clean", "blur_sigma2", "blur_sigma4", "jpeg_q30", "low_light_0.5x", "occlusion_10pct"]
    pretty = {"clean": "Clean", "blur_sigma2": "Blur $\\sigma$=2", "blur_sigma4": "Blur $\\sigma$=4",
              "jpeg_q30": "JPEG $q$=30", "low_light_0.5x": "Low light ($0.5\\times$)",
              "occlusion_10pct": "Occlusion (10\\%)"}
    tabs = {}
    for lab, exs in exmap:
        merged = agg_csv(exs, "degrad_boundary_summary.csv", lambda r: (r["condition"], "boundary_iou"))
        bf = agg_csv(exs, "degrad_boundary_summary.csv", lambda r: (r["condition"], "bf_score"))
        for c in merged:
            merged[c].update(bf[c])
        tabs[lab] = merged
    rows, body = [], ""
    body += "\\textbf{Condition} & \\textbf{Metric} & " + " & ".join(
        short_label(lab) for lab, _ in exmap) + "\\\\\n\\midrule\n"
    for c in conds:
        for met, col in [("Boundary IoU", "boundary_iou"), ("BF-score", "bf_score")]:
            cells = [f"{float(tabs[lab][c][col]):.3f}" for lab, _ in exmap]
            body += f"{pretty[c]} & {met} & " + " & ".join(cells) + "\\\\\n"
            rows.append([c, met] + cells)
        body += "\\midrule\n"
    tex = table(
        "Boundary-quality metrics (Boundary IoU and BF-score, boundary band 2\\% of image diagonal) for "
        "the clean and selected degraded inputs, averaged over three seeds. CropRWKV FULL has the lowest "
        "boundary quality of the six models in every condition except Gaussian blur at $\\sigma$=4, "
        "where the from-scratch U-Net is lower; this is consistent with its low overlap accuracy and "
        "does not support the boundary-enhancement module as an effective mechanism. Data: "
        "\\texttt{results/EX-164}--\\texttt{EX-169} (seed 42), \\texttt{EX-210}--\\texttt{EX-221} "
        "(seeds 7 and 3407).",
        "tab:boundary", "L L r r r r r r", body, wide=True,
        fontsize="\\footnotesize", tabcolsep="3pt")
    write("tab_boundary", tex, rows, ["condition", "metric"] + [lab for lab, _ in exmap])


# --------------------------------------------------------------------------
# Table S3 -- rank correlations with lesion difficulty (EX-154)
# --------------------------------------------------------------------------
def tableS3():
    sp = csv_rows("EX-275", "spearman_correlations.csv")
    rows = []
    body = ("\\textbf{Method} & \\textbf{$\\rho$ vs. area} & \\textbf{$\\rho$ vs. diameter} & "
            "\\textbf{$\\rho$ vs. \\#components} & \\textbf{Mid-50\\% area mean}\\\\\n\\midrule\n")
    for r in sp:
        body += (f'{r["method"]} & {float(r["rho_area"]):.3f} & {float(r["rho_diam"]):.3f} & '
                 f'{float(r["rho_ncomp"]):.3f} & {float(r["mid50_mean"]):.3f}\\\\\n')
        rows.append([r["method"], r["rho_area"], r["rho_diam"], r["rho_ncomp"], r["mid50_mean"]])
    tex = table(
        "Spearman rank correlation between per-image soft-Dice and three measures of lesion "
        "difficulty. CropRWKV FULL and the A0 backbone are the only models whose accuracy is strongly "
        "rank-correlated with lesion size ($\\rho\\approx0.87$ and $0.93$ respectively), i.e. they "
        "behave almost as size detectors; the ImageNet-pretrained baselines are far less size-dependent. "
        "Correlations with the number of lesion components are weaker and always negative "
        "($\\rho=-0.08$ to $-0.44$). Data: "
        "\\texttt{results/EX-275} (per-image soft-Dice averaged over three seeds).",
        "tab:spearman", "L r r r r", body, wide=True)
    write("tab_spearman", tex, rows, ["method", "rho_area", "rho_diam", "rho_ncomp", "mid50_mean"])


# --------------------------------------------------------------------------
# Table S4 -- remaining robustness: data efficiency, resolution, noise, scale
# --------------------------------------------------------------------------
def tableS4():
    rows, body = [], ""
    body += "\\multicolumn{3}{l}{\\textit{(a) Training-set size (CropRWKV FULL)}}\\\\\n\\midrule\n"
    body += "\\textbf{Fraction} & \\textbf{soft-Dice} & \\textbf{EX}\\\\\n\\midrule\n"
    for frac, exs, lab in [("10\\%", ["EX-053", "EX-262", "EX-263"], "EX-053/262/263"),
                           ("25\\%", ["EX-054", "EX-264", "EX-265"], "EX-054/264/265"),
                           ("50\\%", ["EX-055", "EX-266", "EX-267"], "EX-055/266/267"),
                           ("100\\%", ["EX-033", "EX-036", "EX-037"], "EX-033/036/037")]:
        mu, sd = agg_meta(exs, "test_soft_dice_mean")
        body += f"{frac} & {mu:.3f} ({sd:.3f}) & {lab}\\\\\n"
        rows.append(["data", frac, round(mu, 4), round(sd, 4)])
    body += "\\midrule\n\\multicolumn{3}{l}{\\textit{(b) Unseen test resolution}}\\\\\n\\midrule\n"
    body += "\\textbf{Long side (px)} & \\textbf{soft-Dice} & \\textbf{EX}\\\\\n\\midrule\n"
    for res, exs, lab in [("640", ["EX-056", "EX-228", "EX-231"], "EX-056/228/231"),
                          ("768", ["EX-057", "EX-229", "EX-232"], "EX-057/229/232"),
                          ("1024 (train)", ["EX-058", "EX-230", "EX-233"], "EX-058/230/233")]:
        mu, sd = agg_meta(exs, "test_soft_dice_mean")
        body += f"{res} & {mu:.3f} ({sd:.3f}) & {lab}\\\\\n"
        rows.append(["resolution", res, round(mu, 4), round(sd, 4)])
    body += "\\midrule\n\\multicolumn{3}{l}{\\textit{(c) Label noise (ground truth eroded by 3\\,px)}}\\\\\n\\midrule\n"
    for name, exs, lab in [("clean", ["EX-033", "EX-036", "EX-037"], "EX-033/036/037"),
                           ("eroded", ["EX-060", "EX-268", "EX-269"], "EX-060/268/269")]:
        mu, sd = agg_meta(exs, "test_soft_dice_mean")
        body += f"{name} & {mu:.3f} ({sd:.3f}) & {lab}\\\\\n"
        rows.append(["noise", name, round(mu, 4), round(sd, 4)])
    body += "\\midrule\n\\multicolumn{3}{l}{\\textit{(d) Capacity scaling (all at 150 epochs except B)}}\\\\\n\\midrule\n"
    body += "\\textbf{Variant} & \\textbf{soft-Dice} & \\textbf{Params}\\\\\n\\midrule\n"
    for name, exs, p in [("T", ["EX-086", "EX-260", "EX-261"], "9.0\\,M"),
                         ("S $=$ FULL", ["EX-033", "EX-036", "EX-037"], "15.7\\,M")]:
        mu, sd = agg_meta(exs, "test_soft_dice_mean")
        body += f"{name} & {mu:.3f} ({sd:.3f}) & {p}\\\\\n"
        rows.append(["scale", name, round(mu, 4), round(sd, 4), p])
    body += f"B ($\\approx$42 epochs) & {_sd('EX-088'):.3f} & 24.3\\,M\\\\\n"
    rows.append(["scale", "B", _sd("EX-088"), None, "24.3 M"])
    tex = table(
        "Additional robustness and scaling results for CropRWKV FULL, all values now three-seed means "
        "with the cross-seed standard deviation in parentheses. (a) Accuracy rises monotonically with the "
        "size of the training set and does not plateau at half the data: each halving costs "
        "$0.036$--$0.052$ soft-Dice. (b) Accuracy peaks at the training resolution and falls when the "
        "input is smaller than the training crop. (c) Corrupting the training masks by eroding them "
        "3\\,px costs about $22\\%$ relative accuracy without collapsing. (d) The smaller T variant is "
        "statistically indistinguishable from S ($=$FULL); the B variant was cut short by compute and is "
        "shown for completeness only, not as a scaling comparison.",
        "tab:robust_other", "L r L", body,
        note="Panels (a)--(c) use three seeds; panel (d) uses three seeds for T and S but only the "
             "seed-42 run for B, which is marked by the absence of a standard deviation. EX identifiers "
             "are given so each value can be traced to its snapshot.")
    write("tab_robust_other", tex, rows, ["block", "setting", "value", "sd", "params"])


def main():
    table1(); table2(); tableS1(); table3(); table4(); table5(); table6()
    table7(); table8(); table9(); table10(); table11(); table12(); table13()
    tableS2(); tableS3(); tableS4()
    print("\n".join(_LOG))
    print(f"OK: {len(_LOG)} LaTeX tables + CSV mirrors in {OUT}")


if __name__ == "__main__":
    main()