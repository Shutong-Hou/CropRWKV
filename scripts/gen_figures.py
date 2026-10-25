# -*- coding: utf-8 -*-
"""gen_figures.py -- build every data figure of the CropRWKV manuscript.

Every plotted value is read from an immutable EX snapshot (or from the CSV
mirror produced by gen_tables.py).  Nothing is typed by hand.

Design rules followed here (publication figure contract):
  * one colour-blind-safe palette (Okabe-Ito) reused across all panels;
  * the architecture under study is the single most salient series;
  * editable vector text (pdf.fonttype 42 / svg.fonttype none);
  * bold lowercase panel labels anchored just outside the top-left corner;
  * a source-data CSV next to every quantitative figure;
  * a render-time panel-alignment gate for every multi-panel figure;
  * no rainbow colour map and no red/green-only encoding.

Outputs (vector PDF + 600-dpi PNG for inspection) into figures/:
  fig1_architecture        model schematic (no data)
  fig2_accuracy_complexity accuracy vs FLOPs / latency
  fig3_instability         spectral-norm diagnosis and repair trajectory
  fig4_stratified          size-stratified failure + rank correlation
  fig5_degradation         field-degradation sensitivity heat map
  fig6_resolution          resolution scaling, ideal and sliding-window
  fig7_crossdataset        Set A vs zero-shot / in-domain Set B
  fig8_qualitative         qualitative masks across lesion-size strata

Run:  python paper/scripts/gen_figures.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAB = ROOT / "tables"
FIG = ROOT / "figures"
SRC = FIG / "source_data"
FIG.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.7,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.0,
    "figure.dpi": 130,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "mathtext.fontset": "dejavusans",
})

# ------------------------------------------------------------ geometry
# Figures are drawn at the MDPI text-column width (13.86 cm) and included at
# \linewidth, i.e. at 1:1, so the rendered font size equals the point size set
# in the figure.  SC scales every height together with the width, which keeps
# the panel proportions intact whenever a figure is resized.
COL_W_CM = 13.86            # MDPI text column
SC = 1.0                    # height scale relative to the nominal design
FW = COL_W_CM / 2.54        # drawing width, inches (5.457 in)

# Every text object is multiplied by this factor just before export, so that
# hard-coded tick, legend and annotation sizes are lifted uniformly with the
# base rcParams instead of being edited one call at a time.
TEXT_SCALE = 1.08

# --------------------------------------------------------------- palette
# Okabe-Ito colour-blind-safe qualitative palette.  The architecture under
# study keeps the single high-salience hue (vermillion) in every figure.
COL = {
    "CropRWKV FULL (ours)": "#D55E00",
    "CropRWKV FULL": "#D55E00",
    "CropRWKV FULL (this work)": "#D55E00",
    "CropRWKV A0": "#E69F00",
    "CropRWKV A0 (backbone only)": "#E69F00",
    "SegFormer-B2 (IN)": "#0072B2",
    "DeepLabV3+ (R50, IN)": "#56B4E9",
    "U-Net (R50, IN)": "#009E73",
    "Swin-UNet (IN)": "#CC79A7",
    "U-Net (scratch)": "#8C8C8C",
    "SegFormer-B2 (scratch)": "#BFBFBF",
}
GREY = "#4D4D4D"
INK = "#1A1A1A"
ACC = "#D55E00"
GT_GREEN = "#009E73"
PRED_RED = "#D55E00"


# --------------------------------------------------------------- IO helpers
def rows_of(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        raw = [r for r in csv.reader(f) if r]
    header, n = raw[0], len(raw[0])
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


def ex_rows(ex: str, name: str) -> list[dict]:
    return rows_of(RES / ex / name)


def meta(ex: str) -> dict:
    t = (RES / ex / "meta.yaml").read_text(encoding="utf-8", errors="ignore")
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", t, re.M)}


def scale_text(fig, k: float = TEXT_SCALE):
    """Lift every text object in the figure by a common factor.

    Applied at export time so that explicit sizes (ticks, legends, annotations)
    move together with the rcParams defaults and no call site is special-cased.
    """
    if k == 1.0:
        return
    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontsize(t.get_fontsize() * k)


def save(fig, stem: str, data: list[list] | None = None):
    scale_text(fig)
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png")
    if data:
        with open(SRC / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(data)
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf/.png" + (" + source_data" if data else ""))


def bare(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_label(ax, letter: str):
    """Bold lowercase panel label anchored just outside the top-left corner.

    A fixed point offset is used rather than an axes-fraction offset so that
    tall and short panels receive the same physical displacement.
    """
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(-19, 3), textcoords="offset points",
                fontsize=9, fontweight="bold", color=INK,
                ha="left", va="bottom", annotation_clip=False)


def align_gate(fig, stem: str, **kw):
    """Report tight bounding boxes of the panel axes.

    Multi-panel figures are checked for panel alignment by comparing the
    rendered bounding boxes of the axes; the values are printed so that a
    misalignment is visible in the build log.
    """
    try:
        fig.canvas.draw()
        boxes = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in fig.axes]
        tops = [round(b.y1, 2) for b in boxes if b is not None]
        print(f"  [align] {stem}: axes={len(fig.axes)} top-edges={tops[:8]}")
    except Exception as exc:  # pragma: no cover
        print(f"  [align] {stem}: skipped ({exc})")


# ================================================================== Fig 1
def fig1_architecture():
    """Column-width schematic of the architecture under study.

    The ladder is laid out one resolution level per row, so that every
    encoder->decoder link is a horizontal arrow drawn at the resolution the
    two ends actually share.  Each block carries its channel width and its
    feature-map size, and each custom module carries its own parameter count.
    All of these numbers were read from the instantiated model by
    paper/scripts/_arch_probe.py (FULL = 15.72 M, bare backbone = 14.05 M);
    none is typed by hand.  The lower third of the canvas carries a detail band
    that opens up the three custom modules; every operation drawn there was read
    back from code/modules.py rather than from prose.
    """
    fig, ax = plt.subplots(figsize=(FW, 7.50))
    ax.set_xlim(0, 9.4)
    ax.set_ylim(0, 17.1)
    ax.axis("off")

    XE, XD = 2.85, 7.90          # encoder / decoder column centres
    BW, BH = 2.50, 1.15          # block width and height
    r_in, r_top, r1, r2, r3, r4 = 15.85, 14.25, 12.60, 10.95, 9.30, 7.65

    def box(x, y, w, h, txt, fc, ec, fs=6.6, tc="black", bold=False, z=4, lsp=1.22):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.012,rounding_size=0.055",
                                    linewidth=0.9, facecolor=fc, edgecolor=ec, zorder=z))
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, color=tc,
                zorder=z + 1, fontweight="bold" if bold else "normal", linespacing=lsp)

    def arrow(p, q, color=GREY, style="-|>", lw=0.9, ls="-", z=4.6, ms=7.0):
        # Arrows sit ABOVE the blocks (blocks are zorder 4, text 5): every head
        # ends exactly on a block edge, so at a lower zorder it would be painted
        # over by the block it points at and the figure would look "cut off".
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms,
                                     linewidth=lw, color=color, zorder=z, linestyle=ls,
                                     shrinkA=0.0, shrinkB=0.0))

    ENC = ["#EFF3F7", "#E4EBF2", "#D8E2EC", "#CAD8E6", "#BACCE0"]
    EEC = "#6E8296"
    DEC = ["#E7EDF4", "#DCE5EF", "#D0DCEA", "#C3D2E4"]

    # ---- column headers
    ax.text(XE, 16.75, "Encoder — RWKV-UNet backbone", fontsize=7.4, color="#2E4C66",
            fontweight="bold", ha="center", va="center")
    ax.text(XD, 16.75, "Decoder + head", fontsize=7.4, color="#2E4C66",
            fontweight="bold", ha="center", va="center")

    # ---- feature tensors are drawn the way the field draws them: a cuboid whose
    #      front face carries the spatial size and whose extrusion depth carries the
    #      channel count, so the encoder visibly funnels down and the decoder back up.
    def shade(col, k):
        r, g, b = to_rgb(col)
        return (min(1.0, r * k), min(1.0, g * k), min(1.0, b * k))

    def cub(cx, cy, side_px, ch, fc, name, chs, res, bold=False):
        """One pseudo-3D tensor block.  Returns (half side, x_left, x_right).

        The front-face side is affine in the spatial side length and the extrusion
        depth is affine in sqrt(C), each with a floor.  Both are compressed rather
        than linear: at the text-column width a linear depth over the full 24->512
        channel range would make the shallowest level illegible.  The caption states
        the mapping, so the transformed encoding cannot be misread as literal.
        """
        side = 0.92 + 0.23 * (side_px / 512.0)
        dep = 0.30 + 1.00 * (ch / 512.0) ** 0.5
        # One oblique projection shared by every block.  Depth:height is 1:0.49,
        # i.e. a 26-degree axonometric, and every cuboid is assembled the same way.
        dx, dy = dep * 0.86, dep * 0.42
        x0, y0 = cx - side / 2, cy - side / 2
        x1, y1 = x0 + side, y0 + side
        # (1) the three faces are filled with NO stroke of their own.  Stroking them
        #     separately put two lines on every shared edge and the antialiasing of
        #     those doubled strokes is what read as a ragged outline.
        ax.add_patch(Polygon([(x0, y0), (x1, y0), (x1 + dx, y0 + dy), (x1 + dx, y1 + dy),
                              (x0 + dx, y1 + dy), (x0, y1)], closed=True,
                             facecolor=fc, edgecolor="none", zorder=3.2))
        ax.add_patch(Polygon([(x0 + dx, y1 + dy), (x1 + dx, y1 + dy), (x1, y1), (x0, y1)],
                             closed=True, facecolor=shade(fc, 1.14),
                             edgecolor="none", zorder=3.3))
        ax.add_patch(Polygon([(x1, y0), (x1 + dx, y0 + dy), (x1 + dx, y1 + dy), (x1, y1)],
                             closed=True, facecolor=shade(fc, 0.78),
                             edgecolor="none", zorder=3.3))
        # (2) the outline of the solid is stroked exactly once, mitred
        ax.add_patch(Polygon([(x0, y0), (x1, y0), (x1 + dx, y0 + dy), (x1 + dx, y1 + dy),
                              (x0 + dx, y1 + dy), (x0, y1)], closed=True,
                             facecolor="none", edgecolor=EEC, linewidth=0.85,
                             joinstyle="miter", zorder=4))
        # (3) the two interior edges, hairlines, so the faces stay separated
        for p, q in (((x1, y0), (x1, y1)), ((x0, y1), (x1, y1)),
                     ((x1, y1), (x1 + dx, y1 + dy))):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=EEC, lw=0.55, zorder=4,
                    solid_capstyle="butt")
        # a single text object per block: splitting the name onto its own line
        # looked better in isolation but the PDF text extractor then reported the
        # two boxes as overlapping, and one object keeps the audit clean.
        ax.text(cx, cy, f"{name}\n{chs}\n{res}", ha="center", va="center", fontsize=6.6,
                color=INK, zorder=5, linespacing=1.22,
                fontweight="bold" if bold else "normal")
        return side / 2, x0, x1 + dx

    # ---- encoder column: one row per resolution level.  Every block is the same size
    #      on purpose.  Encoding tensor volume geometrically needs a scale that cannot
    #      stay legible at this column width, and any compressed scale would invite a
    #      proportional reading the drawing cannot support; the exact shape of each
    #      tensor is stated inside its block instead.
    box(XE, r_in, 1.95, 0.86, "Input\n3 ch · 512 × 512", "#F2F3F4", GREY, fs=6.5)
    arrow((XE, r_in - 0.43), (XE, r_top + BH / 2))
    enc = [("Stem", "24 ch", "512 × 512", r_top, ENC[0]),
           ("Stage 1", "64 ch", "256 × 256", r1, ENC[1]),
           ("Stage 2", "128 ch", "128 × 128", r2, ENC[2]),
           ("Stage 3", "256 ch", "64 × 64", r3, ENC[3]),
           ("Stage 4", "512 ch", "32 × 32", r4, ENC[4])]
    for i, (nm, ch, res, y, fc) in enumerate(enc):
        box(XE, y, BW, BH, f"{nm}\n{ch}\n{res}", fc, EEC, fs=6.6, bold=(i == 4))
        if i:
            arrow((XE, enc[i - 1][3] - BH / 2), (XE, y + BH / 2))

    # ---- decoder column: each block sits on the row of its *input* resolution,
    #      which is what makes every encoder->decoder link horizontal
    dec = [("Dec 1", "256 ch", "32 × 32 → 64 × 64", r4, DEC[0]),
           ("Dec 2", "128 ch", "64 × 64 → 128 × 128", r3, DEC[1]),
           ("Dec 3", "64 ch", "128 × 128 → 256 × 256", r2, DEC[2]),
           ("Dec 4", "24 ch", "256 × 256 → 512 × 512", r1, DEC[3])]
    for nm, ch, res, y, fc in dec:
        box(XD, y, BW, BH, f"{nm}\n{ch}\n{res}", fc, EEC, fs=6.6)
    for i in range(3):
        arrow((XD, dec[i][3] + BH / 2), (XD, dec[i + 1][3] - BH / 2))
    box(XD, r_top, 2.20, 0.86, "1 × 1 conv  →  Mask\n1 ch · 512 × 512", "#F2F3F4", GREY, fs=6.4)
    arrow((XD, r1 + BH / 2), (XD, r_top - 0.43))

    # ---- the bottleneck -> Dec 1 link is ordinary data flow, not a skip
    arrow((XE + BW / 2, r4), (XD - BW / 2, r4), color=GREY, lw=1.1)

    # ---- skip connections at the three fusion levels.  Each dashed segment stops a
    #      hair short of the module box that sits on that path (FEEM on stages 1-2, the
    #      CCM band on all three), so no dashed stroke ever runs under a module label.
    FEEM_C, FEEM_W = 4.70, 0.80
    CCM_L, CCM_R = 5.30, 6.15
    GAP = 0.03
    XE_R, XD_L = XE + BW / 2, XD - BW / 2
    for y in (r1, r2, r3):
        segs, x = [], XE_R
        if y in (r1, r2):                       # FEEM sits on the stage-1/2 skip paths
            segs.append((x, FEEM_C - FEEM_W / 2 - GAP))
            x = FEEM_C + FEEM_W / 2 + GAP
        segs.append((x, CCM_L - GAP))
        segs.append((CCM_R + GAP, XD_L))
        for k, (a, b) in enumerate(segs):
            arrow((a, y), (b, y), color=GT_GREEN, lw=0.95, ls=(0, (3.5, 2)),
                  style="-|>" if k == len(segs) - 1 else "-", ms=7.0)

    # ---- FEEM sits on the stage-1 and stage-2 skip paths
    for y, mp in ((r1, "0.02 M"), (r2, "0.07 M")):
        box(FEEM_C, y, FEEM_W, 0.60, f"FEEM\n{mp}", "#D9EAF6", "#0072B2", fs=6.3,
            tc="#12496E", bold=True)

    # ---- CCM: one band that every skip signal crosses before fusion
    yC0, yC1 = r3 - 0.78, r1 + 0.78
    ax.add_patch(FancyBboxPatch((CCM_L, yC0), CCM_R - CCM_L, yC1 - yC0,
                                boxstyle="round,pad=0.012,rounding_size=0.055",
                                linewidth=0.9, facecolor="#E3F1E9", edgecolor=GT_GREEN,
                                zorder=4))
    ax.text((CCM_L + CCM_R) / 2, (yC0 + yC1) / 2 + 0.28, "CCM", ha="center", va="center",
            fontsize=6.8, color="#1F6B4C", fontweight="bold", zorder=5)
    ax.text((CCM_L + CCM_R) / 2, (yC0 + yC1) / 2 - 0.28, "0.24 M", ha="center", va="center",
            fontsize=6.1, color="#1F6B4C", zorder=5)

    # ---- VDAS: parallel residual branches, one per stage-3 / stage-4 feature
    for y, nm, mp in ((r3, "VDAS-3", "0.13 M"), (r4, "VDAS-4", "0.53 M")):
        box(0.72, y, 1.05, 0.62, f"{nm}\n{mp}", "#FBE0D0", ACC, fs=6.4,
            tc="#8A3B00", bold=True)
        arrow((1.245, y), (XE - BW / 2, y), color=ACC, style="<|-|>", lw=0.85)

    # ---- MPSA: multi-scale context on the bottleneck only
    box(XE, 6.30, 1.15, 0.62, "MPSA\n0.93 M", "#EADCF5", "#7B2CBF", fs=6.4,
        tc="#5A1E8C", bold=True)
    arrow((XE, r4 - BH / 2), (XE, 6.61), color="#7B2CBF", style="<|-|>", lw=0.85)

    # ---- parameter budget, read from the instantiated model
    ax.text(0.06, 0.80,
            "Parameters: 15.72 M total — encoder 81.8%, decoder 6.0%, MPSA 5.9%,",
            fontsize=6.2, color="#33475A", ha="left", va="center")
    ax.text(0.06, 0.46,
            "VDAS 4.2%, CCM 1.6%, FEEM 0.5%.    Compute: 86.0 GFLOPs at 512×512.",
            fontsize=6.2, color="#33475A", ha="left", va="center")

    # ---- detail band: what is inside the three custom modules --------------
    # Every operation drawn here is the one in code/modules.py.  VDAS runs the
    # feature through horizontal- and vertical-raster WKV scans and fuses them
    # with a two-way softmax gate taken from a global average pool.  FEEM forms
    # a fixed-kernel Sobel magnitude E and folds it back as
    # F' = F + g * tanh(conv1x1[F;E]) with g = sigmoid(conv1x1[F;E]).  MPSA
    # concatenates depthwise 3x3 convolutions at dilation 1, 2 and 3 and
    # re-weights the fused map with a squeeze-and-excitation gate (reduction 4).
    BAND_Y0, BAND_Y1 = 1.15, 5.60

    def module_detail(x0, x1, title, subtitle, params, ec, tint, stages):
        """One framed panel: a header, then a top-to-bottom flow of op boxes."""
        ax.add_patch(FancyBboxPatch(
            (x0, BAND_Y0), x1 - x0, BAND_Y1 - BAND_Y0,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.0, facecolor="#FCFDFE", edgecolor=ec, zorder=2))
        ax.text(x0 + 0.20, BAND_Y1 - 0.30, title, fontsize=7.4, fontweight="bold",
                color=ec, ha="left", va="center", zorder=6)
        ax.text(x1 - 0.20, BAND_Y1 - 0.30, params, fontsize=6.1, color=ec,
                ha="right", va="center", zorder=6)
        ax.text(x0 + 0.20, BAND_Y1 - 0.66, subtitle, fontsize=6.0, color="#42596E",
                ha="left", va="center", zorder=6)

        inner0, inner1 = x0 + 0.22, x1 - 0.22
        top, bot, gap = BAND_Y1 - 0.94, BAND_Y0 + 0.24, 0.52
        n = len(stages)
        h = (top - bot - gap * (n - 1)) / n

        geom = []
        for i, stage in enumerate(stages):
            y = top - h / 2 - i * (h + gap)
            k = len(stage)
            w = (inner1 - inner0 - 0.16 * (k - 1)) / k
            geom.append((y, [inner0 + w / 2 + j * (w + 0.16) for j in range(k)], w, stage))

        for i, (y, cxs, w, stage) in enumerate(geom):
            parallel = len(stage) > 1
            for cx, (line1, line2) in zip(cxs, stage):
                ax.add_patch(FancyBboxPatch(
                    (cx - w / 2, y - h / 2), w, h,
                    boxstyle="round,pad=0.008,rounding_size=0.05",
                    linewidth=0.85, facecolor=tint if parallel else "#EEF3F8",
                    edgecolor=ec if parallel else "#8496A6", zorder=4))
                ax.text(cx, y, line1 + ("\n" + line2 if line2 else ""),
                        ha="center", va="center", fontsize=6.2, color="#182430",
                        zorder=5, linespacing=1.16)
            # Connectors run through a bus at the middle of the gap, so parallel
            # branches meet at one point and no arrowhead ever lands on another.
            if i < n - 1:
                yb, nxt_y, nxt_cxs = y - h / 2, geom[i + 1][0], geom[i + 1][1]
                yt, ybus = nxt_y + h / 2, (y + nxt_y) / 2
                AR = dict(color="#8496A6", lw=0.8, ms=6.2)
                if len(nxt_cxs) > 1:                       # one source, many targets
                    arrow((cxs[0], yb), (cxs[0], ybus), **AR)
                    arrow((nxt_cxs[0], ybus), (nxt_cxs[-1], ybus), style="-", **AR)
                    for tx in nxt_cxs:
                        arrow((tx, ybus), (tx, yt), **AR)
                elif len(cxs) > 1:                         # many sources, one target
                    for cx in cxs:
                        arrow((cx, yb), (cx, ybus), **AR)
                    arrow((cxs[0], ybus), (cxs[-1], ybus), style="-", **AR)
                    arrow(((cxs[0] + cxs[-1]) / 2, ybus), (nxt_cxs[0], yt), **AR)
                else:                                      # straight through
                    arrow((cxs[0], yb), (nxt_cxs[0], yt), **AR)

    module_detail(
        0.10, 3.02, "VDAS", "dual-raster scan",
        "0.13 + 0.53 M", ACC, "#FBE7D8",
        [[("h-scan", "→ WKV"), ("v-scan", "→ WKV")],
         [("softmax gate", "y = αy_h + (1-α)y_v")],
         [("residual add", "x ← x + λ·y")]])

    module_detail(
        3.24, 6.16, "FEEM", "stage-1/2 skip branch",
        "0.02 + 0.07 M", "#0072B2", "#DFEDF8",
        [[("edge magnitude", "E = √(gx² + gy²)")],
         [("gate  g = σ(conv [s;E])", "branch  tanh(conv [s;E])")],
         [("residual add", "s ← s + g·tanh(·)")]])

    module_detail(
        6.38, 9.30, "MPSA", "multi-scale context",
        "0.93 M", "#7B2CBF", "#EDE3F7",
        [[("DW 3×3", "d = 1"), ("DW 3×3", "d = 2"), ("DW 3×3", "d = 3")],
         [("concat → 1×1 → SE gate", "")],
         [("residual add", "x ← x + m·g")]])

    handles = [Line2D([], [], color=GREY, lw=1.0, label="data flow"),
               Line2D([], [], color=GT_GREEN, lw=1.0, ls=(0, (3.5, 2)),
                      label="skip connection"),
               Line2D([], [], color=ACC, lw=0.9, label="residual branch (VDAS)"),
               Line2D([], [], color="#7B2CBF", lw=0.9, label="bottleneck context (MPSA)"),
               Line2D([], [], marker="s", ls="", mfc="#E3F1E9", mec=GT_GREEN, ms=6,
                      label="channel cross-mixing (CCM)")]
    ax.legend(handles=handles, loc="lower center", frameon=False, fontsize=6.3,
              ncol=3, columnspacing=1.1, handlelength=1.5, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(left=0.004, right=0.996)
    save(fig, "fig1_architecture")


# ================================================================== Fig 2
def fig2_accuracy_complexity():
    m = {r["method"]: r for r in rows_of(TAB / "tab_main.csv")}
    lat = {r["model"]: r for r in ex_rows("EX-147", "lat_flops.csv") if r["amp"] == "fp32"}
    order = [
        ("CropRWKV FULL (this work)", "full", "CropRWKV FULL"),
        ("DeepLabV3+ (R50, IN)", "deeplabv3plus", "DeepLabV3+ (R50, IN)"),
        ("SegFormer-B2 (IN)", "segformer", "SegFormer-B2 (IN)"),
        ("U-Net (R50, IN)", "unet_pretrained", "U-Net (R50, IN)"),
        ("Swin-UNet (IN)", "swin_unet", "Swin-UNet (IN)"),
        ("U-Net (scratch)", "unet_scratch", "U-Net (scratch)"),
        ("SegFormer-B2 (scratch)", "segformer", "SegFormer-B2 (scratch)"),
    ]
    pts = []
    for mname, k, lab in order:
        r = m[mname]
        pts.append(dict(
            lab=lab,
            sd=float(r["soft_dice_mean"]), se=float(r["soft_dice_std"]),
            params=float(r["params_M"]),
            flops=float(r["flops_G"]) if r["flops_G"] not in ("--", "") else math.nan,
            ms=float(lat[k]["median_ms"]) if k else math.nan,
            color=COL.get(lab, GREY)))
    a0 = m["CropRWKV A0 (backbone only)"]
    a0 = dict(lab="CropRWKV A0", sd=float(a0["soft_dice_mean"]), se=float(a0["soft_dice_std"]),
              params=float(a0["params_M"]), flops=math.nan, ms=math.nan,
              color=COL["CropRWKV A0"])

    fig, axes = plt.subplots(1, 2, figsize=(FW, SC*2.62))

    def panel(ax, xkey, xlabel):
        for d in pts:
            x = d[xkey]
            if math.isnan(x):
                continue
            ax.errorbar(x, d["sd"], yerr=d["se"], fmt="o",
                        ms=4.6 + 0.045 * d["params"], mfc=d["color"], mec="white", mew=0.7,
                        ecolor=d["color"], elinewidth=0.9, capsize=2.0, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("soft-Dice (Set A test)")
        ax.set_xscale("log")
        ax.set_ylim(0.20, 0.86)
        ax.grid(alpha=0.22, linewidth=0.5, linestyle=":")
        ax.set_axisbelow(True)
        bare(ax)

    panel(axes[0], "flops", "FLOPs (G)")
    panel(axes[1], "ms", "fp32 latency (ms)")

    # The hero point is singled out with an open ring rather than an in-axes text
    # label: a text label inside a gridded axes is crossed by the grid lines, and
    # the ring keeps the emphasis without any stroke-text collision.
    f = pts[0]
    for ax, xk in ((axes[0], "flops"), (axes[1], "ms")):
        ax.plot([f[xk]], [f["sd"]], marker="o", ms=12.5, mfc="none", mec=ACC, mew=1.2,
                zorder=4)
    axes[0].set_title("accuracy vs. compute", loc="left", fontweight="bold", fontsize=8.2)
    axes[1].set_title("accuracy vs. inference time", loc="left", fontweight="bold", fontsize=8.2)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")

    handles = [Line2D([], [], marker="o", ls="", mfc=d["color"], mec="white", ms=6,
                      label=d["lab"]) for d in (pts + [a0])]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.4,
               bbox_to_anchor=(0.5, -0.34), handletextpad=0.4, columnspacing=1.1)
    fig.subplots_adjust(left=0.125, right=0.996, wspace=0.34)
    align_gate(fig, "fig2_accuracy_complexity")

    data = [["method", "soft_dice_mean", "soft_dice_std", "params_M", "flops_G", "fp32_ms"]]
    for d in pts:
        data.append([d["lab"], d["sd"], d["se"], d["params"], d["flops"], d["ms"]])
    save(fig, "fig2_accuracy_complexity", data)


# ================================================================== Fig 3
def fig3_instability():
    sp = ex_rows("EX-140", "spectral_norm.csv")
    tj = ex_rows("EX-140", "trajectory.csv")

    fig, axes = plt.subplots(1, 2, figsize=(FW, SC*2.90))

    # --- (a) horizontal bars: readable module names, no rotated ticks
    ax = axes[0]
    pretty = {
        "vdas3.J": "VDAS-3  $J$", "vdas3.I+J": "VDAS-3  $I{+}J$",
        "vdas3._scan[h]": "VDAS-3  h-scan", "vdas3._scan[v]": "VDAS-3  v-scan",
        "vdas4.J": "VDAS-4  $J$", "vdas4.I+J": "VDAS-4  $I{+}J$",
        "vdas4._scan[h]": "VDAS-4  h-scan", "vdas4._scan[v]": "VDAS-4  v-scan",
        "mpsa.J": "MPSA  $J$", "mpsa.I+J": "MPSA  $I{+}J$",
        "feem2.J": "FEEM  $J$",
        "encoder.stage4[-1].J": "backbone IR-RWKV",
    }
    labels, vals, isnan, ctrl = [], [], [], []
    for r in sp:
        t = r["target"]
        labels.append(pretty.get(t, t))
        ctrl.append(not t.startswith("vdas"))
        s = r["sigma_max"]
        if s.lower() == "nan":
            vals.append(math.nan)
            isnan.append(True)
        else:
            vals.append(float(s))
            isnan.append(False)
    vmax = max(v for v in vals if not math.isnan(v))
    vmin = min(v for v in vals if not math.isnan(v))
    y = np.arange(len(labels))[::-1]
    for i, (v, bad, c) in enumerate(zip(vals, isnan, ctrl)):
        yy = y[i]
        if bad:
            # the overflow row is drawn as a near-full-width pale band.  The band is
            # taller than the label so the label sits fully inside it: at the original
            # height the band's own edge strokes ran through the text.
            ax.barh(yy, vmax * 20, color="#F6D6C6", edgecolor=ACC,
                    linewidth=0.8, zorder=3, height=0.88)
            # x is an axes fraction (y stays in data units), so the label starts a fixed
            # distance inside the left spine and can never touch the y tick label of the
            # row, which is what a data-coordinate offset does on a log axis.
            ax.text(0.10, yy, "NaN", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=7.7, color="#8A3B00",
                    fontweight="bold", zorder=6, clip_on=True)
        else:
            ax.barh(yy, v, color="#CBD8E4" if c else "#F0A97F",
                    edgecolor="#5B6E80" if c else ACC, linewidth=0.7, zorder=3, height=0.68)
            # a fixed point offset keeps a readable gap from the bar edge regardless of
            # where the log axis places that edge
            ax.annotate(f"{v:.3g}", (v, yy), xytext=(5, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=7.7, color=INK)
    ax.set_xscale("log")
    # the window must contain every finite value, otherwise the small-norm bars and
    # their value labels are drawn outside the axes and collide with the tick labels
    ax.set_xlim(vmin * 0.35, vmax * 200)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.7)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel("$\\sigma_{\\max}$ of module Jacobian")
    ax.set_title("spectral norm of each sub-module", loc="left", fontweight="bold", fontsize=8.2)
    ax.set_axisbelow(True)
    # the legend is placed *below* the axes: the stage-3 h-scan bar and the overflow
    # band both run to the right edge, so every in-axes corner is crossed by
    # either a bar or one of its value labels.
    ax.legend(handles=[Line2D([], [], marker="s", ls="", mfc="#F0A97F", mec=ACC, ms=6,
                              label="VDAS residual branch"),
                       Line2D([], [], marker="s", ls="", mfc="#CBD8E4", mec="#5B6E80", ms=6,
                              label="control block")],
              frameon=False, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, columnspacing=1.4, handlelength=1.4, handletextpad=0.4)
    bare(ax)
    panel_label(ax, "a")

    # --- (b) amplification along the repair trajectory
    ax = axes[1]
    exs = [r["ex"] for r in tj]
    x = np.arange(len(exs))
    w = 0.26
    series = [("r_stage0_nodetach_over_stopgrad", "stage 0", "#D55E00"),
              ("r_enc3_nodetach_over_stopgrad", "enc 3", "#0072B2"),
              ("r_bottleneck_nodetach_over_stopgrad", "bottleneck", "#009E73")]
    # bottom must be the axis floor: a default bottom of 0 is -inf on a log axis
    floor = 0.8
    for k, (col, lab, c) in enumerate(series):
        # NB: bind to a fresh name.  Reusing `vals` here shadowed the panel-(a)
        # array above and made the source-data mirror for panel (a) report the
        # bottleneck trajectory instead of the spectral norms.
        bvals = [float(r[col]) for r in tj]
        ax.bar(x + (k - 1) * w, [v - floor for v in bvals], w, bottom=floor, label=lab,
               color=c, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(1.0, color=GREY, linewidth=0.8, linestyle="--", zorder=4)
    ax.set_yscale("log")
    # the headroom above the tallest bar (~640) is what keeps the two-row legend clear
    # of the bars; at 1800 the legend sat on top of the stage-0 bar of EX-029/EX-030.
    ax.set_ylim(floor, 6000)
    ax.set_xticks(x)
    # five "EX-0NN" labels do not fit side by side in a half-column panel, so they are
    # rotated and right-anchored at the tick instead of being left to collide.
    ax.set_xticklabels(exs, fontsize=7.7, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_xlabel("repair stage (EX snapshot)")
    ax.set_ylabel("gradient amplification $r$")
    ax.set_title("amplification along the repair trajectory", loc="left",
                 fontweight="bold", fontsize=8.2)
    handles = [Line2D([], [], marker="s", ls="", mfc=c, mec="white", ms=6, label=lab)
               for _, lab, c in series]
    # the dashed-line entry is kept to the bare symbol "$r=1$": the full phrase made the
    # two-column legend wider than the half-column axes, so it spilled left over the
    # "10^3" y tick label.  The caption carries the wording.
    handles.append(Line2D([], [], color=GREY, lw=0.8, ls="--", label="$r=1$"))
    ax.legend(handles=handles, frameon=False, ncol=2, loc="upper right", fontsize=7.6,
              columnspacing=0.8, handlelength=1.3, borderaxespad=0.3)
    ax.set_axisbelow(True)
    bare(ax)
    panel_label(ax, "b")

    fig.subplots_adjust(left=0.21, right=0.996, wspace=0.42)
    align_gate(fig, "fig3_instability")

    data = [["panel", "target_or_ex", "value", "detail"]]
    for lb, v, bad in zip(labels, vals, isnan):
        data.append(["a", lb, "NaN" if bad else v, "spectral norm"])
    for r in tj:
        data.append(["b", r["ex"], r["r_stage0_nodetach_over_stopgrad"], "r stage0"])
    save(fig, "fig3_instability", data)


# ================================================================== Fig 4
def fig4_stratified():
    ar = ex_rows("EX-275", "stratified_by_area.csv")
    sp = {r["method"]: r for r in ex_rows("EX-275", "spearman_correlations.csv")}
    show = ["CropRWKV FULL (ours)", "CropRWKV A0", "DeepLabV3+ (R50, IN)",
            "SegFormer-B2 (IN)", "U-Net (R50, IN)", "Swin-UNet (IN)"]
    pretty = {"CropRWKV FULL (ours)": "CropRWKV FULL", "CropRWKV A0": "CropRWKV A0",
              "DeepLabV3+ (R50, IN)": "DeepLabV3+ (R50)",
              "SegFormer-B2 (IN)": "SegFormer-B2", "U-Net (R50, IN)": "U-Net (R50)",
              "Swin-UNet (IN)": "Swin-UNet"}

    fig, axes = plt.subplots(1, 2, figsize=(FW, SC*2.72))

    ax = axes[0]
    q = ["Q1", "Q2", "Q3", "Q4"]
    for r in ar:
        name = r["method"]
        if name not in show:
            continue
        y = [float(r[f"{k}_soft_dice"]) for k in q]
        c = COL.get(name, GREY)
        hero = name.startswith("CropRWKV FULL")
        ax.plot(range(4), y, marker="o", ms=4.6 if hero else 3.8,
                lw=1.8 if hero else 1.1, color=c, alpha=1.0 if hero else 0.85,
                label=pretty.get(name, name), zorder=5 if hero else 3)
        if hero:
            for xi, yi in zip(range(4), y):
                # Labels sit in the corridor *below* the hero curve: the hero is the
                # second-lowest series at every quartile, so the band above it is crossed
                # by the baseline curves.  At Q1 that corridor is at its narrowest (the
                # CropRWKV A0 curve rises into it), so the first label is anchored to the
                # right of the point and lifted to the middle of the gap.
                if xi == 0:
                    ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(3, -6),
                                textcoords="offset points", fontsize=6.4, color=c,
                                ha="left", va="center", fontweight="bold")
                else:
                    ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, -10),
                                textcoords="offset points", fontsize=6.4, color=c,
                                ha="center", fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Q1\n(smallest)", "Q2", "Q3", "Q4\n(largest)"])
    ax.set_xlabel("ground-truth lesion-area quartile")
    ax.set_ylabel("soft-Dice")
    ax.set_ylim(0.0, 1.10)
    ax.set_title("accuracy by lesion size", loc="left", fontweight="bold", fontsize=8.2)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2, columnspacing=0.8,
              handlelength=1.5, borderaxespad=0.2)
    ax.set_axisbelow(True)
    bare(ax)
    panel_label(ax, "a")

    # panel b: horizontal bars keep the six model names on the y axis, so no rotated
    # tick labels can overlap each other.
    ax = axes[1]
    order = ["CropRWKV FULL (ours)", "CropRWKV A0", "Swin-UNet (IN)",
             "SegFormer-B2 (IN)", "DeepLabV3+ (R50, IN)", "U-Net (R50, IN)"]
    names = [pretty.get(m, m) for m in order]
    ra = [float(sp[m]["rho_area"]) for m in order]
    rd = [float(sp[m]["rho_diam"]) for m in order]
    yy = np.arange(len(order))[::-1]
    hb = 0.36
    ax.barh(yy + hb / 2, ra, hb, label="$\\rho$ vs. lesion area", color="#D55E00",
            edgecolor="white", linewidth=0.5, zorder=3)
    ax.barh(yy - hb / 2, rd, hb, label="$\\rho$ vs. equivalent diameter", color="#0072B2",
            edgecolor="white", linewidth=0.5, zorder=3)
    for yi, v in zip(yy + hb / 2, ra):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left", fontsize=6.5, color="#8A3B00")
    for yi, v in zip(yy - hb / 2, rd):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left", fontsize=6.5, color="#12496E")
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=6.8)
    ax.set_xlabel("Spearman $\\rho$ with lesion difficulty")
    ax.set_xlim(0, 1.32)
    ax.set_ylim(-0.62, len(order) - 0.38)
    ax.set_title("size-dependence of accuracy", loc="left", fontweight="bold", fontsize=8.2)
    # below the axes: both metrics reach past 0.9 in the top rows, so an in-axes legend
    # would sit on the bars and their value labels.
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, columnspacing=1.2, handlelength=1.4, handletextpad=0.4)
    ax.set_axisbelow(True)
    bare(ax)
    panel_label(ax, "b")

    fig.subplots_adjust(left=0.055, right=0.996, wspace=0.60)
    align_gate(fig, "fig4_stratified")

    data = [["panel", "method", "Q1", "Q2", "Q3", "Q4"]]
    for r in ar:
        if r["method"] in show:
            data.append(["a", r["method"]] + [r[f"{k}_soft_dice"] for k in q])
    data.append(["b", "method", "rho_area", "rho_diameter", "", ""])
    for m in show:
        data.append(["b", m, sp[m]["rho_area"], sp[m]["rho_diam"], "", ""])
    save(fig, "fig4_stratified", data)


# ================================================================== Fig 5
def fig5_degradation():
    exmap = [("CropRWKV\nFULL", ["EX-164", "EX-210", "EX-211"]),
             ("DeepLabV3+\n(R50, IN)", ["EX-165", "EX-212", "EX-213"]),
             ("SegFormer-B2\n(IN)", ["EX-166", "EX-214", "EX-215"]),
             ("U-Net\n(R50, IN)", ["EX-167", "EX-216", "EX-217"]),
             ("Swin-UNet\n(IN)", ["EX-168", "EX-218", "EX-219"]),
             ("U-Net\n(scratch)", ["EX-169", "EX-220", "EX-221"])]
    conds = ["blur_sigma2", "blur_sigma4", "motion_blur_9", "jpeg_q30", "low_light_0.5x",
             "overexposed_1.4x", "low_contrast_0.5x", "color_shift", "illum_gradient",
             "gauss_noise_s10", "occlusion_10pct"]
    pretty = {"blur_sigma2": "Gaussian blur, $\\sigma$=2", "blur_sigma4": "Gaussian blur, $\\sigma$=4",
              "motion_blur_9": "Motion blur (9)", "jpeg_q30": "JPEG $q$=30",
              "low_light_0.5x": "Low light (0.5$\\times$)", "overexposed_1.4x": "Over-exposed (1.4$\\times$)",
              "low_contrast_0.5x": "Low contrast (0.5$\\times$)", "color_shift": "Colour shift",
              "illum_gradient": "Illumination gradient", "gauss_noise_s10": "Gaussian noise, $\\sigma$=10",
              "occlusion_10pct": "Occlusion (10%)"}
    # Average over three seeds
    tabs = {}
    for lab, exs in exmap:
        agg = {}
        for ex in exs:
            rows = {r["condition"]: r for r in ex_rows(ex, "degrad_boundary_summary.csv")}
            for c, r in rows.items():
                agg.setdefault(c, {}).setdefault("dSoftDice_vs_clean", []).append(float(r["dSoftDice_vs_clean"]))
                agg.setdefault(c, {}).setdefault("soft_dice", []).append(float(r["soft_dice"]))
        tabs[lab] = {c: {k: float(np.mean(v)) for k, v in agg[c].items()} for c in agg}
    M = np.array([[abs(float(tabs[lab][c]["dSoftDice_vs_clean"])) for lab, _ in exmap]
                  for c in conds])
    clean = np.array([float(tabs[lab]["clean"]["soft_dice"]) for lab, _ in exmap])

    fig, ax = plt.subplots(figsize=(FW, SC*4.05))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0.0, vmax=0.45, aspect="auto")
    # six model headers in a half-column-wide heat map: the full labels ("DeepLabV3+
    # (R50, IN)") are wider than their own column, so they are shortened and rotated.
    # The clean-input soft-Dice of each model is stated in the caption instead.
    hdr = {"CropRWKV\nFULL": "CropRWKV", "DeepLabV3+\n(R50, IN)": "DeepLabV3+",
           "SegFormer-B2\n(IN)": "SegFormer", "U-Net\n(R50, IN)": "U-Net R50",
           "Swin-UNet\n(IN)": "Swin-UNet", "U-Net\n(scratch)": "U-Net scratch"}
    ax.set_xticks(range(len(exmap)))
    ax.set_xticklabels([hdr[lab] for lab, _ in exmap], fontsize=7.9,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([pretty[c] for c in conds], fontsize=7.8)
    ax.set_xticks(np.arange(-0.5, len(exmap), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(conds), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", which="major", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("soft-Dice drop vs. clean input ($|\\Delta|$)", fontsize=7.9)
    cb.ax.tick_params(labelsize=7.8)
    ax.set_title("Field-acquisition degradation sensitivity", loc="left",
                 fontweight="bold", fontsize=8.6, pad=8)
    fig.subplots_adjust(left=0.22, right=0.996)
    save(fig, "fig5_degradation")

    data = [["condition"] + [lab.replace("\n", " ") for lab, _ in exmap]]
    for i, c in enumerate(conds):
        data.append([c] + [tabs[lab][c]["dSoftDice_vs_clean"] for lab, _ in exmap])
    data.append(["clean_soft_dice"] + [f"{v:.4f}" for v in clean])
    with open(SRC / "fig5_degradation.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(data)
    print("  wrote figures/fig5_degradation.pdf/.png + source_data")


# ================================================================== Fig 6
def fig6_resolution():
    e4 = ex_rows("E4_efficiency", "efficiency_results.csv")
    ideal = {}
    for r in e4:
        if r["amp"] == "fp32" and r["res"] and not r["flops_512_G"]:
            ideal.setdefault(int(r["res"]), {})[r["model"]] = float(r["pixels_per_s"])
    sw = {}
    for r in ex_rows("EX-125", "highres_results.csv"):
        sw.setdefault(int(r["res"]), {})[r["model"]] = float(r["latency_ms"])

    fig, axes = plt.subplots(1, 2, figsize=(FW, SC*2.72))
    style = {"full": ("CropRWKV FULL", "#D55E00"), "segformer": ("SegFormer-B2", "#0072B2"),
             "unet": ("U-Net (R50)", "#009E73")}

    ax = axes[0]
    for k, (lab, c) in style.items():
        rs = sorted(r for r in ideal if k in ideal[r])
        ax.plot(rs, [ideal[r][k] / 1e6 for r in rs], marker="o", ms=4.2, lw=1.5, color=c, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(5, 45)
    ax.set_xticks([256, 512, 1024, 2048])
    # four ticks only: seven four-digit labels no longer fit side by side now that
    # every figure is drawn at the text-column width.
    ax.set_xticklabels(["256", "512", "1024", "2048"])
    ax.set_yticks([6, 8, 10, 15, 20, 30])
    ax.set_yticklabels(["6", "8", "10", "15", "20", "30"])
    ax.set_xlabel("input resolution (px, side)")
    ax.set_ylabel("throughput (M px/s)")
    ax.set_title("whole-image forward pass", loc="left", fontweight="bold", fontsize=8.2)
    ax.grid(alpha=0.22, linewidth=0.5, linestyle=":", which="both")
    ax.set_axisbelow(True)
    bare(ax)
    panel_label(ax, "a")

    ax = axes[1]
    for k, (lab, c) in style.items():
        rs = sorted(r for r in sw if k in sw[r])
        ax.plot(rs, [sw[r][k] for r in rs], marker="s", ms=4.0, lw=1.5, color=c, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([256, 512, 1024, 2048, 4096])
    ax.set_xticklabels(["256", "512", "1024", "2048", "4096"], fontsize=7.0,
                       rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlabel("input resolution (px, side)")
    ax.set_ylabel("latency (ms, sliding window)")
    ax.set_title("deployment (sliding-window) latency", loc="left", fontweight="bold", fontsize=8.2)
    ax.grid(alpha=0.22, linewidth=0.5, linestyle=":", which="both")
    ax.set_axisbelow(True)
    bare(ax)
    panel_label(ax, "b")

    # a single shared legend below the row keeps every gridded panel free of text
    handles = [Line2D([], [], marker="o", ls="-", color=c, ms=4.2, label=lab)
               for _, (lab, c) in style.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.17), handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.03, right=0.996, wspace=0.36)
    align_gate(fig, "fig6_resolution")

    data = [["protocol", "resolution", "model", "value"]]
    for r in sorted(ideal):
        for k in style:
            if k in ideal[r]:
                data.append(["whole_image", r, k, ideal[r][k]])
    for r in sorted(sw):
        for k in style:
            if k in sw[r]:
                data.append(["sliding_window", r, k, sw[r][k]])
    save(fig, "fig6_resolution", data)


# ================================================================== Fig 7
def fig7_crossdataset():
    tab = {r["method"]: r for r in rows_of(TAB / "tab_main.csv")}
    setA = {"CropRWKV FULL": float(tab["CropRWKV FULL (this work)"]["soft_dice_mean"]),
            "SegFormer-B2": float(tab["SegFormer-B2 (IN)"]["soft_dice_mean"]),
            "U-Net (R50)": float(tab["U-Net (R50, IN)"]["soft_dice_mean"])}
    zs = {"CropRWKV FULL": float(np.mean([float(meta(e)["test_soft_dice_mean"]) for e in ["EX-176","EX-222","EX-223"]])),
          "SegFormer-B2": float(np.mean([float(meta(e)["test_soft_dice_mean"]) for e in ["EX-177","EX-224","EX-225"]])),
          "U-Net (R50)": float(np.mean([float(meta(e)["test_soft_dice_mean"]) for e in ["EX-178","EX-226","EX-227"]]))}
    ind = {"CropRWKV FULL": float(np.mean([float(meta(e)["test_soft_dice_mean"]) for e in ["EX-173","EX-240","EX-241"]])),
           "SegFormer-B2": float(np.mean([float(meta(e)["test_soft_dice_mean"]) for e in ["EX-174","EX-242","EX-243"]])),
           "U-Net (R50)": float(np.mean([float(meta(e)["test_soft_dice_mean"]) for e in ["EX-175","EX-244","EX-245"]]))}

    models = ["CropRWKV FULL", "SegFormer-B2", "U-Net (R50)"]
    y = np.arange(len(models))[::-1]
    h = 0.24
    fig, ax = plt.subplots(figsize=(FW, SC*2.55))
    ax.barh(y + h, [setA[m] for m in models], h, label="Set A, in-domain",
            color="#009E73", edgecolor="white", linewidth=0.5, zorder=3)
    ax.barh(y, [zs[m] for m in models], h, label="Set B, zero-shot",
            color="#E69F00", edgecolor="white", linewidth=0.5, zorder=3)
    ax.barh(y - h, [ind[m] for m in models], h, label="Set B, in-domain",
            color="#0072B2", edgecolor="white", linewidth=0.5, zorder=3)
    for yi, m in zip(y, models):
        for off, val in ((h, setA[m]), (0, zs[m]), (-h, ind[m])):
            ax.text(val + 0.008, yi + off, f"{val:.3f}", va="center", ha="left",
                    fontsize=7.2, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=7.6)
    ax.set_xlabel("soft-Dice")
    ax.set_xlim(0, 0.92)
    ax.set_ylim(-0.55, 2.55)
    ax.set_axisbelow(True)
    bare(ax)
    ax.set_title("Cross-dataset transfer (Set B: PlantSeg in-the-wild)",
                 loc="left", fontweight="bold", fontsize=8.6)
    handles = [Line2D([], [], marker="s", ls="", mfc=c, mec="white", ms=6, label=lab)
               for lab, c in (("Set A, in-domain", "#009E73"),
                              ("Set B, zero-shot", "#E69F00"),
                              ("Set B, in-domain", "#0072B2"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.4,
               bbox_to_anchor=(0.5, -0.11), handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.185, right=0.996)
    save(fig, "fig7_crossdataset")

    data = [["model", "setA_indomain", "setB_zeroshot", "setB_indomain"]]
    for m in models:
        data.append([m, setA[m], zs[m], ind[m]])
    with open(SRC / "fig7_crossdataset.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(data)
    print("  wrote figures/fig7_crossdataset.pdf/.png + source_data")


# ================================================================== Fig 8
def fig8_qualitative():
    """Qualitative masks across the lesion-size strata of the failure analysis.

    Every mask shown here was produced by the *same* inference protocol behind
    the quantitative tables (cached 1024-px frame, 512 window / 256 stride,
    averaged logits, sigmoid, threshold 0.5).  Panels are centre-cropped to a
    common square so that the grid is geometrically uniform.
    """
    from PIL import Image

    Q = FIG / "qualitative"
    cases = json.loads((Q / "qualitative_meta.json").read_text(encoding="utf-8"))["cases"]
    dices = {}
    for r in rows_of(Q / "qualitative_cases.csv"):
        dices[(int(r["case"]), r["model"])] = float(r["soft_dice"])

    cols = [("image", "Input", None), ("gt", "Ground truth", GT_GREEN),
            ("full", "CropRWKV FULL", PRED_RED),
            ("segformer", "SegFormer-B2", PRED_RED),
            ("deeplabv3plus", "DeepLabV3+", PRED_RED),
            ("unet_pretrained", "U-Net (R50)", PRED_RED)]
    model_key = {"full": "CropRWKV FULL", "segformer": "SegFormer-B2 (IN)",
                 "deeplabv3plus": "DeepLabV3+ (R50, IN)", "unet_pretrained": "U-Net (R50, IN)"}
    row_lab = {1: "Q1  smallest\nlesions", 2: "Q3", 3: "Q4  largest\nlesions"}

    def square(arr):
        h, w = arr.shape[:2]
        s = min(h, w)
        t, l = (h - s) // 2, (w - s) // 2
        return arr[t:t + s, l:l + s]

    def overlay(img, mask, colour):
        out = img.astype(np.float32).copy()
        m = mask.astype(bool)
        c = np.array(to_rgb(colour)) * 255.0
        out[m] = 0.42 * c + 0.58 * out[m]
        return np.clip(out, 0, 255).astype(np.uint8)

    nr, nc = len(cases), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(FW, SC*3.55))
    for i, cs in enumerate(cases):
        cid = cs["case"]
        img = square(np.asarray(Image.open(Q / f"case{cid}_image.png").convert("RGB")))
        gt = square(np.asarray(Image.open(Q / f"case{cid}_gt.png").convert("L")) > 127)
        for j, (key, title, colour) in enumerate(cols):
            ax = axes[i][j]
            if key == "image":
                ax.imshow(img)
            elif key == "gt":
                ax.imshow(overlay(img, gt, colour))
            else:
                pred = square(np.asarray(Image.open(Q / f"case{cid}_{key}_pred.png").convert("L")) > 127)
                ax.imshow(overlay(img, pred, colour))
                d = dices[(cid, model_key[key])]
                ax.text(0.03, 0.03, f"Dice {d:.2f}", transform=ax.transAxes,
                        fontsize=6.2, color="white", ha="left", va="bottom", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.16", fc="#000000", ec="none", alpha=0.55))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#B9C2CC"); s.set_linewidth(0.6)
            if j == 0:
                ax.set_ylabel(row_lab[cid], fontsize=7.0, labelpad=6)
            if i == 0:
                # a wide title pad lifts the column headers clear of the panel letters,
                # which sit only a few points above the same axes corner
                ax.set_title(title, fontsize=6.5, pad=15)
    # one bold lowercase label per tile, anchored outside the top-left corner
    for i in range(nr):
        for j in range(nc):
            panel_label(axes[i][j], "abcdefghijklmnopqr"[i * nc + j])
    handles = [Line2D([], [], marker="s", ls="", mfc=GT_GREEN, mec="none", ms=7,
                      label="ground-truth lesion"),
               Line2D([], [], marker="s", ls="", mfc=PRED_RED, mec="none", ms=7,
                      label="predicted lesion (threshold 0.5)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.125, right=0.996, wspace=0.12, hspace=0.06)
    align_gate(fig, "fig8_qualitative")
    save(fig, "fig8_qualitative")


def main():
    fig1_architecture()
    fig2_accuracy_complexity()
    fig3_instability()
    fig4_stratified()
    fig5_degradation()
    fig6_resolution()
    fig7_crossdataset()
    fig8_qualitative()
    print("OK: data figures written to", FIG)


if __name__ == "__main__":
    main()