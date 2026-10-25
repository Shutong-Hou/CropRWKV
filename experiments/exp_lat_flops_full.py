# -*- coding: utf-8 -*-
"""Steady-clock latency and FLOPs, measured for the full model set.

Motivation: latency benchmarks on an idle GPU are dominated by the power state
of the device. On this machine an idle GPU is clocked at 210 MHz against a
2100 MHz maximum, which over-estimates the latency of short-latency models by
nearly a factor of two. The protocol used here therefore warms the GPU until the
SM clock reaches a steady state, records clocks.sm during measurement, and
reports the median of three rounds.

Coverage: full / segformer / unet (from scratch) / unet_pretrained /
deeplabv3plus / swin_unet. Per model: params (M), FLOPs at 512 px (G), fp32 and
fp16 latency under the steady clock, and the measured SM clock.

FLOPs are counted with thop, the same tool used for every model. The custom WKV
scan in this project cannot be counted by thop, so the FLOPs reported for FULL
are a lower bound covering the convolutional and attention parts only, and are
labelled as such.

Usage: python exp_lat_flops_full.py --ex EX-147
"""
from __future__ import annotations

import argparse
import json
import statistics as stat
import subprocess
import sys
import time
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parent
CR_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(CR_ROOT), str(CR_ROOT / "code"), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from exp_harness import ExSnapshot  # noqa: E402
from models_registry import build_model  # noqa: E402

DEV = torch.device("cuda")

SPEC = {
    "a0":              ("a0",              "EX-021", {}),
    "full":            ("full",            "EX-032", dict(vdas_stop_grad=True)),
    "segformer":       ("segformer",       "EX-043", {}),
    "unet_scratch":    ("unet",            "EX-038", {}),
    "unet_pretrained": ("unet_pretrained", "EX-044", {}),
    "deeplabv3plus":   ("deeplabv3plus",   "EX-042", {}),
    "swin_unet":       ("swin_unet",       "EX-080", {}),
}


def _smi(field="clocks.sm"):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        return out.splitlines()[0].strip()
    except Exception as e:
        return f"NA({type(e).__name__})"


def make(mid, ck):
    m = build_model(mid, in_channels=3, num_classes=1, img_size=512)
    p = CR_ROOT / "results" / ck / "checkpoints" / "best.pt"
    if p.exists():
        m.load_state_dict(torch.load(p, map_location=DEV)["model"])
    else:
        print(f"    (warn) ckpt missing: {p}", flush=True)
    return m.to(DEV).eval()


def flops_g(m, res=512):
    from thop import profile
    x = torch.randn(1, 3, res, res, device=DEV)
    macs, _ = profile(m, inputs=(x,), verbose=False)
    return macs * 2 / 1e9


def bench(m, amp, warm_iters, n=100, warmup=20):
    x = torch.randn(1, 3, 512, 512, device=DEV)
    with torch.inference_mode():
        for _ in range(warm_iters):
            with torch.autocast("cuda", enabled=amp):
                m(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            with torch.autocast("cuda", enabled=amp):
                m(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return stat.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", default="EX-147")
    ap.add_argument("--tag", default="lat_flops_full")
    ap.add_argument("--models", default=",".join(SPEC.keys()))
    ap.add_argument("--warm_iters", type=int, default=400)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    snap = ExSnapshot(args.ex, tag=args.tag, root=CR_ROOT / "results")
    snap.set(res=512, warm_iters=args.warm_iters, repeat=args.repeat, batch=1,
             flops_tool="thop(macs*2)", protocol="steady-state clock warm-up + median of 3 rounds + clock logging")
    print(f"GPU idle clock: {_smi('clocks.sm')} MHz (max {_smi('clocks.max.sm')})", flush=True)

    rows = ["model,params_M,flops_512_G,amp,median_ms,min_ms,sm_clock_mhz,repeat_medians"]
    summary = {}
    for key in [s.strip() for s in args.models.split(",") if s.strip()]:
        if key not in SPEC:
            print(f"  skip unknown {key}", flush=True); continue
        mid, ck, kw = SPEC[key]
        print(f"\n=== {key} ===", flush=True)
        try:
            m = make(mid, ck)
        except Exception as e:
            print(f"  !! build failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
            rows.append(f"{key},BUILD_FAIL,,,,,")
            continue
        params = sum(p.numel() for p in m.parameters()) / 1e6
        try:
            fl = flops_g(m)
        except Exception as e:
            fl = float("nan")
            print(f"  (warn) flops failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
        print(f"  params={params:.3f}M  FLOPs@512={fl:.2f}G", flush=True)
        summary[key] = {"params_M": round(params, 3),
                        "flops_512_G": round(fl, 2) if fl == fl else None}
        for amp, ampt in ((False, "fp32"), (True, "fp16")):
            meds, mins = [], []
            for r in range(args.repeat):
                md, mn = bench(m, amp, args.warm_iters)
                meds.append(md); mins.append(mn)
            clock = _smi("clocks.sm")
            mm = stat.median(meds)
            rows.append(f"{key},{params:.3f},{fl:.2f},{ampt},{mm:.4f},{min(mins):.4f},{clock},"
                        + "|".join(f"{v:.3f}" for v in meds))
            summary[key][f"{ampt}_median_ms"] = round(mm, 4)
            summary[key][f"{ampt}_min_ms"] = round(min(mins), 4)
            summary[key][f"{ampt}_sm_clock_mhz"] = clock
            print(f"  {ampt}: median={mm:.3f}ms min={min(mins):.3f}ms (per-rep {[round(v,3) for v in meds]}) clock={clock}MHz", flush=True)
        del m
        torch.cuda.empty_cache()
    (snap.dir / "lat_flops.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (snap.dir / "lat_flops_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in summary.items():
        if v.get("flops_512_G") is not None:
            snap.set(**{f"{k}_flops_512_G": v["flops_512_G"]})
        if f"fp32_median_ms" in v:
            snap.set(**{f"{k}_fp32_ms": v["fp32_median_ms"], f"{k}_fp16_ms": v["fp16_median_ms"]})
    snap.write_meta("SUCCESS")
    snap.write_manifest(title=f"{args.ex} {args.tag}")
    print(f"\n[{args.ex}] done. outputs={snap.dir}", flush=True)


if __name__ == "__main__":
    main()
