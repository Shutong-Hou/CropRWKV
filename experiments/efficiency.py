# -*- coding: utf-8 -*-
"""Efficiency analysis: parameters, FLOPs, peak memory and latency at 512 px,
for fp32 and fp16, plus a resolution sweep (256/384/512/768/1024/1536/2048).
Each resolution rebuilds the model at the corresponding img_size, so the
parameter differences are recorded rather than assumed.

Note: the custom WKV scan cannot be counted by thop automatically; its cost is
listed separately as WKV-manual and the reported FLOPs cover the convolutional
and attention parts.
"""
import sys, time, json
from pathlib import Path
import torch
from thop import profile

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR.parent / "code"))
from models_registry import build_model

DEV = torch.device("cuda")

def build_full(img_size=512):
    return build_model("full", in_channels=3, num_classes=1, img_size=img_size, vdas_stop_grad=True)

def build_segformer(img_size=512):
    return build_model("segformer", in_channels=3, num_classes=1, img_size=img_size)

def build_unet(img_size=512):
    return build_model("unet", in_channels=3, num_classes=1, img_size=img_size)

BUILDERS = {"full": build_full, "segformer": build_segformer, "unet": build_unet}

def measure_latency(model, x, amp, n=100):
    model.eval()
    with torch.inference_mode():
        for _ in range(20):
            with torch.autocast("cuda", enabled=amp):
                model(x)
    torch.cuda.synchronize()
    ts = []
    with torch.inference_mode():
        for _ in range(n):
            t0 = time.perf_counter()
            with torch.autocast("cuda", enabled=amp):
                model(x)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts)//2]  # median

def measure_mem(model, x, amp):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=amp):
            model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()

def main():
    rows = []
    # ---- 512 baseline: params / FLOPs / memory / latency (fp32 + fp16)
    print("### BASE @512")
    for name in ("full", "segformer", "unet"):
        m = BUILDERS[name](512).to(DEV).eval()
        params = sum(p.numel() for p in m.parameters()) / 1e6
        x0 = torch.randn(1, 3, 512, 512)
        macs, _ = profile(m, inputs=(x0.to(DEV),), verbose=False)
        flops_g = macs * 2 / 1e9
        for amp in (False, True):
            x = torch.randn(1, 3, 512, 512, device=DEV)
            mem = measure_mem(m, x, amp) / 1e6
            lat = measure_latency(m, x, amp)
            rec = {"model": name, "amp": "fp16" if amp else "fp32", "res": 512,
                   "params_M": round(params, 2), "flops_512_G": round(flops_g, 2),
                   "peak_mem_MB": int(mem), "latency_ms": round(lat*1000, 2),
                   "fps": round(1.0/lat, 1), "pixels_per_s": int(512*512/lat)}
            rows.append(rec)
            print(f"  {name} {rec['amp']} @512: {rec['latency_ms']}ms fps={rec['fps']} "
                  f"mem={rec['peak_mem_MB']}MB FLOPs={rec['flops_512_G']}G")
            sys.stdout.flush()
    # ---- resolution scaling curve (fp32): rebuild the model at each res
    print("### SCALING CURVE (fp32)")
    for res in (256, 384, 768, 1024, 1536, 2048):
        for name in ("full", "segformer", "unet"):
            try:
                m = BUILDERS[name](res).to(DEV).eval()
                params = sum(p.numel() for p in m.parameters()) / 1e6
                x = torch.randn(1, 3, res, res, device=DEV)
                lat = measure_latency(m, x, False)
                px = res * res
                rec = {"model": name, "amp": "fp32", "res": res, "params_M": round(params, 2),
                       "latency_ms": round(lat*1000, 2), "fps": round(1.0/lat, 1),
                       "pixels_per_s": int(px/lat)}
                rows.append(rec)
                print(f"  {name} @{res}: {rec['latency_ms']}ms fps={rec['fps']} px/s={rec['pixels_per_s']}")
                sys.stdout.flush()
            except Exception as e:
                print(f"  {name} @{res}: ERR {type(e).__name__}: {str(e)[:80]}")
                torch.cuda.empty_cache()
    # ---- save
    out = Path(CODE_DIR).parent / "results" / "E4_efficiency"
    out.mkdir(parents=True, exist_ok=True)
    keys = ["model","amp","res","params_M","flops_512_G","latency_ms","fps","pixels_per_s","peak_mem_MB"]
    (out / "efficiency_results.csv").write_text(
        ",".join(keys) + "\n" +
        "\n".join(",".join(str(r.get(k, "")) for k in keys) for r in rows) + "\n", encoding="utf-8")
    (out / "efficiency_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")

if __name__ == "__main__":
    main()