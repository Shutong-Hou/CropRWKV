# -*- coding: utf-8 -*-
"""
exp_wkv_denominator.py -- EX-281: measures the minimum absolute value of the WKV closed-form
scan denominator (direct evidence of near-singularity).

Background: the closed-form denominator in wkv_pure is
    denom_t = u*k_t + sum_{s<t} exp(-(t-s)w) * k_s + 1e-6
where k is a **signed** linear projection (no exp applied), so the two terms can cancel and the
denominator can approach the constant floor 1e-6, making y = numer/denom blow up. Section 2 of
the paper therefore attributes the stage-3 vertical-scan NaNs to this near-singularity.

This experiment runs forward passes on all 60 test images of set A with the trained FULL
(EX-063, seed 42), recording per module:
  min |signed_sum|        minimum absolute value of the sum of the two terms (excluding eps)
  mean |term|             mean absolute value of the two terms (magnitude reference)
  min |denom|             minimum absolute value including eps
  cancellation_orders     decimal orders between the magnitude reference and min|signed_sum|

Usage:
  python exp_wkv_denominator.py --ex EX-281
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
CR_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(CR_ROOT), str(CR_ROOT / "code"), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp_harness import ExSnapshot  # noqa: E402
from models_registry import build_model  # noqa: E402
from PIL import Image  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = CR_ROOT / "data" / "kaggle_leafseg" / "cache" / "test" / "images"
CKPT = CR_ROOT / "results" / "EX-063" / "checkpoints" / "best.pt"

REC: dict[str, dict] = {}
CUR = {"tag": "?"}


def _probe_run(w, u, k, v):
    """Same formula as wkv_pure.run, additionally recording denominator statistics."""
    import wkv_pure
    B, T, C = k.shape
    W = w.flatten().to(torch.float64).reshape(1, C)
    U = u.flatten().to(torch.float64).reshape(1, 1, C)
    K = k.float().to(torch.float64)
    V = v.float().to(torch.float64)
    t = torch.arange(T, device=w.device, dtype=torch.float64).view(T, 1)
    g = torch.exp(t * W)
    damp = torch.exp(-t * W)
    G = g.unsqueeze(0)
    preK = torch.cumsum(G * K, dim=1)
    preD = torch.cumsum(G * K * V, dim=1)
    z = torch.zeros_like(preK[:, :1])
    preK_shift = torch.cat([z, preK[:, :-1]], dim=1)
    preD_shift = torch.cat([torch.zeros_like(preD[:, :1]), preD[:, :-1]], dim=1)
    damp_ = damp.unsqueeze(0)

    term_a = U * K
    term_b = damp_ * preK_shift
    signed = term_a + term_b
    denom = signed + wkv_pure.EPS

    rec = REC.setdefault(CUR["tag"], {"min_signed": np.inf, "min_denom": np.inf,
                                      "term_sum": 0.0, "n": 0})
    rec["min_signed"] = min(rec["min_signed"], signed.abs().min().item())
    rec["min_denom"] = min(rec["min_denom"], denom.abs().min().item())
    rec["term_sum"] += 0.5 * (term_a.abs().mean().item() + term_b.abs().mean().item())
    rec["n"] += 1

    numer = U * K * V + damp_ * preD_shift
    return (numer / denom).to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", required=True)
    ap.add_argument("--n_images", type=int, default=60)
    args = ap.parse_args()

    snap = ExSnapshot(args.ex, tag="wkv_denominator_probe")
    snap.write_env_snapshot()
    snap.set(ckpt=str(CKPT), note="forward-only probe; denominator statistics only")

    import wkv_pure
    wkv_pure.run = _probe_run

    model = build_model("full", in_channels=3, num_classes=1, img_size=512,
                        vdas_stop_grad=True, vdas_mode="gated",
                        enable_vdas=True, enable_mpsa=True, enable_feem=True,
                        feem_all=False).to(DEV)
    model.load_state_dict(torch.load(CKPT, map_location=DEV)["model"])
    model.eval()

    # --- tagging: backbone spatial-mix blocks and VDAS scan directions ---
    import modules as mod

    for name, m in model.named_modules():
        cls = m.__class__.__name__.lower()
        if cls.startswith("vrwkv") and hasattr(m, "spatial_decay"):
            m.register_forward_pre_hook(lambda _m, _i, _n=name: CUR.__setitem__("tag", _n))
        if cls == "vdas":
            m._ex_name = name

    _orig_scan = mod.VDAS._scan

    def _tagged_scan(self, x, orient):
        CUR["tag"] = f"{getattr(self, '_ex_name', 'vdas')}.{orient}"
        return _orig_scan(self, x, orient)

    mod.VDAS._scan = _tagged_scan

    imgs = sorted(glob.glob(str(CACHE / "*.jpg")))[: args.n_images]
    print(f"[{args.ex}] probing {len(imgs)} test images")

    with torch.inference_mode():
        for p in imgs:
            im = Image.open(p).convert("RGB").resize((512, 512), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im).transpose(2, 0, 1) / 255.0).float()[None].to(DEV)
            model(x)

    rows = ["module,min_abs_signed_sum,min_abs_denom_with_eps,mean_abs_term,n_calls,cancellation_orders"]
    for nm in sorted(REC):
        r = REC[nm]
        term = r["term_sum"] / max(r["n"], 1)
        orders = float(np.log10(term / max(r["min_signed"], 1e-300))) if r["min_signed"] > 0 else 300.0
        rows.append(f"{nm},{r['min_signed']:.6e},{r['min_denom']:.6e},{term:.6e},{r['n']},{orders:.2f}")
        print(f"  {nm:<28} min|sum|={r['min_signed']:.3e}  min|denom|={r['min_denom']:.3e}  "
              f"term={term:.3e}  orders={orders:.1f}")

    (snap.dir / "denominator_probe.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    worst = min(REC.items(), key=lambda kv: kv[1]["min_signed"])
    snap.set(n_images=len(imgs),
             worst_module=worst[0],
             worst_min_abs_signed_sum=float(worst[1]["min_signed"]),
             worst_min_abs_denom=float(worst[1]["min_denom"]))
    snap.write_meta("SUCCESS")
    snap.write_manifest(title=f"{args.ex} WKV denominator near-singularity probe")
    print(f"[{args.ex}] done. outputs={snap.dir}")


if __name__ == "__main__":
    main()
