# -*- coding: utf-8 -*-
"""Per-epoch gradient-norm trajectories along a training run.

Purpose: the checkpoint-based comparison records the amplification ratio across
several runs; this script records the trajectory *within* one run, so that it
can be seen directly whether the gradient of the non-detached wiring grows over
training and whether the stop-gradient keeps it bounded.

The two arms differ in one wiring detail only:
  arm_nodetach  (vdas_stop_grad=False)
  arm_stopgrad  (vdas_stop_grad=True)
Same seed, same initialisation, same data order (DataLoader uses a seeded
generator). Everything else follows the unified protocol in train.py:
AdamW lr 1e-4 wd 1e-4, batch 8, AMP fp16, focal-Tversky loss, 512 crops with
lesion-biased sampling.

Each epoch records the unscaled gradients of: the first convolution of
encoder.stage0, the final convolution (control), the global gradient norm, and
the train loss / IoU. The first and last epochs also measure the VDAS Jacobian
spectral norms by power iteration, reusing exp_spectral_grad.

Methodological note: with AMP the GradScaler rescales the loss, and its dynamic
behaviour contaminates the raw gradient observation (non-finite batches drive
the scale down, the gradients underflow after unscaling, and the observed norm
collapses to zero). The fp32 arm (--amp 0) is therefore the authoritative
gradient source; the AMP run is retained because its loss and IoU trajectories
remain valid.

Usage: python exp_grad_trajectory.py --ex EX-191 --epochs 12
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent
CR_ROOT = CODE_DIR.parent
for _p in (str(CR_ROOT), str(CR_ROOT / "code"), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import train  # noqa: E402
from exp_harness import ExSnapshot  # noqa: E402
from models_registry import build_model  # noqa: E402
from exp_spectral_grad import spectral_norm  # noqa: E402

DEV = torch.device("cuda")
SEED = 7


def first_conv(mod):
    for sub in mod.modules():
        if isinstance(sub, torch.nn.Conv2d) and sub.weight.requires_grad:
            return sub.weight
    return None


@torch.no_grad()
def encoder_feats(model, x):
    h = x
    for st in ("stage0", "stage1", "stage2", "stage3"):
        for blk in getattr(model.encoder, st):
            h = blk(h)
    enc3 = h
    for blk in model.encoder.stage4:
        h = blk(h)
    return enc3, h


def measure_sigma(model, x, iters=20):
    model.eval()
    with torch.no_grad():
        enc3, bot = encoder_feats(model, x[:1])
    s3, _, st3 = spectral_norm(lambda z: model.vdas3(z), enc3, iters)
    s4, _, st4 = spectral_norm(lambda z: model.vdas4(z), bot, iters)
    model.train()
    return {"sigma_vdas3": (round(s3, 3) if np.isfinite(s3) else "nan"), "status3": st3,
            "sigma_vdas4": (round(s4, 3) if np.isfinite(s4) else "nan"), "status4": st4}


def run_arm(name, stop_grad, epochs, batch_size, lr, subset=1.0, img_size=512, amp=True):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    ids = [l.strip() for l in (train.SPLITS / "train.txt").read_text().splitlines() if l.strip()]
    if subset < 1.0:
        rng = random.Random(42); rng.shuffle(ids); ids = ids[: max(1, int(len(ids) * subset))]
    train.build_cache(sorted(set(ids)))
    tf = {"crop_size": img_size, "scale": (0.5, 1.0), "lesion_bias": 0.9}
    ds = train.NWRDSet(ids, "train", tf)
    g = torch.Generator().manual_seed(SEED)          # identical data order for both arms
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, generator=g)

    m = build_model("full", img_size=img_size, in_channels=3, num_classes=1,
                    vdas_stop_grad=stop_grad).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    rows = []
    probe_x = None
    for ep in range(epochs):
        m.train()
        g_step, g_st0, g_fin, tl, ti, nb, nskip = 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            if probe_x is None:
                probe_x = x[:1].clone()
            with torch.autocast("cuda", enabled=amp):
                loss = train.focal_tversky(m(x), y)
            opt.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)                  # unscale to obtain the true gradients before measuring
            else:
                loss.backward()                       # fp32: no GradScaler dynamic-scaling interference
            tot = 0.0
            for p in m.parameters():
                if p.grad is not None:
                    tot += float(p.grad.detach().norm() ** 2)
            st0 = first_conv(m.encoder.stage0)
            g_step += float(np.sqrt(tot))
            if st0 is not None and st0.grad is not None:
                g_st0 += float(st0.grad.norm())
            if m.final_conv.weight.grad is not None:
                g_fin += float(m.final_conv.weight.grad.norm())
            if not np.isfinite(tot):
                nskip += 1
            if amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            tl += float(loss.item()); ti += float(train.iou_bin(torch.sigmoid(m(x).float()).detach(), y)); nb += 1
        rec = {"arm": name, "epoch": ep,
               "grad_total_l2": round(g_step / max(1, nb), 4),
               "stage0_firstconv_grad_l2": round(g_st0 / max(1, nb), 6),
               "final_conv_grad_l2": round(g_fin / max(1, nb), 6),
               "train_loss": round(tl / max(1, nb), 4),
               "train_iou": round(ti / max(1, nb), 4),
               "nonfinite_batches": nskip}
        if ep in (0, epochs - 1):
            rec.update(measure_sigma(m, probe_x))
        rows.append(rec)
        print(f"  [{name}] ep{ep} grad_total={rec['grad_total_l2']:.3e} "
              f"stage0={rec['stage0_firstconv_grad_l2']:.3e} loss={rec['train_loss']:.4f} "
              f"iou={rec['train_iou']:.4f} nonfinite={nskip}"
              + (f" sigma_vdas3={rec.get('sigma_vdas3')} sigma_vdas4={rec.get('sigma_vdas4')}" if "sigma_vdas4" in rec else ""),
              flush=True)
    del m
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", default="EX-149")
    ap.add_argument("--tag", default="grad_trajectory")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subset", type=float, default=1.0, help="training-subset fraction (1.0 = all 468 images)")
    ap.add_argument("--amp", type=int, default=0,
                    help="1 = AMP fp16 (train.py protocol, but GradScaler dynamic scaling contaminates gradient measurements); 0 = fp32 (default, trustworthy gradients)")
    args = ap.parse_args()

    snap = ExSnapshot(args.ex, tag=args.tag, root=CR_ROOT / "results")
    snap.set(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, subset=args.subset,
             seed=SEED, arms="nodetach(vdas_stop_grad=False) vs stopgrad(True)",
             amp=bool(args.amp),
             protocol="train.py unified protocol (AdamW / 512x512 lesion-biased crops / focal_tversky), gradients recorded per epoch; AMP disabled so gradient values are reliable")
    all_rows = []
    for name, sg in (("nodetach", False), ("stopgrad", True)):
        print(f"\n===== arm {name} (vdas_stop_grad={sg}) =====", flush=True)
        all_rows += run_arm(name, sg, args.epochs, args.batch_size, args.lr, args.subset,
                            amp=bool(args.amp))

    hdr = "arm,epoch,grad_total_l2,stage0_firstconv_grad_l2,final_conv_grad_l2,train_loss,train_iou,nonfinite_batches,sigma_vdas3,sigma_vdas4"
    lines = [hdr]
    for r in all_rows:
        lines.append(",".join(str(r.get(k, "")) for k in hdr.split(",")))
    (snap.dir / "grad_trajectory.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    nd = [r for r in all_rows if r["arm"] == "nodetach"]
    sg = [r for r in all_rows if r["arm"] == "stopgrad"]
    summary = {
        "nodetach": {"grad_total_first": nd[0]["grad_total_l2"], "grad_total_last": nd[-1]["grad_total_l2"],
                     "stage0_first": nd[0]["stage0_firstconv_grad_l2"], "stage0_last": nd[-1]["stage0_firstconv_grad_l2"],
                     "iou_first": nd[0]["train_iou"], "iou_last": nd[-1]["train_iou"],
                     "sigma_vdas4_first": nd[0].get("sigma_vdas4"), "sigma_vdas4_last": nd[-1].get("sigma_vdas4")},
        "stopgrad": {"grad_total_first": sg[0]["grad_total_l2"], "grad_total_last": sg[-1]["grad_total_l2"],
                     "stage0_first": sg[0]["stage0_firstconv_grad_l2"], "stage0_last": sg[-1]["stage0_firstconv_grad_l2"],
                     "iou_first": sg[0]["train_iou"], "iou_last": sg[-1]["train_iou"],
                     "sigma_vdas4_first": sg[0].get("sigma_vdas4"), "sigma_vdas4_last": sg[-1].get("sigma_vdas4")},
    }
    (snap.dir / "grad_trajectory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    snap.set(**{f"{k}_{kk}": vv for k, v in summary.items() for kk, vv in v.items()
                if isinstance(vv, (int, float))})
    snap.write_meta("SUCCESS")
    snap.write_manifest(title=f"{args.ex} {args.tag}")
    print(f"\n[{args.ex}] done. outputs={snap.dir}", flush=True)


if __name__ == "__main__":
    main()
