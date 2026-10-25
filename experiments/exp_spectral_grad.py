# -*- coding: utf-8 -*-
"""Direct measurement of the VDAS spectral norm (power iteration) and of the
back-propagated gradient with and without a stop-gradient on the branch input.

Method:
  1) ``--selftest`` validates the power-iteration implementation against known
     matrices (compared with torch.linalg.svd, relative error reported);
  2) the iteration runs on J^T J (jvp + vjp), with sigma = sqrt(lambda_max);
  3) every measurement reports its convergence status: converged / maxiter /
     degenerate (lambda ~ 0) / nan (numeric overflow); a NaN is never reported
     as a zero;
  4) a random-vector lower bound ||Jv||/||v|| is also given, which does not
     depend on convergence and supports the sigma estimate.

Sub-experiments:
  A1  spectral norms of vdas3/vdas4 (J), of the composite residual (I+J) and of
      each single scan direction _scan[h]/[v]; controls are MPSA, FEEM and the
      backbone IR-RWKV block (same family, contains a WKV scan).
  A2  back-propagated gradient with the branch connected, stop-gradient on the
      branch input, and VDAS removed, under identical weights, input and loss;
      measured both at random initialisation and at trained weights, since the
      ratio depends on the weight state.
  A3  the same key quantities along the repair trajectory, one checkpoint per
      configuration visited.

Usage:
  python exp_spectral_grad.py --ex EX-140 --selftest
"""
from __future__ import annotations

import argparse
import copy
import random as _r
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
CR_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(CR_ROOT), str(CR_ROOT / "code"), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import train  # noqa: E402
from exp_harness import ExSnapshot  # noqa: E402
from models_registry import build_model  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1234


# ---------------------------------------------------------------------------
# Power iteration for sigma_max(d fn / d x) -- power method on J^T J
# ---------------------------------------------------------------------------
def spectral_norm(fn, x, iters=40, tol=1e-5):
    """Returns (sigma, iters_used, status). status is one of converged|maxiter|degenerate|nan."""
    x = x.detach()
    g = torch.Generator(device="cpu").manual_seed(SEED)
    v = torch.randn(x.shape, generator=g).to(x.device).to(x.dtype)
    v = v / v.norm()
    lam, prev, used = 0.0, None, 0
    status = "maxiter"
    for i in range(iters):
        used = i + 1
        try:
            _, jv = torch.autograd.functional.jvp(fn, (x,), (v,), create_graph=False)
            _, w = torch.autograd.functional.vjp(fn, x, v=jv, create_graph=False)
        except RuntimeError as e:                     # e.g. numerical overflow in backpropagation
            return float("nan"), used, f"nan({type(e).__name__})"
        w = w[0] if isinstance(w, tuple) else w
        lam = float(w.norm())
        if not np.isfinite(lam):
            return float("nan"), used, "nan"
        if lam < 1e-30:
            return 0.0, used, "degenerate"
        v = w / lam
        if prev is not None and abs(lam - prev) <= tol * max(lam, 1e-12) and i >= 3:
            status = "converged"
            break
        prev = lam
    return float(np.sqrt(max(lam, 0.0))), used, status


def lower_bounds(fn, x, n=3):
    """Random lower bound on sigma: sigma >= ||Jv||/||v||. Independent of convergence, used as a robust cross-check."""
    out = []
    for k in range(n):
        g = torch.Generator(device="cpu").manual_seed(SEED + 100 + k)
        v = torch.randn(x.shape, generator=g).to(x.device).to(x.dtype)
        v = v / v.norm()
        _, jv = torch.autograd.functional.jvp(fn, (x.detach(),), (v,), create_graph=False)
        out.append(float(jv.norm()))
    return max(out), out


def selftest(verbose=True):
    """Validate the power-iteration implementation on a known matrix: sigma_power vs sigma_svd."""
    torch.manual_seed(0)
    res = []
    for n in (16, 64):
        W = torch.randn(n, n, dtype=torch.float64) / np.sqrt(n)
        W[0] *= 25.0                                    # strong anisotropy to stress-test the power method
        fn = lambda z: (z @ W.T)                        # noqa: E731
        x = torch.randn(n, dtype=torch.float64)
        s, used, st = spectral_norm(fn, x, iters=200, tol=1e-10)
        s_ref = float(torch.linalg.matrix_norm(W, ord=2))
        res.append((n, s, s_ref, abs(s - s_ref) / s_ref, used, st))
        if verbose:
            print(f"  selftest n={n}: power={s:.6f} svd={s_ref:.6f} rel_err={abs(s-s_ref)/s_ref:.2e} "
                  f"iters={used} status={st}", flush=True)
    return res


# ---------------------------------------------------------------------------
# Real features / weights
# ---------------------------------------------------------------------------
def load_full(ckpt=None, img_size=512, **kw):
    m = build_model("full", img_size=img_size, in_channels=3, num_classes=1, **kw).to(DEV)
    if ckpt and Path(ckpt).exists():
        m.load_state_dict(torch.load(ckpt, map_location=DEV)["model"])
    return m


def one_crop(seed=100, img_size=512):
    """Lesion-biased 512x512 crop (same setting as diag2)."""
    _r.seed(seed); np.random.seed(seed)
    import math
    from PIL import Image
    ids = [l.strip() for l in (train.SPLITS / "train.txt").read_text().splitlines() if l.strip()]
    cid = _r.choice(ids)
    img = Image.open(train.CACHE / "train" / "images" / f"{cid}.jpg").convert("RGB")
    msk = Image.open(train.CACHE / "train" / "masks" / f"{cid}.png").convert("L")
    w, h = img.size
    m = np.asarray(msk) > 127
    les = np.argwhere(m)
    r = _r.uniform(0.5, 1.0)
    cw = min(w, max(img_size, int(w * math.sqrt(r))))
    ch = min(h, max(img_size, int(h * math.sqrt(r))))
    if _r.random() < 0.9 and len(les) > 0:
        ly, lx = les[_r.randint(0, len(les) - 1)]
        x = _r.randint(max(0, min(lx, w - cw)), min(max(0, lx), w - cw))
        y = _r.randint(max(0, min(ly, h - ch)), min(max(0, ly), h - ch))
    else:
        x = _r.randint(0, max(0, w - cw)); y = _r.randint(0, max(0, h - ch))
    img = img.crop((x, y, x + cw, y + ch)).resize((img_size, img_size), Image.BILINEAR)
    msk = msk.crop((x, y, x + cw, y + ch)).resize((img_size, img_size), Image.NEAREST)
    t = np.asarray(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    mm = ((np.asarray(msk).astype(np.float32) / 255.0) > 0.5).astype(np.float32)
    return (torch.tensor(t).unsqueeze(0).to(DEV),
            torch.tensor(mm).unsqueeze(0).unsqueeze(0).to(DEV))


@torch.no_grad()
def encoder_feats(model, x):
    f, h = {}, x
    for st, key in [("stage0", None), ("stage1", "enc1"), ("stage2", "enc2"),
                    ("stage3", "enc3"), ("stage4", "bottleneck")]:
        for blk in getattr(model.encoder, st):
            h = blk(h)
        if key:
            f[key] = h
    return f


# ---------------------------------------------------------------------------
# A1 Spectral norm
# ---------------------------------------------------------------------------
def measure(model, feats, iters, want_bounds=True):
    rows = []

    def rec(name, fn, x, note=""):
        s, used, st = spectral_norm(fn, x, iters)
        lb, allv = (lower_bounds(fn, x, 3) if want_bounds else (float("nan"), []))
        rows.append({"target": name, "sigma_max": (round(s, 4) if np.isfinite(s) else "nan"),
                     "status": st, "iters": used, "shape": str(tuple(x.shape)),
                     "sigma_lowerbound_rand": round(lb, 4) if np.isfinite(lb) else "nan",
                     "note": note})
        print(f"  {name:<34} sigma={rows[-1]['sigma_max']} ({st}, it={used}) "
              f"lb_rand={rows[-1]['sigma_lowerbound_rand']} {tuple(x.shape)}", flush=True)

    for tag, mod, key in [("vdas3", model.vdas3, "enc3"), ("vdas4", model.vdas4, "bottleneck")]:
        x = feats[key]
        rec(f"{tag}.J", lambda z: mod(z), x, "VDAS scan operator")
        rec(f"{tag}.I+J", lambda z: z + mod(z), x, "composite residual")
        rec(f"{tag}._scan[h]", lambda z: mod._scan(z, "h"), x, "single direction: horizontal")
        rec(f"{tag}._scan[v]", lambda z: mod._scan(z, "v"), x, "single direction: vertical")
    xb = feats["bottleneck"]
    rec("mpsa.J", lambda z: model.mpsa(z), xb, "reference: dilated depthwise + SE")
    rec("mpsa.I+J", lambda z: z + model.mpsa(z), xb, "reference")
    rec("feem2.J", lambda z: model.feem2(z), feats["enc2"], "reference: edge enhancement")
    blk = list(model.encoder.stage4)[-1]
    rec("encoder.stage4[-1].J", lambda z: blk(z), xb, "reference: same-family IR-RWKV block from the backbone")
    return rows


# ---------------------------------------------------------------------------
# A2 Gradient quantification
# ---------------------------------------------------------------------------
def grad_compare(ckpt, img_size=512, seed=7):
    """Backpropagation gradient l2 for three wiring variants, same data/loss. ckpt=None means random initialisation."""
    torch.manual_seed(seed); np.random.seed(seed); _r.seed(seed)
    base = load_full(ckpt, img_size, vdas_stop_grad=False)
    sd = copy.deepcopy(base.state_dict())
    del base
    torch.cuda.empty_cache()
    x, y = one_crop(100, img_size)

    def first_conv(container):
        for sub in container.modules():
            if isinstance(sub, torch.nn.Conv2d) and sub.weight.requires_grad:
                return sub.weight
        return None

    res = {}
    for tag, kw in [("nodetach", dict(vdas_stop_grad=False, enable_vdas=True)),
                    ("stopgrad", dict(vdas_stop_grad=True, enable_vdas=True)),
                    ("vdasOFF", dict(vdas_stop_grad=True, enable_vdas=False))]:
        torch.manual_seed(seed); np.random.seed(seed); _r.seed(seed)   # same drop_path mask
        m = build_model("full", img_size=img_size, in_channels=3, num_classes=1, **kw).to(DEV)
        m.load_state_dict(sd)
        m.train()
        g = {}
        def mk(nm):
            def _h(mod, gin, gout):
                if gout[0] is not None:
                    g[nm] = float(gout[0].norm())
            return _h
        h1 = list(m.encoder.stage3)[-1].register_full_backward_hook(mk("dL/d_enc3"))
        h2 = list(m.encoder.stage4)[-1].register_full_backward_hook(mk("dL/d_bottleneck"))
        m.zero_grad(set_to_none=True)
        loss = train.focal_tversky(m(x), y)
        loss.backward()
        h1.remove(); h2.remove()
        for st in ("stage0", "stage1", "stage2", "stage3", "stage4"):
            w = first_conv(getattr(m.encoder, st))
            if w is not None and w.grad is not None:
                g[f"{st}.firstconv"] = float(w.grad.norm())
        g["final_conv"] = float(m.final_conv.weight.grad.norm())
        if kw["enable_vdas"]:
            g["vdas3.key"] = float(m.vdas3.key.weight.grad.norm())
            g["vdas3.value"] = float(m.vdas3.value.weight.grad.norm())
        g["loss"] = float(loss.item())
        res[tag] = g
        print(f"  [{tag}] loss={g['loss']:.4f} stage0={g.get('stage0.firstconv', float('nan')):.4e} "
              f"dL/d_enc3={g.get('dL/d_enc3', float('nan')):.4e} "
              f"dL/d_bot={g.get('dL/d_bottleneck', float('nan')):.4e}", flush=True)
        del m; torch.cuda.empty_cache()
    return res


def ratio_rows(res, state):
    a, b, c = res["nodetach"], res["stopgrad"], res["vdasOFF"]
    out = ["state,quantity,nodetach,stopgrad,vdasOFF,ratio_nodetach_over_stopgrad,ratio_nodetach_over_off"]
    for k in [q for q in a if q != "loss"]:
        av, bv, cv = a.get(k, float("nan")), b.get(k, float("nan")), c.get(k, float("nan"))
        out.append(f"{state},{k},{av:.6e},{bv:.6e},{cv:.6e},{av/max(bv,1e-30):.3f},{av/max(cv,1e-30):.3f}")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", default="EX-140")
    ap.add_argument("--tag", default="spectral_grad")
    ap.add_argument("--ckpt", default=str(CR_ROOT / "results" / "EX-032" / "checkpoints" / "best.pt"))
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--traj", default="EX-025,EX-029,EX-030,EX-031,EX-032")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--selftest_only", action="store_true")
    args = ap.parse_args()

    print("=== Methodological self-check: power iteration vs SVD ===", flush=True)
    st_res = selftest()
    ok = all(r[3] < 1e-6 for r in st_res)
    print(f"  selftest {'PASS' if ok else 'FAIL'} (max rel_err={max(r[3] for r in st_res):.2e})", flush=True)
    if args.selftest_only:
        return

    snap = ExSnapshot(args.ex, tag=args.tag, root=CR_ROOT / "results")
    snap.set(ckpt=args.ckpt, power_iters=args.iters, precision="fp32",
             selftest_max_rel_err=max(r[3] for r in st_res), mode="eval(spectral)/train(grad)")
    (snap.dir / "selftest.csv").write_text(
        "n,power,svd,rel_err,iters,status\n" +
        "\n".join(f"{r[0]},{r[1]:.8f},{r[2]:.8f},{r[3]:.3e},{r[4]},{r[5]}" for r in st_res) + "\n",
        encoding="utf-8")

    print(f"\n[{args.ex}] A1 measured spectral norms (trained weights {Path(args.ckpt).parent.parent.name})", flush=True)
    model = load_full(args.ckpt, args.img_size).eval()
    x, _ = one_crop(100, args.img_size)
    feats = encoder_feats(model, x)
    print("  feature shapes:", {k: tuple(v.shape) for k, v in feats.items()}, flush=True)
    rows = measure(model, feats, args.iters)
    (snap.dir / "spectral_norm.csv").write_text(
        "target,sigma_max,status,iters,shape,sigma_lowerbound_rand,note\n" +
        "\n".join(f"{r['target']},{r['sigma_max']},{r['status']},{r['iters']},{r['shape']},"
                  f"{r['sigma_lowerbound_rand']},{r['note']}" for r in rows) + "\n", encoding="utf-8")
    del model; torch.cuda.empty_cache()

    print(f"\n[{args.ex}] A2 gradient quantification (random initialisation)", flush=True)
    g_rand = grad_compare(None, args.img_size)
    print(f"[{args.ex}] A2 gradient quantification (trained weights EX-032)", flush=True)
    g_trained = grad_compare(args.ckpt, args.img_size)
    allg = ratio_rows(g_rand, "random_init") + ratio_rows(g_trained, "trained_EX032")
    (snap.dir / "grad_compare.csv").write_text("\n".join(allg) + "\n", encoding="utf-8")
    for ln in allg:
        print("  " + ln, flush=True)

    _traj = [] if args.traj.strip().lower() in ("none", "-") else \
        [t.strip() for t in args.traj.split(",") if t.strip()]
    print(f"\n[{args.ex}] A3 along the training trajectory", flush=True)
    tr = ["ex,sigma_vdas3,sigma_vdas4,sigma_I+J_vdas4,r_stage0_nodetach_over_stopgrad,"
          "r_enc3_nodetach_over_stopgrad,r_bottleneck_nodetach_over_stopgrad"]
    for ex in _traj:
        ck = CR_ROOT / "results" / ex / "checkpoints" / "best.pt"
        if not ck.exists():
            tr.append(f"{ex},MISSING,,,,,"); print(f"  {ex}: MISSING", flush=True); continue
        try:
            m2 = load_full(str(ck), args.img_size).eval()
            f2 = encoder_feats(m2, x)
            s3, _, st3 = spectral_norm(lambda z: m2.vdas3(z), f2["enc3"], args.iters)
            s4, _, st4 = spectral_norm(lambda z: m2.vdas4(z), f2["bottleneck"], args.iters)
            sc, _, stc = spectral_norm(lambda z: z + m2.vdas4(z), f2["bottleneck"], args.iters)
            gg = grad_compare(str(ck), args.img_size)
            a, b = gg["nodetach"], gg["stopgrad"]
            r0 = a["stage0.firstconv"] / max(b["stage0.firstconv"], 1e-30)
            re3 = a["dL/d_enc3"] / max(b["dL/d_enc3"], 1e-30)
            rbt = a["dL/d_bottleneck"] / max(b["dL/d_bottleneck"], 1e-30)
            tr.append(f"{ex},{s3:.3f},{s4:.3f},{sc:.3f},{r0:.3f},{re3:.3f},{rbt:.3f}")
            print(f"  {ex}: sig_vdas3={s3:.3g}({st3}) sig_vdas4={s4:.3g}({st4}) sig(I+J)={sc:.3g}({stc}) "
                  f"r_stage0={r0:.2f}x r_enc3={re3:.2f}x r_bot={rbt:.2f}x", flush=True)
            del m2; torch.cuda.empty_cache()
        except Exception as e:
            tr.append(f"{ex},ERR,{type(e).__name__},,,,")
            print(f"  {ex}: ERR {type(e).__name__}: {str(e)[:90]}", flush=True)
    (snap.dir / "trajectory.csv").write_text("\n".join(tr) + "\n", encoding="utf-8")

    snap.set(n_targets=len(rows), selftest_pass=bool(ok))
    snap.write_meta("SUCCESS")
    snap.write_manifest(title=f"{args.ex} {args.tag}")
    print(f"\n[{args.ex}] done. outputs={snap.dir}", flush=True)


if __name__ == "__main__":
    main()
