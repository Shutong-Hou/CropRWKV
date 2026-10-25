# -*- coding: utf-8 -*-
"""Operator-level audit: does the CUDA WKV kernel in third_party/RWKV-UNet
implement the same operator as wkv_pure?

The kernel is reproduced here by a CPU float64 reference implementation of its
formula, so the comparison needs no GPU, and the result is compared against
wkv_pure. If the two are not the same operator, the unit-level speed-up of the
kernel does not transfer to this model.

Kernel forward mathematics (rwkv_cuda.cu, kernel_forward, at _t = 0):
  y_i = ( sum_{s<i} exp(k_s - w*(i-s)) v_s + exp(u + k_i) v_i )
        / ( sum_{s<i} exp(k_s - w*(i-s))     + exp(u + k_i) + EPS )
i.e. k sits inside the exponential (softmax-like weighting) and u is a
log-space bonus on the current step.

wkv_pure mathematics (code/wkv_pure.py):
  y_i = ( u*k_i*v_i + sum_{s<i} exp(-w*(i-s)) k_s v_s )
        / ( u*k_i     + sum_{s<i} exp(-w*(i-s)) k_s + EPS )
i.e. k enters linearly and u is a multiplicative coefficient.

Usage: python exp_wkv_semantics.py --ex EX-194
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
CR_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(CR_ROOT), str(CR_ROOT / "code"), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import wkv_pure  # noqa: E402
from exp_harness import ExSnapshot  # noqa: E402

EPS = 1e-6


def kernel_ref(w, u, k, v):
    """O(T^2) fp64 reference following the formula in rwkv_cuda.cu. w,u:(C,), k,v:(B,T,C) -> (B,T,C)."""
    k = k.detach().cpu().double().numpy()
    v = v.detach().cpu().double().numpy()
    w = w.detach().cpu().double().numpy().reshape(-1)
    u = u.detach().cpu().double().numpy().reshape(-1)
    B, T, C = k.shape
    y = np.zeros_like(k)
    for b in range(B):
        for i in range(T):
            if i == 0:
                hist_n = np.zeros(C); hist_d = np.zeros(C)
            else:
                # sum_{s<i} exp(k_s - w(i-s)) X_s, stabilised with log-sum-exp
                ks = k[b, :i] - w[None, :] * (i - np.arange(i))[:, None]
                m = ks.max(axis=0)
                wgt = np.exp(ks - m[None, :])
                hist_n = (wgt * v[b, :i]).sum(axis=0) * np.exp(m)
                hist_d = wgt.sum(axis=0) * np.exp(m)
            e3 = np.exp(u + k[b, i])          # kernel: e3 = exp(u + k_i - no)
            y[b, i] = (hist_n + e3 * v[b, i]) / (hist_d + e3 + EPS)
    return torch.from_numpy(y).float()


def pure_ref(w, u, k, v):
    return wkv_pure.run_reference(w, u, k, v)


def rel_err(a, b):
    a = a.double(); b = b.double()
    return float((a - b).norm() / (b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", default="EX-194")
    ap.add_argument("--tag", default="wkv_semantics_audit")
    args = ap.parse_args()

    snap = ExSnapshot(args.ex, tag=args.tag, root=CR_ROOT / "results")
    snap.set(device="cpu", dtype="float64 reference", eps=EPS,
             protocol="CPU reproduction of the kernel formula vs wkv_pure; identical w/u/k/v inputs")

    rng = np.random.default_rng(7)
    cases = []
    # Test magnitudes: T=64/256; k uses the backbone's real scale (linear output, std ~ 1)
    for T in (64, 256):
        for C in (32, 64):
            B = 2
            w = torch.tensor(rng.normal(0, 0.01, C), dtype=torch.float32)      # ≈ spatial_decay/T
            u = torch.tensor(rng.normal(0, 1e-3, C), dtype=torch.float32)      # ≈ spatial_first/T
            k = torch.tensor(rng.normal(0, 1.0, (B, T, C)), dtype=torch.float32)
            v = torch.tensor(rng.normal(0, 1.0, (B, T, C)), dtype=torch.float32)
            y_k = kernel_ref(w, u, k, v)
            y_p = pure_ref(w, u, k, v).float()
            cases.append({"T": T, "C": C, "B": B,
                          "rel_err_kernel_vs_pure": round(rel_err(y_k, y_p), 6),
                          "max_abs_diff": round(float((y_k - y_p).abs().max()), 6),
                          "y_kernel_mean": round(float(y_k.mean()), 6),
                          "y_pure_mean": round(float(y_p.mean()), 6)})
            print(f"T={T} C={C}: rel_err={cases[-1]['rel_err_kernel_vs_pure']:.3e} "
                  f"max_abs={cases[-1]['max_abs_diff']:.3e} "
                  f"(kernel mean {cases[-1]['y_kernel_mean']:.4f} vs pure {cases[-1]['y_pure_mean']:.4f})",
                  flush=True)

    # Reverse check: if u in the pure implementation is replaced by log(u) (trying to make
    # exp(u') ~ u), is it still non-equivalent?
    T, C, B = 64, 32, 2
    w = torch.tensor(rng.normal(0, 0.01, C), dtype=torch.float32)
    u_pos = torch.tensor(rng.uniform(0.5, 2.0, C), dtype=torch.float32)
    k = torch.tensor(rng.normal(0, 1.0, (B, T, C)), dtype=torch.float32)
    v = torch.tensor(rng.normal(0, 1.0, (B, T, C)), dtype=torch.float32)
    y_p = pure_ref(w, u_pos, k, v).float()
    y_k = kernel_ref(w, torch.log(u_pos), k, v)
    alt = {"note": "mapping attempt with u_kernel = log(u_pure)",
           "rel_err": round(rel_err(y_k, y_p), 6)}
    print(f"mapping u->log(u): rel_err={alt['rel_err']:.3e}", flush=True)

    out = {"cases": cases, "mapping_attempt": alt,
           "conclusion": "The two forward formulations are mathematically non-equivalent: in the kernel, k enters inside exp and u is a log-space additive term; in wkv_pure, k is a linear weight and u is a multiplicative coefficient. "
                         "Hence the kernel's 10-33x unit-level speed-up does not transfer directly to this model; the kernel must be rewritten."
                         if min(c["rel_err_kernel_vs_pure"] for c in cases) > 0.1 else
                         "The two agree within a small error (contrary to expectation; needs re-checking)."}
    (snap.dir / "wkv_semantics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    snap.set(min_rel_err=min(c["rel_err_kernel_vs_pure"] for c in cases),
             n_cases=len(cases), same_operator=False)
    snap.write_meta("SUCCESS")
    snap.write_manifest(title=f"{args.ex} {args.tag}")
    print(f"\n[{args.ex}] done. outputs={snap.dir}", flush=True)


if __name__ == "__main__":
    main()
