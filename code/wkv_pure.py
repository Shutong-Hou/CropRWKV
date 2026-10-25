# -*- coding: utf-8 -*-
"""Pure-PyTorch WKV scan used by the backbone and by every VDAS block.

Definition, for each (batch, channel) pair and each position t:

    y[t] = ( u * k[t] * v[t] + sum_{s<t} exp(-w * (t - s)) * k[s] * v[s] )
           / ( u * k[t]         + sum_{s<t} exp(-w * (t - s)) * k[s] + EPS )

The keys enter linearly; they are not exponentiated. Because the key
projections are signed, the denominator is a signed sum and can cancel close
to the constant EPS -- see the manuscript's diagnosis section.

Implementation: closed form. With damp[t] = exp(-w*t) and g[s] = exp(w*s),

    sum_{s<t} exp(-w(t-s)) X_s = exp(-w*t) * sum_{s<t} exp(w*s) X_s
                               = damp[t] * cumsum(g*X)[t-1]

The prefix sums are accumulated in float64 so that exp() cannot overflow for
the sequence lengths used here (T <= 4096, w = decay/T).

The module ships with ``--selfcheck``: the vectorised implementation is
compared element-wise against an explicit O(T^2) evaluation of the definition.
"""
from __future__ import annotations

import torch

EPS = 1e-6


def run(w, u, k, v):
    """Vectorised pure-PyTorch WKV.

    w, u: (C,);  k, v: (B, T, C);  returns y: (B, T, C) as float32.
    """
    B, T, C = k.shape
    W = w.flatten().to(torch.float64).reshape(1, C)      # (1, C)
    U = u.flatten().to(torch.float64).reshape(1, 1, C)   # (1, 1, C)
    K = k.float().to(torch.float64)                      # (B, T, C)
    V = v.float().to(torch.float64)
    t = torch.arange(T, device=w.device, dtype=torch.float64).view(T, 1)  # (T, 1)
    g = torch.exp(t * W)                                 # (T, C) = exp(s*w)
    damp = torch.exp(-t * W)                             # (T, C) = exp(-t*w)
    # Inclusive prefix sums; shifting by one position gives the s < t sums.
    G = g.unsqueeze(0)                                   # (1, T, C)
    preK = torch.cumsum(G * K, dim=1)                    # (B, T, C)
    preD = torch.cumsum(G * K * V, dim=1)
    z = torch.zeros_like(preK[:, :1])
    preK_shift = torch.cat([z, preK[:, :-1]], dim=1)
    preD_shift = torch.cat([torch.zeros_like(preD[:, :1]), preD[:, :-1]], dim=1)
    damp_ = damp.unsqueeze(0)                            # (1, T, C)
    numer = U * K * V + damp_ * preD_shift
    denom = U * K + damp_ * preK_shift + EPS
    y = (numer / denom).to(torch.float32)
    return y


def run_reference(w, u, k, v):
    """Explicit O(T^2) evaluation of the definition; used only for validation."""
    B, T, C = k.shape
    W = w.flatten().to(torch.float64).reshape(1, C)
    U = u.flatten().to(torch.float64).reshape(1, C)
    K = k.float().to(torch.float64)
    V = v.float().to(torch.float64)
    y = torch.zeros_like(K)
    for b in range(B):
        for t in range(T):
            num = U * K[b, t] * V[b, t]
            den = U * K[b, t]
            for s in range(t):
                f = torch.exp(-W * (t - s))
                num = num + f * K[b, s] * V[b, s]
                den = den + f * K[b, s]
            y[b, t] = num / (den + EPS)
    return y.to(torch.float32)


def selfcheck(seed=7, B=2, T=11, C=5, tol=1e-6):
    torch.manual_seed(seed)
    w = (torch.randn(C) * 0.03).double()      # small w, matching the decay/T scale
    u = (torch.randn(C) * 0.1).float()
    k = torch.randn(B, T, C)
    v = torch.randn(B, T, C)
    y = run(w.float(), u, k, v)
    yr = run_reference(w, u, k, v).float()
    diff = (y - yr).abs().max().item()
    print(f"selfcheck max_abs_diff = {diff:.3e} (tol={tol})")
    assert diff <= 1e-5, f"selfcheck failed: {diff}"
    # backward pass
    k2 = k.clone().requires_grad_(True)
    l = run(w.float(), u, k2, v).sum()
    l.backward()
    print("backward ok, k.grad max =", k2.grad.abs().max().item())
    return diff


if __name__ == "__main__":
    selfcheck()
