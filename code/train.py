# -*- coding: utf-8 -*-
"""Unified training harness shared by every method in the benchmark.

Protocol (identical for all seven methods):
  input     512x512 random crops sampled from the high-resolution images;
            test-time sliding window with stride = half window, plain argmax
  optimiser AdamW, lr 1e-4, weight decay 1e-4, cosine schedule, 500-iteration warmup
  schedule  --epochs default 150, early stopping with patience 30 on the
            validation metric
  batch     8, AMP fp16 (the WKV scan itself runs in fp32)
  loss      0.5 * BCE(logits) + 0.5 * Dice, binary
  augments  flip / rot90 / HSV / RandomResizedCrop(0.5-1.0)
  seed      --seed

Resuming:
  ``best.pt`` and ``last.pt`` are written every epoch (model / optimiser /
  scheduler / epoch / best metric / RNG state). ``--resume <ckpt>`` restores
  the random-network state plus optimiser, scheduler, epoch counter and RNG.

Output (results/<EX-ID>/):
  terminal.log / meta.yaml / env_snapshot.txt / command.txt / manifest.sha256
  metrics.csv   per-epoch loss and IoU
  checkpoints/{last,best}.pt
  Per-image test metrics are produced separately by ``eval.py``.

Preprocessing:
  The source images are roughly 6000x4000. On first use the training split is
  cached as JPEG files with the long side at 1024 px, identically for every
  method, so that 512 crops can be sampled efficiently; the pixels are only
  downsampled, never altered otherwise.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from exp_harness import ExSnapshot
from models_registry import build_model

SPLITS = PROJECT_ROOT / "results" / "EX-020" / "splits"
SRC = PROJECT_ROOT / "data" / "kaggle_leafseg" / "extracted" / "data" / "data"
CACHE = PROJECT_ROOT / "data" / "kaggle_leafseg" / "cache"
CACHE_SIDE = 1024

# Dataset switch; the plantdoc entry reproduces the benchmark configuration and
# the plantseg entry is the external-validation dataset.
DATASETS = {
    "plantdoc": dict(src=PROJECT_ROOT / "data" / "kaggle_leafseg" / "extracted" / "data" / "data",
                     cache=PROJECT_ROOT / "data" / "kaggle_leafseg" / "cache",
                     splits=PROJECT_ROOT / "results" / "EX-020" / "splits"),
    "plantseg": dict(src=PROJECT_ROOT / "data" / "plantseg" / "raw" / "data",
                     cache=PROJECT_ROOT / "data" / "plantseg" / "cache",
                     splits=PROJECT_ROOT / "data" / "plantseg" / "splits"),
}


# ---------------------------------------------------------------------------
# Cache preprocessing (training sub-images only, for efficient 512x512 sampling)
# ---------------------------------------------------------------------------
def build_cache(ids, progress_every=20):
    (CACHE / "train" / "images").mkdir(parents=True, exist_ok=True)
    (CACHE / "train" / "masks").mkdir(parents=True, exist_ok=True)
    made = 0
    for i, imid in enumerate(ids):
        img_p = CACHE / "train" / "images" / f"{imid}.jpg"
        msk_p = CACHE / "train" / "masks" / f"{imid}.png"
        if img_p.exists() and msk_p.exists():
            continue
        im = Image.open(SRC / "images" / f"{imid}.jpg")
        mk = Image.open(SRC / "masks" / f"{imid}.png").convert("L")
        w, h = im.size
        s = CACHE_SIDE / max(w, h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        im = im.resize((nw, nh), Image.BILINEAR)
        mk = mk.resize((nw, nh), Image.NEAREST)  # nearest-neighbour resize keeps the mask binary
        im.save(img_p, "JPEG", quality=95)
        mk.save(msk_p, "PNG")
        made += 1
        if (i + 1) % progress_every == 0:
            print(f"  cache {i+1}/{len(ids)}", flush=True)
    print(f"cache ready ({made} newly built, {len(ids)} total)", flush=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NWRDSet(Dataset):
    def __init__(self, ids, split="train", transform=None, gt_erode=0, shuffle_labels=False):
        assert split == "train", "this harness shares the training-subset cache for both training and validation"
        self.ids = ids
        self.transform = transform
        self.gt_erode = gt_erode
        self.shuffle_labels = shuffle_labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        imid = self.ids[idx]
        img = Image.open(CACHE / "train" / "images" / f"{imid}.jpg").convert("RGB")
        msk = Image.open(CACHE / "train" / "masks" / f"{imid}.png").convert("L")
        w, h = img.size
        sz = self.transform["crop_size"] if self.transform else 512
        scale = self.transform["scale"] if self.transform else (0.5, 1.0)
        # Lesion-biased sampling, applied identically to every method: with
        # probability lesion_bias the crop window is anchored on a lesion pixel.
        m_arr = np.asarray(msk) > 127
        lesion_xy = np.argwhere(m_arr)   # (N,2) rows=[y,x]
        r = random.uniform(*scale)
        cw = min(w, max(sz, int(w * math.sqrt(r))))
        ch = min(h, max(sz, int(h * math.sqrt(r))))
        bi = (self.transform or {}).get("lesion_bias", 0.9)
        if bi > 0 and random.random() < bi and len(lesion_xy) > 0:
            ly, lx = lesion_xy[random.randint(0, len(lesion_xy) - 1)]
            x = random.randint(max(0, min(lx, w - cw)), min(max(0, lx), w - cw))
            y = random.randint(max(0, min(ly, h - ch)), min(max(0, ly), h - ch))
        else:
            x = random.randint(0, max(0, w - cw))
            y = random.randint(0, max(0, h - ch))
        img = img.crop((x, y, x + cw, y + ch)).resize((sz, sz), Image.BILINEAR)
        msk = msk.crop((x, y, x + cw, y + ch)).resize((sz, sz), Image.NEAREST)

        if self.transform:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT); msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.rotate(90, expand=False); msk = msk.rotate(90, expand=False)
            if random.random() < 0.5:
                img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
        t = np.asarray(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        m = (np.asarray(msk).astype(np.float32) / 255.0)
        m = (m > 0.5).astype(np.float32)
        if self.gt_erode > 0:  # morphological erosion of the GT mask, radius gt_erode px
            import cv2
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.gt_erode + 1, 2 * self.gt_erode + 1))
            m = cv2.erode(m, k, iterations=1)
            m = (m > 0.5).astype(np.float32)
        if self.shuffle_labels:  # sanity check: large random roll breaks spatial correspondence (lesion ratio preserved)
            _rng = np.random.RandomState(idx * 9973 + 17)
            m = np.roll(np.roll(m, _rng.randint(1, m.shape[0]), axis=0), _rng.randint(1, m.shape[1]), axis=1)
        return torch.tensor(t), torch.tensor(m).unsqueeze(0)


# ---------------------------------------------------------------------------
# Loss: 0.5*Focal + 0.5*(1 - Soft-Tversky)
# ---------------------------------------------------------------------------
def focal_tversky(logits, target, alpha=0.25, gamma=2.0, beta=0.7):
    """Unified loss robust to class imbalance: 0.5*Focal + 0.5*(1 - Soft-Tversky).
    Focal: alpha/gamma suppress the abundant easy background; Tversky: beta < 0.5 favours
    recall (small targets). Following the original NWRD authors' use of focal-family losses;
    applied identically to all methods."""
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal = -alpha_t * (1 - pt).pow(gamma) * torch.log(pt.clamp_min(1e-6))
    focal = focal.mean()
    tp = (p * target).sum(); fp = (p * (1 - target)).sum(); fn = ((1 - p) * target).sum()
    tversky = (tp + 1.0) / (tp + beta * fn + (1 - beta) * fp + 1.0)
    return 0.5 * focal + 0.5 * (1.0 - tversky)


def iou_bin(pred, target):
    p = (pred > 0.5).float()
    inter = (p * target).sum().float()
    union = (p + target).clamp(max=1).sum().float()
    return (inter + 1e-6) / (union + 1e-6)


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", default="EX-004")
    ap.add_argument("--tag", default="baseline_a0_sanity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--warmup_iters", type=int, default=500)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--loss", default="focal_tversky", help="focal_tversky (default) or bce_dice")
    ap.add_argument("--model", default="a0", help="model name: a0 / unet / deeplabv3plus / segformer / swinunet / full")
    ap.add_argument("--lesion_bias", type=float, default=0.9, help="lesion-biased sampling probability")
    ap.add_argument("--no_mpsa", action="store_true", help="disable MPSA (leave-one-out)")
    ap.add_argument("--no_vdas", action="store_true", help="disable VDAS (leave-one-out)")
    ap.add_argument("--no_feem", action="store_true", help="disable FEEM (leave-one-out)")
    ap.add_argument("--vdas_rw", type=float, default=1.0, help="VDAS residual weight lambda")
    ap.add_argument("--vdas_stop_grad", action="store_true", help="stop-gradient on the VDAS branch input")
    ap.add_argument("--vdas_mode", default="gated", help="scan ablation: gated / h_only / v_only / equal")
    ap.add_argument("--feem_all", action="store_true", help="apply FEEM at all four stages (default: stages 2 and 3)")
    ap.add_argument("--shuffle_labels", action="store_true", help="sanity check: roll the training masks")
    ap.add_argument("--pretrained", action="store_true", help="load ImageNet encoder weights for the pretrained baselines (smp)")
    ap.add_argument("--gt_erode", type=int, default=0, help="label noise: morphological erosion radius of the GT mask in px, 0 = off")
    ap.add_argument("--subset_frac", type=float, default=1.0, help="training-subset fraction {0.1, 0.25, 0.5, 1.0}")
    ap.add_argument("--embed_dims", default="", help="capacity scaling: comma-separated stage widths, e.g. 32,64,128,256 (empty = 64,128,256,512)")
    ap.add_argument("--resume", default="", help="path to a last/best checkpoint to resume from")
    ap.add_argument("--dataset", default="plantdoc", choices=sorted(DATASETS),
                    help="plantdoc (benchmark) or plantseg (external validation)")
    ap.add_argument("--clip_grad", type=float, default=0.0,
                    help="global gradient-norm clipping threshold; 0 disables clipping")
    ap.add_argument("--nan_guard", action="store_true",
                    help="skip the optimiser step when the loss is non-finite; off by default")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader workers (default 0)")
    args = ap.parse_args()
    global SRC, CACHE, SPLITS
    _d = DATASETS[args.dataset]
    SRC, CACHE, SPLITS = _d["src"], _d["cache"], _d["splits"]
    print(f"[{args.ex}] dataset={args.dataset} cache={CACHE}", flush=True)

    snap = ExSnapshot(args.ex, tag=args.tag, root=PROJECT_ROOT / "results")
    snap.set(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
             img_size=args.img_size, model=args.model, optimizer="AdamW",
             loss=args.loss, protocol="v4_unified_lesionbiased", lesion_bias=args.lesion_bias)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.ex}] device={dev}", flush=True)

    train_ids = [l.strip() for l in (SPLITS / "train.txt").read_text().splitlines() if l.strip()]
    if args.subset_frac < 1.0:  # data-efficiency subset
        rng = random.Random(42)
        rng.shuffle(train_ids)
        train_ids = train_ids[: max(1, int(len(train_ids) * args.subset_frac))]
        print(f"[{args.ex}] E5c subset_frac={args.subset_frac} -> {len(train_ids)} train ids", flush=True)
    val_ids = [l.strip() for l in (SPLITS / "val.txt").read_text().splitlines() if l.strip()]
    if not train_ids:
        print("!! no train ids"); sys.exit(2)
    print(f"[{args.ex}] cache preprocessing train ids={len(train_ids)}", flush=True)
    build_cache(sorted(set(train_ids + val_ids)))

    transform = {"crop_size": args.img_size, "scale": (0.5, 1.0), "lesion_bias": args.lesion_bias}
    tr_ds = NWRDSet(train_ids, "train", transform, gt_erode=args.gt_erode, shuffle_labels=args.shuffle_labels)
    va_ds = NWRDSet(val_ids, "train", transform)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    _mk = dict(img_size=args.img_size, in_channels=3, num_classes=1, pretrained=args.pretrained)
    if args.model in ("full", "crw-rwkv-full"):
        _mk.update(enable_mpsa=not args.no_mpsa, enable_vdas=not args.no_vdas, enable_feem=not args.no_feem,
                   vdas_residual_weight=args.vdas_rw, vdas_stop_grad=args.vdas_stop_grad,
                   vdas_mode=args.vdas_mode, feem_all=args.feem_all)
        if args.embed_dims:
            _mk["embed_dims"] = tuple(int(x) for x in args.embed_dims.split(",") if x.strip())
    model = build_model(args.model, **_mk).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    snap.set(n_params=n_params)
    print(f"[{args.ex}] model=A0 params={n_params} ids:train={len(train_ids)} val={len(val_ids)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    iters_per_ep = max(1, len(tr_dl))
    total_iters = args.epochs * iters_per_ep
    warmup = min(args.warmup_iters, max(1, total_iters // 4))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, it / warmup) * (0.5 * (1 + math.cos(math.pi * min(it, total_iters) / total_iters))))

    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    start_ep = 0
    best_loss = float("inf")
    skipped_steps = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"] + 1
        best_loss = ck["best"]
        try:
            _r = ck["rng"]
            if _r is not None and isinstance(_r, torch.Tensor) and _r.dtype == torch.uint8:
                torch.set_rng_state(_r.cpu() if _r.is_cuda else _r)
            else:
                print(f"[{args.ex}] warning: skip non-ByteTensor rng resume", flush=True)
        except Exception as _e:
            print(f"[{args.ex}] warning: rng resume skipped ({_e})", flush=True)
        print(f"[{args.ex}] resumed from {args.resume}, epoch={start_ep}, best={best_loss:.4f}", flush=True)

    # record the exact command for reproducibility
    import sys as _s
    cmd = f"python {_s.argv[0]} " + " ".join(_s.argv[1:])
    (snap.dir / "command.txt").write_text(cmd + "\n", encoding="utf-8")

    ckpt_dir = snap.dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    snap.write_env_snapshot()

    wait = 0
    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        model.train()
        run_loss, run_iou, nb = 0.0, 0.0, 0
        for bi, (x, m) in enumerate(tr_dl):
            x, m = x.to(dev), m.to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                logits = model(x)
                loss = focal_tversky(logits, m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            if args.nan_guard and not bool(torch.isfinite(loss)):
                skipped_steps += 1
                scaler.update()
            else:
                scaler.step(opt)
                scaler.update()
            sched.step()
            # synchronously evaluate the current batch for progress display (no gradients)
            with torch.no_grad():
                run_loss += loss.item(); run_iou += float(iou_bin(torch.sigmoid(logits.float()), m)); nb += 1
            if (bi + 1) % 5 == 0:
                print(f"  ep{ep} it{bi+1}/{len(tr_dl)} loss={loss.item():.4f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        # validation
        model.eval()
        v_loss, v_iou, vnb = 0.0, 0.0, 0
        with torch.no_grad():
            for x, m in va_dl:
                x, m = x.to(dev), m.to(dev)
                with torch.autocast("cuda", enabled=dev.type == "cuda"):
                    logits = model(x)
                    v_loss += focal_tversky(logits, m).item()
                v_iou += float(iou_bin(torch.sigmoid(logits.float()), m)); vnb += 1
        v_loss /= max(1, vnb); v_iou /= max(1, vnb)
        metrics.append({"epoch": ep, "train_loss": run_loss / max(1, nb),
                        "train_iou": run_iou / max(1, nb), "val_loss": v_loss, "val_iou": v_iou})
        print(f"[{args.ex}] ep{ep} train_loss={run_loss/max(1,nb):.4f} train_iou={run_iou/max(1,nb):.4f} "
              f"val_loss={v_loss:.4f} val_iou={v_iou:.4f}", flush=True)

        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "epoch": ep, "best": best_loss, "rng": torch.get_rng_state()}
        torch.save(ck, ckpt_dir / "last.pt")
        if v_loss < best_loss:
            best_loss = v_loss; torch.save(ck, ckpt_dir / "best.pt")
            wait = 0
        else:
            wait += 1
        (snap.dir / "metrics.csv").write_text(
            "epoch,train_loss,train_iou,val_loss,val_iou\n" +
            "\n".join(f"{r['epoch']},{r['train_loss']:.6f},{r['train_iou']:.6f},{r['val_loss']:.6f},{r['val_iou']:.6f}" for r in metrics) + "\n",
            encoding="utf-8")
        if wait >= args.patience:
            print(f"[{args.ex}] early-stop at ep{ep}"); break

    snap.set(best_val_loss=best_loss, train_seconds=round(time.time() - t0, 1),
             skipped_nonfinite_steps=skipped_steps, clip_grad=args.clip_grad,
             nan_guard=bool(args.nan_guard))
    snap.write_meta("SUCCESS")
    snap.write_manifest(title=f"{args.ex} {args.tag}")
    print("=" * 60)
    print(f"[{args.ex}] done. best_val_loss={best_loss:.4f}. outputs={snap.dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()