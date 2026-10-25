# -*- coding: utf-8 -*-
"""Acquire and standardise the second dataset (PlantSeg) for external validation.

Source: the public test-set release ``Voxel51/PlantSeg-Test`` on HuggingFace --
1,200 images, 115 disease classes, collected in the wild; original paper Wei et
al., arXiv:2409.04038; dataset descriptor Sci Data 2026, 13:205. Licensed
CC BY-NC-ND 4.0 (research use only; not redistributed here).

Steps:
  1) download the images and samples.json;
  2) parse samples.json: per-sample plant / disease / mask_ratio / detections /
     segmentations (polylines);
  3) rasterise the normalised polygons into a binary lesion mask of the same
     size as the image (union of polygons) and store it as PNG;
  4) report counts of annotated vs box-only samples and cross-check the
     rasterised lesion fraction against the recorded mask_ratio;
  5) split 70/15/15 stratified by lesion fraction and write the split files;
  6) build a cache with max_side <= 1024 (downsampling only), with the same
     layout as the primary dataset so that train.py / eval.py work unchanged.

Output: results/EX-159/ (meta.yaml, command.txt, manifest.sha256,
prepare_report.txt, plantseg_stats.json, rasterization_check.csv); the data go
to data/plantseg/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from exp_harness import ExSnapshot

DATA = PROJECT_ROOT / "data" / "plantseg"
RAW = DATA / "raw"
MASKS = DATA / "masks"
CACHE = DATA / "cache"
SPLITS = DATA / "splits"
CACHE_SIDE = 1024
REPO = "Voxel51/PlantSeg-Test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", default="EX-159")
    ap.add_argument("--tag", default="plantseg_second_dataset_prepare")
    ap.add_argument("--cache_side", type=int, default=CACHE_SIDE)
    args = ap.parse_args()
    snap = ExSnapshot(args.ex, tag=args.tag, root=PROJECT_ROOT / "results")
    (snap.dir / "command.txt").write_text(
        f'{sys.executable} experiments/{Path(__file__).name} --ex {args.ex}\n',
        encoding="utf-8")
    snap.set(source=f"huggingface:{REPO}", upstream="PlantSeg (Wei et al., arXiv:2409.04038)",
             license="CC BY-NC-ND 4.0 (research use; not redistributed)",
             cache_side=args.cache_side, split="70/15/15 stratified by lesion fraction")

    # 1) Download
    # Note: snapshot_download's HEAD request fails in this environment (LocalEntryNotFoundError),
    # but the HTTP API and direct CDN access work, so list files via the API and download them
    # directly with a thread pool, retrying each file up to 3 times and logging failures.
    import subprocess as _sp
    import time as _time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"downloading {REPO} -> {RAW}", flush=True)
    _req_url = f"https://huggingface.co/api/datasets/{REPO}"
    _apijson = RAW / "_api_files.json"
    _meta = None
    for _k in range(4):
        _p = _sp.run(["curl.exe", "-sL", "--max-time", "60", "-o", str(_apijson), _req_url],
                     capture_output=True, text=True)
        try:
            if _p.returncode == 0:
                _meta = json.loads(_apijson.read_text(encoding="utf-8"))
                if _meta and "siblings" in _meta:
                    break
        except Exception:
            _meta = None
        print(f"  api listing retry {_k+1} (rc={_p.returncode})", flush=True)
        _time.sleep(3 * (_k + 1))
    if not _meta or "siblings" not in _meta:
        raise SystemExit("failed to fetch dataset file listing via curl")
    files = [s["rfilename"] for s in _meta["siblings"]
             if s["rfilename"].startswith("data/")
             or s["rfilename"] in ("samples.json", "metadata.json", "README.md")]
    print(f"files to fetch: {len(files)}", flush=True)

    def _valid(dst):
        """A file is valid only if it exists and decodes with PIL (guards against truncated files or error pages saved as .jpg)."""
        if not dst.exists() or dst.stat().st_size == 0:
            return False
        if dst.suffix.lower() in (".jpg", ".jpeg", ".png"):
            try:
                with Image.open(dst) as _im:
                    _im.verify()
                return True
            except Exception:
                return False
        return True

    def _fetch(rel):
        dst = RAW / rel
        if _valid(dst):
            return "skip"
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}"
        # use curl.exe: urllib is frequently interrupted by this CDN; curl follows redirects more reliably
        for k in range(3):
            try:
                p = _sp.run(["curl.exe", "-sL", "--max-time", "60", "-o", str(dst), url],
                            capture_output=True, text=True)
                if p.returncode == 0 and _valid(dst):
                    return "ok"
                raise RuntimeError(f"curl rc={p.returncode}")
            except Exception as e:
                try:
                    if dst.exists():
                        dst.unlink()      # delete the incomplete file so it cannot corrupt downstream processing
                except Exception:
                    pass
                if k == 2:
                    return f"fail:{rel}:{e}"
                _time.sleep(1.5 * (k + 1))
        return "fail"

    _done, _fails = 0, []
    with ThreadPoolExecutor(max_workers=12) as _ex:
        _futs = {_ex.submit(_fetch, rel): rel for rel in files}
        for _fu in as_completed(_futs):
            _res = _fu.result()
            _done += 1
            if str(_res).startswith("fail"):
                _fails.append(str(_res))
            if _done % 200 == 0:
                print(f"  fetched {_done}/{len(files)} (fails={len(_fails)})", flush=True)
    print(f"download done: {_done} files, fails={len(_fails)}", flush=True)
    if _fails:
        print("  first failures:", _fails[:3], flush=True)
        (snap.dir / "download_failures.txt").write_text("\n".join(_fails), encoding="utf-8")
    samples_p = RAW / "samples.json"
    if not samples_p.exists():
        raise SystemExit("samples.json not found after download")
    js = json.loads(samples_p.read_text(encoding="utf-8"))
    samples = js["samples"]
    print(f"samples: {len(samples)}", flush=True)

    # 2)+3) Rasterisation
    MASKS.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    check_rows = ["sample_id,plant,disease,mask_ratio_field,rasterized_frac,abs_diff,polygons,img_w,img_h"]
    recs = []
    for s in samples:
        fp = s.get("filepath", "")
        img_p = RAW / fp
        if not img_p.exists():
            stats["missing_image"] += 1
            continue
        sid = Path(fp).stem
        try:
            with Image.open(img_p) as im:
                w, h = im.size
        except Exception:
            stats["corrupt_image"] += 1      # count honestly instead of skipping silently
            continue
        segs = (s.get("segmentations") or {}).get("polylines") or []
        if len(segs) == 0:
            stats["no_segmentation"] += 1
            continue
        m = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(m)
        n_poly = 0
        for seg in segs:
            pts = seg.get("points") or []
            if not pts:
                continue
            ring = pts[0] if isinstance(pts[0][0], (list, tuple)) else pts
            xy = [(float(p[0]) * w, float(p[1]) * h) for p in ring if len(p) >= 2]
            if len(xy) >= 3:
                dr.polygon(xy, fill=255)
                n_poly += 1
        arr = np.asarray(m)
        frac = float((arr > 127).mean())
        if frac <= 0:
            stats["empty_after_rasterize"] += 1
            continue
        Image.fromarray(arr).save(MASKS / f"{sid}.png", "PNG")
        mr = s.get("mask_ratio")
        check_rows.append(f"{sid},{s.get('plant','')},{s.get('disease','')},{mr},{frac:.6f},"
                          f"{abs(float(mr)-frac) if mr is not None else ''},{n_poly},{w},{h}")
        recs.append(dict(sid=sid, frac=frac, plant=s.get("plant", ""), disease=s.get("disease", ""),
                         img_p=str(img_p), w=w, h=h))
        stats["ok"] += 1
        if len(segs) > 1:
            stats["multi_polygon"] += 1

    print(f"rasterized ok={stats['ok']} no_segmentation={stats['no_segmentation']} "
          f"empty={stats['empty_after_rasterize']} missing_image={stats['missing_image']}", flush=True)

    # 4) Consistency checks
    diffs = [float(r.split(",")[5]) for r in check_rows[1:] if r.split(",")[5] not in ("", "None")]
    fracs = np.array([r["frac"] for r in recs])
    stat = {
        "n_samples_in_json": len(samples), "n_rasterized": len(recs),
        "n_no_segmentation": stats["no_segmentation"], "n_empty_after_rasterize": stats["empty_after_rasterize"],
        "n_missing_image": stats["missing_image"], "n_multi_polygon": stats["multi_polygon"],
        "n_plants": len({r["plant"] for r in recs}), "n_diseases": len({r["disease"] for r in recs}),
        "lesion_frac": {"mean": float(fracs.mean()), "median": float(np.median(fracs)),
                        "p25": float(np.percentile(fracs, 25)), "p75": float(np.percentile(fracs, 75)),
                        "min": float(fracs.min()), "max": float(fracs.max()),
                        "n_lt5pct": int((fracs < 0.05).sum()), "n_gt30pct": int((fracs > 0.30).sum())},
        "mask_ratio_vs_rasterized": {"n": len(diffs), "mean_abs_diff": float(np.mean(diffs)),
                                     "max_abs_diff": float(np.max(diffs))},
        "resolution": {"n_unique": len({(r['w'], r['h']) for r in recs}),
                       "median_w": int(np.median([r["w"] for r in recs])),
                       "median_h": int(np.median([r["h"] for r in recs]))},
        "top_plants": Counter([r["plant"] for r in recs]).most_common(12),
    }
    (snap.dir / "plantseg_stats.json").write_text(json.dumps(stat, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    (snap.dir / "rasterization_check.csv").write_text("\n".join(check_rows) + "\n", encoding="utf-8")

    # 5) Stratified 70/15/15 split
    recs_sorted = sorted(recs, key=lambda r: r["frac"])
    n = len(recs_sorted)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    # interleaved assignment keeps the lesion-fraction distribution consistent across the three sets
    buckets = {"train": [], "val": [], "test": []}
    plan = ["train", "train", "train", "val", "train", "train", "train", "test",
            "train", "train", "train", "val", "train", "train", "train", "test",
            "train", "train", "train", "val"]
    for i, r in enumerate(recs_sorted):
        buckets[plan[i % len(plan)]].append(r["sid"])
    SPLITS.mkdir(parents=True, exist_ok=True)
    for k, v in buckets.items():
        (SPLITS / f"{k}.txt").write_text("\n".join(v) + "\n", encoding="utf-8")
    print(f"splits: train={len(buckets['train'])} val={len(buckets['val'])} test={len(buckets['test'])}",
          flush=True)

    # 6) Cache (downsampling only)
    by_id = {r["sid"]: r for r in recs}
    for split, ids in buckets.items():
        for d in ("images", "masks"):
            (CACHE / split / d).mkdir(parents=True, exist_ok=True)
        for sid in ids:
            r = by_id[sid]
            ip = CACHE / split / "images" / f"{sid}.jpg"
            mp = CACHE / split / "masks" / f"{sid}.png"
            if ip.exists() and mp.exists():
                continue
            im = Image.open(r["img_p"]).convert("RGB")
            w, h = im.size
            s = min(1.0, args.cache_side / max(w, h))
            nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
            im.resize((nw, nh), Image.BILINEAR).save(ip, "JPEG", quality=95)
            Image.open(MASKS / f"{sid}.png").convert("L").resize((nw, nh), Image.NEAREST).save(mp, "PNG")
        print(f"cache {split} ready ({len(ids)})", flush=True)

    rep = [f"source: {REPO} (PlantSeg official test split, in-the-wild, CC BY-NC-ND 4.0)",
           f"samples in json={len(samples)}; rasterized={len(recs)}; "
           f"no segmentation={stats['no_segmentation']}; empty after rasterize={stats['empty_after_rasterize']}",
           f"plants={stat['n_plants']} diseases={stat['n_diseases']}",
           f"lesion frac mean={fracs.mean():.4f} median={np.median(fracs):.4f} "
           f"lt5%={int((fracs<0.05).sum())} gt30%={int((fracs>0.30).sum())}",
           f"rasterized vs mask_ratio: n={len(diffs)} mean|diff|={np.mean(diffs):.4f} max|diff|={np.max(diffs):.4f}",
           f"resolution unique={stat['resolution']['n_unique']} median="
           f"{stat['resolution']['median_w']}x{stat['resolution']['median_h']}",
           "NOTE: images cached with downscale-only (no upsampling), unlike the PlantDoc pipeline."]
    (snap.dir / "prepare_report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    snap.write_env_snapshot()
    snap.write_meta("SUCCESS")
    snap.write_manifest(title=f"{args.ex} {args.tag}")
    print(f"[{args.ex}] done -> {snap.dir}")


if __name__ == "__main__":
    main()
