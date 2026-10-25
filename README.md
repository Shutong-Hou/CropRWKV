# CropRWKV — linear-complexity (RWKV) segmentation for crop leaf-lesion phenotyping

Code and source-data records for the manuscript

> *Linear-Complexity (RWKV) Segmentation for Crop Leaf-Lesion Phenotyping:
> A Multi-Seed Benchmark, a Diagnosed Mixed-Precision Training Failure, and
> Practical Guidance*

submitted to *Plants* (MDPI).

---

## What is in this release

```
code/                     model definitions, training and evaluation entry points
  wkv_pure.py             pure-PyTorch WKV scan (the operator used everywhere)
  modules.py              VDAS / FEEM / MPSA
  backbone_a0.py          A0 baseline (bare RWKV-UNet backbone)
  backbone_full.py        FULL (A0 + the three modules)
  models_registry.py      factory for every compared architecture
  swin_unet_model.py      Swin-UNet baseline
  train.py                unified training protocol, identical for all methods
  eval.py                 sliding-window evaluation, per-image metrics
  exp_harness.py          per-experiment snapshot harness
  diag_vdas.py            forward-hook inspection of the VDAS outputs
  diag2_grad_compare.py   gradient comparison with VDAS on / off

experiments/              one script per experiment reported in the paper
  exp_tables_stats.py         main table, bootstrap CIs, Wilcoxon / Holm tests
  exp_stratified_failure.py   size-stratified analysis and rank correlations
  exp_degradation_boundary.py field degradations and boundary metrics
  exp_spectral_grad.py        Jacobian spectral norms, gradient decomposition
  exp_grad_trajectory.py      per-epoch gradient trajectories, fp32 / fp16
  exp_wkv_denominator.py      measured minima of the WKV denominator
  exp_wkv_semantics.py        operator-level audit of the public CUDA kernel
  exp_lat_flops_full.py       steady-clock latency and FLOPs
  exp_qualitative.py          prediction masks for the qualitative figure
  exp_plantseg_prepare.py     second dataset: download, rasterise, split, cache
  efficiency.py               parameters, FLOPs, memory, resolution sweep

third_party/RWKV-UNet/    public RWKV-UNet implementation (see its LICENSE)
scripts/                  table and figure generators for the manuscript
tables/                   source-data CSV for every manuscript table
figures/source_data/      source-data CSV for every quantitative figure
```

Each file in `tables/` and `figures/source_data/` holds the exact values printed
in the corresponding manuscript table or plotted in the corresponding figure, in
machine-readable form. They are the frozen record of the runs reported in the
paper.

## Data layout expected by the code

The scripts resolve every path relative to the repository root. To re-run
training or evaluation, place the datasets as

```
data/kaggle_leafseg/extracted/data/data/   Set A images and masks (Kaggle Leaf Disease Segmentation, CC0-1.0)
data/kaggle_leafseg/cache/                 cache built on first use (long side 1024 px)
data/plantseg/raw/data/                    Set B (PlantSeg test release, CC BY-NC-ND 4.0)
results/                                   one directory per experiment identifier
  EX-020/splits/                           the 468/60/60 stratified split used for Set A
```

Set A is the public Kaggle *Leaf Disease Segmentation Dataset* (CC0-1.0). Set B is
the public PlantSeg test-set release (`Voxel51/PlantSeg-Test`, CC BY-NC-ND 4.0),
used for research only and **not redistributed here**; `experiments/exp_plantseg_prepare.py`
downloads it, rasterises the instance masks into binary lesion masks, applies the
75/15/10 stratified split used in the paper and builds the cache.

## Running the code

```bash
pip install -r requirements.txt

# quick check of the WKV scan against its explicit O(T^2) definition
python code/wkv_pure.py

# train one method under the unified protocol
python code/train.py --ex EX-001 --model full --seed 42 \
    --epochs 150 --batch_size 8 --img_size 512 --lr 1e-4 \
    --loss focal_tversky --lesion_bias 0.9

# evaluate a checkpoint on the test split
python code/eval.py --ex EX-002 --model full \
    --ckpt results/EX-001/checkpoints/best.pt --img_size 512
```

Every run writes a snapshot directory `results/<EX-ID>/` containing `meta.yaml`,
`command.txt`, `env_snapshot.txt`, `manifest.sha256`, the per-epoch `metrics.csv`
and the checkpoints, so that a reported number can always be traced back to the
run that produced it.

## Reproducing the tables and figures

`scripts/gen_tables.py` and `scripts/gen_figures.py` regenerate the manuscript
tables and figures. They read the per-experiment snapshot tree under `results/`,
which is too large to ship in this repository; the frozen source-data records in
`tables/` and `figures/source_data/` are provided instead, so that every value in
the paper can be checked without re-running any experiment. The full snapshot
tree is available from the corresponding author on request.

## Third-party code

`third_party/RWKV-UNet/` is the public RWKV-UNet implementation of
Jiang et al. (<https://arxiv.org/abs/2501.08458>), MIT licensed; the original
LICENSE file is shipped alongside. The sources are used as published -- the only
additions are two empty `__init__.py` files so that the two packages import
cleanly -- and its `RUN_CUDA` hook is redirected to the pure-PyTorch scan in
`code/wkv_pure.py`, because the public CUDA kernel implements a different
operator (exponential key weights) and is therefore not used for any result in
the paper.

## License

The code in `code/`, `experiments/` and `scripts/` is released under the MIT
License (see `LICENSE`).

`third_party/RWKV-UNet/` is third-party code and remains under its original MIT
License (Copyright (c) 2025 Juntao Jiang); its own `LICENSE` file is included in
that directory.

The source-data records in `tables/`, `figures/source_data/` and
`results/EX-020/splits/` contain only the aggregated values reported in the
paper and the split identifiers used here, and are provided for verification of
the published results. The licenses of the original datasets continue to apply
to the underlying data (Set A: CC0-1.0; Set B: CC BY-NC-ND 4.0, not
redistributed here).
