# BGKD: Boundary-Guided Knowledge Distillation

Code and per-patch metrics for the paper *"Boundary-guided knowledge distillation
for accurate and compact nuclei instance segmentation on consumer GPUs"*.

A 7.9M-parameter U-Net detector (BG-UNet) makes boundary accuracy the
first-class objective at three levels: multi-scale Sobel gating on skip
connections (SobelGate-MS), an uncertainty-band reverse attention (RAB), and
a composite boundary-band loss with per-instance erosion semantics.
Boundary-guided knowledge distillation (BGKD) transfers the released NuLite
teacher's logit and decoder-stage feature knowledge into the student,
concentrated entirely on the boundary band. On the official three-fold split
of PanNuke the distilled model reaches 24.04% PQ / 12.95% mPQ+, surpassing
re-implemented HoVer-Net, CellViT-256, and CellViT-256-SAM while training,
distillation, and inference fit on a single 8 GB consumer GPU (RTX 4060
Laptop); the released model further runs at 99 patches/s (116 mJ per patch)
on an embedded Jetson Orin Nano Super.

## Repository structure

```
q2pkg/                 all training / evaluation / analysis code
  train.py             training entry (architecture + loss flags)
  boundary_distill.py  BGKD distillation training
  models.py, modules.py        BG-UNet, SobelGate-MS, RAB, loss terms
  pannuke_data.py, convert_pannuke_v1.py, convert_consep.py   data pipeline
  metrics_nuclei.py, infer_consep.py, finetune_consep.py      evaluation
  hovernet_adapter.py, cellvit_adapter.py, cellvit_sam_adapter.py,
  ddbunet_adapter.py, nulite_adapter.py                        baselines
  export_onnx.py, jetson_*.py, jetson_*.sh                     ONNX/TensorRT edge deployment
  run_stats.py, stats_tests.py  paired statistics (Wilcoxon-Holm, bootstrap CIs)
  make_tables.py, make_figures.py, make_qualitative.py         regenerate every table/figure
runs/                  per-run configuration (JSON) and per-patch metrics (CSV)
                       for all reported experiments (ablations, seeds,
                       lambda sweeps, baselines, CoNSeP transfer, Jetson bench)
```

## Requirements

- Python >= 3.10 with PyTorch (CUDA). Development used Python 3.13 on a
  single 8 GB RTX 4060 Laptop; no data-center GPU is required.
- numpy, scipy, pandas, opencv-python, pillow, tqdm (see imports).

## Data

- **PanNuke v1** — download from the Warwick TIA Lab release, then convert:
  `python q2pkg/convert_pannuke_v1.py --help`
- **CoNSeP** — download from the authors' release, then convert:
  `python q2pkg/convert_consep.py --help`

## Usage

Train the full method (SobelGate-MS + RAB + boundary loss) on fold 1:

```
python q2pkg/train.py --data-root <pannuke-npz-dir> --model own_baseline \
  --fold 1 --sobel-gate --rab --boundary-loss --out runs/full_f1
```

Add distillation (BGKD, requires the released NuLite checkpoint):

```
python q2pkg/boundary_distill.py --help
```

All entry scripts expose `--help`. Every table and figure in the paper can be
regenerated from the released `runs/` metrics with:

```
python q2pkg/make_tables.py
python q2pkg/make_figures.py
python q2pkg/run_stats.py
```

## License

Released under the MIT License (see [LICENSE](LICENSE)). Baseline
architectures adapted in `q2pkg/*_adapter.py` follow the licenses of their
original repositories (HoVer-Net, CellViT, NuLite, DDBU-Net).
