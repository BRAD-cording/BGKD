# -*- coding: utf-8 -*-
"""P2-1 模拟审稿的三个数据核验 (CPU-only, 不碰 GPU):
1. §4.6 引用但未入 stats_facts.json 的 AJI/F1/BF1/HD95 统计声明
2. §4.7 组织级声明 (10/19 best, bile-duct 并列, testis/pancreas/cervix 边际, uterus/esophagus/thyroid)
3. §3.4 "5-px per-instance 边界带 ≈27.6% 面积" 覆盖率
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(r"C:\runs")
TBL = Path(r"c:\Users\HP\OneDrive\Desktop\Paper\paper\tables")
PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))

DIR_OVERRIDE = {"own_baseline": {1: "w1_own_f1"},
                "cellvit_sam256": {f: f"cellvit_sam_f{f}" for f in (1, 2, 3)}}

# ---------- 1. stats CSV 声明核验 ----------
print("=" * 60)
print("[1] §4.6 统计声明核验 (来自 stats_full_vs_*.csv)")
checks = [("full_nokd", "bf1_2px", "+0.15pt, p_holm=0.29"),
          ("nulite_s",  "hd95",    "-0.23px, p_holm=0.10")]
for base in ("full_nokd", "hovernet", "cellvit_sam256", "cellvit_t256", "nulite_s", "nulite_t", "ddbunet"):
    df = pd.read_csv(TBL / f"stats_full_vs_{base}.csv")
    sig = df[df.metric.isin(["aji", "f1_det", "bf1_2px"])]
    print(f"\nFull vs {base}: AJI/F1/BF1 显著性: "
          + ", ".join(f"{r.metric}={r.significant}(p={r.p_holm:.3g})" for r in sig.itertuples()))
    r_pq = df[df.metric == "pq"].iloc[0]
    print(f"  PQ p_holm={r_pq.p_holm:.3g}")
for base, metric, claim in checks:
    df = pd.read_csv(TBL / f"stats_full_vs_{base}.csv")
    r = df[df.metric == metric].iloc[0]
    scale = 100 if metric != "hd95" else 1
    print(f"\nclaim {base}/{metric} {claim}: 实际 mean_diff={r.mean_diff*scale:+.2f} p_holm={r.p_holm:.3g}")

# ---------- 2. 组织级声明核验 ----------
print("\n" + "=" * 60)
print("[2] §4.7 组织级声明核验 (逐组织 PQ, 三折 patch 汇总)")
def load_pooled(name):
    dfs = []
    for f in (1, 2, 3):
        d = DIR_OVERRIDE.get(name, {}).get(f, f"{name}_f{f}")
        dfs.append(pd.read_csv(RUNS / d / "val_metrics.csv"))
    return pd.concat(dfs, ignore_index=True)

methods = {"full": "Ours(full)", "hovernet": "HoVer-Net", "cellvit_sam256": "CellViT-SAM",
           "cellvit_t256": "CellViT-256", "nulite_s": "NuLite-S", "own_baseline": "Ours(base)"}
tissue_pq = {}
for m in methods:
    df = load_pooled(m)
    tissue_pq[m] = df.groupby("group")["pq"].mean()

import json as _json
tissues = _json.load(open(r"c:\Users\HP\OneDrive\Desktop\Paper\paper\paper_facts.json"))["tissues"]
n_best, ties = 0, []
for g in sorted(tissue_pq["full"].index):
    vals = {m: tissue_pq[m].get(g, np.nan) * 100 for m in methods}
    best = max(vals.values())
    winners = [methods[m] for m in vals if np.isclose(vals[m], best, atol=1e-9)]
    name = tissues[int(g)] if int(g) < len(tissues) else f"g{g}"
    if "Ours(full)" in winners:
        n_best += 1 if len(winners) == 1 else 0
        if len(winners) > 1:
            ties.append(name)
    line = f"  {name:14s} best={'/'.join(winners):12s}"
    for m in ("hovernet", "cellvit_sam256"):
        line += f" {methods[m]}={vals[m]:.2f}"
    line += f" Ours(full)={vals['full']:.2f}"
    if name in ("Testis", "Pancreatic", "Cervix", "Uterus", "Esophagus", "Thyroid", "Bile-duct"):
        second = sorted(v for mm, v in vals.items() if methods[mm] not in winners)[-1]
        print(line + f"  margin=+{best-second:.2f}")
print(f"Ours(full) 独占 best 的组织数 = {n_best}; 并列 best = {ties}")

# ---------- 3. 边界带覆盖率 ----------
print("\n" + "=" * 60)
print("[3] §3.4 5-px per-instance 边界带覆盖率 (PanNuke 全部 patch)")
import torch
import torch.nn.functional as F

def band_fraction(inst: np.ndarray, r: int = 5) -> float:
    L = torch.from_numpy(inst.astype(np.int64))[None, None]
    fg = (L > 0).float()
    dil = F.max_pool2d(fg, 2 * r + 1, stride=1, padding=r)
    # per-instance erosion: 邻域内 max/min 标签均等于自身标签
    mx = F.max_pool2d(L.float(), 2 * r + 1, stride=1, padding=r)
    mn = -F.max_pool2d(-L.float(), 2 * r + 1, stride=1, padding=r)
    interior = ((mn == L.float()) & (mx == L.float()) & (L > 0)).float()
    band = (dil > 0) & (interior < 0.5)
    return float(band.float().mean())

fracs = []
for f in (1, 2, 3):
    z = np.load(rf"C:\data\pannuke\fold{f}.npz")
    masks = z["masks"]  # (N,256,256,6) uint16
    for i in range(masks.shape[0]):
        inst = masks[i, :, :, 0] + masks[i, :, :, 1] + masks[i, :, :, 2] \
             + masks[i, :, :, 3] + masks[i, :, :, 4]
        fracs.append(band_fraction(inst))
    print(f"  fold{f}: n={len(fracs)} running_mean={np.mean(fracs):.4f}")
print(f"最终 5-px per-instance 边界带平均覆盖率 = {np.mean(fracs)*100:.1f}% (论文写 27.6%)")
