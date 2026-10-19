# -*- coding: utf-8 -*-
"""数据集事实统计 → paper/dataset_facts.json
* 各折 patch 数 (由 val_metrics.csv 行数)
* 5px 逐实例腐蚀边界带面积占比 (BGKD motivation 的 X% 实测)
"""
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
from boundary_distill import boundary_band  # noqa: E402
import torch  # noqa: E402

RUNS = Path(r"C:\runs")
OUT = Path(r"c:\Users\HP\OneDrive\Desktop\Paper\paper\dataset_facts.json")

fold_counts = {}
for f in (1, 2, 3):
    n = sum(1 for _ in open(RUNS / f"full_f{f}" / "val_metrics.csv", encoding="utf-8")) - 1
    fold_counts[f] = n

# ---- 边界带面积占比 (fold1 验证折, 逐实例腐蚀语义, r=5) ----
with np.load(r"C:\data\pannuke\fold1.npz") as z:
    masks = z["masks"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(masks), size=500, replace=False)
    frac, frac_pos = [], []
    for i in idx:
        m = masks[i].astype(np.int64)
        inst = np.zeros(m.shape[:2], np.uint32)
        for c in range(6):
            mm = m[..., c] > 0
            inst[mm] = (m[..., c][mm] + (c + 1) * (1 << 16)).astype(np.uint32)
        lab = torch.from_numpy(inst.astype(np.int64))[None, None]
        band = boundary_band(lab, band_px=5)[0, 0].numpy()
        fg = inst > 0
        frac.append(band.mean())
        if fg.any():
            frac_pos.append(band[fg].mean())
facts = {
    "fold_val_counts": fold_counts,
    "total_patches": int(sum(fold_counts.values())),
    "band_area_frac_mean": float(np.mean(frac)),
    "band_area_frac_within_fg": float(np.mean(frac_pos)),
}
OUT.write_text(json.dumps(facts, indent=2), encoding="utf-8")
print(json.dumps(facts, indent=2))
