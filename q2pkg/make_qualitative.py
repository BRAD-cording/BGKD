# -*- coding: utf-8 -*-
"""P1-3 定性图: H&E / GT 叠加 / 预测叠加 / Sobel 门控热图
=====================================================
数据: full_f1 (seed0 fold1) 的 val_metrics.csv 逐 patch PQ → 选 2 个最优 + 2 个失败案例;
模型: own_full 架构 (SG@s2 + RAB), 装载 checkpoint.pt (旧键名 sobel_gate.* → sobel_gates.s2.*);
推理: C:\\data\\pannuke fold1, Reinhard 归一化 (run 内 stain_stats.json, 与验证一致)。
输出: paper/figures/fig_qualitative.pdf + stdout 选片信息 (供图注引用)。
前提: 独占 GPU。
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models import build_model
from metrics_nuclei import _instance_class, decode_instances
from pannuke_data import apply_reinhard, load_fold_npz, merge_masks

RUN = Path(r"C:\runs\full_f1")
DATA = Path(r"C:\data\pannuke\fold1.npz")
OUT = Path(r"C:\Users\HP\OneDrive\Desktop\Paper\paper\figures\fig_qualitative.pdf")
N_GT_MIN, N_PRED_MIN = 15, 5


def pick_patches():
    rows = list(csv.DictReader(open(RUN / "val_metrics.csv")))
    rows = [r for r in rows if int(r["n_gt"]) >= N_GT_MIN and int(r["n_pred"]) >= N_PRED_MIN]
    rows.sort(key=lambda r: float(r["pq"]))
    worst = rows[:2]
    best = rows[-2:][::-1]
    return best + worst


def overlay_instances(inst, img, nolabel_mask=None):
    """灰度 H&E + 逐实例黄金比例色相染色; no-label 忽略区域以浅灰标出 (评估排除)。"""
    out = (img * 0.55 + 0.45).clip(0, 1).copy()
    if nolabel_mask is not None:
        out[nolabel_mask] = 0.82 * out[nolabel_mask] + 0.18
    ids = np.unique(inst)
    ids = ids[ids > 0]
    rng = np.random.default_rng(0)
    rng.shuffle(ids)                                   # 打乱相邻 ID 的颜色相关性
    for i, pid in enumerate(ids):
        hue = (i * 0.61803) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
        m = inst == pid
        out[m] = 0.35 * out[m] + 0.65 * np.array([r, g, b])
    return out


def filter_eval_instances(inst, type_map):
    """复刻评估口径: 逐实例多数类 >5 (no-label) 的实例剔除, 返回显示用 inst。"""
    disp = inst.copy()
    for pid in np.unique(inst[inst > 0]):
        if _instance_class(inst, type_map, int(pid)) > 5:
            disp[inst == pid] = 0
    return disp


def main():
    assert torch.cuda.is_available()
    used = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]
    assert used < 1.5 * 2**30, f"GPU 非独占 ({used/2**30:.2f} GB)"

    picks = pick_patches()
    z = load_fold_npz(DATA)
    images, masks, types = z["images"], z["masks"], z["types"]
    tissue_names = sorted(set(types.tolist()))

    model = build_model("own_baseline", use_sobel_gate=True, use_rab=True).cuda().eval()
    sd = torch.load(RUN / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
    sd = {k.replace("sobel_gate.", "sobel_gates.s2."): v for k, v in sd.items()}
    model.load_state_dict(sd)

    stain_stats = __import__("json").loads((RUN / "stain_stats.json").read_text())
    panels = []                                        # (tissue, pq, img, gt_ov, pd_ov, gate)
    with torch.inference_mode():
        for r in picks:
            idx = int(r["patch_id"].split("_")[1])
            img_u8 = apply_reinhard(images[idx], stain_stats)
            img = torch.from_numpy(img_u8).permute(2, 0, 1).float().cuda() / 255.0
            out = model(img[None])
            np_prob = torch.sigmoid(out["np_logits"])[0, 0].cpu().numpy()
            hv = out["hv"][0].cpu().numpy()
            pred = decode_instances(np_prob, hv)
            pred_type = out["type_logits"][0].argmax(0).cpu().numpy()
            gt_inst, gt_type = merge_masks(masks[idx])
            nolabel = gt_type == 6                      # 忽略区域 (评估双侧排除)
            gt_disp = filter_eval_instances(gt_inst.astype(np.int32), gt_type)
            pd_disp = filter_eval_instances(pred, pred_type)
            gate = F.interpolate(out["gate_w"], size=(256, 256), mode="bilinear",
                                 align_corners=False)[0, 0].cpu().numpy()
            img_f = img_u8.astype(np.float32) / 255.0
            panels.append((types[idx], float(r["pq"]) * 100, img_f,
                           overlay_instances(gt_disp, img_f, nolabel),
                           overlay_instances(pd_disp, img_f, nolabel), gate))
            del out
            torch.cuda.empty_cache()

    tags = ["best", "best", "worst", "worst"]
    fig, axes = plt.subplots(4, 4, figsize=(9.2, 9.6))
    for i, (tissue, pq, im, gto, pdo, gate) in enumerate(panels):
        tn = tissue_names[int(np.where(np.array(tissue_names) == tissue)[0][0])] if False else tissue
        for j, (panel, cmap) in enumerate([(im, None), (gto, None), (pdo, None), (gate, "magma")]):
            ax = axes[i, j]
            ax.imshow(panel, cmap=cmap, vmin=0 if j == 3 else None, vmax=1 if j == 3 else None)
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(["H&E", "Ground truth", "Prediction",
                              "Sobel gate $W$"][j], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{tags[i]}  {tn.replace('_', ' ')}\nPQ {pq:.1f}", fontsize=8)
    fig.subplots_adjust(wspace=0.03, hspace=0.10, left=0.09, right=0.99, top=0.96, bottom=0.01)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".png"), dpi=140)      # 目检用
    print(f"已保存 {OUT}")
    for (tissue, pq, *_), tag in zip(panels, tags):
        print(f"[{tag}] {tissue}  PQ={pq:.1f}")


if __name__ == "__main__":
    main()
