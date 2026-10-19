# -*- coding: utf-8 -*-
"""双轨评估协议实现 (T1, W1 检查点) —— 对应执行包 §2.2
=========================================================
实例轨: marker-controlled watershed 解码 → PQ / mPQ+ / AJI / F1@IoU0.5
边界轨: Hungarian 匹配(TP) 实例轮廓上 → HD95 / ASSD / BF1@2px; 带内 Dice k∈{2,5,10}

实现约束:
  * 仅依赖 numpy/scipy (环境无 skimage/cv2); 解码用 scipy.ndimage.watershed_ift。
  * mPQ+ 在 5 个 PanNuke 核类别上计算; no-label(通道6) 实例不入 PQ/AJI/F1。
  * 解码阈值与后处理仅允许在 fold1-val 上调优 (协议 R6/§2.2)。
  * W11 如需与 CellViT 官方 eval 对齐, 仅替换本文件解码入口, 指标函数复用。
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

__all__ = ["decode_instances", "instance_metrics_row", "evaluate_pair",
           "N_EVAL_CLASSES"]

N_EVAL_CLASSES = 5          # neoplastic..epithelial (no-label 不评估)


# ---------------------------------------------------------------- 解码
def _markers_from_hv(np_bin, energy, min_marker_area=3, energy_thresh=0.3):
    """核内低能量(距边界远)连通域作为分水岭标记; 每实例一个标记。"""
    low = np_bin & (energy < energy_thresh)
    lab, n = ndi.label(low)
    if n == 0:
        return lab, 0
    areas = np.bincount(lab.ravel())
    remove = np.zeros(n + 1, bool)
    remove[np.where(areas < min_marker_area)[0]] = True
    remove[0] = False
    if remove.any():
        lab[np.isin(lab, np.where(remove)[0])] = 0
        lab, n = ndi.label(lab > 0)
    return lab, n


def decode_instances(np_prob, hv, np_thresh=0.5, min_area=5):
    """np_prob (H,W) 概率, hv (2,H,W) 回归图 → 实例标签图 (H,W) int32 (0=背景)。

    流程: 前景阈值 → HV 能量面(中心≈0/边界≈1) → 低能量标记 →
    watershed_ift 洪填(限前景) → 小实例滤除。
    """
    np_bin = np_prob > np_thresh
    if not np_bin.any():
        return np.zeros(np_prob.shape, np.int32)
    h0 = ndi.gaussian_filter(hv[0].astype(np.float32), 1.0)
    h1 = ndi.gaussian_filter(hv[1].astype(np.float32), 1.0)
    energy = np.sqrt(h0 * h0 + h1 * h1)
    markers, n_mark = _markers_from_hv(np_bin, energy)
    if n_mark == 0:                                  # 退化: 整片前景一个标记
        markers, _ = ndi.label(np_bin)
        markers = markers.astype(np.int32)
    else:
        mark_in = markers.astype(np.int32)
        mark_in[~np_bin] = -1                        # 前景外作为背景标记, 防外溢
        land = np.clip(energy * 255.0, 0, 255).astype(np.uint8)
        markers = ndi.watershed_ift(land, mark_in).astype(np.int32)
        markers[~np_bin] = 0
    ids, counts = np.unique(markers[markers > 0], return_counts=True)
    keep = ids[counts >= min_area]
    out = np.zeros_like(markers)
    for new, old in enumerate(keep, 1):
        out[markers == old] = new
    return out


# ---------------------------------------------------------------- 轮廓与距离
def _boundary(mask):
    return mask & ~ndi.binary_erosion(mask, np.ones((3, 3), bool))


def _directed_dists(a_mask, b_mask):
    """返回 (a 边界到 b 边界的逐像素距离数组, b 边界到 a 边界的)。"""
    ba, bb = _boundary(a_mask), _boundary(b_mask)
    da = ndi.distance_transform_edt(~ba)[bb]
    db = ndi.distance_transform_edt(~bb)[ba]
    return da, db


# ---------------------------------------------------------------- 单 patch 指标
def _contingency_iou(gt, pred):
    g_ids = np.unique(gt); g_ids = g_ids[g_ids > 0]
    p_ids = np.unique(pred); p_ids = p_ids[p_ids > 0]
    g_idx = {v: i for i, v in enumerate(g_ids)}
    p_idx = {v: i for i, v in enumerate(p_ids)}
    G, P = len(g_ids), len(p_ids)
    inter = np.zeros((G, P), np.int64)
    if G and P:
        mul = int(pred.max()) + 1
        pair = (gt.astype(np.int64) * mul + pred).ravel()
        pair_id, cnt = np.unique(pair, return_counts=True)
        gi = np.array([g_idx.get(int(x // mul), -1) for x in pair_id])
        pi = np.array([p_idx.get(int(x % mul), -1) for x in pair_id])
        ok = (gi >= 0) & (pi >= 0)
        inter[gi[ok], pi[ok]] = cnt[ok]
    a_gt = np.array([(gt == v).sum() for v in g_ids], np.int64) if G else np.zeros(0, np.int64)
    a_pd = np.array([(pred == v).sum() for v in p_ids], np.int64) if P else np.zeros(0, np.int64)
    union = a_gt[:, None] + a_pd[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return g_ids, p_ids, iou, inter, a_gt, a_pd


def _hungarian_tp(iou, thr=0.5):
    """返回 [(g_i, p_i, iou)] 满足 IoU≥thr 的最大权匹配。"""
    if iou.size == 0:
        return []
    gi, pi = linear_sum_assignment(-iou)
    return [(int(g), int(p), float(iou[g, p])) for g, p in zip(gi, pi) if iou[g, p] >= thr]


def _instance_class(inst, type_map, inst_id):
    m = inst == inst_id
    vals = type_map[m]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0
    return int(np.bincount(vals).argmax())          # 1..5 五类 / 6=no-label


def instance_metrics_row(gt_inst, gt_type, pred_inst, pred_type,
                         patch_id="", group=-1, band_ks=(2, 5, 10),
                         np_iou_thr=0.5):
    """双轨指标, 单 patch 一行 dict。指标含义与方向见 §2.2 (HD95/ASSD 越低越好)。

    * gt_type/pred_type: (H,W) 0=背景,1..5=类别,6=no-label
    * PQ: 全类 micro 聚合; mpq_plus: 5 类宏平均 (GT∪Pred 中出现的类)
    * 边界轨: Hungarian TP 对的轮廓上计算, patch 内取均值
    """
    row = {"patch_id": patch_id, "group": group}
    g_ids, p_ids, iou, inter, a_gt, a_pd = _contingency_iou(gt_inst, pred_inst)

    gt_class = {int(v): _instance_class(gt_inst, gt_type, v) for v in g_ids}
    pd_class = {int(v): _instance_class(pred_inst, pred_type, v) for v in p_ids}

    keep_g = [i for i, v in enumerate(g_ids) if gt_class[int(v)] <= N_EVAL_CLASSES]
    keep_p = [i for i, v in enumerate(p_ids) if pd_class[int(v)] <= N_EVAL_CLASSES]
    if keep_g and keep_p:
        iou_kept = iou[np.ix_(keep_g, keep_p)]
        inter_kept = inter[np.ix_(keep_g, keep_p)]
    else:
        iou_kept = np.zeros((len(keep_g), len(keep_p)))
        inter_kept = np.zeros((len(keep_g), len(keep_p)), np.int64)
    a_gt_k, a_pd_k = a_gt[keep_g], a_pd[keep_p]
    pairs = _hungarian_tp(iou_kept, np_iou_thr)
    G, P = len(keep_g), len(keep_p)
    TP, FP, FN = len(pairs), P - len(pairs), G - len(pairs)
    row["n_gt"], row["n_pred"], row["tp"] = G, P, TP

    if G + P == 0:                                   # 双空 patch: 不参与统计
        return {**row, **{k: np.nan for k in (
            "pq", "mpq_plus", "aji", "f1_det", "hd95", "assd", "bf1_2px",
            *[f"dice{int(k)}px" for k in band_ks])}}

    sum_iou = sum(p[2] for p in pairs)
    row["pq"] = sum_iou / (TP + 0.5 * FP + 0.5 * FN) if (TP + FP + FN) else np.nan
    row["f1_det"] = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else np.nan

    # --- mPQ+ (5 类宏平均) ---
    g_ids_k = [g_ids[i] for i in keep_g]
    p_ids_k = [p_ids[i] for i in keep_p]
    pqs = []
    for c in range(1, N_EVAL_CLASSES + 1):
        gi = [i for i, v in enumerate(g_ids_k) if gt_class[int(v)] == c]
        pi = [j for j, v in enumerate(p_ids_k) if pd_class[int(v)] == c]
        sub = iou_kept[np.ix_(gi, pi)] if gi and pi else np.zeros((0, 0))
        pr = _hungarian_tp(sub, np_iou_thr)
        tp, fp, fn = len(pr), len(pi) - len(pr), len(gi) - len(pr)
        if tp + fp + fn == 0:
            continue                                 # 该类双侧均无 → 不进宏平均
        pqs.append(sum(p[2] for p in pr) / (tp + 0.5 * fp + 0.5 * fn))
    row["mpq_plus"] = float(np.mean(pqs)) if pqs else np.nan

    # --- AJI (官方 per-pred 贪心语义: 每个 pred 取 IoU 最大且未被占用的 gt) ---
    if G or P:
        used_gt, inter_acc, union_acc = set(), 0, 0
        for p in range(P):
            if not G:
                break
            g_best = int(np.argmax(iou_kept[:, p]))
            v = float(iou_kept[g_best, p])
            if v <= 0 or g_best in used_gt:
                continue
            used_gt.add(g_best)
            i_ab = int(inter_kept[g_best, p])
            u_ab = int(a_gt_k[g_best]) + int(a_pd_k[p]) - i_ab
            inter_acc += i_ab
            union_acc += u_ab
        union_acc += sum(int(a_gt_k[g]) for g in range(G) if g not in used_gt)
        row["aji"] = inter_acc / union_acc if union_acc else np.nan
    else:
        row["aji"] = np.nan

    # --- 边界轨 (TP 对) ---
    hd, asd, bf = [], [], []
    for g, p, _ in pairs:
        gm = gt_inst == g_ids_k[g]
        pm = pred_inst == p_ids_k[p]
        d_gp, d_pg = _directed_dists(gm, pm)
        both = np.concatenate([d_gp, d_pg])
        hd.append(float(np.percentile(both, 95)))
        asd.append(float(both.mean()))
        prec = float((d_gp <= 2).mean()) if d_gp.size else np.nan
        rec = float((d_pg <= 2).mean()) if d_pg.size else np.nan
        bf.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else np.nan)
    row["hd95"] = float(np.mean(hd)) if hd else np.nan
    row["assd"] = float(np.mean(asd)) if asd else np.nan
    row["bf1_2px"] = float(np.mean(bf)) if bf else np.nan

    # --- 带内 Dice (matched 实例掩膜 ∪, 轮廓带 k px) ---
    if pairs:
        gm_all = np.isin(gt_inst, [g_ids_k[g] for g, _, _ in pairs])
        pm_all = np.isin(pred_inst, [p_ids_k[p] for _, p, _ in pairs])
        cont = _boundary(gm_all) | _boundary(pm_all)
        dist = ndi.distance_transform_edt(~cont)
        gt_fg, pd_fg = gm_all, pm_all
        for k in band_ks:
            band = dist <= k
            denom = (gt_fg & band).sum() + (pd_fg & band).sum()
            row[f"dice{int(k)}px"] = (
                2.0 * (gt_fg & pd_fg & band).sum() / denom if denom else np.nan)
    else:
        for k in band_ks:
            row[f"dice{int(k)}px"] = np.nan
    return row


def evaluate_pair(np_prob, hv, gt_inst, gt_type, pred_type_map=None,
                  np_thresh=0.5, **kw):
    """便捷入口: 模型输出 (np_prob, hv) → 解码 → 指标行。
    pred_type_map 可为 None (用 gt_type 代替 —— 仅适用于 type 无关的冒烟检查)。"""
    pred = decode_instances(np_prob, hv, np_thresh=np_thresh)
    if pred_type_map is None:
        pred_type_map = gt_type
    return pred, instance_metrics_row(gt_inst, gt_type, pred, pred_type_map, **kw)
