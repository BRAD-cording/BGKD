#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W1 冒烟测试 —— 数据管线 / 双轨评估 / 自家基线 / 训练入口
=============================================================
对应执行包 W1 检查点「指标模块单测通过」:
  1. npz_hv       官方格式 npz 读取 + 6 通道掩膜合并 + HV 图几何正确性
  2. reinhard     折内拟合 → 逐 patch 迁移 (LAB 均值对齐参考统计)
  3. decode_met   解码复原两实例; 完美预测下 PQ/AJI/F1/BF1/带内Dice=1;
                  位移预测下指标正确退化 (双轨灵敏度)
  4. dataset      Dataset 端到端: 形状/取值域/增广确定性
  5. model_loss   LightUNet 三头 + 特征抽头形状; 复合损失有限 + 反传有限;
                  SobelGate/RAB 变体前向正常
  6. train_e2e    train.py 子进程 1 epoch 冒烟 (合成 3 折): checkpoint/日志/指标落盘

运行: python scripts/q2pkg/smoke_test_w1.py   期望 6 行 OK + ALL W1 TESTS PASSED
"""
import json
import pathlib
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from metrics_nuclei import decode_instances, instance_metrics_row  # noqa: E402
from models import BoundaryCompositeLoss, LightUNet                # noqa: E402
from pannuke_data import (PANNUKE_CHANNELS, PanNukeDataset,        # noqa: E402
                          apply_reinhard, compute_hv_map, fit_reinhard,
                          load_fold_npz, make_synthetic_fold, merge_masks)


# ---------------------------------------------------------------- 1
def test_npz_hv():
    masks = np.zeros((1, 8, 8, 6), np.uint16)
    masks[0, 0:4, 0:4, 0] = 1                    # ch0 实例1
    masks[0, 4:8, 0:4, 1] = 1                    # ch1 实例1
    masks[0, 0:4, 4:8, 1] = 2                    # ch1 实例2
    inst, tmap = merge_masks(masks[0])
    assert len(np.unique(inst[inst > 0])) == 3, "实例 ID 应跨通道唯一"
    assert (tmap[0:4, 0:4] == 1).all() and (tmap[4:8, 0:4] == 2).all()
    assert (tmap[0:4, 4:8] == 2).all() and (tmap[inst == 0] == 0).all()

    sq = np.zeros((64, 64), np.int64)
    sq[15:48, 15:48] = 1                       # 奇数边长 33px → 质心恰在像素 (31,31)
    hv = compute_hv_map(sq)
    assert hv.shape == (2, 64, 64) and np.abs(hv).max() <= 1.0
    assert np.abs(hv[:, 31, 31]).max() < 1e-6, "实例中心 HV 应≈0"
    assert hv[0, 31, 46] > 0.9 and hv[0, 31, 15] < -0.9, "水平分量方向错误"
    assert hv[1, 46, 31] > 0.9 and hv[1, 15, 31] < -0.9, "垂直分量方向错误"

    with tempfile.TemporaryDirectory() as td:
        p = make_synthetic_fold(pathlib.Path(td) / "fold1.npz", n=6, seed=0)
        d = load_fold_npz(p)
        assert set(d["images"].shape[1:]) == {256, 256} or d["images"].shape[1] == 256
        assert d["masks"].shape[-1] == 6
        assert len(PANNUKE_CHANNELS) == 6


# ---------------------------------------------------------------- 2
def test_reinhard():
    rng = np.random.default_rng(0)
    imgs = np.clip(rng.normal(140, 30, (16, 64, 64, 3)), 0, 255).astype(np.uint8)
    stats = fit_reinhard(imgs, max_patches=8, seed=0)
    out = apply_reinhard(imgs[0], stats)
    assert out.shape == imgs[0].shape and out.dtype == np.uint8
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from pannuke_data import _rgb_to_lab
    lab_after = _rgb_to_lab(out.astype(np.float32) / 255.0).reshape(-1, 3).mean(0)
    ref = np.asarray(stats["mean"], np.float32)
    assert np.abs(lab_after - ref).max() < 2.0, \
        f"迁移后 LAB 均值应≈参考统计: {lab_after} vs {ref}"
    blank = np.full((64, 64, 3), 255, np.uint8)
    assert apply_reinhard(blank, stats).shape == blank.shape   # 退化 guard 不崩


# ---------------------------------------------------------------- 3
def test_decode_met():
    gt = np.zeros((256, 256), np.int64)
    gt[40:80, 40:80] = 1
    gt[120:160, 120:160] = 2
    tmap = np.zeros((256, 256), np.int64)
    tmap[gt == 1] = 1
    tmap[gt == 2] = 2

    np_prob = (gt > 0).astype(np.float32)
    hv = compute_hv_map(gt)
    pred = decode_instances(np_prob, hv, np_thresh=0.5)
    assert len(np.unique(pred[pred > 0])) == 2, f"解码应复原 2 实例, 得到 {np.unique(pred)}"
    row = instance_metrics_row(gt, tmap, pred, tmap, patch_id="p0", group=0)
    for k in ("pq", "mpq_plus", "aji", "f1_det", "bf1_2px", "dice5px"):
        assert row[k] == 1.0, f"完美预测 {k}={row[k]} 应为 1"
    assert row["hd95"] < 1e-6 and row["assd"] < 1e-6

    pred2 = gt.copy()                                        # B 实例向下移 3px
    pred2[120:160, 120:160] = 0
    pred2[123:163, 120:160] = 2
    row2 = instance_metrics_row(gt, tmap, pred2, tmap, patch_id="p1", group=0)
    assert 0.85 < row2["pq"] < 1.0, f"位移后 pq={row2['pq']}"
    assert row2["hd95"] > 0 and row2["bf1_2px"] < 1.0, "边界轨应反映位移"
    empty = decode_instances(np.zeros((256, 256), np.float32), hv)
    assert empty.max() == 0


# ---------------------------------------------------------------- 4
def test_dataset():
    with tempfile.TemporaryDirectory() as td:
        for f in (1, 2, 3):
            make_synthetic_fold(pathlib.Path(td) / f"fold{f}.npz", n=8, seed=f)
        ds = PanNukeDataset(td, [1], stain="reinhard", train=False, seed=0)
        it = ds[0]
        assert it["img"].shape == (3, 256, 256) and it["img"].max() <= 1.0
        assert it["np"].shape == (1, 256, 256) and it["hv"].shape == (2, 256, 256)
        assert it["inst"].shape == (256, 256) and it["type"].shape == (256, 256)
        assert set(np.unique(it["type"].numpy())) <= set(range(7))
        assert it["group"] in (0, 1)
        dtr = PanNukeDataset(td, [1], stain="reinhard", train=True, seed=0)
        a, b = dtr[3], dtr[3]                                # 同 epoch 两次取样一致
        assert torch.equal(a["inst"], b["inst"]) and torch.equal(a["img"], b["img"])
        dtr.set_epoch(1)
        c = dtr[3]
        assert (not torch.equal(a["inst"], c["inst"])) or (not torch.equal(a["img"], c["img"]))


# ---------------------------------------------------------------- 5
def test_model_loss():
    torch.manual_seed(0)
    B = 2
    inst = torch.zeros(B, 256, 256, dtype=torch.int64)
    inst[:, 40:80, 40:80] = 1
    inst[:, 120:160, 120:160] = 2
    tmap = (inst > 0).long()
    batch = {"np": (inst > 0).float().unsqueeze(1),
             "hv": torch.randn(B, 2, 256, 256) * 0.1,
             "inst": inst, "type": tmap}
    for flags in (dict(), dict(use_sobel_gate=True), dict(use_rab=True),
                  dict(use_sobel_gate=True, use_rab=True)):
        model = LightUNet(n_type=7, base=16, **flags)
        out = model(torch.rand(B, 3, 256, 256))
        assert out["np_logits"].shape == (B, 1, 256, 256)
        assert out["hv"].shape == (B, 2, 256, 256)
        assert out["type_logits"].shape == (B, 7, 256, 256)
        assert out["feats"]["s2"].shape == (B, 64, 64, 64)
        assert out["feats"]["s3"].shape == (B, 128, 32, 32)
        assert out["feats"]["s4"].shape == (B, 256, 16, 16)
        if flags.get("use_sobel_gate"):
            assert out["gate_w"] is not None
    model = LightUNet(n_type=7, base=16, use_sobel_gate=True, use_rab=True)
    crit = BoundaryCompositeLoss(use_boundary=True)
    loss, parts = crit(model(torch.rand(B, 3, 256, 256)), batch)
    assert torch.isfinite(loss).item(), "loss 非有限"
    assert all(torch.isfinite(v).item() for v in parts.values())
    assert parts["l_bnd"].item() > 0, "边界项应被启用"
    loss.backward()
    assert torch.isfinite(model.np_head.weight.grad).all().item()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 8e6, f"轻量基线参数量异常: {n_params/1e6:.1f}M"


# ---------------------------------------------------------------- 6
def test_train_e2e():
    with tempfile.TemporaryDirectory() as td:
        for f in (1, 2, 3):
            make_synthetic_fold(pathlib.Path(td) / f"fold{f}.npz", n=8, seed=f)
        run = pathlib.Path(td) / "run"
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve().parent / "train.py"),
               "--data-root", td, "--fold", "1", "--epochs", "1", "--batch", "4",
               "--limit", "8", "--workers", "0", "--val-every", "1",
               "--base", "16", "--boundary-loss", "--seed", "0", "--out", str(run)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, f"train.py 失败:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        assert (run / "checkpoint.pt").exists()
        assert (run / "stain_stats.json").exists()
        assert (run / "val_metrics.csv").exists()
        log = list(csv_reader(run / "epoch_log.csv"))
        assert len(log) == 1 and float(log[0]["sec"]) > 0, f"epoch 日志行数异常: {len(log)}"
        summary = json.loads((run / "summary.json").read_text())
        assert 0.0 <= summary.get("est_hours_per_fold_50ep", -1)


def csv_reader(path):
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


_TESTS = [test_npz_hv, test_reinhard, test_decode_met, test_dataset,
          test_model_loss, test_train_e2e]


def main():
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"[w1] {fn.__name__[5:]:<12s} OK")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"[w1] {fn.__name__[5:]:<12s} FAIL")
            traceback.print_exc()
    if failed:
        print(f"{failed}/{len(_TESTS)} test(s) FAILED")
        sys.exit(1)
    print("ALL W1 TESTS PASSED")


if __name__ == "__main__":
    main()
