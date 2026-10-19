#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q2pkg 冒烟测试（2026-09-27 升级版）
====================================
覆盖执行包 §3.1–§3.4 四个可执行模块, 对应附录 B 的 6 项断言:

  1. band        —— 逐实例腐蚀 B = D_r(M) \\ E_r^(i)(M): 实例间接触界面像素必须在
                    带内（回归点: union 掩膜腐蚀会漏掉界面）, 深部内部与带互斥,
                    外侧环在带内, 取值 ⊆ {0,1}; 传入 union 0/1 掩膜时优雅退化为
                    普通腐蚀语义（界面像素=0）
  2. bgkd        —— batch=2, teacher 特征/logit 与 student 空间网格和通道数双重
                    不对齐（回归点: _match_spatial 空间对齐 + 1×1 adapter 通道
                    对齐）, loss/l_fe/l_kd 有限, backward 后 student 梯度有限、
                    adapter 收到非零梯度
  3. sobel_gate  —— batch=4 深度固定 Sobel 卷积（回归点: 旧实现 (1,B·C,H,W)+
                    groups=C 仅 batch=1 合法）, 门控 b=+2 初始化 → W=σ(2)≈0.8808
                    恒等缩放, 梯度有限
  4. rab         —— 尾卷积零初始化 → 嵌入初期严格恒等(torch.equal); 不确定带
                    归一化后 amax≈1; 反向传播后 f.grad 严格恒 1
  5. stats       —— 切片级聚合配对 Wilcoxon 主检验: 24 slide × 10 patch 合成数据,
                    验证 n_slide/n_patch/p_slide/p_holm/rank-biserial 效应量
  6. summarize   —— 延迟-能耗汇总: median/P99/能耗主口径 E=P̄·t̄(energy_mean_mJ)
                    与协议守门(warmup≥50, iters≥500)

运行: python scripts/q2pkg/smoke_test.py
期望: 6 行 "[test] ... OK" + "ALL TESTS PASSED"
"""
import math
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from boundary_distill import BGKDLoss, boundary_band              # noqa: E402
from jetson_latency_bench import summarize                        # noqa: E402
from modules import SobelGateMS, UncertaintyBandReverseAttention  # noqa: E402
from stats_tests import paired_compare                            # noqa: E402


# ---------------------------------------------------------------- 1. 逐实例腐蚀
def test_band():
    L = torch.zeros(2, 32, 32)
    L[:, 8:24, 0:16] = 1.0      # 实例 A
    L[:, 8:24, 16:32] = 2.0     # 实例 B(与 A 在 col 15/16 接触)

    band = boundary_band(L, band_px=2).squeeze(1)           # (2,32,32)

    # 回归点: 实例间接触界面必须纳入边界带(batch 两元素都检查)
    assert (band[:, 12, 15] == 1.0).all(), "实例 A 侧界面像素(12,15)不在带内"
    assert (band[:, 12, 16] == 1.0).all(), "实例 B 侧界面像素(12,16)不在带内"
    # 深部内部与带互斥
    assert (band[:, 15, 5] == 0.0).all(), "实例 A 深部内部被误纳入带内"
    assert (band[:, 15, 26] == 0.0).all(), "实例 B 深部内部被误纳入带内"
    # 外侧环(膨胀环)在带内
    assert (band[:, 7, 5] == 1.0).all(), "外侧环像素(7,5)不在带内"
    assert (band[:, 24, 26] == 1.0).all(), "外侧环像素(24,26)不在带内"
    # 取值域
    assert set(band.unique().tolist()) <= {0.0, 1.0}

    # 优雅降级: 传入 union 0/1 掩膜 → 退化为普通腐蚀语义, 界面像素=0
    band_u = boundary_band((L > 0).float(), band_px=2).squeeze(1)
    assert (band_u[:, 12, 15] == 0.0).all(), "union 掩膜退化语义不符"


# ---------------------------------------------------------------- 2. BGKD 组合
def test_bgkd():
    torch.manual_seed(0)
    B, n_cls = 2, 6
    bgkd = BGKDLoss(
        student_channels={"s2": 16, "s3": 32, "s4": 64},
        teacher_channels={"s2": 32, "s3": 32, "s4": 48},
        stages=("s2", "s3", "s4"), band_px=5,
        temperature=4.0, gamma=0.7, lambda_fe=0.3, lambda_kd=0.3)

    s_feats = {
        "s2": torch.randn(B, 16, 64, 64, requires_grad=True),
        "s3": torch.randn(B, 32, 32, 32, requires_grad=True),
        "s4": torch.randn(B, 64, 16, 16, requires_grad=True),
    }
    # teacher 空间网格与通道均不对齐 → 回归点: _match_spatial + 1×1 adapter
    t_feats = {
        "s2": torch.randn(B, 32, 48, 48),
        "s3": torch.randn(B, 32, 32, 32),
        "s4": torch.randn(B, 48, 16, 16),
    }
    s_logits = torch.randn(B, n_cls, 64, 64, requires_grad=True)
    t_logits = torch.randn(B, n_cls, 32, 32)      # teacher head 网格不一致

    # 全分辨率实例标签: 两个接触实例(挤拥组织场景)。
    # 实例边缘取 16 的倍数, 保证边界带落在 s4(16×16) nearest 采样格点上,
    # 避免"小网格上整带被采样跳过 → adapter 梯度恒 0"的假阴性
    lab = torch.zeros(B, 256, 256)
    lab[:, 32:96, 32:80] = 1.0
    lab[:, 32:96, 80:128] = 2.0

    loss, parts = bgkd(s_feats, t_feats, s_logits, t_logits, lab)
    assert torch.isfinite(loss).item(), "loss 非有限"
    assert torch.isfinite(parts["l_fe"]).item(), "l_fe 非有限"
    assert torch.isfinite(parts["l_kd"]).item(), "l_kd 非有限"

    loss.backward()
    assert torch.isfinite(s_logits.grad).all().item(), "s_logits.grad 非有限"
    for k, f in s_feats.items():
        assert f.grad is not None and torch.isfinite(f.grad).all().item(), \
            f"s_feats[{k}] 梯度缺失/非有限"
    # 1×1 adapter(s2: 16→32, s4: 64→48)必须收到非零梯度
    for name, p in bgkd.imit.adapters.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0, \
            f"adapter {name} 无梯度"


# ---------------------------------------------------------------- 3. SobelGateMS
def test_sobel_gate():
    torch.manual_seed(0)
    g = torch.randn(4, 8, 32, 32, requires_grad=True)   # batch=4: 回归修复②
    mod = SobelGateMS(8)
    out, w = mod(g)

    w_expect = torch.sigmoid(torch.tensor(2.0)).item()  # σ(2) ≈ 0.880797
    assert abs(w_expect - 0.880797) < 1e-6
    assert torch.allclose(w, torch.full_like(w, w_expect), atol=1e-6), \
        "零权重+偏置2 初始化下 W 应恒等于 σ(2)"
    assert torch.allclose(out, g * w_expect, atol=1e-6), \
        "嵌入初期应为恒等缩放 out = g·σ(2)"

    out.sum().backward()
    assert torch.isfinite(g.grad).all().item(), "g.grad 非有限"


# ---------------------------------------------------------------- 4. RAB
def test_rab():
    torch.manual_seed(0)
    f = torch.randn(2, 16, 32, 32, requires_grad=True)
    p = torch.rand(2, 1, 32, 32)        # 必须与 f 同分辨率(RAB 不做内部插值)
    rab = UncertaintyBandReverseAttention(16)

    out, a = rab(f, p, return_attn=True)
    assert torch.equal(out, f), "尾卷积零初始化下嵌入初期应严格恒等"
    assert torch.allclose(a.amax(dim=(2, 3)), torch.ones(2, 1), atol=1e-5), \
        "逐样本 amax 归一化后注意力峰值应≈1"

    out.sum().backward()
    assert torch.equal(f.grad, torch.ones_like(f)), "初始恒等下 f.grad 应严格恒 1"


# ---------------------------------------------------------------- 5. 配对统计
def test_stats():
    rng = np.random.default_rng(0)
    n_slide, n_patch = 24, 10
    delta = 0.045 + 0.005 * rng.standard_normal()       # 真实差异(逐 slide 一致)
    rows_a, rows_b = [], []
    for i in range(n_slide):
        slide_eff = 0.05 * rng.standard_normal()        # slide 随机效应(配对抵消)
        for j in range(n_patch):
            pid = f"p{i:02d}_{j:02d}"
            base = 0.75 + slide_eff + 0.02 * rng.standard_normal()
            rows_b.append({"patch_id": pid, "slide_id": f"s{i:02d}", "dice": base})
            rows_a.append({"patch_id": pid, "slide_id": f"s{i:02d}",
                           "dice": min(base + delta + 0.01 * rng.standard_normal(), 1.0)})
    df_a, df_b = pd.DataFrame(rows_a), pd.DataFrame(rows_b)

    res = paired_compare(df_a, df_b, ["dice"], "patch_id", "slide_id")
    row = res.iloc[0]
    assert row["n_slide"] == 24, f"n_slide={row['n_slide']}"
    assert row["n_patch"] == 240, f"n_patch={row['n_patch']}"
    assert row["p_slide"] < 0.01, f"切片级主检验 p={row['p_slide']}"
    assert row["p_holm"] < 0.05, f"Holm 校正 p={row['p_holm']}"
    assert row["effect_rank_biserial"] > 0.5, \
        f"rank-biserial={row['effect_rank_biserial']}"


# ---------------------------------------------------------------- 6. 延迟-能耗汇总
def test_summarize():
    lat = [120.0] * 470 + [180.0] * 30                  # 500 次测量
    rep = summarize(lat, power_w=7.5, warmup=50, iters=500)
    assert rep["n_measured"] == 500
    assert math.isclose(rep["median_ms"], 120.0, rel_tol=1e-9)
    assert math.isclose(rep["p99_ms"], 180.0, rel_tol=1e-9)
    assert math.isclose(rep["mean_ms"], 123.6, rel_tol=1e-9)
    assert math.isclose(rep["energy_median_mJ"], 900.0, rel_tol=1e-9)
    assert math.isclose(rep["energy_mean_mJ"], 7.5 * rep["mean_ms"], rel_tol=1e-12)
    assert rep["protocol_ok"] is True
    # 协议守门: 预热/迭代不足必须判 False
    assert summarize([120.0] * 10, warmup=5, iters=10)["protocol_ok"] is False


_TESTS = [test_band, test_bgkd, test_sobel_gate, test_rab, test_stats, test_summarize]


def main():
    failed = 0
    for fn in _TESTS:
        name = fn.__name__[5:]                          # 去掉 "test_" 前缀
        try:
            fn()
            print(f"[test] {name:<14s} OK")
        except Exception:                               # noqa: BLE001
            failed += 1
            print(f"[test] {name:<14s} FAIL")
            traceback.print_exc()
    if failed:
        print(f"{failed}/{len(_TESTS)} test(s) FAILED")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
