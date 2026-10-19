#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配对统计检验（主检验=切片级聚合, 参考=patch级）—— 对应执行包 §2.2 统计协议与审查表 F 区。
修复原审稿意见 P0-3（无统计检验）与"伪重复"问题。

用途: 两份逐 patch 指标 CSV 的配对比较, 处理"同一 slide 的 patch 不独立"的伪重复。
输入: 每个CSV一行一个patch, 至少包含: <id_col>, <group_col>, 以及各 metric 列
用法:
  python stats_tests.py --file ours.csv baseline.csv --names Ours Baseline \
      --metrics dice bf1_2px pq --id-col patch_id --group-col slide_id \
      --alpha 0.05 --out stats_results.csv

输出(每 metric 一行):
  切片级 mean±std(两方法) | 相对提升% | bootstrap 95%CI(切片级均差) |
  切片级 Wilcoxon p(主) | Holm 校正 p | patch 级参考 p | rank-biserial 效应量 |
  n_slide / n_patch / n_effective / significant
说明: 逐折 mean±std 请由 3 个折的训练日志另行汇总（本脚本专注配对检验）。
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats


def rank_biserial(d: np.ndarray):
    """Wilcoxon 符号秩的 rank-biserial 效应量（+1=全为正差异, −1=全为负）。"""
    d = d[d != 0]
    n = d.size
    if n == 0:
        return np.nan, 0
    r = stats.rankdata(np.abs(d))
    w_pos = r[d > 0].sum()
    w_neg = r[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg)), n


def bootstrap_ci(d: np.ndarray, iters: int = 10000, seed: int = 0, alpha: float = 0.05):
    """切片级均差的 percentile bootstrap 95% CI（效应量报告, 审查表 F5）。"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(iters, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm_correct(pvals):
    """Holm-Bonferroni 逐步校正（族=预定义的主指标集合）。"""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    out = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def _safe_wilcoxon(d: np.ndarray) -> float:
    if d.size == 0 or not np.any(d != 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    except (ValueError, np.linalg.LinAlgError):
        return 1.0


def paired_compare(a: pd.DataFrame, b: pd.DataFrame, metrics, id_col, group_col,
                   seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """a/b: 逐 patch 指标表（同 patch_id 配对）。返回每 metric 一行的结果表。"""
    rows = []
    for metric in metrics:
        m = a[[id_col, group_col, metric]].merge(
            b[[id_col, metric]], on=id_col, suffixes=("_a", "_b"))
        d_patch = (m[metric + "_a"] - m[metric + "_b"]).to_numpy()

        # --- patch 级（仅参考, 存在伪重复） ---
        p_patch = _safe_wilcoxon(d_patch)

        # --- 切片级聚合（主检验: 同 slide 的 patch 先求均值再配对） ---
        sa = m.groupby(group_col)[metric + "_a"].mean()
        sb = m.groupby(group_col)[metric + "_b"].mean()
        d_slide = (sa - sb).to_numpy()
        p_slide = _safe_wilcoxon(d_slide)

        rb, n_eff = rank_biserial(d_slide)
        mean_diff = float(d_slide.mean())
        base = float(sb.mean())
        rel = 100.0 * mean_diff / base if base != 0 else np.nan
        lo, hi = bootstrap_ci(d_slide, seed=seed, alpha=alpha)

        rows.append(dict(
            metric=metric,
            mean_a=float(sa.mean()), std_a=float(sa.std(ddof=1)),
            mean_b=base, std_b=float(sb.std(ddof=1)),
            rel_improve_pct=rel,
            mean_diff=mean_diff, ci_lo=lo, ci_hi=hi,
            p_slide=p_slide, p_patch_ref=p_patch,
            effect_rank_biserial=rb,
            n_slide=int(sa.size), n_patch=int(m.shape[0]), n_effective=n_eff,
        ))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm_correct(df["p_slide"].to_numpy())
    df["significant"] = df["p_holm"] < alpha
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", nargs=2, required=True, help="ours.csv baseline.csv")
    ap.add_argument("--names", nargs=2, default=None)
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--id-col", default="patch_id")
    ap.add_argument("--group-col", default="slide_id")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="结果CSV路径(可选)")
    args = ap.parse_args()

    a = pd.read_csv(args.file[0])
    b = pd.read_csv(args.file[1])
    df = paired_compare(a, b, args.metrics, args.id_col, args.group_col,
                        seed=args.seed, alpha=args.alpha)

    names = args.names or ["method_a", "method_b"]
    show = df.copy()
    for c in ("p_slide", "p_patch_ref", "p_holm"):
        show[c] = show[c].map(lambda v: f"{v:.3g}")
    cols = ["metric", "mean_a", "std_a", "mean_b", "std_b", "rel_improve_pct",
            "ci_lo", "ci_hi", "p_slide", "p_holm", "p_patch_ref",
            "effect_rank_biserial", "n_slide", "n_patch", "significant"]
    print(f"[paired comparison] {names[0]} vs {names[1]} "
          f"(主检验=切片级 Wilcoxon, Holm 校正, alpha={args.alpha})")
    print(show[cols].to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
