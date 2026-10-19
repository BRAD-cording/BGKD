# -*- coding: utf-8 -*-
"""投稿前审计 P1 补充核查: 子代理发现的 6 项疑点逐项复算。只读。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(r"C:\runs")
PAPER = Path(r"c:\Users\HP\OneDrive\Desktop\Paper\paper")

print("== 1. Jetson spread 复算 (p99-median / std / p95-median) ==")
for p in sorted((RUNS / "jetson_bench").glob("bench_*.json")):
    r = json.load(open(p, encoding="utf-8"))
    print(f"  {p.stem[6:]:16s} p99-med={r['p99_ms']-r['median_ms']:.3f}  "
          f"std={r['std_ms']:.3f}  p95-med={r['p95_ms']-r['median_ms']:.3f}")

print("== 3. seed mPQ+ 核查 (claims: seed-1 fold min < nokd mean; seed variance spans nokd mean) ==")
s = {k: [json.load(open(RUNS / f"{k}_f{f}" / "summary.json"))["mpq_plus"] * 100
         for f in (1, 2, 3)]
     for k in ("full", "full_s1", "full_s2", "full_nokd")}
print(f"  seed fold means: s0={np.mean(s['full']):.2f} s1={np.mean(s['full_s1']):.2f} "
      f"s2={np.mean(s['full_s2']):.2f}  nokd mean={np.mean(s['full_nokd']):.2f}")
print(f"  per-seed fold mins: {min(s['full']):.2f} {min(s['full_s1']):.2f} {min(s['full_s2']):.2f}")
print(f"  seed-mean range: [{min(np.mean(s[k]) for k in ('full','full_s1','full_s2')):.2f}, "
      f"{max(np.mean(s[k]) for k in ('full','full_s1','full_s2')):.2f}]")

print("== 6. KD 敏感性切片 (fold-1 PQ%) ==")
lam = {t: json.load(open(RUNS / f"{t}_f1" / "summary.json"))["pq"] * 100
       for t in ("dsens_a", "dsens_b", "dsens_c", "dsens_d", "full")}
a, b, c, d, full = (lam[t] for t in ("dsens_a", "dsens_b", "dsens_c", "dsens_d", "full"))
print(f"  (0.1,0.1)={a:.2f} (0.1,0.5)={b:.2f} (0.5,0.1)={c:.2f} (0.5,0.5)={d:.2f} def={full:.2f}")
print(f"  lkd 0.1→0.5: @lfe0.1={b-a:+.2f} @lfe0.5={d-c:+.2f} | lfe 0.1→0.5: @lkd0.1={c-a:+.2f} @lkd0.5={d-b:+.2f}")
print(f"  grid band vs default: {max(abs(x - full) for x in (a, b, c, d)):.2f}")

print("== 7. 28% 边界带面积来源 ==")
df = json.load(open(PAPER / "dataset_facts.json", encoding="utf-8"))
def walk(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                walk(v, pre + k + ".")
            elif "band" in k.lower() or "area" in k.lower() or "cover" in k.lower():
                print(f"  {pre}{k} = {v}")
walk(df)

print("== 8. T19 计数比 (claims: base 0.03-0.36, 一 tile 8 vs 245; full 1.0-2.3; AJI 34.7/23.1) ==")
for m in ("full", "own", "sam"):
    df = pd.read_csv(RUNS / f"t19_ft_{m}" / "consep_ft_reinhard.csv")
    ratio = df["n_pred"] / df["n_gt"]
    print(f"  {m}: ratio [{ratio.min():.2f}, {ratio.max():.2f}]  AJI={df['aji'].mean()*100:.1f}")
dfb = pd.read_csv(RUNS / "t19_ft_own" / "consep_ft_reinhard.csv")
print("  base 8 vs 245 存在?", bool(((dfb["n_pred"] == 8) & (dfb["n_gt"] == 245)).any()))
