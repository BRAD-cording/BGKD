# -*- coding: utf-8 -*-
"""汇总 C:\runs 所有已完成 run 的 PQ/mPQ+。用法: python q2_status.py"""
import glob
import json
import os
import statistics


def pick(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


rows = []
for sj in glob.glob(r"C:\runs\*_f?\summary.json") + glob.glob(r"C:\runs\lam*_f?\summary.json") + glob.glob(r"C:\runs\bnd_*\summary.json"):
    name = os.path.basename(os.path.dirname(sj))
    try:
        s = json.load(open(sj, encoding="utf-8"))
    except Exception:
        continue
    pq = pick(s, "test_pq", "pq", "val_pq", "final_pq")
    mpq = pick(s, "test_mpq_plus", "mpq_plus", "val_mpq_plus", "final_mpq_plus")
    if pq is None:
        continue
    rows.append((name, pq, mpq))

for name, pq, mpq in sorted(rows):
    print(f"{name:16s} PQ={pq:.4f} mPQ+={mpq:.4f}" if mpq is not None else f"{name:16s} PQ={pq:.4f}")

# 3 折均值（同名前缀分组）
groups = {}
for name, pq, mpq in rows:
    g = name.rsplit("_f", 1)[0]
    groups.setdefault(g, []).append(pq)
print("--- 3 折均值 ---")
for g, pqs in sorted(groups.items()):
    if len(pqs) >= 2:
        print(f"{g:14s} n={len(pqs)} meanPQ={statistics.mean(pqs):.4f}")
