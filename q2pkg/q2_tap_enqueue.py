# -*- coding: utf-8 -*-
"""检查/入队 SobelGate 位置消融 (P1-1)。用法: python q2_tap_enqueue.py inspect|apply

单变量协议: base + SG@tap only (--sobel-gate --sobel-tap X), 其余全默认
(λ₂=2.0, 无 RAB/边界/KD, batch 32, seed 0) —— 与 abl_sg_f* (tap=s2) 可比。
位置消融 s2 行 = 已完成 abl_sg_f{1,2,3}, 不重跑。
"""
import json
import os
import sys

QUEUE = r"C:\Users\HP\OneDrive\Desktop\Paper\q2_queue.json"
RUNS = r"C:\runs"

COMMON = ["--data-root", "C:\\data\\pannuke", "--epochs", "50", "--workers", "0",
          "--model", "own_baseline", "--batch", "32", "--sobel-gate"]


def make(tap, fold):
    name = f"sg_{tap}_f{fold}"
    out = os.path.join(RUNS, name)
    cmd = COMMON + ["--fold", str(fold), "--sobel-tap", tap, "--out", out]
    return {"name": name, "cmd": cmd, "out": out, "gate": False}   # A 类盲跑


def status():
    q = json.load(open(QUEUE, encoding="utf-8"))
    for i, t in enumerate(q):
        done = os.path.exists(os.path.join(t["out"], "summary.json"))
        print(f"{i:2d} {'D' if done else '.'} {'G' if t.get('gate') else ' '} {t['name']}")
    n_done = sum(1 for t in q if os.path.exists(os.path.join(t["out"], "summary.json")))
    print(f"--- {n_done}/{len(q)} 完成")


def apply():
    q = json.load(open(QUEUE, encoding="utf-8"))
    names = {t["name"] for t in q}
    new = [make(tap, f) for tap in ("s3", "s4", "all") for f in (1, 2, 3)
           if f"sg_{tap}_f{f}" not in names]
    if not new:
        print("位置消融已全部在队, 无操作")
        return
    q.extend(new)                                   # 追加到队尾 (59 项已完成会被幂等跳过)
    with open(QUEUE, "w", encoding="utf-8") as f:   # 无 BOM
        json.dump(q, f, ensure_ascii=False, indent=2)
    print(f"已追加 {len(new)} 项: {[t['name'] for t in new]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    {"inspect": status, "apply": apply}[mode]()
