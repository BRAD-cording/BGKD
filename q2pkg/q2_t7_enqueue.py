# -*- coding: utf-8 -*-
"""检查/入队 T7 基线任务。用法: python q2_t7_enqueue.py inspect|apply"""
import json
import os
import sys

QUEUE = r"C:\Users\HP\OneDrive\Desktop\Paper\q2_queue.json"
RUNS = r"C:\runs"

COMMON = ["--data-root", "C:\\data\\pannuke", "--epochs", "50", "--workers", "0"]

T7 = [
    # (name, model, batch) — batch 按 2026-10-05 独占剖析 (C:\runs\t7_profile.json)
    ("cellvit_t256_f1", "cellvit_t256", 8),
    ("cellvit_t256_f2", "cellvit_t256", 8),
    ("cellvit_t256_f3", "cellvit_t256", 8),
    ("cellvit_sam_f1", "cellvit_sam256", 4),
    ("cellvit_sam_f2", "cellvit_sam256", 4),
    ("cellvit_sam_f3", "cellvit_sam256", 4),
    ("ddbunet_f1", "ddbunet", 32),
    ("ddbunet_f2", "ddbunet", 32),
    ("ddbunet_f3", "ddbunet", 32),
]


def make(name, model, batch):
    out = os.path.join(RUNS, name)
    cmd = COMMON + ["--model", model, "--fold", name[-1], "--batch", str(batch),
                    "--out", out]
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
    new = [make(*x) for x in T7 if x[0] not in names]
    if not new:
        print("T7 已全部在队, 无操作")
        return
    # 插入点: 最后一个 abl_* (消融 B) 之后、蒸馏链 (full*) 之前
    last_abl = max(i for i, t in enumerate(q) if t["name"].startswith("abl_"))
    q[last_abl + 1:last_abl + 1] = new
    with open(QUEUE, "w", encoding="utf-8") as f:   # 无 BOM
        json.dump(q, f, ensure_ascii=False, indent=2)
    print(f"已插入 {len(new)} 项于位置 {last_abl + 1}: {[t['name'] for t in new]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    {"inspect": status, "apply": apply}[mode]()
