# -*- coding: utf-8 -*-
"""把 full×3 + full_nokd×3 提前到消融 B 之后 (T7 之前)。
用法: python q2_reorder_full.py inspect|apply"""
import json
import sys

QUEUE = r"C:\Users\HP\OneDrive\Desktop\Paper\q2_queue.json"
MOVE = {"full_f1", "full_f2", "full_f3", "full_nokd_f1", "full_nokd_f2", "full_nokd_f3"}


def main(mode):
    q = json.load(open(QUEUE, encoding="utf-8"))
    names = [t["name"] for t in q]
    if mode == "inspect":
        for i, t in enumerate(q):
            print(f"{i:2d} {t['name']}")
        return
    moving = [t for t in q if t["name"] in MOVE]
    rest = [t for t in q if t["name"] not in MOVE]
    # 插入点: 最后一个 abl_* 之后
    last_abl = max(i for i, t in enumerate(rest) if t["name"].startswith("abl_"))
    rest[last_abl + 1:last_abl + 1] = moving
    with open(QUEUE, "w", encoding="utf-8") as f:
        json.dump(rest, f, ensure_ascii=False, indent=2)
    print(f"已移动 {len(moving)} 项至位置 {last_abl + 1}: {[t['name'] for t in moving]}")
    print("新顺序前 20 项:", [t["name"] for t in rest[:20]])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "inspect")
