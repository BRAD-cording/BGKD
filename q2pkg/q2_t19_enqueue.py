# -*- coding: utf-8 -*-
r"""T19 CoNSeP 微调 3 runs 入队 (幂等, inspect|apply)。

用法:
  python q2_t19_enqueue.py inspect
  python q2_t19_enqueue.py apply
"""
import json
import sys
from pathlib import Path

Q = Path(__file__).resolve().parent.parent.parent / "q2_queue.json"

TASKS = [
    {"name": "t19_ft_full", "entry": "scripts/q2pkg/finetune_consep.py",
     "cmd": ["--pretrained", "C:/runs/full_f2", "--out", "C:/runs/t19_ft_full"],
     "out": "C:/runs/t19_ft_full", "gate": False},
    {"name": "t19_ft_own", "entry": "scripts/q2pkg/finetune_consep.py",
     "cmd": ["--pretrained", "C:/runs/own_baseline_f2", "--out", "C:/runs/t19_ft_own"],
     "out": "C:/runs/t19_ft_own", "gate": False},
    {"name": "t19_ft_sam", "entry": "scripts/q2pkg/finetune_consep.py",
     "cmd": ["--pretrained", "C:/runs/cellvit_sam_f2", "--out", "C:/runs/t19_ft_sam"],
     "out": "C:/runs/t19_ft_sam", "gate": False},
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    q = json.loads(Q.read_text(encoding="utf-8"))
    if isinstance(q, dict):                               # 兼容 {"tasks":[...]} 包装
        q = q["tasks"]
    names = {t["name"] for t in q}
    if mode == "inspect":
        print(f"queue: {len(q)} tasks, done="
              f"{sum(1 for t in q if t.get('done'))}")
        for t in TASKS:
            print(f"  {'EXISTS' if t['name'] in names else 'MISSING'}: {t['name']}")
        return
    if mode == "apply":
        added = 0
        for t in TASKS:
            if t["name"] not in names:
                q.append(t)
                added += 1
        if added:
            Q.write_text(json.dumps(q, indent=2), encoding="utf-8")
        print(f"added {added} tasks -> total {len(q)}")
        return
    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
