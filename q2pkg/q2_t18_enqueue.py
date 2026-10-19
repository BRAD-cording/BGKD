#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T18 入队: CoNSeP 零样本推理 (infer_consep.py) 追加到 q2_queue.json 尾部.

幂等: name=t18_consep 已存在则跳过。当前 runner 内存中持有旧列表,
本任务将由 sg_* 收官后 watchdog 拉起的新 runner 执行 (改队列文件需重启 runner)。
"""
import json
import sys
from pathlib import Path

Q = Path(__file__).resolve().parents[2] / "q2_queue.json"
NAME = "t18_consep"
TASK = {
    "name": NAME,
    "entry": "scripts/q2pkg/infer_consep.py",
    "cmd": ["--data", "C:/data/consep/consep_test.npz",
            "--out", "C:/runs/t18_consep"],
    "out": "C:/runs/t18_consep",
    "gate": False,
}

q = json.loads(Q.read_text(encoding="utf-8"))
if any(t.get("name") == NAME for t in q):
    print(f"[skip] {NAME} 已在队列 (共 {len(q)} 项)")
    sys.exit(0)
q.append(TASK)
Q.write_text(json.dumps(q, indent=1), encoding="utf-8")
print(f"[apply] {NAME} 已入队尾部 (共 {len(q)} 项)")
print(json.dumps(TASK, indent=1))
