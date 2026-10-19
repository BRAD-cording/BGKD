# -*- coding: utf-8 -*-
"""24h GPU 队列 runner —— 链式执行 + 幂等续跑 + B 类决策闸门 + 无人值守 watchdog
=============================================================
设计（对应执行包 §1.2 24h 队列模式）:
  * A 类任务(可盲跑): 逐项链式执行, 崩溃/掉电后自动从断点续(train.py checkpoint resume)。
  * B 类闸门(gate=true): 跑完即停, 写 gate_pending.txt; 人工审查决策后手动重跑继续。
  * 幂等: 任务完成标志 = <out>/summary.json 存在(train.py 末尾产物), 重入自动跳过。
  * 单实例锁: queue_runner.lock 记录 PID, 防止 watchdog 与手动运行双开。
  * --auto 模式(供计划任务每 15min 调用): 有闸门待决策/失败退避期内/锁被占 → 静默退出;
    否则拉起队列。失败重试 3 次后写 last_fail.json, 1h 退避, 供计划任务节奏重试。
  * 高速自愈: 启动时自检电源计划(非高性能即纠正)+训练子进程钉 P-core
    (Q2_AFFINITY_MASK 可覆盖, 0=不钉)。
  * 训练产物统一在 C:\\runs(本地盘), 禁止 OneDrive 目录(同步锁文件曾致训练挂死)。

用法:
  python scripts/q2pkg/queue_runner.py --queue q2_queue.json            # 手动主模式
  python scripts/q2pkg/queue_runner.py --queue q2_queue.json --auto     # watchdog 模式
  python scripts/q2pkg/queue_runner.py --queue q2_queue.json --list     # 查看队列进度
  python scripts/q2pkg/queue_runner.py --queue q2_queue.json --from nulite_t_f2   # 强制从指定任务起

队列文件格式(JSON 数组, 每项):
  {"name": "nulite_t_f1", "cmd": [...args 传给 train.py...], "out": "C:/runs/nulite_t_f1", "gate": false}
  cmd 不含 python/train.py 前缀; out 为 --out 目录(绝对路径, 位于本地盘); gate=true 表示跑完后停下等人工决策。
"""
import argparse
import atexit
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # Paper/ (代码所在, 允许 OneDrive)
RUNS_ROOT = Path(os.environ.get("Q2_RUNS_ROOT", r"C:\runs"))   # 训练产物: 本地盘
LOGDIR = RUNS_ROOT / "queue_logs"
TRAIN_ENTRY = ROOT / "scripts" / "q2pkg" / "train.py"
LOCK = RUNS_ROOT / "queue_runner.lock"
GATE_MARK = RUNS_ROOT / "gate_pending.txt"
FAIL_MARK = RUNS_ROOT / "last_fail.json"
FAIL_BACKOFF_SEC = 3600      # auto 模式失败退避
MAX_ATTEMPTS = 3             # 单次运行内每任务尝试次数
RETRY_WAIT_SEC = 60
# --- 24h 高速自愈配置 (2026-09-28 实证: BIOS 重启后 Windows 电源计划重置回「平衡」
# → CPU 不 boost(~2GHz), 单线程数据准备掉速 4.8×: 635s vs 133s/epoch) ---
HIGH_PERF_GUID = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"   # Windows 内置「高性能」计划
OVERLAY_BEST = "ded574b5-45a0-4f42-8737-46345c09c238"     # 电源模式「最佳性能」
AFFINITY_MASK = int(os.environ.get("Q2_AFFINITY_MASK", "0xFFFF"), 0)
# 0xFFFF = 逻辑核 0-15 (i9-14900HX 8 P-core ×2 HT); 防调度器把训练线程放到 E-core
# (实测钉 P-core 221→133s/epoch)。设 0 关闭; 其他 CPU 拓扑用 env 覆盖。


def ensure_perf_power() -> None:
    """防复发自检: 活动电源计划非「高性能」即纠正 + overlay 设「最佳性能」。"""
    if os.name != "nt":
        return
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                             text=True, timeout=15).stdout
        if HIGH_PERF_GUID not in out:
            subprocess.run(["powercfg", "/setactive", HIGH_PERF_GUID],
                           capture_output=True, timeout=15)
            print("[queue] 电源计划已恢复为「高性能」(掉速自检触发)", flush=True)
        subprocess.run(["powercfg", "/overlaysetactive", OVERLAY_BEST],
                       capture_output=True, timeout=15)
    except Exception as e:  # noqa: BLE001
        print(f"[queue] 电源自检失败(不影响队列): {e}", flush=True)


def _pin_affinity(p: subprocess.Popen) -> None:
    """训练子进程钉到 P-core, 防 E-core 调度掉速 (仅 Windows)。"""
    if os.name != "nt" or not AFFINITY_MASK:
        return
    try:
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1FFFFF, False, p.pid)
        if h:
            k32.SetProcessAffinityMask(h, AFFINITY_MASK)
            k32.CloseHandle(h)
    except Exception:  # noqa: BLE001
        pass


def out_path(t) -> Path:
    p = Path(t["out"])
    return p if p.is_absolute() else RUNS_ROOT / p


def load_queue(path: Path):
    tasks = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    for t in tasks:
        for k in ("name", "cmd", "out"):
            if k not in t:
                raise ValueError(f"队列项缺少字段 {k}: {t}")
        if t["name"] in seen:
            raise ValueError(f"任务名重复: {t['name']}")
        seen.add(t["name"])
    return tasks


def is_done(t) -> bool:
    return (out_path(t) / "summary.json").exists()


def _pid_alive(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return pid == os.getpid()
    try:
        if os.name == "nt":
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
            if h:
                k32.CloseHandle(h)
                return True
            return False
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def acquire_lock() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1
        if _pid_alive(pid):
            print(f"[queue] 已有实例在运行 (pid={pid}), 本轮退出", flush=True)
            sys.exit(0)
        print(f"[queue] 清理残留锁 (pid={pid} 已不存在)")
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(lambda: LOCK.exists() and LOCK.unlink(missing_ok=True))


def run_task(t) -> int:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = LOGDIR / f"{t['name']}.log"
    entry = ROOT / t["entry"] if t.get("entry") else TRAIN_ENTRY
    full = [sys.executable, str(entry)] + [str(c) for c in t["cmd"]]
    print(f"[queue] 启动 {t['name']}: {' '.join(str(c) for c in t['cmd'])}", flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {t['name']} start {time.strftime('%F %T')} =====\n")
        f.write(" ".join(str(c) for c in full) + "\n")
        f.flush()
        p = subprocess.Popen(full, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        _pin_affinity(p)
        rc = p.wait()
    dt = time.time() - t0
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"===== {t['name']} end rc={rc} ({dt/3600:.2f}h) =====\n")
    print(f"[queue] {t['name']} 结束 rc={rc} 用时 {dt/3600:.2f}h", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", default="q2_queue.json")
    ap.add_argument("--list", action="store_true", help="只显示进度, 不执行")
    ap.add_argument("--auto", action="store_true",
                    help="watchdog 模式: 闸门待决/退避期/锁被占时静默退出")
    ap.add_argument("--from", dest="force_from", default=None,
                    help="强制从指定任务名开始执行(忽略其已完成状态)")
    args = ap.parse_args()

    if args.auto:
        # pythonw 无控制台, stdout 重定向到 watchdog 日志
        LOGDIR.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(LOGDIR / "watchdog.log", "a",
                                       encoding="utf-8", buffering=1)
        print(f"\n===== watchdog tick {time.strftime('%F %T')} =====", flush=True)

    qpath = Path(args.queue)
    if not qpath.is_absolute():
        qpath = ROOT / qpath
    tasks = load_queue(qpath)

    if args.list:
        for t in tasks:
            mark = "✅ done" if is_done(t) else ("⛔ gate" if t.get("gate") else "▶ pending")
            print(f"{mark:8s} {t['name']:24s} -> {out_path(t)}")
        n_done = sum(1 for t in tasks if is_done(t))
        print(f"--- {n_done}/{len(tasks)} 完成")
        return

    if args.auto:
        if GATE_MARK.exists():
            print(f"[auto] 闸门待人工决策: {GATE_MARK.read_text(encoding='utf-8').strip()}"
                  " — 跳过本轮 (决策后手动运行一次本命令继续)", flush=True)
            return
        if FAIL_MARK.exists():
            try:
                d = json.loads(FAIL_MARK.read_text(encoding="utf-8"))
                if time.time() - d.get("ts", 0) < FAIL_BACKOFF_SEC:
                    print(f"[auto] {d.get('task')} 失败退避中, 跳过本轮", flush=True)
                    return
            except Exception:  # noqa: BLE001
                FAIL_MARK.unlink(missing_ok=True)
    if not args.auto and GATE_MARK.exists():
        GATE_MARK.unlink(missing_ok=True)   # 手动运行 = 人工已决策, 过闸

    ensure_perf_power()      # 24h 高速防复发自检 (幂等, 失败不影响队列)
    acquire_lock()

    start = 0
    if args.force_from:
        names = [t["name"] for t in tasks]
        if args.force_from not in names:
            sys.exit(f"[queue] --from 任务不存在: {args.force_from}")
        start = names.index(args.force_from)
    else:
        for i, t in enumerate(tasks):
            if not is_done(t):
                start = i
                break
        else:
            print("[queue] 队列全部完成", flush=True)
            return

    for t in tasks[start:]:
        if is_done(t) and args.force_from is None:
            continue
        rc = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            rc = run_task(t)
            if rc == 0:
                break
            print(f"[queue] {t['name']} 第 {attempt}/{MAX_ATTEMPTS} 次失败 (rc={rc}), "
                  f"checkpoint 续训重试", flush=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_SEC)
        if rc != 0:
            FAIL_MARK.write_text(json.dumps({"task": t["name"], "ts": time.time()}),
                                 encoding="utf-8")
            print(f"[queue] 任务失败: {t['name']} 已重试 {MAX_ATTEMPTS} 次, 写入退避标记; "
                  f"日志: {LOGDIR / (t['name'] + '.log')}", flush=True)
            sys.exit(rc)
        FAIL_MARK.unlink(missing_ok=True)
        if not is_done(t):
            print(f"[queue] 警告: {t['name']} rc=0 但未见 summary.json, 视为未完成", flush=True)
            sys.exit(2)
        if t.get("gate"):
            GATE_MARK.write_text(t["name"], encoding="utf-8")
            print(f"\n[queue] 🚪 B 类闸门: {t['name']} 已完成, 请审查 {out_path(t)} 中的结果",
                  flush=True)
            print("[queue] 决策后手动运行本命令继续队列(已完成的项自动跳过)", flush=True)
            return
    print("[queue] 队列全部完成", flush=True)


if __name__ == "__main__":
    main()
