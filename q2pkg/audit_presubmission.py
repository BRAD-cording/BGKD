# -*- coding: utf-8 -*-
"""投稿前审计 P0: C:\\runs 全量遍历 + 协议断言 + 表格再生一致性。
输出: PASS/FAIL 明细, 供 go/no-go 判定。只读检查, 不修改任何数据。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
RUNS = Path(r"C:\runs")
PAPER = PKG.parent.parent / "paper"

import make_tables as MT  # noqa: E402

issues, notes = [], []


def check(cond, msg):
    if cond:
        print(f"  [PASS] {msg}")
    else:
        issues.append(msg)
        print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------- 1. 目录遍历
print("== 1. C:\\runs 目录遍历 ==")
all_dirs = sorted(d for d in RUNS.iterdir() if d.is_dir())
run_dirs = {d.name: d for d in all_dirs if (d / "summary.json").exists()}
aux_dirs = [d.name for d in all_dirs if d.name not in run_dirs]
print(f"  run dirs={len(run_dirs)}, aux dirs={aux_dirs}")

# 期望 run 集合 (与 make_tables 完全同源)
expected = set()
for name in MT.FOLD_GROUPS:
    for f in (1, 2, 3):
        d = MT.DIR_OVERRIDE.get(name, {}).get(f, f"{name}_f{f}")
        expected.add(d)
for name in MT.FOLD1_ONLY:
    expected.add(f"{name}_f1")  # FOLD1_ONLY 目录名带 _f1 后缀
expected |= {"t18_consep", "t19_ft_full", "t19_ft_own", "t19_ft_sam"}
missing = expected - set(run_dirs)
extra = set(run_dirs) - expected
check(not missing, f"期望 run 全部存在 (缺: {sorted(missing)})")
if extra:
    notes.append(f"额外 run 目录 (不进表, 需人工确认无关键结果遗漏): {sorted(extra)}")
    print(f"  [NOTE] 额外目录: {sorted(extra)}")

# 健康标志
bad_health = [n for n, d in run_dirs.items()
              if (d / "last_fail.json").exists() or (d / "gate_pending.txt").exists()]
check(not bad_health, f"无 last_fail.json / gate_pending.txt (异常: {bad_health})")

# ---------------------------------------------------------------- 2. 指标完整性
print("== 2. summary.json 指标完整性与量程 ==")
REQ = ("pq", "mpq_plus", "aji", "f1_det", "hd95", "assd",
       "bf1_2px", "dice2px", "dice5px", "dice10px")
XDATASET = {"t18_consep", "t19_ft_full", "t19_ft_own", "t19_ft_sam"}
bad_metrics = []
for n, d in sorted(run_dirs.items()):
    if n.startswith("smoke") or n in XDATASET:
        continue  # smoke=冒烟测试(非论文数据); T18/T19 用各自 schema 单独核
    try:
        s = json.load(open(d / "summary.json", encoding="utf-8"))
        for k in REQ:
            v = s.get(k)
            if v is None or not np.isfinite(v):
                bad_metrics.append(f"{n}.{k}=None"); continue
            if k in MT.PCT and not (0.0 <= v <= 1.0):
                bad_metrics.append(f"{n}.{k}={v} 越界")
            if k in MT.DIST and not (0 < v < 200):
                bad_metrics.append(f"{n}.{k}={v} 越界")
        if not (d / "args.json").exists() and n not in ("t18_consep", "t19_ft_full", "t19_ft_own", "t19_ft_sam"):
            bad_metrics.append(f"{n} 缺 args.json")
    except Exception as e:  # noqa: BLE001
        bad_metrics.append(f"{n} 读取失败: {e}")
check(not bad_metrics, f"全部 run 关键指标非空且量程正常 (异常 {len(bad_metrics)} 项)")
for m in bad_metrics[:10]:
    print(f"         - {m}")

# per_tissue 依赖
miss_vm = [f"{r}_f{f}" for r in ("full", "full_nokd", "own_baseline",
                                 "hovernet", "nulite_s", "cellvit_sam256")
           for f in (1, 2, 3)
           if not (run_dirs.get(MT.DIR_OVERRIDE.get(r, {}).get(f, f"{r}_f{f}")
                                or f"{r}_f{f}", RUNS) / "val_metrics.csv").exists()]
check(not miss_vm, f"per_tissue 依赖的 val_metrics.csv 齐全 (缺: {miss_vm})")

# ---------------------------------------------------------------- 3. 协议断言
print("== 3. 协议一致性断言 (单变量 λ2=2 / Full λ2=4 + KD 0.3/0.3) ==")
try:
    MT.check_protocol()
except AssertionError as e:
    issues.append(f"协议断言失败: {e}")
    print(f"  [FAIL] {e}")

# ---------------------------------------------------------------- 4. T18/T19/Jetson 内部一致性
print("== 4. T18/T19/Jetson 内部一致性 ==")
s18 = json.load(open(RUNS / "t18_consep" / "summary.json", encoding="utf-8"))
rec18 = {(r["model"], r["stain"]): r for r in s18["runs"]}
check(len(rec18) == 6, f"T18 六组合齐全 ({len(rec18)}/6)")
miss18 = [f"{m}.{st}.{k}" for m in MT.CONSEP_MODELS for st in ("reinhard", "none")
          for k in ("consep_pq", "R", "consep_aji", "consep_f1_det",
                    "consep_hd95", "consep_bf1_2px")
          if rec18[(m, st)].get(k) is None]
check(not miss18, f"T18 指标完整 (缺: {miss18})")
miss19 = [f"{m}.{g}.{k}" for m in ("full", "own", "sam")
          for g in ("ft_reinhard", "ft_none")
          for k in ("pq", "aji", "f1_det", "hd95", "bf1_2px")
          if json.load(open(RUNS / f"t19_ft_{m}" / "summary.json",
                            encoding="utf-8"))[g].get(k) is None]
check(not miss19, f"T19 指标完整 (缺: {miss19})")
zs_ok = all(rec18[(m, "reinhard")]["R"] < 0.40 for m in MT.CONSEP_MODELS)
check(zs_ok, "T18 全部 R < 0.40 预注册门槛 (与正文无泛化声明一致)")

pannuke3 = {}
for m, grp in (("full_f2", "full"), ("own_baseline_f2", "own_baseline"),
               ("cellvit_sam_f2", "cellvit_sam256")):
    vals = [json.load(open(RUNS / MT.DIR_OVERRIDE.get(grp, {}).get(f, f"{grp}_f{f}") /
                           "summary.json", encoding="utf-8"))["pq"] for f in (1, 2, 3)]
    pannuke3[m] = float(np.mean(vals))
for m in MT.CONSEP_MODELS:
    r = rec18[(m, "reinhard")]
    calc = r["consep_pq"] / pannuke3[m]
    check(abs(calc - r["R"]) < 0.005, f"T18 {m} R 复算一致 ({calc:.4f} vs {r['R']:.4f})")

ft_ok = True
for m in ("full", "own", "sam"):
    s = json.load(open(RUNS / f"t19_ft_{m}" / "summary.json", encoding="utf-8"))
    calc = s["ft_reinhard"]["pq"] / s["pannuke_pq_3fold"]
    if abs(calc - s["ft_retention"]) >= 0.005:
        ft_ok = False
        issues.append(f"t19_ft_{m} retention 复算不一致 ({calc:.4f} vs {s['ft_retention']:.4f})")
    # 与 T18 零样本记录交叉一致
    zs = rec18[({"full": "full_f2", "own": "own_baseline_f2",
                 "sam": "cellvit_sam_f2"}[m], "reinhard")]
    if abs(zs["consep_pq"] - s["zero_shot_pq"]) > 1e-6:
        ft_ok = False
        issues.append(f"t19_ft_{m} zero_shot_pq 与 T18 不一致")
check(ft_ok, "T19 retention 复算一致 + 与 T18 零样本记录交叉一致")

jb = sorted((RUNS / "jetson_bench").glob("bench_*.json"))
check(len(jb) == 7, f"Jetson bench 7/7 ({len(jb)}/7)")
dt_max, proto_bad = 0.0, []
for p in jb:
    r = json.load(open(p, encoding="utf-8"))
    if not (r.get("n_warmup") == 50 and r.get("n_measured") == 500
            and r.get("protocol_ok") is True):
        proto_bad.append(p.name)
    for zone in ("gpu-thermal", "tj-thermal"):
        dt = r["temps_end_c"][zone] - r["temps_start_c"][zone]
        dt_max = max(dt_max, dt)
check(not proto_bad, f"Jetson 50 warmup/500 iters/protocol_ok 全部满足 (异常: {proto_bad})")
check(dt_max <= 1.6, f"Jetson 温升 ΔT={dt_max:.2f}°C ≤ 1.6°C (正文声明无热节流)")

# ---------------------------------------------------------------- 5. 表格再生一致性 (单一数据源验证)
print("== 5. 表格再生一致性: 论文在用表 vs 由 runs 重生 ==")
tex_files = sorted((PAPER / "tables").glob("*.tex"))
old = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in tex_files}
facts_old = hashlib.md5((PAPER / "paper_facts.json").read_bytes()).hexdigest()
MT.main()  # 重新生成全部表 + facts (会覆盖 paper/tables, 内容应逐字节一致)
new = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in tex_files}
facts_new = hashlib.md5((PAPER / "paper_facts.json").read_bytes()).hexdigest()
diff = [n for n in old if old[n] != new.get(n)]
diff += [n for n in new if n not in old]
check(not diff, f"10 个 .tex 表逐字节一致 (diff: {diff})")
check(facts_old == facts_new, "paper_facts.json 逐字节一致")

# ---------------------------------------------------------------- 6. 汇总
print("\n== 审计汇总 ==")
print(f"issues={len(issues)}")
for i in issues:
    print(f"  [ISSUE] {i}")
for n_ in notes:
    print(f"  [NOTE] {n_}")
json.dump({"issues": issues, "notes": notes},
          open(PKG / "audit_p0_result.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
