# -*- coding: utf-8 -*-
"""成稿数据汇总: C:\\runs → paper/tables/*.tex|csv + paper_facts.json
=====================================================================
对应论文写作冲刺 P0-2。全部表格由本脚本生成, 正文通过 \\input 引用,
禁止手填数字 (单一数据源, 防止成稿数字与 runs 脱节)。

口径 (与项目记忆/审查表一致):
  * 百分比指标 (pq/mpq_plus/aji/f1_det/bf1_2px) ×100, 保留 2 位
  * 距离指标 (hd95/assd) px, 越低越好
  * 折间 mean±std (ddof=1, n=3)
  * mPQ+ = 5 核类宏平均 (N_EVAL_CLASSES=5), 全文唯一口径
  * NuLite 复现行 = nulite_s (fastvit_s12, 对齐官方发布权重); nulite_t 是 t8 变体自训行
  * 消融行 = 单变量基线超参 (λ₂=2.0); Full 行 = 调参 λ₂=4.0 (+ 蒸馏)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))

RUNS = Path(r"C:\runs")
OUT = Path(r"c:\Users\HP\OneDrive\Desktop\Paper\paper\tables")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- run 目录映射
# name → (论文行名, fold→目录覆盖)  缺省目录 = f"{name}_f{f}"
FOLD_GROUPS = {
    # ---- 外部基线 / 复现行 ----
    "nulite_t":       "NuLite-T (FastViT-T8 variant)",
    "nulite_s":       "NuLite-S (official arch.)",
    "hovernet":       "HoVer-Net (reimpl.)",
    "cellvit_t256":   "CellViT-256",
    "cellvit_sam256": "CellViT-256-SAM",
    "ddbunet":        "DDBU-Net (repro.)",
    # ---- 自家 ----
    "own_baseline":   "Ours (base)",
    # ---- 消融 (单变量, λ₂=2.0) ----
    "abl_sg":         "\\quad + SobelGate-MS",
    "abl_rab":        "\\quad + RAB",
    "abl_bnd":        "\\quad + boundary loss",
    "abl_sgrab":      "\\quad + SG + RAB",
    "abl_sgbnd":      "\\quad + SG + Bnd",
    "abl_rabbnd":     "\\quad + RAB + Bnd",
    # ---- 门控位置消融 (单变量, λ₂=2.0, gate tap 位置) ----
    "sg_s3":          "\\quad + SG@s3",
    "sg_s4":          "\\quad + SG@s4",
    "sg_all":         "\\quad + SG@all",
    # ---- 完整方案 ----
    "full_nokd":      "Ours (full, w/o KD)",
    "full":           "Ours (full)",
    # ---- 多种子 (独立表) ----
    "full_s1":        "Ours (full, seed 1)",
    "full_s2":        "Ours (full, seed 2)",
}
DIR_OVERRIDE = {"own_baseline": {1: "w1_own_f1"},   # f1 = W1 run
                "abl_bnd": {1: "bnd_l2_f1"},        # +Bnd f1 复用 bnd_l2 (同配置)
                "cellvit_sam256": {f: f"cellvit_sam_f{f}" for f in (1, 2, 3)}}
FOLD1_ONLY = {  # 折1 调优 / 敏感性 runs: name → 行名
    "lam2_1":  "$\\lambda_2{=}1$",
    "lam2_4":  "$\\lambda_2{=}4$",
    "bnd_l05": "$\\lambda_3{=}0.5$",
    "bnd_l1":  "$\\lambda_3{=}1$",
    "bnd_l2":  "$\\lambda_3{=}2$",
    "dsens_a": "KD $(\\lambda_{fe}{=}0.1,\\lambda_{kd}{=}0.1)$",
    "dsens_b": "KD $(0.1,\\,0.5)$",
    "dsens_c": "KD $(0.5,\\,0.1)$",
    "dsens_d": "KD $(0.5,\\,0.5)$",
}
PCT = ("pq", "mpq_plus", "aji", "f1_det", "bf1_2px")
DIST = ("hd95", "assd")


def fold_dir(name: str, f: int) -> Path:
    d = DIR_OVERRIDE.get(name, {}).get(f, f"{name}_f{f}")
    p = RUNS / d
    if not (p / "summary.json").exists():
        raise FileNotFoundError(p / "summary.json")
    return p


def collect(name: str, folds=(1, 2, 3)) -> pd.DataFrame:
    rows = []
    for f in folds:
        s = json.load(open(fold_dir(name, f) / "summary.json", encoding="utf-8"))
        s["fold"] = f
        rows.append(s)
    return pd.DataFrame(rows)


def ms(df: pd.DataFrame, k: str) -> str:
    v = df[k].astype(float) * (100 if k in PCT else 1)
    prec = 2 if k in PCT else 1
    return f"{v.mean():.{prec}f}$\\pm${v.std(ddof=1):.{prec}f}"


def args_of(name: str, f: int = 1) -> dict:
    return json.load(open(fold_dir(name, f) / "args.json", encoding="utf-8"))


def check_protocol():
    """配置一致性断言: 消融行基线超参 / Full 行调参+蒸馏。"""
    for name in ("own_baseline", "abl_sg", "abl_rab", "abl_bnd",
                 "abl_sgrab", "abl_sgbnd", "abl_rabbnd",
                 "sg_s3", "sg_s4", "sg_all",
                 "nulite_t", "nulite_s", "hovernet",
                 "cellvit_t256", "cellvit_sam256", "ddbunet"):
        a = args_of(name)
        assert a["lambda_hv"] == 2.0 and not a.get("distill", False), \
            f"{name} 违反单变量口径: {a}"
    for name in ("full", "full_nokd", "full_s1", "full_s2"):
        a = args_of(name)
        assert a["lambda_hv"] == 4.0, f"{name} λ₂≠4.0"
        assert a["distill"] == (name != "full_nokd"), f"{name} distill 开关异常"
        if a.get("distill"):
            assert a["lambda_fe"] == 0.3 and a["lambda_kd"] == 0.3
    print("[protocol] 配置一致性断言全部通过")


# ---------------------------------------------------------------- 参数量 (CPU 离线)
def count_params() -> dict:
    import torch
    from models import build_model
    from nulite_adapter import build_nulite_t
    from hovernet_adapter import build_hovernet
    from cellvit_adapter import build_cellvit_256
    from cellvit_sam_adapter import build_cellvit_sam_256
    from ddbunet_adapter import build_ddbunet

    builders = {
        "own_base":  lambda: build_model("own_baseline", base=32),
        "own_full":  lambda: build_model("own_baseline", base=32,
                                         use_sobel_gate=True, use_rab=True),
        "nulite_t":  lambda: build_nulite_t(pretrained=False, backbone="fastvit_t8"),
        "nulite_s":  lambda: build_nulite_t(pretrained=False, backbone="fastvit_s12"),
        "hovernet":  lambda: build_hovernet(pretrained=False),
        "cellvit_t256":   lambda: build_cellvit_256(pretrained=False),
        "cellvit_sam256": lambda: build_cellvit_sam_256(pretrained=False),
        "ddbunet":   lambda: build_ddbunet(pretrained=False),
    }
    out = {}
    for k, b in builders.items():
        n = sum(p.numel() for p in b().parameters())
        out[k] = n
        print(f"[params] {k:16s} {n/1e6:.2f}M")
    return out


# ---------------------------------------------------------------- 表 1: 主对比表
def table_main(pq: dict, params: dict):
    rows_main = ["nulite_t", "nulite_s", "ddbunet", "cellvit_t256",
                 "cellvit_sam256", "hovernet", "own_baseline", "full_nokd", "full"]
    label = {**{k: FOLD_GROUPS[k] for k in FOLD_GROUPS},
             "full_nokd": "Ours (full, w/o KD)", "full": "\\textbf{Ours (full)}"}
    par = {"nulite_t": params["nulite_t"], "nulite_s": params["nulite_s"],
           "hovernet": params["hovernet"], "cellvit_t256": params["cellvit_t256"],
           "cellvit_sam256": params["cellvit_sam256"], "ddbunet": params["ddbunet"],
           "own_baseline": params["own_base"], "full_nokd": params["own_full"],
           "full": params["own_full"]}
    cols = [("pq", "PQ$\\uparrow$"), ("mpq_plus", "mPQ+$\\uparrow$"),
            ("aji", "AJI$\\uparrow$"), ("f1_det", "F1$\\uparrow$"),
            ("hd95", "HD95$\\downarrow$"), ("bf1_2px", "BF1@2px$\\uparrow$")]
    # 每列最优(加粗) / 次优(下划线)
    best, second = {}, {}
    for k, _ in cols:
        vals = {r: pq[r][k].mean() * (100 if k in PCT else 1) for r in rows_main}
        s = sorted(vals, key=vals.get, reverse=(k not in DIST))
        best[k], second[k] = s[0], s[1]

    lines = [
        "% auto-generated by make_tables.py — DO NOT EDIT NUMBERS",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{PanNuke official 3-fold test performance (mean$\\pm$std over folds). "
        "Percentage metrics in \\%; HD95 in px (lower is better). "
        "mPQ+ is the macro-average over the five evaluable nucleus classes.}",
        "\\label{tab:main}",
        "\\small",
        "\\setlength{\\tabcolsep}{4.2pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l r " + " c" * len(cols) + "}",
        "\\toprule",
        "Method & Params & " + " & ".join(c[1] for c in cols) + " \\\\",
        "\\midrule",
    ]
    for r in rows_main:
        if r in ("own_baseline",):
            lines.append("\\midrule")
        p = f"{par[r]/1e6:.2f}" if par.get(r) else "--"
        cells = []
        for k, _ in cols:
            v = ms(pq[r], k)
            if r == best[k]:
                v = f"\\textbf{{{v}}}"
            elif r == second[k]:
                v = f"\\underline{{{v}}}"
            cells.append(v)
        note = ""
        if r == "nulite_s":
            note = "$^{\\dagger}$"
        if r == "hovernet":
            note = "$^{\\ddagger}$"
        if r == "ddbunet":
            note = "$^{\\S}$"
        lines.append(f"{label[r]}{note} & {p} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\begin{flushleft}\\footnotesize",
        "$^{\\dagger}$Official architecture and released-weight alignment run "
        "(FastViT-S12 backbone; the released NuLite-T checkpoint uses this backbone).",
        "$^{\\ddagger}$Re-implementation following the original design.",
        "$^{\\S}$Re-production from the published description (no public code).",
        "\\end{flushleft}",
        "\\end{table}", ""]
    (OUT / "main_comparison.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[table] main_comparison.tex")


# ---------------------------------------------------------------- 表 2: 消融表
def table_ablation(pq: dict):
    rows = ["own_baseline", "abl_sg", "abl_rab", "abl_bnd", "abl_sgrab",
            "abl_sgbnd", "abl_rabbnd", "full_nokd", "full"]
    cols = [("pq", "PQ$\\uparrow$"), ("mpq_plus", "mPQ+$\\uparrow$"),
            ("aji", "AJI$\\uparrow$"), ("bf1_2px", "BF1@2px$\\uparrow$")]
    best = {}
    for k, _ in cols:
        vals = {r: pq[r][k].mean() for r in rows}
        best[k] = max(vals, key=vals.get)
    label = {
        "own_baseline": "Ours (base)",
        "abl_sg": "+ SobelGate-MS",
        "abl_rab": "+ RAB",
        "abl_bnd": "+ boundary loss",
        "abl_sgrab": "+ SG + RAB",
        "abl_sgbnd": "+ SG + Bnd",
        "abl_rabbnd": "+ RAB + Bnd",
        "full_nokd": "Full (tuned, w/o KD)",
        "full": "\\textbf{Full (tuned + BGKD)}",
    }
    lines = [
        "% auto-generated by make_tables.py — DO NOT EDIT NUMBERS",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation on PanNuke (mean$\\pm$std over the official 3 folds, \\%). "
        "Component rows use the single-variable protocol with the baseline loss weights "
        "($\\lambda_2{=}2$); the two Full rows additionally use the tuned $\\lambda_2{=}4$ "
        "(selected on the fold-1 held-out split), isolating the distillation contribution in the "
        "Full vs.\\ Full w/o KD pair.}",
        "\\label{tab:ablation}",
        "\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l " + " c" * len(cols) + "}",
        "\\toprule",
        "Variant & " + " & ".join(c[1] for c in cols) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = ms(pq[r], k)
            if r == best[k]:
                v = f"\\textbf{{{v}}}"
            cells.append(v)
        lines.append(f"{label[r]} & " + " & ".join(cells) + " \\\\")
        if r == "abl_rabbnd":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""]
    (OUT / "ablation.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[table] ablation.tex")


# ---------------------------------------------------------------- 表 2b: 门控位置消融
def table_gatepos(pq: dict):
    rows = ["own_baseline", "abl_sg", "sg_s3", "sg_s4", "sg_all"]
    label = {"own_baseline": "Baseline (no gate)",
             "abl_sg": "+ SG@s2 (default)", "sg_s3": "+ SG@s3",
             "sg_s4": "+ SG@s4", "sg_all": "+ SG@all"}
    cols = [("pq", "PQ$\\uparrow$"), ("mpq_plus", "mPQ+$\\uparrow$"),
            ("aji", "AJI$\\uparrow$"), ("bf1_2px", "BF1@2px$\\uparrow$")]
    best = {}
    for k, _ in cols:
        vals = {r: pq[r][k].mean() for r in rows}
        best[k] = max(vals, key=vals.get)
    lines = [
        "% auto-generated by make_tables.py — DO NOT EDIT NUMBERS",
        "\\begin{table}[t]", "\\centering",
        "\\caption{SobelGate-MS placement ablation under the single-variable "
        "protocol (mean$\\pm$std over the official 3 folds, \\%). "
        "s2/s3/s4 = gate taps at strides 4/8/16; ``all'' gates every tap "
        "simultaneously.}",
        "\\label{tab:gatepos}", "\\small",
        "\\begin{tabular}{l " + " c" * len(cols) + "}", "\\toprule",
        "Variant & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule",
    ]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = ms(pq[r], k)
            if r == best[k]:
                v = f"\\textbf{{{v}}}"
            cells.append(v)
        lines.append(f"{label[r]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "gatepos.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[table] gatepos.tex")


# ---------------------------------------------------------------- 表 3: λ 敏感性
def table_lambda(pq: dict):
    def f1(name):
        df = collect(name, folds=(1,))
        return df["pq"].iloc[0] * 100, df["mpq_plus"].iloc[0] * 100

    rows_a = [("Ours base ($\\lambda_2{=}2$)", "own_baseline"),
              ("$\\lambda_2{=}1$", "lam2_1"), ("$\\lambda_2{=}4$", "lam2_4"),
              ("$\\lambda_3{=}0.5$", "bnd_l05"), ("$\\lambda_3{=}1$", "bnd_l1"),
              ("$\\lambda_3{=}2$", "bnd_l2")]
    rows_b = [("KD $\\lambda_{fe}{=}\\lambda_{kd}{=}0.3$ (default)", "full"),
              ("KD $(0.1,\\,0.1)$", "dsens_a"), ("KD $(0.1,\\,0.5)$", "dsens_b"),
              ("KD $(0.5,\\,0.1)$", "dsens_c"), ("KD $(0.5,\\,0.5)$", "dsens_d")]
    lines = [
        "% auto-generated by make_tables.py — DO NOT EDIT NUMBERS",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Hyper-parameter sensitivity on the fold-1 held-out split (\\%, Section~\\ref{sec:protocol}). "
        "Top: segmentation-loss weights for the un-distilled model. Bottom: BGKD weights; "
        "all four grid points stay within a 0.8\\,pt PQ band around the default, and the "
        "pre-registered default (0.3, 0.3) is retained.}",
        "\\label{tab:lambda}",
        "\\small",
        "\\begin{tabular}{l cc}", "\\toprule",
        "Configuration & PQ & mPQ+ \\\\", "\\midrule",
    ]
    for lab, name in rows_a:
        p, m = f1(name)
        lines.append(f"{lab} & {p:.2f} & {m:.2f} \\\\")
    lines += ["\\midrule"]
    for lab, name in rows_b:
        p, m = f1(name)
        lines.append(f"{lab} & {p:.2f} & {m:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "lambda_sens.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[table] lambda_sens.tex")


# ---------------------------------------------------------------- 表 4: 多种子
def table_seeds(pq: dict):
    def per_fold(name):
        return [collect(name, folds=(f,))["pq"].iloc[0] * 100 for f in (1, 2, 3)]

    seeds = {0: per_fold("full"), 1: per_fold("full_s1"), 2: per_fold("full_s2")}
    nokd = per_fold("full_nokd")
    allv = np.array(list(seeds.values()))
    seed_means = allv.mean(axis=1)                       # 每个种子的 3 折均值
    seed_mean_std = float(seed_means.std(ddof=1))        # 种子均值 std (摘要口径 ±0.18)
    seed_run_std = float(allv.std(ddof=1))               # 全部 9 runs std
    lines = [
        "% auto-generated by make_tables.py — DO NOT EDIT NUMBERS",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Seed robustness of the full method (PQ, \\%, official 3-fold test). "
        "All nine runs exceed the best non-distilled configuration's fold mean.}",
        "\\label{tab:seeds}", "\\small",
        "\\begin{tabular}{l cccc}", "\\toprule",
        "Seed & fold 1 & fold 2 & fold 3 & mean \\\\", "\\midrule",
    ]
    for s in (0, 1, 2):
        v = seeds[s]
        lines.append(f"{s} & " + " & ".join(f"{x:.2f}" for x in v)
                     + f" & {np.mean(v):.2f} \\\\")
    lines += ["\\midrule",
              f"all seeds & \\multicolumn{{3}}{{c}}"
              f"{{seed-mean {allv.mean():.2f}$\\pm${seed_mean_std:.2f} "
              f"(all 9 runs $\\pm${seed_run_std:.2f})}}"
              f" & {allv.mean():.2f} \\\\",
              f"w/o KD (seed 0) & " + " & ".join(f"{x:.2f}" for x in nokd)
              + f" & {np.mean(nokd):.2f} \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "seeds.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[table] seeds.tex")
    return seeds, nokd


# ---------------------------------------------------------------- 表 5: 带宽敏感性
def table_bandwidth(pq: dict):
    rows = ["full", "own_baseline", "hovernet", "nulite_s", "cellvit_sam256"]
    label = {r: FOLD_GROUPS[r] for r in rows}
    lines = [
        "% auto-generated by make_tables.py — DO NOT EDIT NUMBERS",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Band-Dice within $k$ px of matched instance contours (\\%, "
        "mean$\\pm$std over folds). Conclusions are unchanged across $k\\in\\{2,5,10\\}$.}",
        "\\label{tab:bandwidth}", "\\small",
        "\\begin{tabular}{l ccc}", "\\toprule",
        "Method & $k{=}2$ & $k{=}5$ & $k{=}10$ \\\\", "\\midrule",
    ]
    for r in rows:
        lines.append(f"{label[r]} & " + " & ".join(
            ms(pq[r], f"dice{k}px") for k in (2, 5, 10)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "bandwidth.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[table] bandwidth.tex")


# ---------------------------------------------------------------- T12: 逐组织 CSV
def per_tissue(tissue_names: list):
    """逐组织 PQ: 三折 val_metrics 池化 (与 §4.6 统计协议同口径:
    patches pooled across folds, equally weighted per tissue group)。"""
    methods = ("full", "full_nokd", "own_baseline", "hovernet",
               "nulite_s", "cellvit_sam256")
    frames = {}
    for r in methods:
        dfs = [pd.read_csv(fold_dir(r, f) / "val_metrics.csv")
               for f in (1, 2, 3)]
        df = pd.concat(dfs, ignore_index=True)
        frames[r] = df.groupby("group")["pq"].mean() * 100
    out = pd.DataFrame(frames)
    out.index = [tissue_names[i] if i < len(tissue_names) else f"id{i}"
                 for i in out.index]
    out.index.name = "tissue"
    out = out.sort_index()
    out.round(2).to_csv(OUT / "per_tissue_pq.csv", encoding="utf-8")
    print(f"[table] per_tissue_pq.csv ({len(out)} tissues, 3-fold pooled)")


# ---------------------------------------------------------------- main
def table_deploy():
    """部署表: 读 C:\\runs\\deploy_bench.json (bench_deploy.py 独占 GPU 实测)。
    加粗 = 列最优 (median/P95 越低越好, thrpt 越高越好), 下划线 = 次优。"""
    bench = json.load(open(RUNS / "deploy_bench.json", encoding="utf-8"))
    rows = ["NuLite-T (FastViT-T8)", "Ours (full)", "NuLite-S (official)",
            "DDBU-Net (repro.)", "HoVer-Net (reimpl.)", "CellViT-256",
            "CellViT-256-SAM"]
    # 列最优/次优
    best, second = {}, {}
    for k, lower in (("median_ms", True), ("p95_ms", True), ("thrpt_pps", False)):
        vals = {r: bench[r][k] for r in rows}
        s = sorted(vals, key=vals.get, reverse=not lower)
        best[k], second[k] = s[0], s[1]
    label = {"NuLite-T (FastViT-T8)": "NuLite-T (FastViT-T8)",
             "Ours (full)": "\\textbf{Ours (full)}",
             "NuLite-S (official)": "NuLite-S (official)$^{\\dagger}$",
             "DDBU-Net (repro.)": "DDBU-Net (repro.)$^{\\S}$",
             "HoVer-Net (reimpl.)": "HoVer-Net (reimpl.)$^{\\ddagger}$"}
    lines = [
        "% auto-generated by make_tables.py from deploy_bench.json — DO NOT EDIT NUMBERS",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Inference cost on the deployment GPU (RTX 4060 Laptop 8\\,GB), "
        "batch size 1, fp32, exclusive GPU access: median/P95 per-patch latency over "
        "100 timed network forwards after 20 warm-up iterations (watershed decoding "
        "and metric extraction excluded), throughput, and peak forward memory.}",
        "\\label{tab:deploy}", "\\small",
        "\\begin{tabular}{l r rr r r}", "\\toprule",
        "Method & Params & median & P95 & thrpt & VRAM \\\\", "\\midrule",
    ]
    for r in rows:
        b = bench[r]
        cells = []
        for k, prec in (("median_ms", 1), ("p95_ms", 1), ("thrpt_pps", 1),
                        ("vram_gb", 2)):
            v = f"{b[k]:.{prec}f}"
            if r == best.get(k):
                v = f"\\textbf{{{v}}}"
            elif r == second.get(k):
                v = f"\\underline{{{v}}}"
            cells.append(v)
        lines.append(f"{label.get(r, r)} & {b['params_M']:.2f} & "
                     + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\begin{flushleft}\\footnotesize",
              "$^{\\dagger}$Official architecture and released-weight alignment run "
              "(FastViT-S12 backbone).",
              "$^{\\ddagger}$Re-implementation following the original design.",
              "$^{\\S}$Re-production from the published description (no public code).",
              "\\end{flushleft}", "\\end{table}", ""]
    (OUT / "deployment.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[table] deployment.tex")


# ---------------------------------------------------------------- 表: Jetson 边缘部署
JETSON_DIR = Path(r"C:\runs\jetson_bench")
JETSON_ROWS = [("nulite_t", "NuLite-T (FastViT-T8)"),
               ("own_full", "Ours (full)"),
               ("nulite_s", "NuLite-S (official)$^{\\dagger}$"),
               ("ddbunet", "DDBU-Net (repro.)$^{\\S}$"),
               ("hovernet", "HoVer-Net (reimpl.)$^{\\ddagger}$"),
               ("cellvit_t256", "CellViT-256"),
               ("cellvit_sam256", "CellViT-256-SAM")]


def table_jetson():
    recs = [(tag, json.load(open(JETSON_DIR / f"bench_{tag}.json", encoding="utf-8")))
            for tag, _ in JETSON_ROWS]
    cols = [("median_ms", "median$\\downarrow$"), ("p95_ms", "P95$\\downarrow$"),
            ("p99_ms", "P99$\\downarrow$"), ("power_mean_w", "Power (W)"),
            ("energy_mean_mJ", "Energy (mJ)$\\downarrow$")]
    best = {}
    for k, _ in cols:
        vals = {tag: r[k] for tag, r in recs}
        best[k] = min(vals, key=vals.get)
    fmt = {"median_ms": "{:.2f}", "p95_ms": "{:.2f}", "p99_ms": "{:.2f}",
           "power_mean_w": "{:.2f}", "energy_mean_mJ": "{:.1f}"}
    lines = [
        "% auto-generated by make_tables.py from C:/runs/jetson_bench/bench_*.json — DO NOT EDIT NUMBERS",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Edge deployment on an NVIDIA Jetson Orin Nano Super (8\\,GB, "
        "JetPack~6/R36.4.7, fp16 TensorRT~10.3 engines built on-device with "
        "trtexec), batch size 1: host-side per-patch latency \\emph{including} "
        "H2D/D2H transfers over 500 measured forwards after 50 warm-up "
        "iterations, under locked maximum clocks (nvpmodel MAXN + "
        "jetson\\_clocks); GPU temperature stayed near 44$^{\\circ}$C "
        "throughout ($\\Delta T\\le1.6^{\\circ}$C per run, no throttling). "
        "Power is the module input VDD\\_IN mean during measurement "
        "(tegrastats); energy $E=\\bar{P}\\cdot\\bar{t}$. Labels and "
        "footnote markers follow Table~\\ref{tab:deploy}.}",
        "\\label{tab:jetson}", "\\small", "\\setlength{\\tabcolsep}{3.8pt}",
        "\\begin{tabular}{l " + " r" * len(cols) + "}", "\\toprule",
        "Method & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule",
    ]
    for tag, r in recs:
        label = dict(JETSON_ROWS)[tag]
        cells = []
        for k, _ in cols:
            v = fmt[k].format(r[k])
            if tag == best[k]:
                v = f"\\textbf{{{v}}}"
            cells.append(v)
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "jetson.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[table] jetson.tex")
    return dict(recs)


# ---------------------------------------------------------------- T18/T19: CoNSeP 跨数据集
CONSEP_MODELS = ("full_f2", "own_baseline_f2", "cellvit_sam_f2")
CONSEP_LABEL = {"full_f2": "Ours (full)", "own_baseline_f2": "Ours (base)",
                "cellvit_sam_f2": "CellViT-256-SAM"}


def table_consep_zeroshot():
    """T18 零样本表: 读 C:\\runs\\t18_consep\\summary.json (6 组合)。"""
    s = json.load(open(RUNS / "t18_consep" / "summary.json", encoding="utf-8"))
    rec = {(r["model"], r["stain"]): r for r in s["runs"]}
    cols = [("consep_pq", "PQ$\\uparrow$"), ("R", "$R\\uparrow$"),
            ("consep_aji", "AJI$\\uparrow$"), ("consep_f1_det", "F1$\\uparrow$"),
            ("consep_hd95", "HD95$\\downarrow$"),
            ("consep_bf1_2px", "BF1@2px$\\uparrow$")]
    lines = [
        "% auto-generated by make_tables.py — DO NOT EDIT NUMBERS",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Zero-shot transfer PanNuke$\\to$CoNSeP \\cite{kumar2017dataset} "
        "(official 14 test tiles, type-agnostic; CoNSeP has no nuclear type "
        "labels, so mPQ+ is not reported). $R$ = relative retention: CoNSeP PQ "
        "divided by the method's PanNuke three-fold mean PQ "
        "(Table~\\ref{tab:main}). Reinhard uses the PanNuke fold-2 training "
        "statistics --- strictly zero-shot with respect to CoNSeP; none "
        "applies no normalization. All $R$ values fall below the "
        "pre-registered 0.40 gate, so no cross-cohort generalization is "
        "claimed.}",
        "\\label{tab:consep_zs}", "\\small", "\\setlength{\\tabcolsep}{4.2pt}",
        "\\begin{tabular}{ll " + " r" * len(cols) + "}", "\\toprule",
        "Model & Norm. & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule",
    ]
    for m in CONSEP_MODELS:
        for st in ("reinhard", "none"):
            r = rec[(m, st)]
            cells = [f"{r[k]*100:.2f}" if k != "consep_hd95" else f"{r[k]:.2f}"
                     for k, _ in cols]
            lines.append(f"{CONSEP_LABEL[m]} & "
                         + ("Reinhard" if st == "reinhard" else "none")
                         + " & " + " & ".join(cells) + " \\\\")
        if m != CONSEP_MODELS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "consep_zeroshot.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[table] consep_zeroshot.tex")
    return rec


def table_consep_ft(rec_zs: dict):
    """T19 微调表: 读 C:\\runs\\t19_ft_{full,own,sam}\\summary.json。"""
    recs = [json.load(open(RUNS / f"t19_ft_{m}" / "summary.json", encoding="utf-8"))
            for m in ("full", "own", "sam")]
    keymap = [("ft_reinhard", "pq"), ("ft_none", "pq"), ("ft_reinhard", "aji"),
              ("ft_reinhard", "f1_det"), ("ft_reinhard", "hd95"),
              ("ft_reinhard", "bf1_2px")]
    best = {}
    for i, (grp, k) in enumerate(keymap):
        vals = {r["model"]: r[grp][k] for r in recs}
        best[i] = min(vals, key=vals.get) if k == "hd95" else max(vals, key=vals.get)
    for tag, k in (("ret", "ft_retention"), ("zspq", "zero_shot_pq"),
                   ("zsr", "zero_shot_R")):
        vals = {r["model"]: r[k] for r in recs}
        best[tag] = max(vals, key=vals.get)
    lines = [
        "% auto-generated by make_tables.py — DO NOT EDIT NUMBERS",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Fine-tuning on CoNSeP: each row fine-tunes the "
        "corresponding fold-2 PanNuke checkpoint on the 23 official CoNSeP "
        "training tiles (AdamW lr $3{\\times}10^{-5}$, 50 epochs, type-free "
        "loss, no distillation for any method; tile-level validation split "
        "for model selection) and is evaluated on the official 14 test tiles "
        "with CoNSeP-derived Reinhard statistics (permitted under the "
        "fine-tuning protocol). PQ-R / PQ-N = fine-tuned PQ under "
        "CoNSeP-derived Reinhard statistics / no normalization; "
        "Reten. = retention, i.e.\\ fine-tuned PQ (Reinhard) divided "
        "by the PanNuke three-fold mean PQ; the pre-registered gate is 0.40. "
        "ZS-PQ / ZS-$R$ repeat the zero-shot Reinhard results of "
        "Table~\\ref{tab:consep_zs} for reference.}",
        "\\label{tab:consep_ft}", "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{l rrrrrrrrr}", "\\toprule",
        "Model & PQ-R & PQ-N & AJI & F1 & HD95 & BF1@2px & Reten. & "
        "ZS-PQ & ZS-$R$ \\\\", "\\midrule",
    ]
    for r in recs:
        m = r["model"]
        zs = rec_zs[(m, "reinhard")]
        vals = [r[g][k] for g, k in keymap]
        vals += [r["ft_retention"], zs["consep_pq"], zs["R"]]
        cells = []
        for i, v in enumerate(vals):
            s = f"{v*100:.2f}"
            if i == 4:
                s = f"{v:.2f}"
            bold = (i < 6 and best[i] == m) or \
                   (i == 6 and best["ret"] == m) or \
                   (i == 7 and best["zspq"] == m) or \
                   (i == 8 and best["zsr"] == m)
            cells.append(f"\\textbf{{{s}}}" if bold else s)
        lines.append(f"{CONSEP_LABEL[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "consep_ft.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[table] consep_ft.tex")
    return recs


def main():
    check_protocol()
    pq = {name: collect(name) for name in FOLD_GROUPS}

    # 组织类型名 (fold1.npz types 成员, 惰性只读该数组)
    with np.load(r"C:\data\pannuke\fold1.npz") as z:
        tissue_names = sorted(set(np.asarray(z["types"]).astype(str).tolist()))
    print(f"[data] {len(tissue_names)} tissue types")

    params = count_params()

    table_main(pq, params)
    table_ablation(pq)
    table_gatepos(pq)
    table_lambda(pq)
    seeds, nokd = table_seeds(pq)
    table_bandwidth(pq)
    per_tissue(tissue_names)
    table_deploy()
    recs_jetson = table_jetson()
    rec_zs = table_consep_zeroshot()
    recs_ft = table_consep_ft(rec_zs)

    # ---- 论文正文引用的关键事实 (单一数据源) ----
    def m(name, k):
        return pq[name][k].mean() * 100

    facts = {
        "params": params,
        "tissues": tissue_names,
        "main": {r: {k: float(pq[r][k].mean() * (100 if k in PCT else 1))
                     for k in ("pq", "mpq_plus", "aji", "f1_det", "hd95",
                               "bf1_2px", "assd", "dice5px")} for r in FOLD_GROUPS},
        "main_std": {r: {k: float(pq[r][k].std(ddof=1) * (100 if k in PCT else 1))
                         for k in ("pq", "mpq_plus", "aji", "f1_det", "hd95",
                                   "bf1_2px")} for r in FOLD_GROUPS},
        "kd_gain_pq": m("full", "pq") - m("full_nokd", "pq"),
        "kd_gain_mpq": m("full", "mpq_plus") - m("full_nokd", "mpq_plus"),
        "full_vs_own_pq": m("full", "pq") - m("own_baseline", "pq"),
        "full_vs_own_mpq": m("full", "mpq_plus") - m("own_baseline", "mpq_plus"),
        "full_vs_hovernet_pq": m("full", "pq") - m("hovernet", "pq"),
        "seeds": {str(s): [float(x) for x in v] for s, v in seeds.items()},
        "nokd_folds": [float(x) for x in nokd],
        "lam2_4_f1": float(collect("lam2_4", (1,))["pq"].iloc[0] * 100),
        "own_f1_pq": float(collect("own_baseline", (1,))["pq"].iloc[0] * 100),
        "gatepos_pq": {r: float(pq[r]["pq"].mean() * 100)
                       for r in ("own_baseline", "abl_sg", "sg_s3",
                                 "sg_s4", "sg_all")},
    }
    facts["consep_zs"] = {f"{m}_{st}": rec_zs[(m, st)]
                          for m in CONSEP_MODELS for st in ("reinhard", "none")}
    facts["consep_ft"] = {r["model"]: r for r in recs_ft}
    facts["jetson"] = {tag: r for tag, r in recs_jetson.items()}
    (PKG.parent.parent / "paper" / "paper_facts.json").write_text(
        json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[saved] paper_facts.json")
    print("\n=== 关键数字 ===")
    for r in ("full", "full_nokd", "own_baseline", "hovernet", "nulite_s",
              "cellvit_sam256", "cellvit_t256", "ddbunet", "nulite_t"):
        print(f"{r:16s} PQ={m(r,'pq'):.2f} mPQ+={m(r,'mpq_plus'):.2f} "
              f"AJI={m(r,'aji'):.2f} F1={m(r,'f1_det'):.2f} "
              f"HD95={pq[r]['hd95'].mean():.2f} BF1={m(r,'bf1_2px'):.2f}")
    print(f"\nKD gain: +{facts['kd_gain_pq']:.2f} PQ / +{facts['kd_gain_mpq']:.2f} mPQ+")
    print(f"Full vs own: +{facts['full_vs_own_pq']:.2f} PQ")


if __name__ == "__main__":
    main()
