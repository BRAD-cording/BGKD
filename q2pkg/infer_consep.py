#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T18: CoNSeP 跨数据集零样本泛化推理 (执行包 W12, CMIG 门槛实验)
================================================================
协议:
  * 模型 = PanNuke 训练 checkpoint.pt 末轮权重, 零样本直推 CoNSeP Test (27 tiles)
  * 推理: 1000×1000 tile, 滑窗 256/stride 128, 重叠区 sigmoid 概率 + hv 回归均值
  * 染色双口径:
      reinhard — 参考统计 = 该 run 自身 PanNuke 训练折 stain_stats.json (严格零样本, 主口径)
      none     — 原图直推 (对照口径)
  * 评估: type-agnostic (gt_type=全 1, pred_type=全 1) → PQ/AJI/F1/HD95/BF1@2px;
    mPQ+ 因类系不同不报 (CoNSeP 14 类与 PanNuke 5 类不可映射)
  * 聚合: 逐 tile nanmean (与 PanNuke summary.json 口径一致) + micro PQ/F1 副口径
  * 产出: <out>/consep_<model>_<stain>.csv 逐 tile 指标
          <out>/consep_<model>_<stain>_summary.json 单组合汇总
          <out>/summary.json 全局保持率报告 (R = CoNSeP PQ / PanNuke 3 折均值 PQ;
          全部 R<0.40 → verdict=fallback_single_dataset, 供 W12 决策)
  * 幂等: per-combo summary 已存在自动跳过; GPU 任务重复入队安全

用法:
  # CPU 冒烟 (主链运行期安全):
  python infer_consep.py --data C:/data/consep/consep_test.npz \
      --runs C:/runs/full_f2 --stains none --out C:/runs/t18_smoke \
      --device cpu --max-tiles 2
  # 队列全量 (3 runs × 2 stains + 保持率报告):
  python infer_consep.py --data C:/data/consep/consep_test.npz --out C:/runs/t18_consep
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                                                # runner/手动双环境编码兜底
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from models import build_model                      # noqa: E402
from pannuke_data import apply_reinhard             # noqa: E402
from metrics_nuclei import (decode_instances,       # noqa: E402
                            instance_metrics_row)

KEY_METRICS = ("pq", "aji", "f1_det", "hd95", "bf1_2px")


# ---------------------------------------------------------------- 模型装载
def load_model(run_dir: Path, device: str):
    args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    model = build_model(args["model"],
                        base=args.get("base", 32),
                        use_sobel_gate=bool(args.get("sobel_gate", False)),
                        use_rab=bool(args.get("rab", False)),
                        sobel_tap=args.get("sobel_tap") or "s2")
    ckpt = run_dir / "checkpoint.pt"
    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    if any(k.startswith("sobel_gate.") for k in sd):
        # P1 位置消融改造前的旧键名 (单 gate = s2 = 历史行为): 重映射到新 ModuleDict
        sd = {k.replace("sobel_gate.", "sobel_gates.s2.", 1): v
              for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    n_ep = int(ck.get("epoch", -1))
    print(f"[model] {run_dir.name}: {args['model']} (epoch {n_ep}, "
          f"sg={args.get('sobel_gate', False)} rab={args.get('rab', False)})",
          flush=True)
    return model, args


def load_stain_stats(run_dir: Path, args) -> dict | None:
    """reinhard 口径必须复用该 run 训练期的 PanNuke 训练折参考统计。"""
    if args.get("stain", "reinhard") != "reinhard":
        return None
    p = run_dir / "stain_stats.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} 缺失 — run 未用 reinhard, 无法执行主口径")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 滑窗推理
@torch.no_grad()
def infer_tile(model, img_u8: np.ndarray, device: str, win=256, stride=128,
               batch=8):
    """(H,W,3) uint8 → (prob (H,W) float32, hv (2,H,W) float32) 重叠平均。"""
    H, W = img_u8.shape[:2]
    img = torch.from_numpy(img_u8.astype(np.float32) / 255.0) \
               .permute(2, 0, 1).unsqueeze(0)                   # (1,3,H,W)
    ys = list(range(0, H - win + 1, stride))
    xs = list(range(0, W - win + 1, stride))
    if ys[-1] != H - win:
        ys.append(H - win)
    if xs[-1] != W - win:
        xs.append(W - win)
    windows = [(y, x) for y in ys for x in xs]

    prob_sum = torch.zeros((1, H, W), dtype=torch.float32)
    hv_sum = torch.zeros((2, H, W), dtype=torch.float32)
    cnt = torch.zeros((H, W), dtype=torch.float32)
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch]
        x = torch.cat([img[:, :, y:y + win, x0:x0 + win] for y, x0 in chunk], 0)
        x = x.to(device, non_blocking=True)
        if device == "cuda":
            x = x.to(memory_format=torch.channels_last)
        with torch.autocast(device.split(":")[0], dtype=amp_dtype,
                            enabled=device == "cuda"):
            out = model(x)
        p = torch.sigmoid(out["np_logits"].float()).cpu()[:, 0]     # (b,win,win)
        hv = out["hv"].float().cpu()                                # (b,2,win,win)
        for j, (y, x0) in enumerate(chunk):
            prob_sum[:, y:y + win, x0:x0 + win] += p[j]
            hv_sum[:, y:y + win, x0:x0 + win] += hv[j]
            cnt[y:y + win, x0:x0 + win] += 1
    prob = (prob_sum / cnt)[0].numpy()
    hv = (hv_sum / cnt.unsqueeze(0)).numpy()
    return prob, hv


# ---------------------------------------------------------------- 单组合评估
def eval_combo(model, stain_stats, data: dict, stain: str, device: str,
               batch: int, max_tiles: int, tag: str, out_dir: Path):
    images, inst_masks = data["images"], data["inst_masks"]
    tile_ids = [str(t) for t in data["tile_ids"]]
    n = len(images) if max_tiles <= 0 else min(max_tiles, len(images))
    rows, micro = [], {"tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0}
    t0 = time.time()
    for i in range(n):
        img = images[i]
        if stain == "reinhard":
            img = apply_reinhard(img, stain_stats)
        prob, hv = infer_tile(model, img, device, batch=batch)
        pred = decode_instances(prob, hv, np_thresh=0.5)
        gt_inst = inst_masks[i].astype(np.int64)
        ones = np.ones(gt_inst.shape, np.uint8)          # type-agnostic: 全 1
        row = instance_metrics_row(gt_inst, ones, pred, ones,
                                   patch_id=tile_ids[i], group=-1)
        rows.append(row)
        tp = row["tp"]
        fp, fn = row["n_pred"] - tp, row["n_gt"] - tp
        micro["tp"] += tp; micro["fp"] += fp; micro["fn"] += fn
        if not np.isnan(row.get("pq", np.nan)):
            micro["sum_iou"] += row["pq"] * (tp + 0.5 * fp + 0.5 * fn)
        print(f"  [{tag}] {tile_ids[i]}: pq={row.get('pq', float('nan')):.4f} "
              f"n_gt={row['n_gt']} n_pred={row['n_pred']}", flush=True)

    tile_mean = {k: float(np.nanmean([r.get(k, np.nan) for r in rows]))
                 for k in KEY_METRICS}
    den_pq = micro["tp"] + 0.5 * micro["fp"] + 0.5 * micro["fn"]
    micro_pq = micro["sum_iou"] / den_pq if den_pq else float("nan")
    den = 2 * micro["tp"] + micro["fp"] + micro["fn"]
    micro_f1 = 2 * micro["tp"] / den if den else float("nan")
    summary = {"model": tag, "stain": stain, "n_tiles": n,
               "tile_mean": tile_mean,
               "micro": {"pq": micro_pq, "f1": micro_f1},
               "wall_sec": round(time.time() - t0, 1)}
    csv_p = out_dir / f"consep_{tag}_{stain}.csv"
    keys = ["patch_id", "group", "n_gt", "n_pred", "tp", *KEY_METRICS, "assd"]
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out_dir / f"consep_{tag}_{stain}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[combo] {tag}/{stain}: PQ(tile-mean)={tile_mean['pq']:.4f} "
          f"micro_PQ={micro_pq:.4f} AJI={tile_mean['aji']:.4f} "
          f"F1={tile_mean['f1_det']:.4f} HD95={tile_mean['hd95']:.2f} "
          f"BF1={tile_mean['bf1_2px']:.4f} ({summary['wall_sec']}s)", flush=True)
    return summary


# ---------------------------------------------------------------- 保持率报告
# 早期 (W1) 目录命名: own 基线 fold-1 存为 w1_own_f1 (PQ 0.2282), f2/f3 为 own_baseline_*
LEGACY_FOLDS = {"own_baseline": ["w1_own_f1", "own_baseline_f2", "own_baseline_f3"]}


def pannuke_ref_pq(run_dir: Path) -> float:
    """PanNuke 3 折 tile-mean PQ (与 summary.json 口径一致)。"""
    m = re.match(r"^(.*)_f\d$", run_dir.name)
    base = m.group(1) if m else run_dir.name
    names = LEGACY_FOLDS.get(base) or [f"{base}_f{f}" for f in (1, 2, 3)]
    vals = []
    for nm in names:
        p = run_dir.parent / nm / "summary.json"
        if p.exists():
            vals.append(float(json.loads(p.read_text())["pq"]))
    if not vals:
        raise FileNotFoundError(f"{run_dir.parent}/{base}_f*/summary.json 无折结果")
    if len(vals) < len(names):
        print(f"[warn] {base}: 仅 {len(vals)}/{len(names)} 折 summary 可用 "
              f"({'/'.join(n for n in names if (run_dir.parent / n / 'summary.json').exists())})",
              flush=True)
    return float(np.mean(vals))


def build_report(combos, run_dirs, out_dir: Path):
    report = {"runs": [], "verdict_threshold": 0.40}
    all_r = []
    for run_dir, stain in run_dirs:
        s = combos.get((run_dir.name, stain))
        if s is None:
            continue
        ref = pannuke_ref_pq(run_dir)
        pq = s["tile_mean"]["pq"]
        r = pq / ref if ref > 0 else float("nan")
        entry = {"model": run_dir.name, "stain": stain,
                 "pannuke_pq": ref, "consep_pq": pq, "R": r,
                 **{f"consep_{k}": s["tile_mean"][k] for k in KEY_METRICS}}
        if stain == "reinhard":
            all_r.append(r)
        report["runs"].append(entry)
    by_model = {}
    for e in report["runs"]:
        by_model.setdefault(e["model"], {})[e["stain"]] = e["R"]
    for mdl, d in by_model.items():
        d["delta_R"] = (d.get("reinhard", float("nan")) -
                        d.get("none", float("nan")))
    report["retention_by_model"] = by_model
    report["verdict"] = ("fallback_single_dataset"
                         if (all_r and all(r < report["verdict_threshold"]
                                           for r in all_r))
                         else "proceed_cross_dataset")
    (out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    return report


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="consep_test.npz (convert_consep.py)")
    ap.add_argument("--runs", nargs="*", default=[
        r"C:\runs\full_f2", r"C:\runs\own_baseline_f2", r"C:\runs\cellvit_sam_f2"],
        help="checkpoint 所在 run 目录 (含 args.json/stain_stats.json)")
    ap.add_argument("--stains", nargs="*", default=["reinhard", "none"])
    ap.add_argument("--out", default=r"C:\runs\t18_consep")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8, help="滑窗批大小 (SAM 推理 8 足够)")
    ap.add_argument("--max-tiles", type=int, default=0, help=">0 仅前 N tile (冒烟)")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(a.data)
    print(f"[data] {a.data}: {len(data['images'])} tiles "
          f"shape={data['images'].shape[1:]} inst_max="
          f"{int(data['inst_masks'].max())}", flush=True)

    combos = {}
    for rd in a.runs:
        run_dir = Path(rd)
        model, args = load_model(run_dir, a.device)
        stain_stats = load_stain_stats(run_dir, args)
        tag = run_dir.name
        for stain in a.stains:
            sm = out_dir / f"consep_{tag}_{stain}_summary.json"
            if sm.exists():
                combos[(tag, stain)] = json.loads(sm.read_text())
                print(f"[skip] {sm.name} 已存在", flush=True)
                continue
            combos[(tag, stain)] = eval_combo(
                model, stain_stats, data, stain, a.device,
                a.batch, a.max_tiles, tag, out_dir)
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()

    report = build_report(combos, [(Path(r), s) for r in a.runs
                                   for s in a.stains], out_dir)
    for e in report["runs"]:
        print(f"[report] {e['model']:18s} {e['stain']:9s} "
              f"CoNSeP PQ={e['consep_pq']:.4f} PanNuke PQ={e['pannuke_pq']:.4f} "
              f"R={e['R']:.3f}", flush=True)
    for mdl, d in report["retention_by_model"].items():
        print(f"[report] {mdl}: ΔR(reinhard-none)={d['delta_R']:+.3f}", flush=True)
    print(f"[verdict] {report['verdict']}", flush=True)


if __name__ == "__main__":
    main()
