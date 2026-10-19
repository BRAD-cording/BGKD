#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W1 训练入口 —— 自家基线 1 折冒烟 / 后续全部 4060 夜间队列
=============================================================
对应执行包 §1.4 W1 与铁律 §1.5:
  * 固定 --seed; 每 epoch 覆盖存 checkpoint(可断点续训 R-4); epoch 计时+GPU 温度日志
  * PanNuke uint8 全量内存缓存; bfloat16 autocast(Ada 无需 GradScaler); channels_last
  * 染色参考统计仅在训练折估计, 持久化到 run 目录, val 共用同一份 (§2.2 防泄漏)
  * 结束时: val 双轨指标落盘 val_metrics.csv + summary.json
    (含 mean epoch 时间 —— 用于回填 §1.2 全表估时)

用法 (W1 冒烟):
  python scripts/q2pkg/train.py --data-root D:/PanNuke --fold 1 --epochs 50 \
      --out runs/w1_own_f1
消融开关 (W3/W4): --sobel-gate / --rab / --boundary-loss
"""
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
# runner 重定向日志默认 cp1252, 中文 print (如 "[resume] 从 epoch N 续训") 会崩 → 强制 UTF-8
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from boundary_distill import BGKDLoss  # noqa: F401  (W5 蒸馏在此接入)
from metrics_nuclei import instance_metrics_row, decode_instances
from models import BoundaryCompositeLoss, build_model
from pannuke_data import PanNukeDataset, build_fold_split
from train_opt_template import seed_everything  # noqa: F401  (TF32/cudnn 设置在其导入时生效)


def _keep_awake() -> None:
    """防睡眠/防冻结: 本机为 Modern Standby 笔记本, 空闲后 DAM 冻结后台训练
    (2026-09-29 夜实证: 冻结 15.5h 仅推进 6 epochs)。训练进程持系统唤醒锁,
    仅对本进程生效, 不影响日常用机睡眠策略; 进程退出锁自动释放。"""
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_AWAYMODE_REQUIRED = 0x80000000, 0x1, 0x40
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)


def gpu_temp():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return float("nan")


def save_checkpoint(path: Path, state: dict) -> None:
    """原子写 + 上一 epoch 轮转: 进程被杀/掉电最坏丢 1 epoch, 绝不损坏唯一 checkpoint。"""
    tmp, prev = path.with_suffix(".tmp"), path.with_name("checkpoint_prev.pt")
    torch.save(state, tmp)
    if path.exists():
        os.replace(path, prev)
    os.replace(tmp, path)


def _truncate_log(log_path: Path, start_ep: int) -> None:
    """resume 后删除 epoch > start_ep 的残留行(崩溃窗口期可能已多写)。"""
    rows = list(csv.DictReader(open(log_path)))
    if not rows:
        return
    kept = [r for r in rows if int(r["epoch"]) <= start_ep]
    if len(kept) != len(rows):
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(kept)
        print(f"[resume] 清理 epoch_log 残留 {len(rows) - len(kept)} 行")


def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="含 fold1/2/3.npz 的目录")
    ap.add_argument("--model", default="own_baseline")
    ap.add_argument("--fold", type=int, default=1, help="验证折 (1/2/3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stain", choices=["reinhard", "none"], default="reinhard")
    ap.add_argument("--out", default="runs/w1_run")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help=">0 时截断样本数(冒烟用)")
    ap.add_argument("--base", type=int, default=32, help="LightUNet 基础通道")
    ap.add_argument("--sobel-gate", action="store_true")
    ap.add_argument("--sobel-tap", choices=["s2", "s3", "s4", "all"], default="s2",
                    help="SobelGate 接入的 skip 位置 (仅 own_baseline 生效; s2=历史行为)")
    ap.add_argument("--rab", action="store_true")
    ap.add_argument("--boundary-loss", action="store_true")
    ap.add_argument("--lambda-np", type=float, default=1.0)
    ap.add_argument("--lambda-hv", type=float, default=2.0)
    ap.add_argument("--lambda-bnd", type=float, default=1.0)
    ap.add_argument("--distill", action="store_true",
                    help="W5: BGKD 蒸馏 (teacher=NuLite-T 官方权重, student=当前 --model)")
    ap.add_argument("--teacher-ckpt", default=r"C:\data\weights\NuLite-T-Weights.pth")
    ap.add_argument("--lambda-fe", type=float, default=0.3)
    ap.add_argument("--lambda-kd", type=float, default=0.3)
    ap.add_argument("--no-resume", action="store_true")
    return ap.parse_args()


def make_loaders(args):
    train_folds, val_folds = build_fold_split(args.data_root, args.fold)
    stain_json = Path(args.out) / "stain_stats.json"
    ds_train = PanNukeDataset(args.data_root, train_folds, stain=args.stain,
                              stain_stats_json=stain_json, train=True,
                              stain_seed=args.seed, seed=args.seed)
    ds_val = PanNukeDataset(args.data_root, val_folds, stain=args.stain,
                            stain_stats_json=stain_json, train=False)
    if args.limit > 0:
        for d in (ds_train, ds_val):
            d.images = d.images[:args.limit]
            d.masks = d.masks[:args.limit]
            d.types = d.types[:args.limit]
            d.tissue_ids = d.tissue_ids[:args.limit]
            d.fold_of = d.fold_of[:args.limit]
    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          drop_last=True, **common)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, **common)
    return ds_train, ds_val, dl_train, dl_val


@torch.no_grad()
def evaluate(model, dl_val, criterion, device, np_thresh=0.5):
    """val 损失 + 双轨逐 patch 指标行。"""
    model.eval()
    val_loss, rows = 0.0, []
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    for batch in dl_val:
        x = batch["img"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.split(":")[0], dtype=amp_dtype,
                            enabled=device == "cuda"):
            out = model(x)
            loss, _ = criterion(out, {k: (v.to(device) if torch.is_tensor(v) else v)
                                      for k, v in batch.items()})
        val_loss += float(loss) * len(batch["img"])
        np_prob = torch.sigmoid(out["np_logits"].float()).cpu().numpy()[:, 0]
        hv_pred = out["hv"].float().cpu().numpy()
        type_pred = out["type_logits"].float().argmax(1).cpu().numpy()
        for i in range(len(np_prob)):
            rows.append(instance_metrics_row(
                batch["inst"][i].numpy(), batch["type"][i].numpy(),
                decode_instances(np_prob[i], hv_pred[i], np_thresh=np_thresh),
                type_pred[i],
                patch_id=batch["patch_id"][i], group=int(batch["group"][i])))
    return val_loss / max(len(dl_val.dataset), 1), rows


def main():
    args = get_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _keep_awake()
    ds_train, ds_val, dl_train, dl_val = make_loaders(args)
    model = build_model(args.model, base=args.base,
                        use_sobel_gate=args.sobel_gate, use_rab=args.rab,
                        sobel_tap=args.sobel_tap
                        ).to(device).to(memory_format=torch.channels_last)
    criterion = BoundaryCompositeLoss(args.lambda_np, args.lambda_hv, args.lambda_bnd,
                                      use_boundary=args.boundary_loss).to(device)
    teacher, bgkd = None, None
    if args.distill:
        from nulite_adapter import _EMBED_DIMS, build_nulite_t, load_nulite_t_weights
        from boundary_distill import BGKDLoss
        # 官方发布权重 arch=NuLite-T 但 backbone=fastvit_s12 (ckpt config 实测)
        t_bb = "fastvit_s12"
        teacher = build_nulite_t(backbone=t_bb, pretrained=False).to(device).eval()
        rep = load_nulite_t_weights(teacher, args.teacher_ckpt)
        print(f"[distill] teacher=NuLite-T 官方权重({t_bb}) 装载 missing={len(rep['missing'])} "
              f"unexpected={len(rep['unexpected'])}", flush=True)
        b = args.base
        t_dims = _EMBED_DIMS[t_bb]
        bgkd = BGKDLoss(student_channels={"s2": 4 * b, "s3": 8 * b, "s4": 16 * b},
                        teacher_channels={"s2": t_dims[0], "s3": t_dims[1], "s4": t_dims[2]},
                        lambda_fe=args.lambda_fe, lambda_kd=args.lambda_kd).to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_ep, log_path = 0, out / "epoch_log.csv"
    ckpt_path = out / "checkpoint.pt"
    ck = None
    if not args.no_resume:                                   # R-4 断点续训(主→备回退)
        for cand in (ckpt_path, out / "checkpoint_prev.pt"):
            if not cand.exists():
                continue
            try:
                # map_location="cpu": RNG 状态必须留在 CPU(cuda 张量过不了 set_rng_state);
                # opt/sched 状态由 load_state_dict 自动搬回参数所在设备
                ck = torch.load(cand, map_location="cpu", weights_only=False)
                if cand != ckpt_path:
                    print(f"[resume] checkpoint.pt 损坏, 已回退 {cand.name}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"[resume] {cand.name} 加载失败(可能写入中被杀): {e}")
    if ck is not None:
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start_ep = ck["epoch"]
        torch.set_rng_state(ck["rng"])
        print(f"[resume] 从 epoch {start_ep} 续训")
        _truncate_log(log_path, start_ep)
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "lr", "train_loss", "l_seg", "l_hv", "l_bnd", "l_type",
                 "sec", "gpu_temp_c"])

    for ep in range(start_ep, args.epochs):
        model.train()
        ds_train.set_epoch(ep)
        t0, seen, acc = time.perf_counter(), 0, {}
        for batch in dl_train:
            x = batch["img"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                preds = model(x)
                loss, parts = criterion(preds, y)
                if bgkd is not None:
                    with torch.inference_mode():
                        t_out = teacher(x)
                    # 8ch 混合 logits: [np双通道(±z 差值约定, 两侧一致), type 前 6 通道]
                    s_kd = torch.cat([-preds["np_logits"], preds["np_logits"],
                                      preds["type_logits"][:, :6]], dim=1)
                    t_kd = torch.cat([-t_out["np_logits"], t_out["np_logits"],
                                      t_out["type_logits"][:, :6].clone()], dim=1)
                    l_bd, bd_parts = bgkd(preds["feats"],
                                          {k: v.clone() for k, v in t_out["feats"].items()},
                                          s_kd, t_kd, y["inst"])
                    loss = loss + l_bd
                    parts.update(bd_parts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bs = len(batch["img"])
            seen += bs
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + float(v) * bs
        sched.step()
        sec = time.perf_counter() - t0
        train_loss = acc["l_seg"] / max(seen, 1) + acc["l_hv"] / max(seen, 1) \
            + acc["l_bnd"] / max(seen, 1) + 0.5 * acc["l_type"] / max(seen, 1)
        temp = gpu_temp()
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep + 1, f"{sched.get_last_lr()[0]:.2e}", f"{train_loss:.5f}"]
                + [f"{acc[k] / max(seen, 1):.5f}" for k in ("l_seg", "l_hv", "l_bnd", "l_type")]
                + [f"{sec:.1f}", temp])
        save_checkpoint(ckpt_path,
                        {"model": model.state_dict(), "opt": optimizer.state_dict(),
                         "sched": sched.state_dict(), "epoch": ep + 1,
                         "rng": torch.get_rng_state(), "args": vars(args)})
        msg = f"[ep {ep + 1}/{args.epochs}] loss={train_loss:.4f} {sec:.1f}s temp={temp:.0f}C"
        if bgkd is not None:
            msg += f" l_fe={acc.get('l_fe', 0) / max(seen, 1):.4f} l_kd={acc.get('l_kd', 0) / max(seen, 1):.4f}"
        if (ep + 1) % args.val_every == 0 or ep + 1 == args.epochs:
            vl, _ = evaluate(model, dl_val, criterion, device)
            msg += f" val_loss={vl:.4f}"
        print(msg, flush=True)

    # ---- 最终评估落盘 (W1: 也作为解码阈值调优前的基准行) ----
    _, rows = evaluate(model, dl_val, criterion, device)
    with open(out / "val_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    metrics = [k for k in rows[0] if k not in ("patch_id", "group")]
    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in metrics}
    ep_secs = [float(r["sec"]) for r in csv.DictReader(open(log_path))]
    summary["mean_epoch_sec"] = float(np.mean(ep_secs)) if ep_secs else None
    summary["est_hours_per_fold_50ep"] = summary["mean_epoch_sec"] * 50 / 3600 if ep_secs else None
    if bgkd is not None:                                     # 蒸馏 run 元数据
        summary["distill"] = {"teacher": "nulite_t_official",
                              "lambda_fe": args.lambda_fe, "lambda_kd": args.lambda_kd,
                              "mean_l_fe": float(acc.get("l_fe", 0) / max(seen, 1)),
                              "mean_l_kd": float(acc.get("l_kd", 0) / max(seen, 1))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] summary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
