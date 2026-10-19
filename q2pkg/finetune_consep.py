# -*- coding: utf-8 -*-
r"""T19: CoNSeP 微调实验 — PanNuke 预训练权重 → CoNSeP train 微调 → 官方 test 评估。

协议 (预注册, 与 T18 零样本形成 zero-shot vs fine-tuned 对照):
  * 数据: consep_finetune.npz 官方 train 27 tiles (1000^2);
          train = tile 1..23 (随机 256 crop + 随机翻转), 内部 val = tile 24..27 (tile 级划分防泄漏)
  * 微调: 加载 PanNuke run 的 checkpoint (load_model 含旧键名 remap), AdamW lr=3e-5
          (base 1/10), cosine, 50 epochs, bf16 autocast; 不开蒸馏 (三模型公平)
  * loss: NP(BCE+Dice) + HV(核内 masked MSE) + 边界带 BCE (原配置) + ddbunet edge;
          无 type 项 (CoNSeP 无类型标注, 评估 type-agnostic)
  * 归一化: CoNSeP train 23 tiles Reinhard 统计 (微调协议合法, 区别于 T18 严格零样本)
  * 模型选择: 每 val_every epochs 在 4 val tiles 滑窗 tile-mean PQ, best.pt (原子写)
  * 评估: 官方 test 14 tiles, 复用 infer_consep.eval_combo (reinhard 主口径 + none 对照)
  * summary.json: ft 指标 + PanNuke 保持率 + T18 零样本对照; 存在即幂等跳过

用法 (队列): python finetune_consep.py --pretrained C:/runs/full_f2 --out C:/runs/t19_ft_full
冒烟 (CPU):  python finetune_consep.py --pretrained C:/runs/full_f2 --out C:/tmp/t19smoke --device cpu --smoke
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from boundary_distill import boundary_band  # noqa: E402
from infer_consep import (eval_combo, infer_tile, load_model,  # noqa: E402
                          pannuke_ref_pq)
from metrics_nuclei import decode_instances, instance_metrics_row  # noqa: E402
from models import BoundaryCompositeLoss  # noqa: E402
from pannuke_data import apply_reinhard, compute_hv_map  # noqa: E402

WIN, STRIDE = 256, 128
N_TRAIN_TILES = 23                                        # tile 1..23 / val 24..27
T18_SUMMARY = Path(r"C:\runs\t18_consep\summary.json")


class ConSePFtDataset(Dataset):
    """官方 train tiles → 随机 256 crop + 翻转; 无 type 标签 (type-agnostic 微调)。"""

    def __init__(self, images, insts, stats, crops_per_tile=32, seed=0):
        self.images, self.insts = images, insts
        self.stats = stats
        self.crops_per_tile = crops_per_tile
        self.epoch = 0
        self.seed = seed

    def __len__(self):
        return len(self.images) * self.crops_per_tile

    def __getitem__(self, i):
        t, _ = divmod(i, self.crops_per_tile)
        img, inst = self.images[t], self.insts[t]
        H, W = inst.shape
        rng = np.random.default_rng((self.seed, self.epoch, i))
        for _ in range(10):                               # 避免全空 crop
            y0, x0 = int(rng.integers(0, H - WIN + 1)), int(rng.integers(0, W - WIN + 1))
            ic, mc = img[y0:y0 + WIN, x0:x0 + WIN], inst[y0:y0 + WIN, x0:x0 + WIN]
            if (mc > 0).any():
                break
        for axis in (0, 1):
            if rng.random() < 0.5:
                ic, mc = np.flip(ic, axis).copy(), np.flip(mc, axis).copy()
        if self.stats is not None:
            ic = apply_reinhard(ic, self.stats)
        hv = compute_hv_map(mc.astype(np.int64))
        return {
            "img": torch.from_numpy(np.ascontiguousarray(ic.transpose(2, 0, 1))).float() / 255.0,
            "np": torch.from_numpy((mc > 0).astype(np.float32)).unsqueeze(0),
            "hv": torch.from_numpy(hv),
            "inst": torch.from_numpy(mc.astype(np.int64)),
        }


def ft_loss(out, batch, lam_np, lam_hv, lam_bnd, use_boundary, band_px=2):
    """BoundaryCompositeLoss 的 type-free 版 (逻辑逐项对齐, 去掉 0.5·type-CE)。"""
    np_gt, inst = batch["np"], batch["inst"]
    device = np_gt.device
    bce = F.binary_cross_entropy_with_logits(out["np_logits"], np_gt, reduction="none")
    l_seg = bce.mean() + BoundaryCompositeLoss._soft_dice(out["np_logits"], np_gt)
    mask = (inst > 0).float()
    l_hv = (F.mse_loss(out["hv"], batch["hv"], reduction="none")
            .mean(dim=1, keepdim=True) * mask).sum() / mask.sum().clamp(min=1.0)
    l_bnd = torch.zeros((), device=device)
    if use_boundary:
        band = boundary_band(inst, band_px)
        l_bnd = (bce * band).sum() / band.sum().clamp(min=1.0)
    l_edge = torch.zeros((), device=device)
    if "edge_logits" in out:
        band_e = boundary_band(inst, band_px)
        with torch.no_grad():
            p_np = torch.sigmoid(out["np_logits"].detach())
            w_hard = 1.0 + (1.0 - (2 * p_np - 1).abs()) * band_e
        l_edge = (F.binary_cross_entropy_with_logits(out["edge_logits"], band_e,
                                                     reduction="none") * w_hard).sum() \
            / band_e.sum().clamp(min=1.0)
    loss = lam_np * l_seg + lam_hv * l_hv + lam_bnd * l_bnd + l_edge
    return loss, {"l_seg": l_seg.item(), "l_hv": l_hv.item(),
                  "l_bnd": l_bnd.item(), "l_edge": l_edge.item()}


@torch.no_grad()
def val_pq(model, images, insts, stats, device, batch=8):
    """4 val tiles 滑窗 → tile-mean PQ (模型选择用)。"""
    model.eval()
    pqs = []
    for img, inst in zip(images, insts):
        if stats is not None:
            img = apply_reinhard(img, stats)
        prob, hv = infer_tile(model, img, device, win=WIN, stride=STRIDE, batch=batch)
        pred = decode_instances(prob, hv, np_thresh=0.5)
        ones = np.ones(inst.shape, np.uint8)
        row = instance_metrics_row(inst.astype(np.int64), ones, pred, ones,
                                   patch_id="val", group=-1)
        pqs.append(row.get("pq", np.nan))
    model.train()
    return float(np.nanmean(pqs))


def save_best(path: Path, model, meta: dict):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), **meta}, tmp)
    os.replace(tmp, path)                                 # 原子写


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True, help="PanNuke run 目录 (args.json+checkpoint.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=r"C:\data\consep\consep_finetune.npz")
    ap.add_argument("--test-data", default=r"C:\data\consep\consep_test.npz")
    ap.add_argument("--stats", default=r"C:\data\consep\stain_stats_consep.json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--wd", type=float, default=None, help="默认继承预训练 args.wd")
    ap.add_argument("--crops-per-tile", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU 冒烟: 2 epochs/4 crops/val 1 tile/test 2 tiles")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "summary.json").exists():                   # 幂等
        print(f"[skip] {out}/summary.json 已存在")
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    dev = args.device
    epochs = 2 if args.smoke else args.epochs
    crops = 4 if args.smoke else args.crops_per_tile
    n_val = 1 if args.smoke else None

    model, pargs = load_model(Path(args.pretrained), dev)
    model.train()
    lam_np = float(pargs.get("lambda_np", 1.0))
    lam_hv = float(pargs.get("lambda_hv", 2.0))
    lam_bnd = float(pargs.get("lambda_bnd", 1.0))
    use_bnd = bool(pargs.get("boundary_loss", False))
    wd = args.wd if args.wd is not None else float(pargs.get("wd", 1e-4))
    stats = json.loads(Path(args.stats).read_text(encoding="utf-8")) \
        if Path(args.stats).exists() else None

    z = np.load(args.data)
    images, insts = z["images"], z["inst_masks"]
    order = np.argsort(z["tile_ids"])                     # 按 tile 编号 1..27
    images, insts = images[order], insts[order]
    tr_im, tr_in = images[:N_TRAIN_TILES], insts[:N_TRAIN_TILES]
    va_im, va_in = images[N_TRAIN_TILES:], insts[N_TRAIN_TILES:]
    if n_val:
        va_im, va_in = va_im[:n_val], va_in[:n_val]
    print(f"[data] train={len(tr_im)} tiles / val={len(va_im)} tiles", flush=True)

    ds = ConSePFtDataset(tr_im, tr_in, stats, crops_per_tile=crops, seed=args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=0, pin_memory=(dev == "cuda"),
                                     drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    amp_dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    best = -1.0
    log_p = out / "epoch_log.csv"
    if not log_p.exists():
        log_p.write_text("epoch,lr,train_loss,l_seg,l_hv,l_bnd,val_pq,sec\n",
                         encoding="utf-8")
    for ep in range(epochs):
        ds.epoch = ep
        t0 = time.time()
        acc, seen = {"l_seg": 0.0, "l_hv": 0.0, "l_bnd": 0.0}, 0
        for batch in dl:
            x = batch["img"].to(dev, non_blocking=True)
            y = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items() if k != "img"}
            if dev == "cuda":
                x = x.to(memory_format=torch.channels_last)
            with torch.autocast(dev.split(":")[0], dtype=amp_dtype,
                                enabled=dev == "cuda"):
                outd = model(x)
                loss, parts = ft_loss(outd, y, lam_np, lam_hv, lam_bnd, use_bnd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            seen += 1
            for k in acc:
                acc[k] += parts[k]
        sched.step()
        sec = time.time() - t0
        vpq = val_pq(model, va_im, va_in, stats, dev) \
            if (ep + 1) % args.val_every == 0 or ep + 1 == epochs else float("nan")
        with open(log_p, "a", encoding="utf-8") as f:
            f.write(f"{ep + 1},{sched.get_last_lr()[0]:.2e},{loss.item():.4f},"
                    f"{acc['l_seg'] / max(seen, 1):.5f},{acc['l_hv'] / max(seen, 1):.5f},"
                    f"{acc['l_bnd'] / max(seen, 1):.5f},{vpq:.4f},{sec:.1f}\n")
        print(f"[ep {ep + 1}/{epochs}] loss={loss.item():.4f} {sec:.1f}s val_pq={vpq:.4f}",
              flush=True)
        if not np.isnan(vpq) and vpq > best:
            best = vpq
            save_best(out / "best.pt", model,
                      {"epoch": ep + 1, "val_pq": vpq, "pretrained": args.pretrained})
            print(f"  [best] val_pq={vpq:.4f} (ep {ep + 1})", flush=True)

    if not (out / "best.pt").exists():                    # 从未触发 best (理论不达)
        save_best(out / "best.pt", model, {"epoch": epochs, "val_pq": best,
                                           "pretrained": args.pretrained})

    # ---------------- test 评估 (best.pt, CoNSeP 统计 reinhard + none 对照) ----
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[test] best.pt = ep{ck['epoch']} val_pq={ck['val_pq']:.4f}", flush=True)
    zt = np.load(args.test_data)
    test_data = {"images": zt["images"], "inst_masks": zt["inst_masks"],
                 "tile_ids": zt["tile_ids"]}
    n_test = 2 if args.smoke else -1
    res = {}
    for stain in ("reinhard", "none"):
        st = stats if stain == "reinhard" else None
        s = eval_combo(model, st, test_data, stain, dev,
                       batch=8, max_tiles=n_test, tag="ft", out_dir=out)
        res[stain] = s["tile_mean"]

    ref_pq = pannuke_ref_pq(Path(args.pretrained))
    ft_ret = res["reinhard"]["pq"] / ref_pq if ref_pq else float("nan")
    summary = {
        "model": Path(args.pretrained).name,
        "pannuke_pq_3fold": ref_pq,
        "ft_reinhard": res["reinhard"], "ft_none": res["none"],
        "ft_retention": ft_ret,
        "best_epoch": ck["epoch"], "best_val_pq": ck["val_pq"],
    }
    if T18_SUMMARY.exists():                              # 零样本对照
        t18 = json.loads(T18_SUMMARY.read_text(encoding="utf-8"))
        for r in t18.get("runs", []):
            if r["model"] == Path(args.pretrained).name and r["stain"] == "reinhard":
                summary["zero_shot_pq"] = r["consep_pq"]
                summary["zero_shot_R"] = r["R"]
                break
    (out / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] {out.name}: ft_PQ={res['reinhard']['pq']:.4f} "
          f"retention={ft_ret:.3f} (zero-shot R="
          f"{summary.get('zero_shot_R', float('nan')):.3f})", flush=True)


if __name__ == "__main__":
    main()
