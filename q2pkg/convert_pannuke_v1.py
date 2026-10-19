#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PanNuke v1 官方 zip (分体 float64 npy) → 管线约定 fold{f}.npz
================================================================
官方 Warwick 直链 fold_{1,2,3}.zip 解压结构 (实测):
  Fold {f}/images/fold{f}/images.npy  (N,256,256,3) float64 0..255  ~4GB/折
  Fold {f}/images/fold{f}/types.npy   (N,)        组织类型字符串索引
  Fold {f}/masks/fold{f}/masks.npy    (N,256,256,6) float64 实例ID(1-based, 通道0背景) ~8GB/折
  (README 掩膜通道: 0 Neoplastic / 1 Inflammatory / 2 Connective / 3 Dead /
   4 Epithelial / 5 no-label —— 与 pannuke_data.PANNUKE_CHANNELS 一致)

本脚本: mmap 读入 → 降精度 (images→uint8, masks→uint16) → 打包为
  data/fold{f}.npz {images uint8, types, masks uint16}
单折内存峰值 ~2GB (uint16 masks) + page cache, 32GB 机器安全。
用法:
  python convert_pannuke_v1.py --data-root <解压根目录> --out <输出目录> [--folds 1 2 3]
"""
import argparse
from pathlib import Path

import numpy as np


def convert_fold(src_root: Path, fold: int, out_dir: Path):
    img_p = src_root / f"Fold {fold}" / "images" / f"fold{fold}" / "images.npy"
    typ_p = src_root / f"Fold {fold}" / "images" / f"fold{fold}" / "types.npy"
    msk_p = src_root / f"Fold {fold}" / "masks" / f"fold{fold}" / "masks.npy"
    for p in (img_p, typ_p, msk_p):
        if not p.exists():
            raise FileNotFoundError(f"缺少 {p}")
    out_p = out_dir / f"fold{fold}.npz"
    if out_p.exists():
        print(f"[skip] {out_p} 已存在")
        return

    imgs = np.load(img_p, mmap_mode="r")
    types = np.load(typ_p, mmap_mode="r")
    masks = np.load(msk_p, mmap_mode="r")
    n = imgs.shape[0]
    assert imgs.shape[1:] == (256, 256, 3), imgs.shape
    assert masks.shape == (n, 256, 256, 6), masks.shape
    assert len(types) == n, (len(types), n)
    print(f"[fold{fold}] N={n}; 转换 images float64→uint8 …")

    images_u8 = np.asarray(imgs, dtype=np.float32).round().clip(0, 255).astype(np.uint8)
    del imgs
    print(f"[fold{fold}] masks float64→uint16 (实例 ID 上限校验) …")
    m_max = float(np.max(masks))
    assert m_max < 2 ** 16, f"实例 ID {m_max} 超 uint16"
    masks_u16 = np.asarray(masks, dtype=np.uint16)
    del masks
    print(f"[fold{fold}] 打包 → {out_p} (savez_compressed, CPU 数分钟) …")
    np.savez_compressed(out_p, images=images_u8, types=types, masks=masks_u16)
    mb = out_p.stat().st_size / 1e6
    print(f"[fold{fold}] 完成: {mb:.0f} MB; types 分布: "
          f"{dict(zip(*[a.tolist() for a in np.unique(np.asarray(types), return_counts=True)]))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="zip 解压根目录 (含 Fold 1/2/3)")
    ap.add_argument("--out", required=True, help="输出目录 (fold{f}.npz)")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.folds:
        convert_fold(Path(a.data_root), f, out)


if __name__ == "__main__":
    main()
