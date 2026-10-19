#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CoNSeP Test (NucFuseRank original_split tif+npy) → 管线约定 consep_test.npz  (T18, W12)
=================================================================================
实测格式 (NucFuseRank Kaggle 镜像, 8.86GB zip):
  original_split/original_split/CoNSeP/test/test_{1..14}.tif   (1000,1000,3) uint8
  original_split/original_split/CoNSeP/test/test_{1..14}.npy   (1000,1000) float64
      实例图: 0=背景, 1..K 逐 tile 重编号 (官方 inst_map 同构, 无 type_map → type-agnostic)
官方 Warwick 直链有登录墙, NucFuseRank 的 original_split 保留官方 test tile 编号, 14 tiles。

本脚本: 打包为 <out>/consep_test.npz
  {images uint8, inst_masks uint16, type_maps uint8(全 0 占位), tile_ids}
T18 评估为 type-agnostic (只报 PQ/AJI/F1/HD95/BF1, 不报 mPQ+), type_maps 仅存档。

用法:
  python convert_consep.py --src <.../original_split/original_split/CoNSeP/test> \
      --out C:\data\consep
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="CoNSeP test 目录 (含 test_*.tif/.npy)")
    ap.add_argument("--out", required=True, help="输出目录 (consep_test.npz)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "consep_test.npz"
    if dst.exists():
        print(f"[skip] {dst} 已存在")
        return

    tifs = sorted(src.glob("test_*.tif"),
                  key=lambda p: int(p.stem.split("_")[1]))
    if not tifs:
        raise FileNotFoundError(f"{src} 下无 test_*.tif")
    images, inst, ids = [], [], []
    for p in tifs:
        npy = p.with_suffix(".npy")
        if not npy.exists():
            raise FileNotFoundError(npy)
        img = np.asarray(Image.open(p))
        m = np.load(npy)
        assert img.shape == (1000, 1000, 3) and img.dtype == np.uint8, (p.name, img.shape)
        assert m.shape == (1000, 1000), (npy.name, m.shape)
        m = m.astype(np.int64)
        i_max = int(m.max())
        assert i_max < 2 ** 16, (npy.name, "实例 ID 超 uint16")
        # ID 规整: 去空洞重编号 (官方 inst_map 1..K 连续, 此处仅保险)
        uniq = np.unique(m)
        uniq = uniq[uniq > 0]
        remap = np.zeros(i_max + 1, np.uint16)
        remap[uniq] = np.arange(1, len(uniq) + 1, dtype=np.uint16)
        images.append(img)
        inst.append(remap[m])
        ids.append(p.stem)
        print(f"  {p.name}: {len(uniq)} 实例 (raw max {i_max})", flush=True)
    np.savez_compressed(dst,
                        images=np.stack(images),
                        inst_masks=np.stack(inst),
                        type_maps=np.zeros((len(images), 1000, 1000), np.uint8),
                        tile_ids=np.array(ids))
    mb = dst.stat().st_size / 1e6
    n_inst = [int(x.max()) for x in inst]
    print(f"[done] {dst.name}: {mb:.0f} MB, N={len(images)} tiles, "
          f"实例数/Tile: min={min(n_inst)} max={max(n_inst)}")


if __name__ == "__main__":
    main()
