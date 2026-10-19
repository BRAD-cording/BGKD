# -*- coding: utf-8 -*-
r"""T19 prep: 从 NucFuseRank zip 提取 CoNSeP 官方 train 27 tiles → consep_finetune.npz
+ CoNSeP 染色统计 (train 前 23 tiles)。

微调协议 (预注册):
  * train = train_1..train_23 (官方编号), 内部 val = train_24..train_27 (tile 级划分, 防泄漏)
  * test = 官方 test_1..test_14 (T18 已转换, 不动)
  * 染色归一化: CoNSeP train 23 tiles 统计 (微调协议下合法, 区别于 T18 严格零样本口径)
输出: C:\data\consep\consep_finetune.npz (images uint8/inst uint16/tile_ids)
      C:\data\consep\stain_stats_consep.json
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pannuke_data import fit_reinhard  # noqa: E402

ZIP = Path(r"C:\data\consep\nucfuserank.zip")
PREFIX = "custom_split/custom_split/CoNSeP/train/"
OUT_NPZ = Path(r"C:\data\consep\consep_finetune.npz")
OUT_STATS = Path(r"C:\data\consep\stain_stats_consep.json")
N_TRAIN_TILES = 23                                        # train_1..23 / val_24..27


def main():
    from PIL import Image
    z = zipfile.ZipFile(ZIP)
    tifs = sorted((n for n in z.namelist()
                   if n.startswith(PREFIX) and n.endswith(".tif")),
                  key=lambda s: int(Path(s).stem.split("_")[1]))
    assert len(tifs) == 27, f"expect 27 train tiles, got {len(tifs)}"
    print(f"[prep] found {len(tifs)} train tiles", flush=True)

    images, insts, ids = [], [], []
    for name in tifs:
        tid = int(Path(name).stem.split("_")[1])
        img = np.asarray(Image.open(io.BytesIO(z.read(name))).convert("RGB"),
                         dtype=np.uint8)
        mname = name.replace(".tif", ".npy")
        inst = np.load(io.BytesIO(z.read(mname)), allow_pickle=True)
        inst = np.ascontiguousarray(inst.astype(np.int64))
        # 实例 ID 紧凑重编号 (0=bg)
        uniq = np.unique(inst)
        uniq = uniq[uniq > 0]
        remap = np.zeros(int(inst.max()) + 1, dtype=np.uint16)
        remap[uniq] = np.arange(1, len(uniq) + 1, dtype=np.uint16)
        inst_c = remap[inst]
        images.append(img)
        insts.append(inst_c)
        ids.append(tid)
        print(f"  tile {tid:2d}: {img.shape} inst={len(uniq)}", flush=True)

    images = np.stack(images)                             # (27,1000,1000,3) uint8
    insts = np.stack(insts)                               # (27,1000,1000) uint16
    ids = np.asarray(ids, dtype=np.int32)
    np.savez_compressed(OUT_NPZ, images=images, inst_masks=insts, tile_ids=ids)
    print(f"[prep] wrote {OUT_NPZ} ({OUT_NPZ.stat().st_size / 2**20:.0f} MB)",
          flush=True)

    # 染色统计: 仅 train 23 tiles (val/test 不可见)
    fit_reinhard(images[:N_TRAIN_TILES], out_json=str(OUT_STATS))
    print(f"[prep] wrote {OUT_STATS} (fit on {N_TRAIN_TILES} train tiles)",
          flush=True)


if __name__ == "__main__":
    main()
