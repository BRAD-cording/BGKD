# -*- coding: utf-8 -*-
"""PanNuke 数据管线 (W1) —— 对应执行包 §1.4 W1 与 §2.2 防泄漏声明
=====================================================================
输入: 官方 PanNuke v1 三个 fold npz (fold1/2/3.npz), 每个含:
  images : (N,256,256,3) uint8
  types  : (N,) 组织类型字符串 (19 类)
  masks  : (N,256,256,6) 实例通道, 通道顺序固定:
           0=Neoplastic 1=Inflammatory 2=Connective 3=Dead 4=Epithelial 5=No-Label
           每通道内为实例 ID (1-based)

折协议 (官方 3 折交叉, 铁律 R6):
  以 fold v 为验证/测试折时, 训练折 = 其余两折; 超参仅在 fold1-val 调优。

防泄漏设计 (§2.2 "Implementation and leakage control"):
  * Reinhard 染色归一化: 参考统计(LAB 均值/方差)仅在每折组合的【训练折】上估计,
    持久化为 JSON, val/test 用同一参考; --stain none 时完全不启用。
  * 几何增广 (翻转+90°旋转) 对 image/inst/type 同步施加, HV 图在增广后重算。

统计分组说明: PanNuke 公开版不含 slide 溯源, stats_tests.py 的 group_col
默认用组织类型 (tissue type) 作聚合组; 若获得 slide 映射可在 Dataset 处替换。
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "PANNUKE_CHANNELS", "EVAL_CLASSES", "load_fold_npz", "merge_masks",
    "compute_hv_map", "fit_reinhard", "apply_reinhard",
    "build_fold_split", "PanNukeDataset", "make_synthetic_fold",
]

PANNUKE_CHANNELS = ["neoplastic", "inflammatory", "connective", "dead",
                    "epithelial", "no_label"]      # masks 第 5 通道 = 无标注核
EVAL_CLASSES = PANNUKE_CHANNELS[:5]                # mPQ+ 评估类别 (5 类)
N_TYPE = 7                                          # type 头通道: 0=背景, 1..5=五类, 6=no-label

# ---------------------------------------------------------------- npz 读取与合并
def load_fold_npz(path):
    """读取官方 fold npz → dict(images, types, masks)。校验形状与通道数。"""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        if not {"images", "types", "masks"} <= keys:
            raise ValueError(f"{path} 缺少 images/types/masks 键, 实际: {sorted(keys)}")
        images = np.ascontiguousarray(z["images"])
        types = np.asarray(z["types"]).astype(str)
        masks = np.ascontiguousarray(z["masks"])
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"{path} images 形状异常: {images.shape}")
    if masks.ndim != 4 or masks.shape[-1] != 6:
        raise ValueError(f"{path} masks 应为 (N,256,256,6), 实际: {masks.shape}")
    if len(types) != len(images) or len(masks) != len(images):
        raise ValueError(f"{path} images/types/masks 数量不一致")
    return {"images": images, "types": types, "masks": masks}


def merge_masks(masks_i):
    """单 patch 6 通道实例掩膜 → (inst uint32, type uint8)。

    inst: 跨通道唯一实例 ID = v + (c+1)*2**16 (c=通道号, v=通道内 ID, uint16 上限 65535);
    type: 0=背景, 1..5=五类, 6=no-label。
    """
    masks_i = masks_i.astype(np.int64, copy=False)          # (256,256,6)
    inst = np.zeros(masks_i.shape[:2], np.uint32)
    type_map = np.zeros(masks_i.shape[:2], np.uint8)
    for c in range(6):
        m = masks_i[..., c] > 0
        inst[m] = (masks_i[..., c][m] + (c + 1) * (1 << 16)).astype(np.uint32)
        type_map[m] = c + 1                                  # 1..5 五类, 6=no-label
    return inst, type_map


# ---------------------------------------------------------------- HV 回归图 (HoVer-Net 语义)
def compute_hv_map(inst):
    """由实例标签图计算 HV 图 (2,H,W) float32, 取值 [-1,1]。

    每实例: H=x−cx, V=y−cy, 除以该实例最大幅度, 再整图归一到 [-1,1]。
    背景为 0。增广后调用 ⇒ HV 永远与当前几何一致。
    """
    h, w = inst.shape
    hv = np.zeros((2, h, w), np.float32)
    ids = np.unique(inst)
    ids = ids[ids > 0]
    if ids.size == 0:
        return hv
    ys, xs = np.nonzero(inst > 0)
    vals = inst[ys, xs]
    order = np.argsort(vals, kind="stable")
    ys, xs, vals = ys[order], xs[order], vals[order]
    splits = np.searchsorted(vals, ids, side="right")
    start = 0
    maxmag = 1.0
    for i, end in enumerate(splits):
        sy, sx = ys[start:end], xs[start:end]
        cy, cx = sy.mean(), sx.mean()
        dh = sx - cx            # 水平分量
        dv = sy - cy            # 垂直分量
        m = max(dh.max() - dh.min(), dv.max() - dv.min(), 1e-8) / 2.0
        hv[0, sy, sx] = dh / m
        hv[1, sy, sx] = dv / m
        mag = max(np.abs(dh / m).max(), np.abs(dv / m).max())
        maxmag = max(maxmag, mag)
        start = end
    hv /= maxmag
    np.clip(hv, -1.0, 1.0, out=hv)
    return hv


# ---------------------------------------------------------------- Reinhard 染色归一化 (折内拟合)
def _rgb_to_lab(rgb):       # rgb float32 [0,1] → LAB
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]], np.float32) / 0.95047
    xyz = np.maximum(xyz, 1e-6)
    f = np.where(xyz > 0.008856, xyz ** (1 / 3), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def fit_reinhard(images, max_patches=512, seed=0, out_json=None):
    """在【训练折】子集上估计 LAB 参考统计。返回并可持久化 {'mean': [L,a,b], 'std': [...]}。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))[:max_patches]
    labs = [_rgb_to_lab(images[i].astype(np.float32) / 255.0) for i in idx]
    lab = np.concatenate([l.reshape(-1, 3) for l in labs], axis=0)
    stats = {"mean": lab.mean(0).tolist(), "std": lab.std(0).tolist()}
    if out_json:
        Path(out_json).write_text(json.dumps(stats, indent=2))
    return stats


def apply_reinhard(image_u8, stats):
    """单 patch (H,W,3) uint8 → 把该 patch 自身 LAB 均值/方差迁移到训练折参考。"""
    lab = _rgb_to_lab(image_u8.astype(np.float32) / 255.0)
    mean = np.asarray(stats["mean"], np.float32)
    std = np.asarray(stats["std"], np.float32)
    pm = lab.reshape(-1, 3).mean(0)
    ps = lab.reshape(-1, 3).std(0)
    if np.any(ps < 1e-3):          # 近空白 patch: 不迁移, 避免放大噪声
        return image_u8
    lab_n = (lab - pm) / ps * std + mean
    # 反变换 LAB → XYZ → RGB (lab_n 通道: 0=L, 1=a, 2=b)
    f = (lab_n[..., 0] + 16) / 116
    xyz = np.stack([f + lab_n[..., 1] / 500, f, f - lab_n[..., 2] / 200], axis=-1)
    xyz = np.where(xyz > 0.206893, xyz ** 3, (xyz - 16 / 116) / 7.787)
    xyz *= 0.95047
    lin = xyz @ np.array([[3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660, 1.8760108, 0.0415560],
                          [0.0556434, -0.2040259, 1.0572252]], np.float32)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------- 折划分与数据集
def build_fold_split(root, val_fold=1):
    """官方 3 折交叉: val_fold 为验证折, 训练折 = 其余两折。返回 (train_folds, [val_fold])。"""
    assert val_fold in (1, 2, 3)
    train_folds = [f for f in (1, 2, 3) if f != val_fold]
    return train_folds, [val_fold]


class PanNukeDataset(Dataset):
    """内存缓存 uint8 全量 (铁律 2); 返回已含增广与 HV 的训练样本。

    __getitem__ 返回 dict:
      img   : (3,256,256) float32 ∈[0,1]
      np    : (1,256,256) float32 {0,1}      核前景
      hv    : (2,256,256) float32 [-1,1]
      inst  : (256,256)   int64  实例 ID (>0)
      type  : (256,256)   int64  0=背景,1..5=五类,6=no-label
      group : int                  统计聚合组 (默认=组织类型序号)
      patch_id : str               如 "fold1_0123"
    """

    def __init__(self, root, folds, stain="reinhard", stain_stats_json=None,
                 train=False, stain_fit_patches=512, stain_seed=0, seed=0):
        self.root = Path(root)
        self.train = train
        self.stain_mode = stain
        parts = [load_fold_npz(self.root / f"fold{f}.npz") for f in folds]
        self.images = np.concatenate([p["images"] for p in parts])     # uint8 缓存
        self.masks = np.concatenate([p["masks"] for p in parts])
        self.types = np.concatenate([p["types"] for p in parts])
        self.fold_of = np.concatenate(
            [np.full(len(p["images"]), f, int) for p, f in zip(parts, folds)])
        self.tissue_names = sorted(set(self.types.tolist()))
        self.tissue_ids = np.array(
            [self.tissue_names.index(t) for t in self.types], int)

        self.stain_stats = None
        if stain == "reinhard":
            if stain_stats_json and Path(stain_stats_json).exists():
                self.stain_stats = json.loads(Path(stain_stats_json).read_text())
            else:
                self.stain_stats = fit_reinhard(
                    self.images, max_patches=stain_fit_patches,
                    seed=stain_seed, out_json=stain_stats_json)
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, ep):                     # 增广随 epoch 变化, 保证多样性
        self._epoch = int(ep)

    def __len__(self):
        return len(self.images)

    def _augment(self, img, inst, type_map, rng):
        k = int(rng.integers(0, 4))              # rot90 ×k
        if k:
            img = np.rot90(img, k).copy()
            inst = np.rot90(inst, k).copy()
            type_map = np.rot90(type_map, k).copy()
        for axis in (0, 1):
            if rng.random() < 0.5:
                img = np.flip(img, axis).copy()
                inst = np.flip(inst, axis).copy()
                type_map = np.flip(type_map, axis).copy()
        return img, inst, type_map

    def __getitem__(self, i):
        img = self.images[i]
        if self.stain_stats is not None:
            img = apply_reinhard(img, self.stain_stats)
        inst, type_map = merge_masks(self.masks[i])
        if self.train:
            rng = np.random.default_rng((self.seed, self._epoch, i))
            img, inst, type_map = self._augment(img, inst, type_map, rng)
        hv = compute_hv_map(inst)
        return {
            "img": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0,
            "np": torch.from_numpy((inst > 0).astype(np.float32)).unsqueeze(0),
            "hv": torch.from_numpy(hv),
            "inst": torch.from_numpy(inst.astype(np.int64)),
            "type": torch.from_numpy(type_map.astype(np.int64)),
            "group": int(self.tissue_ids[i]),
            "patch_id": f"fold{self.fold_of[i]}_{i:05d}",
        }


# ---------------------------------------------------------------- 合成折 (单测/冒烟, 官方格式)
def make_synthetic_fold(path, n=16, size=256, seed=0):
    """生成官方格式的小型合成 npz: 每 patch 2-4 个不相交方块核, 2 种组织类型。"""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, size, size, 3), np.uint8)
    masks = np.zeros((n, size, size, 6), np.uint16)
    types = np.array(["tissue_a", "tissue_b"])[(rng.random(n) < 0.5).astype(int)]
    for i in range(n):
        bg = 90 + int(rng.integers(0, 40))
        images[i] = (bg,) * 3
        for c in range(6):                       # 前 3 patch 用 no-label 通道覆盖 c=5
            if c == 5 and i >= 3:
                continue
            for _ in range(int(rng.integers(0, 3)) + 1):
                ch, cw = int(rng.integers(12, 26)), int(rng.integers(12, 26))
                y0 = int(rng.integers(0, size - ch - 1)); x0 = int(rng.integers(0, size - cw - 1))
                masks[i, y0:y0 + ch, x0:x0 + cw, c] = _next_id(masks[i, ..., c])
                col = 60 + int(rng.integers(0, 120))
                images[i, y0:y0 + ch, x0:x0 + cw] = (col, col // 2, col)
        images[i] = np.clip(images[i] + rng.normal(0, 4, images[i].shape), 0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=images, types=types, masks=masks)
    return path


def _next_id(chan):
    return chan.max() + 1
