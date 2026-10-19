# -*- coding: utf-8 -*-
r"""CellViT-256 自包含适配 (T7 SOTA 基线, W7) —— build_model("cellvit_t256")
=================================================================
官方参考: TIO-IKIM/CellViT (Hörst et al., "CellViT: Vision Transformers
for Precise Cell Segmentation and Classification", MedIA 2024)。官方源码
已存 third_party/CellViT-main, 本文件按其 models/segmentation/cellvit.py
(CellViT256) + cell_segmentation/utils.py (Conv2DBlock/Deconv2DBlock) +
models/encoders/VIT/vits_histo.py (VisionTransformer) 逐层对照自包含实现,
不 import 官方仓库 (其依赖 einops/segment_anything 与本机环境不兼容)。

结构 (官方 CellViT-256):
  * Encoder : ViT-S/256 (HIPT DINO 预训练, embed 384 / depth 12 / heads 6 /
              patch 16, extract_layers=[3,6,9,12]; cls token + 1D pos embed)
  * Decoder : UNet 式 —— 共享 decoder0(输入图 Conv 3→32→64) 与
              decoder1/2/3(对 skip token 逐级 Deconv 上采样);
              三分支各自独立 upsampling branch (bottleneck_upsampler→
              decoder3/2/1_upsampler→decoder0_header), 官方分支不共享
  * 三分支  : nuclei_binary (2ch) / hv (2ch) / nuclei_type (n_type ch)
              (官方还有 tissue_types 分类头, 本协议无 tissue 监督管线,
              head=Identity 不采用 —— 论文协议差异声明, 不影响实例分割)

接口对齐 models.py 约定 (与 own_baseline / hovernet 可互换):
  np_logits   (B,1,H,W) : 官方 binary 2 通道 softmax 头转 sigmoid 等价差值 z1−z0
  hv          (B,2,H,W) : 官方 hv 直通 (语义同 pannuke_data.compute_hv_map)
  type_logits (B,7,H,W) : 0=背景 1..5 五类 6=no-label
  feats {s2,s3,s4}      : s4=bottleneck token (384@16), s3/s4 取 binary 分支
                          上采样中间量 b4(312@8)/b3(256@4), stride 4/8/16;
                          BGKD student 侧 1×1 adapter + nearest 自动对齐
  gate_w      None      : CellViT 无 SobelGate (消融开关仅 own_baseline)
输入约定: 外部管线给 [0,1]; 内部 (x−0.5)/0.5 归一化 (官方默认 Normalize)。
预训练: HIPT ViT-256 DINO 权重 (mahmoodlab/HIPT HIPT_4K/Checkpoints/
  vit256_small_dino.pth, 已存 C:\data\weights\vit256_small_dino.pth),
  取 ckpt["teacher"], 剥离 module./backbone. 前缀, strict=False 装载。
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CellViT256", "build_cellvit_256"]

HIPT_VIT256_CKPT = r"C:\data\weights\vit256_small_dino.pth"


class Conv2DBlock(nn.Module):
    """官方 Conv2DBlock: Conv3×3 → BN → ReLU → Dropout (官方 utils.py)。"""

    def __init__(self, cin: int, cout: int, dropout: float = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout), nn.ReLU(True), nn.Dropout(dropout))

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """官方 Deconv2DBlock: ConvT2×2s2 → Conv3×3 → BN → ReLU → Dropout。"""

    def __init__(self, cin: int, cout: int, dropout: float = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 2, stride=2),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout), nn.ReLU(True), nn.Dropout(dropout))

    def forward(self, x):
        return self.block(x)


class ViTAttention(nn.Module):
    """官方 Attention (vits_histo.py): qkv 联合投影, scale=head_dim^-0.5。"""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class ViTBlock(nn.Module):
    """官方 Block: norm1→attn→残差→norm2→MLP(GELU)→残差。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = ViTAttention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSmall256(nn.Module):
    """官方 ViTCellViT/VisionTransformer (ViT-S/256): cls token + 1D pos embed,
    返回 (head 输出, norm 后 cls, 各 extract_layers 原始块输出——不加最终 norm,
    与官方一致: 解码 skip 用未归一化特征, 仅分类头用 norm 后 token)。"""

    def __init__(self, img_size: int = 256, patch_size: int = 16,
                 embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 extract_layers=(3, 6, 9, 12)):
        super().__init__()
        self.patch_size = patch_size
        self.extract_layers = tuple(extract_layers)
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [ViTBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()          # tissue 分类头不采用 (协议差异声明)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def prepare_tokens(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        return x + self.pos_embed

    def forward(self, x):
        x = self.prepare_tokens(x)
        extracted = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i + 1 in self.extract_layers:
                extracted.append(x)        # 原始块输出 (未过最终 norm, 官方一致)
        xn = self.norm(x)
        return self.head(xn[:, 0]), xn[:, 0], extracted


class CellViT256(nn.Module):
    """CellViT-256 (官方结构): ViT-S/256 encoder + UNet 解码 + 三分支。"""

    def __init__(self, n_type: int = 7):
        super().__init__()
        self.patch_size = 16
        self.embed_dim = 384
        self.skip_dim_11, self.skip_dim_12, self.bottleneck_dim = 256, 128, 312
        self.encoder = ViTSmall256()

        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32), Conv2DBlock(32, 64))              # 输入图 skip
        self.decoder1 = nn.Sequential(                            # z1 skip 分支
            Deconv2DBlock(self.embed_dim, self.skip_dim_11),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12),
            Deconv2DBlock(self.skip_dim_12, 128))
        self.decoder2 = nn.Sequential(                            # z2 skip 分支
            Deconv2DBlock(self.embed_dim, self.skip_dim_11),
            Deconv2DBlock(self.skip_dim_11, 256))
        self.decoder3 = nn.Sequential(                            # z3 skip 分支
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim))

        self.bin_upsampler = self._upsampling_branch(2)
        self.hv_upsampler = self._upsampling_branch(2)
        self.type_upsampler = self._upsampling_branch(n_type)

        self.register_buffer("im_mean", torch.full((1, 3, 1, 1), 0.5))
        self.register_buffer("im_std", torch.full((1, 3, 1, 1), 0.5))

    def _upsampling_branch(self, num_classes: int) -> nn.ModuleDict:
        """官方 create_upsampling_branch: 每分支独立上采样路径 (OrderedDict
        bottleneck_upsampler→decoder3/2/1_upsampler→decoder0_header)。"""
        sd11, sd12, bneck = self.skip_dim_11, self.skip_dim_12, self.bottleneck_dim
        return nn.ModuleDict({
            "bottleneck_upsampler": nn.ConvTranspose2d(
                self.embed_dim, bneck, 2, stride=2),
            "decoder3_upsampler": nn.Sequential(
                Conv2DBlock(bneck * 2, bneck), Conv2DBlock(bneck, bneck),
                Conv2DBlock(bneck, bneck),
                nn.ConvTranspose2d(bneck, 256, 2, stride=2)),
            "decoder2_upsampler": nn.Sequential(
                Conv2DBlock(256 * 2, 256), Conv2DBlock(256, 256),
                nn.ConvTranspose2d(256, 128, 2, stride=2)),
            "decoder1_upsampler": nn.Sequential(
                Conv2DBlock(128 * 2, 128), Conv2DBlock(128, 128),
                nn.ConvTranspose2d(128, 64, 2, stride=2)),
            "decoder0_header": nn.Sequential(
                Conv2DBlock(64 * 2, 64), Conv2DBlock(64, 64),
                nn.Conv2d(64, num_classes, 1)),
        })

    def _forward_upsample(self, z0, z1, z2, z3, z4, br: nn.ModuleDict):
        """官方 _forward_upsample: 共享 decoder0-3 预处理 + 分支专属上采样。"""
        b4 = br["bottleneck_upsampler"](z4)                       # 312@8
        b3 = br["decoder3_upsampler"](torch.cat([self.decoder3(z3), b4], 1))  # 256@4
        b2 = br["decoder2_upsampler"](torch.cat([self.decoder2(z2), b3], 1))  # 128@2
        b1 = br["decoder1_upsampler"](torch.cat([self.decoder1(z1), b2], 1))  # 64@1
        b0 = self.decoder0(z0)                                    # 64@256
        return br["decoder0_header"](torch.cat([b0, b1], 1)), b4, b3

    def forward(self, x):
        assert x.shape[-2] % self.patch_size == 0 and x.shape[-1] % self.patch_size == 0
        xin = (x - self.im_mean) / self.im_std
        _, _, z = self.encoder(xin)                               # z=[z1,z2,z3,z4]
        p = self.patch_size
        patch_dim = [int(d / p) for d in x.shape[-2:]]
        z1, z2, z3, z4 = [
            t[:, 1:, :].transpose(-1, -2).reshape(-1, self.embed_dim, *patch_dim)
            for t in z]

        bin2, b4, b3 = self._forward_upsample(
            x, z1, z2, z3, z4, self.bin_upsampler)
        hv = self._forward_upsample(x, z1, z2, z3, z4, self.hv_upsampler)[0]
        tp = self._forward_upsample(x, z1, z2, z3, z4, self.type_upsampler)[0]
        return {"np_logits": bin2[:, 1:2] - bin2[:, 0:1],
                "hv": hv,
                "type_logits": tp,
                "feats": {"s2": b3, "s3": b4, "s4": z4},   # stride 4/8/16
                "gate_w": None}


def build_cellvit_256(n_type: int = 7, pretrained: bool = True, **kw) -> CellViT256:
    m = CellViT256(n_type=n_type, **kw)
    if pretrained:
        ckpt_path = Path(HIPT_VIT256_CKPT)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"HIPT ViT-256 DINO 权重缺失: {ckpt_path} (训练前必须就位, 禁止静默随机初始化)")
        # HIPT 官方 checkpoint 含 numpy 标量 (torch 2.6 weights_only 默认拦截);
        # 来源 mahmoodlab 官方仓库, trusted → weights_only=False
        sd = torch.load(str(ckpt_path), map_location="cpu",
                        weights_only=False)["teacher"]
        sd = {k.replace("module.", "").replace("backbone.", ""): v
              for k, v in sd.items()}
        # DINO 原始命名 → 本实现命名 (patch_embed.proj.*→patch_embed.*,
        # mlp.fc1/fc2→mlp.0/mlp.2); head.mlp*/last_layer 为 DINO 头, 忽略
        sd = {k.replace("patch_embed.proj.", "patch_embed.")
                .replace("mlp.fc1.", "mlp.0.").replace("mlp.fc2.", "mlp.2."): v
              for k, v in sd.items()}
        # 该 ckpt 的 pos_embed 为 197 token (ViT-S/16 @224 DINO 预训练);
        # 官方经 DINO 式运行时 bicubic 插值适配 256 输入 —— 此处改为装载后
        # 一次性插值到 16×16 网格 (数值等价, 前向零开销)
        pos = sd.pop("pos_embed", None)
        rep = m.encoder.load_state_dict(sd, strict=False)
        if pos is not None:
            npatch = m.encoder.pos_embed.shape[1] - 1
            n = pos.shape[1] - 1
            if n != npatch:
                side, tside = int(n ** 0.5), int(npatch ** 0.5)
                p = pos[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
                p = F.interpolate(p, size=(tside, tside), mode="bicubic",
                                  align_corners=False)
                pos = torch.cat(
                    [pos[:, :1], p.permute(0, 2, 3, 1).reshape(1, npatch, -1)],
                    dim=1)
            m.encoder.pos_embed.data.copy_(pos)
        print(f"[cellvit256] HIPT ViT-256 装载 missing={len(rep.missing_keys)} "
              f"unexpected={len(rep.unexpected_keys)}: "
              f"missing={rep.missing_keys} unexpected={rep.unexpected_keys}",
              flush=True)
    return m
