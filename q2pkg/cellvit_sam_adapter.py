# -*- coding: utf-8 -*-
r"""CellViT-256-SAM 自包含适配 (T7 SOTA 基线, FKDNuSeg 替代) —— build_model("cellvit_sam256")
=================================================================
官方参考: TIO-IKIM/CellViT (Hörst et al., MedIA 2024) 的 CellViTSAM
(vit_structure="SAM-B")。third_party/CellViT-main 已存官方源码, 本文件按
其 cellvit.py (CellViTSAM) + models/encoders/VIT/SAM/image_encoder.py
(Meta 官方 SAM ViT-B encoder) + SAM/utils.py (MLPBlock/LayerNorm2d)
逐层对照自包含实现, 不 import 官方仓库 (依赖 segment_anything/einops)。

结构 (官方 CellViT-256-SAM):
  * Encoder : SAM ViT-B (embed 768 / depth 12 / heads 12, patch 16;
              窗口注意力 window=14 + 全局注意力层 [2,5,8,11]; 分解式
              相对位置编码 rel_pos_h/w —— ckpt 按 1024 输入 (64×64 网格,
              rel 127) 存储, 运行时 get_rel_pos 线性插值到实际尺寸,
              与官方 SAM 行为一致); abs pos embed 按 ViTCellViTDeit 的
              官方裁剪语义取 64×64 网格左上 16×16; neck 768→256 仅产
              classifier 向量, 解码 skip 用 block 原始输出 (B,H,W,C)
  * Decoder : 与 CellViT-256 同构 (embed_dim=768 → skip 512/256,
              bottleneck 512), 三分支独立上采样路径
  * 三分支  : binary 2ch / hv 2ch / type n_type ch (tissue 分类头不采用,
              协议差异声明, 同 cellvit_adapter)

接口对齐 models.py 约定:
  np_logits (B,1,H,W) binary 2ch softmax→sigmoid 等价差值 / hv (B,2,H,W)
  type_logits (B,7,H,W) / feats {s2:b3 256@4, s3:b4 512@8, s4:z4 768@16}
  gate_w None
输入约定: 外部管线给 [0,1]; 内部 (x−0.5)/0.5 (官方 CellViT 默认 Normalize)。
预训练: Meta sam_vit_b_01ec64.pth (dl.fbaipublicfiles.com, 已存
  C:\data\weights\sam_vit_b_01ec64.pth); 取顶层 image_encoder.* 前缀,
  strict=False 装载 (prompt_encoder/mask_decoder 丢弃)。
"""
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CellViTSAM256", "build_cellvit_sam_256"]

SAM_VITB_CKPT = r"C:\data\weights\sam_vit_b_01ec64.pth"


class Conv2DBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout), nn.ReLU(True), nn.Dropout(dropout))

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 2, stride=2),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout), nn.ReLU(True), nn.Dropout(dropout))

    def forward(self, x):
        return self.block(x)


class MLPBlock(nn.Module):
    """SAM MLPBlock: lin1→GELU→lin2 (官方命名)。"""

    def __init__(self, embedding_dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    """SAM LayerNorm2d (ConvNeXt 语义, 官方命名 weight/bias)。"""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int,
                       pad_hw: Tuple[int, int], hw: Tuple[int, int]):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """官方 get_rel_pos: rel pos 维度不符时线性插值 (SAM 1024→256 的适配路径)。"""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode="linear")
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """官方分解式相对位置编码 (mvitv2)。"""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)
    return attn


class SAMAttention(nn.Module):
    """SAM Attention: rel_pos_h/w 按 input_size (ckpts=64) 注册。"""

    def __init__(self, dim: int, num_heads: int, input_size: Tuple[int, int]):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


class SAMBlock(nn.Module):
    """SAM Block: window 注意力 (window>0) 或全局 (window=0)。"""

    def __init__(self, dim: int, num_heads: int, window_size: int, input_size: Tuple[int, int]):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SAMAttention(dim, num_heads,
                                 input_size if window_size == 0 else (window_size, window_size))
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLPBlock(dim, int(dim * 4))
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        H, W = x.shape[1], x.shape[2]
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class SAMImageEncoder(nn.Module):
    """SAM ImageEncoderViT (B,H,W,C token 语义); 返回 (neck cls 均值, neck 输出,
    各 extract_layers 的 block 原始输出 B,H,W,C —— 与 ViTCellViTDeit 官方一致,
    解码 skip 用 block 输出而非 neck 输出)。"""

    def __init__(self, img_size: int, patch_size: int, embed_dim: int, depth: int,
                 num_heads: int, extract_layers=(3, 6, 9, 12),
                 global_attn_indexes=(2, 5, 8, 11), window_size: int = 14,
                 out_chans: int = 256):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        self.extract_layers = tuple(extract_layers)
        self.blocks = nn.ModuleList([
            SAMBlock(embed_dim, num_heads,
                     window_size if i not in global_attn_indexes else 0,
                     (img_size // patch_size, img_size // patch_size))
            for i in range(depth)])
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, 1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, 3, padding=1, bias=False),
            LayerNorm2d(out_chans))

    def forward(self, x):
        x = self.patch_embed(x)                      # B,C,H,W
        token = x.shape[-1]
        x = x.permute(0, 2, 3, 1) + self.pos_embed[:, :token, :token, :]  # 官方裁剪语义
        extracted = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i + 1 in self.extract_layers:
                extracted.append(x)                  # B,H,W,C
        neck_out = self.neck(x.permute(0, 3, 1, 2))
        cls_mean = neck_out.mean(dim=(2, 3))
        return cls_mean, neck_out, extracted


class CellViTSAM256(nn.Module):
    """CellViT-256-SAM (官方 CellViTSAM/SAM-B): SAM ViT-B encoder + UNet 解码。"""

    def __init__(self, n_type: int = 7):
        super().__init__()
        self.patch_size = 16
        self.embed_dim = 768
        self.skip_dim_11, self.skip_dim_12, self.bottleneck_dim = 512, 256, 512
        self.encoder = SAMImageEncoder(img_size=1024, patch_size=16, embed_dim=768,
                                       depth=12, num_heads=12)

        self.decoder0 = nn.Sequential(Conv2DBlock(3, 32), Conv2DBlock(32, 64))
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12),
            Deconv2DBlock(self.skip_dim_12, 128))
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11),
            Deconv2DBlock(self.skip_dim_11, 256))
        self.decoder3 = nn.Sequential(Deconv2DBlock(self.embed_dim, self.bottleneck_dim))

        self.bin_upsampler = self._upsampling_branch(2)
        self.hv_upsampler = self._upsampling_branch(2)
        self.type_upsampler = self._upsampling_branch(n_type)

        self.register_buffer("im_mean", torch.full((1, 3, 1, 1), 0.5))
        self.register_buffer("im_std", torch.full((1, 3, 1, 1), 0.5))

    def _upsampling_branch(self, num_classes: int) -> nn.ModuleDict:
        sd11, sd12, bneck = self.skip_dim_11, self.skip_dim_12, self.bottleneck_dim
        return nn.ModuleDict({
            "bottleneck_upsampler": nn.ConvTranspose2d(self.embed_dim, bneck, 2, stride=2),
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
        b4 = br["bottleneck_upsampler"](z4)                       # 512@8
        b3 = br["decoder3_upsampler"](torch.cat([self.decoder3(z3), b4], 1))  # 256@4
        b2 = br["decoder2_upsampler"](torch.cat([self.decoder2(z2), b3], 1))  # 128@2
        b1 = br["decoder1_upsampler"](torch.cat([self.decoder1(z1), b2], 1))  # 64@1
        b0 = self.decoder0(z0)                                    # 64@256
        return br["decoder0_header"](torch.cat([b0, b1], 1)), b4, b3

    def forward(self, x):
        assert x.shape[-2] % self.patch_size == 0 and x.shape[-1] % self.patch_size == 0
        xin = (x - self.im_mean) / self.im_std
        _, _, z = self.encoder(xin)
        # 官方 CellViTSAM: skip token 直接 permute 成 B,C,H,W (16×16)
        z1, z2, z3, z4 = [t.permute(0, 3, 1, 2) for t in z]

        bin2, b4, b3 = self._forward_upsample(x, z1, z2, z3, z4, self.bin_upsampler)
        hv = self._forward_upsample(x, z1, z2, z3, z4, self.hv_upsampler)[0]
        tp = self._forward_upsample(x, z1, z2, z3, z4, self.type_upsampler)[0]
        return {"np_logits": bin2[:, 1:2] - bin2[:, 0:1],
                "hv": hv,
                "type_logits": tp,
                "feats": {"s2": b3, "s3": b4, "s4": z4},   # stride 4/8/16
                "gate_w": None}


def build_cellvit_sam_256(n_type: int = 7, pretrained: bool = True, **kw) -> CellViTSAM256:
    m = CellViTSAM256(n_type=n_type, **kw)
    if pretrained:
        ckpt_path = Path(SAM_VITB_CKPT)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAM-B 权重缺失: {ckpt_path} (训练前必须就位, 禁止静默随机初始化)")
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # 完整 SAM ckpt: 仅取 image_encoder.* (prompt/mask decoder 丢弃)
        sd = {k[len("image_encoder."):]: v for k, v in sd.items()
              if k.startswith("image_encoder.")}
        # 本实现 patch_embed 为裸 Conv2d, ckpt 为 PatchEmbed.proj 包裹 → 重命名
        sd = {k.replace("patch_embed.proj.", "patch_embed."): v
              for k, v in sd.items()}
        rep = m.encoder.load_state_dict(sd, strict=False)
        print(f"[cellvit_sam256] SAM-B image_encoder 装载 missing={len(rep.missing_keys)} "
              f"unexpected={len(rep.unexpected_keys)}: "
              f"missing={rep.missing_keys} unexpected={rep.unexpected_keys}", flush=True)
    return m
