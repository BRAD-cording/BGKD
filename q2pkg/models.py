# -*- coding: utf-8 -*-
"""自家基线模型 + 复合边界损失 (W1) —— 对应执行包 T2/T6 消融开关设计
=====================================================================
* LightUNet: 轻量编码-解码 (~4.7M params, 4060 bf16 峰值 <2GB), 三输出头:
    np_logits (B,1,H,W)   核前景
    hv         (B,2,H,W)  HoVer 回归图
    type_logits(B,7,H,W)  0=背景 1..5=五类 6=no-label
  特征抽头 feats {s2:stride4, s3:stride8, s4:stride16} 供 BGKD (§3.1) 使用。
* 消融开关 (T6 矩阵, 基线=全关):
    use_sobel_gate → SobelGate-MS 接 s2 特征 (§3.2)
    use_rab        → 不确定带反向注意力接解码输出 (§3.2)
    复合损失的 λ₃ 边界项由 train.py --boundary-loss 控制
* 复合损失: L = λ₁·(BCE+Dice) + λ₂·HV-MSE + λ₃·边界带 BCE (+ 0.5·type CE 辅助)
  边界带复用 §3.1 boundary_band (逐实例腐蚀语义)。λ 默认值为占位,
  按协议仅允许在 fold1-val 上调优后冻结 (W2 M1)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from boundary_distill import boundary_band
from modules import SobelGateMS, UncertaintyBandReverseAttention

__all__ = ["LightUNet", "BoundaryCompositeLoss", "build_model", "N_TYPE"]


def _conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class LightUNet(nn.Module):
    def __init__(self, n_type: int = 7, base: int = 32,
                 use_sobel_gate: bool = False, use_rab: bool = False, in_ch: int = 3,
                 sobel_tap: str = "s2"):
        super().__init__()
        b = base
        self.enc1 = _conv(in_ch, b)          # s1 256×256
        self.enc2 = _conv(b, 2 * b)          # s2' 128
        self.enc3 = _conv(2 * b, 4 * b)      # s2  64   ← BGKD tap
        self.enc4 = _conv(4 * b, 8 * b)      # s3  32   ← BGKD tap
        self.enc5 = _conv(8 * b, 16 * b)     # s4  16   ← BGKD tap
        self.pool = nn.MaxPool2d(2)
        self.use_sobel_gate = use_sobel_gate
        # 位置消融: SobelGate 接入的 skip 位置 ("s2"/"s3"/"s4"/"all", 默认 s2=历史行为)
        assert sobel_tap in ("s2", "s3", "s4", "all"), sobel_tap
        self.sobel_taps = (["s2", "s3", "s4"] if sobel_tap == "all" else [sobel_tap])
        if use_sobel_gate:
            ch = {"s2": 4 * b, "s3": 8 * b, "s4": 16 * b}
            self.sobel_gates = nn.ModuleDict(
                {t: SobelGateMS(ch[t]) for t in self.sobel_taps})
        self.dec4 = _conv(16 * b + 8 * b, 8 * b)
        self.dec3 = _conv(8 * b + 4 * b, 4 * b)
        self.dec2 = _conv(4 * b + 2 * b, 2 * b)
        self.dec1 = _conv(2 * b + b, b)
        self.use_rab = use_rab
        if use_rab:
            self.rab = UncertaintyBandReverseAttention(b)
        self.np_head = nn.Conv2d(b, 1, 1)
        self.hv_head = nn.Conv2d(b, 2, 1)
        self.type_head = nn.Conv2d(b, n_type, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))                     # s2
        w_gate = None
        if self.use_sobel_gate:
            if "s2" in self.sobel_taps:
                e3, w_gate = self.sobel_gates["s2"](e3)   # W≈0.88 恒等起点
            e4 = self.enc4(self.pool(e3))                 # s3
            if "s3" in self.sobel_taps:
                e4, _ = self.sobel_gates["s3"](e4)
            e5 = self.enc5(self.pool(e4))                 # s4
            if "s4" in self.sobel_taps:
                e5, _ = self.sobel_gates["s4"](e5)
        else:
            e4 = self.enc4(self.pool(e3))                 # s3
            e5 = self.enc5(self.pool(e4))                 # s4

        f = self.dec4(torch.cat([F.interpolate(e5, scale_factor=2, mode="nearest"), e4], 1))
        f = self.dec3(torch.cat([F.interpolate(f, scale_factor=2, mode="nearest"), e3], 1))
        f = self.dec2(torch.cat([F.interpolate(f, scale_factor=2, mode="nearest"), e2], 1))
        f = self.dec1(torch.cat([F.interpolate(f, scale_factor=2, mode="nearest"), e1], 1))

        np_logits = self.np_head(f)
        if self.use_rab:                                  # p 与 f 同分辨率 (RAB 不内部插值)
            f = self.rab(f, np_logits)
            np_logits = self.np_head(f)                   # 细化后再出最终 NP 头
        return {"np_logits": np_logits,
                "hv": self.hv_head(f),
                "type_logits": self.type_head(f),
                "feats": {"s2": e3, "s3": e4, "s4": e5},
                "gate_w": w_gate}


class BoundaryCompositeLoss(nn.Module):
    """L = λ₁·(BCE+Dice) + λ₂·HV-MSE(核内) + λ₃·边界带BCE + 0.5·type-CE

    * λ₁/λ₂/λ₃ 默认值为占位 —— 仅 fold1-val 调优后冻结 (执行包 §2.2/C3)
    * use_boundary=False 即消融矩阵的「基线」; True 即「＋边界损失」配置
    """

    def __init__(self, lambda_np=1.0, lambda_hv=2.0, lambda_bnd=1.0,
                 band_px=2, use_boundary=False, n_type=7):
        super().__init__()
        self.l_np, self.l_hv, self.l_bnd = lambda_np, lambda_hv, lambda_bnd
        self.band_px, self.use_boundary, self.n_type = band_px, use_boundary, n_type
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def _soft_dice(logits, target, eps=1.0):
        p = torch.sigmoid(logits)
        num = 2.0 * (p * target).sum(dim=(2, 3))
        den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
        return 1.0 - (num / den).mean()

    def forward(self, out, batch):
        np_gt = batch["np"]
        inst = batch["inst"]
        device = np_gt.device
        l_seg = self.bce(out["np_logits"], np_gt).mean() + self._soft_dice(out["np_logits"], np_gt)

        mask = (inst > 0).float()
        denom = mask.sum().clamp(min=1.0)
        l_hv = (F.mse_loss(out["hv"], batch["hv"], reduction="none").mean(dim=1, keepdim=True)
                * mask).sum() / denom

        if self.use_boundary:
            band = boundary_band(inst, self.band_px)                     # (B,1,H,W) {0,1}
            bce_map = self.bce(out["np_logits"], np_gt)
            l_bnd = (bce_map * band).sum() / band.sum().clamp(min=1.0)
        else:
            l_bnd = torch.zeros((), device=device)

        # DDBU-Net 边界辅助解码器 (ICVRV 2025): edge_logits 由 GT 边界带监督,
        # 带内按预测硬度 (1−|2p−1|) 自适应加权 —— "boundary-aware loss 按公开
        # 描述复现"; 无该键的模型此项恒 0 (零影响)。
        l_edge = torch.zeros((), device=device)
        if "edge_logits" in out:
            band_e = boundary_band(inst, self.band_px)
            with torch.no_grad():
                p_np = torch.sigmoid(out["np_logits"].detach())
                w_hard = 1.0 + (1.0 - (2 * p_np - 1).abs()) * band_e
            l_edge = (self.bce(out["edge_logits"], band_e) * w_hard).sum() \
                / band_e.sum().clamp(min=1.0)

        l_type = F.cross_entropy(out["type_logits"], batch["type"])

        loss = (self.l_np * l_seg + self.l_hv * l_hv + self.l_bnd * l_bnd
                + 0.5 * l_type + l_edge)
        return loss, {"l_seg": l_seg.detach(), "l_hv": l_hv.detach(),
                      "l_bnd": l_bnd.detach(), "l_type": l_type.detach(),
                      "l_edge": l_edge.detach()}


def build_model(name: str, **kw) -> nn.Module:
    """模型注册制: own_baseline / nulite_t / hovernet (官方结构适配, 见 *_adapter.py)。"""
    if name == "own_baseline":
        return LightUNet(**kw)
    if name == "nulite_t":
        from nulite_adapter import build_nulite_t
        for k in ("base", "use_sobel_gate", "use_rab", "in_ch", "sobel_tap"):   # LightUNet 专属参数
            kw.pop(k, None)
        return build_nulite_t(**kw)
    if name == "nulite_s":                                         # W8: fastvit_s12 (官方 NuLite-S 变体)
        from nulite_adapter import build_nulite_t
        for k in ("base", "use_sobel_gate", "use_rab", "in_ch", "sobel_tap"):
            kw.pop(k, None)
        kw.pop("backbone", None)
        return build_nulite_t(backbone="fastvit_s12", **kw)
    if name == "hovernet":
        from hovernet_adapter import build_hovernet
        for k in ("base", "use_sobel_gate", "use_rab", "in_ch", "sobel_tap"):   # LightUNet 专属参数
            kw.pop(k, None)
        return build_hovernet(**kw)
    if name == "cellvit_t256":                                     # T7/W7: CellViT-256 (官方 MedIA 2024)
        from cellvit_adapter import build_cellvit_256
        for k in ("base", "use_sobel_gate", "use_rab", "in_ch", "sobel_tap"):   # LightUNet 专属参数
            kw.pop(k, None)
        return build_cellvit_256(**kw)
    if name == "cellvit_sam256":                                   # T7: CellViT-256-SAM (FKDNuSeg 替代)
        from cellvit_sam_adapter import build_cellvit_sam_256
        for k in ("base", "use_sobel_gate", "use_rab", "in_ch", "sobel_tap"):
            kw.pop(k, None)
        return build_cellvit_sam_256(**kw)
    if name == "ddbunet":                                          # T7/W9: DDBU-Net (ICVRV 2025 复现)
        from ddbunet_adapter import build_ddbunet
        for k in ("base", "use_sobel_gate", "use_rab", "in_ch", "sobel_tap"):
            kw.pop(k, None)
        return build_ddbunet(**kw)
    raise KeyError(f"未知模型: {name}")
