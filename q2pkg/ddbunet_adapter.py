# -*- coding: utf-8 -*-
r"""DDBU-Net 自包含适配 (T7 SOTA 基线, W9) —— build_model("ddbunet")
=================================================================
官方参考: Xie & Liu, "Dual-Decoder Network with Boundary-Enhanced
Attention for Nucleus Segmentation", ICVRV 2025 (DOI 10.1109/
ICVRV67992.2025.00033)。该会议论文无公开代码 (IEEE 付费), 本文件按
论文摘要/结构图披露的核心组件自包含复现 (同 HoVer-Net 先例):

  1) 轻量双解码器: 主解码器 (np/hv/type 三头) + 边界引导辅助解码器
     (显式预测边缘图, 由 GT 边界带监督 —— 经 models.BoundaryCompositeLoss
     新增 "edge_logits" 门控项实现, 对其他模型零影响)
  2) CAF (Convolutional Attention Fusion) 内嵌于跳连: 通道注意力
     (GAP→MLP→sigmoid) + 空间注意力 (mean/max→conv→sigmoid) 对
     解码上采样特征与编码 skip 特征做加权融合后 concat
  3) boundary-aware loss: 边界带 BCE 权重按预测不确定度 (硬度)
     自适应抬升, 强调难分辨边界像素

协议差异声明 (论文全文不可得, 复现以摘要披露组件为准):
  * backbone 未在摘要指明 → 取轻量 ResNet18 (ImageNet 预训练, timm),
    符合 "lightweight" 定位; 输入/输出协议与全部基线统一 (256 输入,
    np+hv+type 三头, 相同解码), 边缘分图仅作辅助监督不参与解码
  * boundary-aware loss 细节公式不可得 → 按摘要语义实现 (带内
    硬度=1−|2p−1| 加权), 论文中注明按公开描述复现

接口对齐 models.py 约定 (与 own_baseline / hovernet 可互换):
  np_logits (B,1,H,W) sigmoid logits / hv (B,2,H,W) / type_logits (B,7,H,W)
  edge_logits (B,1,H,W) 辅助边缘头 (loss 门控消费, 解码忽略)
  feats {s2,s3,s4} ResNet stage1/2/3 (stride 4/8/16, 64/128/256ch)
  gate_w None
输入约定: 外部管线给 [0,1]; 内部 ImageNet 均值方差归一化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

__all__ = ["DDBUNet", "build_ddbunet"]


class CAF(nn.Module):
    """Convolutional Attention Fusion (ICVRV 2025 摘要): 通道注意力 ×
    空间注意力 加权上采样特征, 再与 skip concat 1×1 融合。"""

    def __init__(self, cin: int):
        super().__init__()
        self.ca_mlp = nn.Sequential(
            nn.Linear(cin, cin // 4), nn.ReLU(inplace=True),
            nn.Linear(cin // 4, cin))
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * cin, cin, 1, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True))

    def forward(self, up, skip):
        # 通道注意力 (作用于上采样特征)
        w = self.ca_mlp(up.mean(dim=(2, 3))).sigmoid().unsqueeze(-1).unsqueeze(-1)
        up = up * w
        # 空间注意力 (mean/max 沿通道 → 7×7 conv)
        s = torch.cat([up.mean(dim=1, keepdim=True),
                       up.max(dim=1, keepdim=True).values], 1)
        up = up * self.sa_conv(s).sigmoid()
        return self.fuse(torch.cat([up, skip], 1))


class DDBUNet(nn.Module):
    """DDBU-Net (轻量): ResNet18 编码 + 主/边界双解码器 + CAF 跳连。"""

    def __init__(self, n_type: int = 7, pretrained: bool = True):
        super().__init__()
        self.n_type = n_type
        self.encoder = timm.create_model(
            "resnet18", features_only=True, out_indices=(0, 1, 2, 3, 4),
            pretrained=pretrained)
        c = [64, 64, 128, 256, 512]                     # stem/s1/s2/s3/s4 通道
        # 主解码器 (CAF 融合跳连; CAF 内部已 concat+1×1 融合, dec 仅精炼)
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.caf4 = CAF(c[3]); self.dec4 = self._dec(c[3], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.caf3 = CAF(c[2]); self.dec3 = self._dec(c[2], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.caf2 = CAF(c[1]); self.dec2 = self._dec(c[1], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.caf1 = CAF(c[0]); self.dec1 = self._dec(c[0], c[0])
        self.up0 = nn.ConvTranspose2d(c[0], 32, 2, stride=2)   # 64@128 → 32@256
        self.dec0 = self._dec(32, 32)
        self.np_head = nn.Conv2d(32, 1, 1)
        self.hv_head = nn.Conv2d(32, 2, 1)
        self.type_head = nn.Conv2d(32, n_type, 1)
        # 边界引导辅助解码器 (单路, 由瓶颈特征逐级上采样预测边缘)
        self.edge_dec = nn.Sequential(
            nn.ConvTranspose2d(c[4], 128, 2, stride=2), nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 2, stride=2), nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 2, stride=2), nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 1))
        self.register_buffer(
            "im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _dec(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        xin = (x - self.im_mean) / self.im_std
        f0, f1, f2, f3, f4 = self.encoder(xin)
        # 边界辅助解码器 (边缘分图, GT 边界带监督)
        edge_logits = self.edge_dec(f4)
        # 主解码器: up → CAF 加权 → concat skip → conv
        d = self.dec4(self.caf4(self.up4(f4), f3))        # 256@16
        d = self.dec3(self.caf3(self.up3(d), f2))         # 128@32
        d = self.dec2(self.caf2(self.up2(d), f1))         # 64@64
        d = self.dec1(self.caf1(self.up1(d), f0))         # 64@128
        d = self.dec0(self.up0(d))                        # 32@256
        return {"np_logits": self.np_head(d),
                "hv": self.hv_head(d),
                "type_logits": self.type_head(d),
                "edge_logits": edge_logits,
                "feats": {"s2": f1, "s3": f2, "s4": f3},   # stride 4/8/16
                "gate_w": None}


def build_ddbunet(n_type: int = 7, pretrained: bool = True, **kw) -> DDBUNet:
    return DDBUNet(n_type=n_type, pretrained=pretrained, **kw)
