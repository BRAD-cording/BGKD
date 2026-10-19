# -*- coding: utf-8 -*-
"""HoVer-Net 自包含适配 (W2 队列前置缺口) —— build_model("hovernet")
=================================================================
官方参考: vqdang/hover_net (Graham et al., "HoVer-Net: Simultaneous
Segmentation and Classification of Nuclei in Histopathological Images",
MedIA 2019, arXiv 1904.10442)。本机无官方源码拷贝, 结构按论文 §2 与
主流 PyTorch 复现 (TissueLab CellViT hovernet_net.py) 逐层对照实现,
不 import 官方仓库 (其依赖 scipy/cv2/cucim 与本机环境不兼容)。

结构 (官方 ResNet34 "fast" 模式):
  * Encoder : ResNet34 (ImageNet 预训练, timm features_only 5 抽头:
              stem 64@128 + stage1-4 = 64@64 / 128@32 / 256@16 / 512@8)
  * Decoder : 5 个 DenseBlock —— 上采样分支 (BN-ReLU-ConvTranspose2×2)
              与 skip-info 分支 (BN-ReLU-Conv3×3 ×2) concat 后 1×1 融合;
              decoder1-4 的 skip = stage3/2/1/stem 特征, decoder5 的 skip
              = 输入图 (官方设计, 3 通道)
  * 三分支  : NP (2ch, softmax 语义) / HoVer (2ch 回归) / NC (n_type 通道,
              输入 = decoder 特征 concat NP logits, 官方级联设计)

接口对齐 models.py 约定 (与 own_baseline / nulite_t 可互换):
  np_logits   (B,1,H,W) : 官方 2 通道 softmax 头转 sigmoid 等价差值 z1−z0
  hv          (B,2,H,W) : 官方 hv 直通 (语义同 pannuke_data.compute_hv_map)
  type_logits (B,7,H,W) : 0=背景 1..5 五类 6=no-label
  feats {s2,s3,s4}      : ResNet stage1/2/3 (stride 4/8/16, 64/128/256ch),
                          BGKD 的 student 侧 1×1 adapter + nearest 自动对齐
  gate_w      None      : HoVer-Net 无 SobelGate (消融开关仅 own_baseline)
输入约定: 外部管线给 [0,1] (pannuke_data /255); 内部 ImageNet 均值方差
  归一化 (timm resnet34 预训练权重原生预处理)。

同协议差异声明 (论文 §2.2 对比可接受): 官方原版 NC 分支 6 通道 (背景+5 类,
no-label 核不参与 NC 监督), 此处直接 7 通道训练 (与 LightUNet 头契约一致,
no-label 类有监督, 无需零权重辅助通道)。
"""
import torch
import torch.nn as nn

import timm

__all__ = ["HoVerNet", "build_hovernet"]


class DenseBlock(nn.Module):
    """官方 decoder 单元: 上采样分支 ∥ skip-info 分支 → concat → 1×1 融合。"""

    def __init__(self, cin_up: int, cin_skip: int, cout: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.BatchNorm2d(cin_up), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(cin_up, cout, 2, stride=2))
        self.skip = nn.Sequential(
            nn.BatchNorm2d(cin_skip), nn.ReLU(inplace=True),
            nn.Conv2d(cin_skip, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * cout, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x_up, x_skip):
        return self.fuse(torch.cat([self.up(x_up), self.skip(x_skip)], 1))


def _branch(cin: int, cout: int) -> nn.Sequential:
    """官方 NP/HV/NC 分支: Conv3×3(64)×2 + 1×1 投影。"""
    return nn.Sequential(
        nn.Conv2d(cin, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.Conv2d(64, cout, 1))


class HoVerNet(nn.Module):
    """HoVer-Net (ResNet34 encoder + 5×DenseBlock decoder + NP/HV/NC 三分支)。"""

    def __init__(self, n_type: int = 7, pretrained: bool = True):
        super().__init__()
        self.n_type = n_type
        self.encoder = timm.create_model(
            "resnet34", features_only=True, out_indices=(0, 1, 2, 3, 4),
            pretrained=pretrained)
        self.decoder1 = DenseBlock(512, 256, 512)   # + stage3 → 512@16
        self.decoder2 = DenseBlock(512, 128, 256)   # + stage2 → 256@32
        self.decoder3 = DenseBlock(256, 64, 128)    # + stage1 → 128@64
        self.decoder4 = DenseBlock(128, 64, 64)     # + stem    →  64@128
        self.decoder5 = DenseBlock(64, 3, 64)       # + 输入图  →  64@256
        self.np_branch = _branch(64, 2)
        self.hv_branch = _branch(64, 2)
        self.nc_branch = _branch(64 + 2, n_type)
        self.register_buffer(
            "im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        xin = (x - self.im_mean) / self.im_std
        f0, f1, f2, f3, f4 = self.encoder(xin)
        d = self.decoder1(f4, f3)
        d = self.decoder2(d, f2)
        d = self.decoder3(d, f1)
        d = self.decoder4(d, f0)
        d = self.decoder5(d, x)                     # skip = [0,1] 原始输入 (官方)
        np2 = self.np_branch(d)                     # (B,2,H,W) softmax 语义
        tp = self.nc_branch(torch.cat([d, np2], 1))  # 官方: NC 级联 NP 输出
        return {"np_logits": np2[:, 1:2] - np2[:, 0:1],
                "hv": self.hv_branch(d),
                "type_logits": tp,
                "feats": {"s2": f1, "s3": f2, "s4": f3},   # stride 4/8/16
                "gate_w": None}


def build_hovernet(n_type: int = 7, pretrained: bool = True, **kw) -> HoVerNet:
    return HoVerNet(n_type=n_type, pretrained=pretrained, **kw)
