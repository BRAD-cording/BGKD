# -*- coding: utf-8 -*-
"""NuLite-T 官方结构适配 (W1 遗留项②) —— build_model("nulite_t")
=================================================================
官方仓库: github.com/CosmoIknosLab/NuLite (arXiv 2408.01797, Tommasino et al. 2024)

变体命名核实 (源码证据):
  * 官方 README 权重链接 = NuLite-T / NuLite-M / NuLite-H (Zenodo 13272655/13272705/13272667)
    → 执行包的 "NuLite-T" 命名正确 (早期 arXiv 版 S/M/H 为旧称)
  * 官方 config_examples/training_example.yaml: model.backbone=fastvit_t8,
    logging group "NuLite-fastvit_t8" → **NuLite-T = fastvit_t8**

实现方式: 自包含复制官方 models/nulite.py 结构 (Conv2DBlock + timm FastViT
features_only encoder + 单共享上采样解码分支 + 三头), 不 import 官方仓库
(官方 import 链含 torchinfo/wandb/cucim 等重依赖, 与本机环境不兼容;
结构逐字对照 third_party/NuLite-main/models/nulite.py @ main)。

接口对齐 models.py 约定 (teacher/student 可互换):
  np_logits   (B,1,H,W) : 官方 2 通道 softmax 头转为 sigmoid 等价差值 z1−z0
  hv          (B,2,H,W) : 官方 hv_map 直通
  type_logits (B,7,H,W) : 官方 6 通道 (0=背景,1..5 五类) + 可学习 no-label 通道
                          (零权重 1×1 conv, bias=−8 初始化 → argmax 不会选 6)
  feats {s2,s3,s4}      : FastViT stage 特征 (stride 4/8/16, t8 通道 48/96/192),
                          BGKD 的 student 侧 1×1 adapter + nearest 自动对齐
输入约定: 外部管线输出 [0,1] (pannuke_data /255); adapter 内部按官方预处理
  归一化 (x−0.5)/0.5 (官方 training_example.yaml transformations.normalize)。

官方全量权重 (Zenodo 13272655) 可经 load_nulite_t_weights() 加载, 供 W5 蒸馏。
"""
import torch
import torch.nn as nn

import timm

__all__ = ["NuLiteT", "build_nulite_t", "load_nulite_t_weights"]


class Conv2DBlock(nn.Module):
    """逐字复制官方 models/utils.py Conv2DBlock (conv+BN+ReLU+dropout)。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                      padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


_EMBED_DIMS = {  # 官方 nulite.py _init_* (t8 起逐变体)
    "fastvit_t8": [48, 96, 192, 384],
    "fastvit_t12": [64, 128, 256, 512],
    "fastvit_s12": [64, 128, 256, 512],
    "fastvit_sa12": [64, 128, 256, 512],
    "fastvit_sa24": [64, 128, 256, 512],
    "fastvit_sa36": [64, 128, 256, 512],
    "fastvit_ma36": [76, 152, 304, 608],
}


class NuLiteT(nn.Module):
    """官方 NuLite (U-Net-like 单共享解码分支 + 三头), 默认 fastvit_t8。

    与官方差异 (均不影响三头分割输出, 复现为同协议双轨对比):
      * 省略 tissue 分类头 —— 本项目训练管线无 tissue CE (执行包 §2.2 同协议要求)
      * type 头扩为 7 通道 (末位 no-label 通道, 见模块 docstring)
      * 输出 key 重命名对齐 models.py: nuclei_binary_map→np_logits(差值形式)
    """

    def __init__(self, n_type: int = 7, backbone: str = "fastvit_t8",
                 pretrained: bool = True, drop_rate: float = 0.0):
        super().__init__()
        if backbone not in _EMBED_DIMS:
            raise KeyError(f"未知 FastViT 变体: {backbone} (可选 {list(_EMBED_DIMS)})")
        self.backbone = backbone
        self.embed_dims = _EMBED_DIMS[backbone]
        self.num_nuclei_classes = 6                       # 官方 PanNuke: 背景+5 类
        self.n_type = n_type
        self.encoder = timm.create_model(
            f"{backbone}.apple_in1k", features_only=True, pretrained=pretrained)
        e = self.embed_dims
        # ---- 官方 create_upsampling_branch (单共享分支, 逐字) ----
        self.decoder = nn.ModuleDict({
            "bottleneck_upsampler": nn.Sequential(
                Conv2DBlock(e[-1], e[-2], dropout=drop_rate),
                nn.ConvTranspose2d(e[-2], e[-2], 2, stride=2)),
            "decoder4_upsampler": nn.Sequential(
                Conv2DBlock(2 * e[-2], e[-2], dropout=drop_rate),
                Conv2DBlock(e[-2], e[-3], dropout=drop_rate),
                nn.ConvTranspose2d(e[-3], e[-3], 2, stride=2)),
            "decoder3_upsampler": nn.Sequential(
                Conv2DBlock(2 * e[-3], e[-3], dropout=drop_rate),
                Conv2DBlock(e[-3], e[-4], dropout=drop_rate),
                nn.ConvTranspose2d(e[-4], e[-4], 2, stride=2)),
            "decoder2_upsampler": nn.Sequential(
                Conv2DBlock(2 * e[-4], e[-4], dropout=drop_rate),
                nn.ConvTranspose2d(e[-4], e[-4], 2, stride=2)),
            "decoder1_upsampler": nn.Sequential(
                Conv2DBlock(e[-4], e[-4], dropout=drop_rate),
                nn.ConvTranspose2d(e[-4], e[-4], 2, stride=2)),
        })
        self.decoder0 = Conv2DBlock(3, e[-4], 3, dropout=drop_rate)
        # ---- 官方三头 (np/hv/tp 同结构: Conv2DBlock(2e, e) + 1×1 conv) ----
        def head(out_ch):
            return nn.Sequential(
                Conv2DBlock(2 * e[-4], e[-4], dropout=drop_rate),
                nn.Conv2d(e[-4], out_ch, 1, stride=1, padding=0))
        self.np_head, self.hv_head, self.tp_head = head(2), head(2), head(6)
        # ---- no-label 通道: 零权重 + bias=−8 (初始恒 −8, 不干扰 argmax) ----
        self.type_extra = nn.Conv2d(6, 1, 1)
        nn.init.zeros_(self.type_extra.weight)
        nn.init.constant_(self.type_extra.bias, -8.0)

    def forward(self, x):
        x = (x - 0.5) / 0.5                               # 官方预处理
        feats = self.encoder(x)                           # t8: 48@64, 96@32, 192@16, 384@8
        z1, z2, z3, z4 = feats
        d = self.decoder
        b5 = d["bottleneck_upsampler"](z4)
        b4 = d["decoder4_upsampler"](torch.cat([z3, b5], dim=1))
        b3 = d["decoder3_upsampler"](torch.cat([z2, b4], dim=1))
        b2 = d["decoder2_upsampler"](torch.cat([z1, b3], dim=1))
        b1 = d["decoder1_upsampler"](b2)
        xt = torch.cat([self.decoder0(x), b1], dim=1)     # 官方: decoder0 输入 forward 入参(归一化后)
        np2 = self.np_head(xt)                            # (B,2,H,W) softmax 语义
        tp6 = self.tp_head(xt)                            # (B,6,H,W)
        return {"np_logits": np2[:, 1:2] - np2[:, 0:1],   # sigmoid(z1−z0)=P(fg)
                "hv": self.hv_head(xt),
                "type_logits": torch.cat([tp6, self.type_extra(tp6)], dim=1),
                "feats": {"s2": z1, "s3": z2, "s4": z3},  # stride 4/8/16 抽头
                "gate_w": None}


def load_nulite_t_weights(model: NuLiteT, ckpt_path: str) -> dict:
    """加载官方 NuLite 权重 (Zenodo 13272655, arch=NuLite-T) 供 W5 蒸馏 teacher。

    实测 (2026-10-05): 官方发布权重的 model.backbone=**fastvit_s12**
    (config 元数据实锤; 此前「NuLite-T=fastvit_t8」系对官方 training_example.yaml
    示例配置的误推断)。故 teacher 须以 NuLiteT(backbone="fastvit_s12") 构造。

    兼容顶层键 {"state_dict"/"model"/"model_state_dict"} 或裸 state_dict;
    官方旧版 timm 键名映射: encoder.fast_vit.→encoder., decoder0.0.→decoder0.;
    classifier_head.* (19 tissue 分类头) 与 type_extra.* 为预期 missing/unexpected。
    """
    import torch as _t
    ck = _t.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck.get("model_state_dict", ck)))
    if not hasattr(sd, "items"):
        raise KeyError(f"checkpoint 无 state_dict: {list(ck.keys())}")
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    sd = {k.replace("encoder.fast_vit.", "encoder.").replace("decoder0.0.", "decoder0."): v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected)}


def build_nulite_t(n_type: int = 7, pretrained: bool = True, **kw) -> NuLiteT:
    return NuLiteT(n_type=n_type, pretrained=pretrained, **kw)
