# -*- coding: utf-8 -*-
"""
两个边界模块（对应执行包 §2.4 的方法段落）:
  * SobelGateMS                     —— 多尺度 Sobel 先验 + 可学习门控（正偏置近恒等初始化）
  * UncertaintyBandReverseAttention —— 4σ(P)(1−σ(P)) 不确定带反向注意力（残差门控, 尾卷积零初始化）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SobelGateMS", "UncertaintyBandReverseAttention"]


# ---------------------------------------------------------------- 固定 Sobel 核（不可学习）
def _norm_kernel(k: torch.Tensor) -> torch.Tensor:
    """L1 归一化到 1, 使 3×3 与 5×5 的响应尺度可比。"""
    return (k / k.abs().sum()).reshape(1, 1, *k.shape)


_SOBEL3_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_SOBEL3_Y = _SOBEL3_X.T.contiguous()
_G5 = torch.tensor([1., 4., 6., 4., 1.])
_D3 = torch.tensor([-1., 0., 1.])
_SOBEL5_X = torch.outer(_G5, _D3)          # 行=高斯平滑, 列=差分 → x 方向梯度
_SOBEL5_Y = torch.outer(_D3, _G5)


def _depthwise_fixed_conv(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """同一固定核对每个通道独立卷积。x:(B,C,H,W), k:(1,1,kh,kw)。

    以 (1,B·C,H,W) + groups=B·C 分组卷积实现——把 batch 折叠进通道维,
    每个 (样本,通道) 对独立做一次固定核卷积, 对任意 batch 均成立;
    旧实现 (1,B·C,H,W)+groups=C 要求权重形状为 (C,B,kh,kw), 仅 batch=1 时合法
    (batch>1 训练会在首次前向报权重形状错误——smoke test 已加入 batch=4 回归测试)。
    """
    b, c, h, w = x.shape
    kh, kw = k.shape[-2:]
    weight = k.reshape(1, 1, kh, kw).expand(b * c, 1, kh, kw).contiguous()
    out = F.conv2d(x.reshape(1, b * c, h, w), weight,
                   padding=(kh // 2, kw // 2), groups=b * c)
    return out.reshape(b, c, h, w)


class SobelGateMS(nn.Module):
    """多尺度 Sobel 门控。

        z = Conv1×1( ReLU( Conv1×1([G, ∇₃G, ∇₅G]) ) )
        W = σ(z + b),   out = G ⊙ W

    * 双尺度梯度: 3×3 捕捉锐利核边界, 5×5（高斯平滑×差分）捕捉较厚/模糊边界
    * b 初始化为 +2.0 → W≈0.88 恒定 → 训练初期近似恒等缩放, 不破坏基线
      （直接回应审稿人"嵌入模块后为什么不掉点"的质疑）
    * 返回 (out, W): W 用于机制可视化（叠加在 H&E 上的门控热图, Discussion 素材）
    """

    def __init__(self, in_channels: int, gate_channels: int = None, bias_init: float = 2.0):
        super().__init__()
        gate_channels = gate_channels or max(in_channels // 4, 16)
        self.proj = nn.Conv2d(in_channels * 5, gate_channels, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Conv2d(gate_channels, 1, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, bias_init)
        for name, k in (("k3x", _SOBEL3_X), ("k3y", _SOBEL3_Y),
                        ("k5x", _SOBEL5_X), ("k5y", _SOBEL5_Y)):
            self.register_buffer(name, _norm_kernel(k), persistent=False)

    def forward(self, g: torch.Tensor):
        g3x = _depthwise_fixed_conv(g, self.k3x)
        g3y = _depthwise_fixed_conv(g, self.k3y)
        g5x = _depthwise_fixed_conv(g, self.k5x)
        g5y = _depthwise_fixed_conv(g, self.k5y)
        z = self.gate(self.act(self.proj(torch.cat([g, g3x, g3y, g5x, g5y], dim=1))))
        w = torch.sigmoid(z)
        return g * w, w


class UncertaintyBandReverseAttention(nn.Module):
    """不确定带反向注意力 (uncertainty-guided reverse attention)。

    朴素反向注意力 A = 1−σ(P) 在纯背景同样 ≈1, 把算力浪费在无核区域;
    本实现用熵最大化指示器 4σ(P)(1−σ(P))（在 σ(P)=0.5 处取峰=1）作带限:

        A_ub(x) = (1 − σ(P(x))) · 4·σ(P(x))·(1 − σ(P(x)))

    注意力精确聚焦"决策不确定的过渡带"（即边界）而非全部背景。

    融合方式为残差门控:  out = F_skip + A_ub ⊙ Refine(F_skip)
    * Refine 尾卷积零初始化 → 模块嵌入初期严格恒等（不扰动基线收敛）
    * p_logits 需先上采样到 F_skip 分辨率再传入
    """

    def __init__(self, in_channels: int, mid_channels: int = None, normalize: bool = True):
        super().__init__()
        mid = mid_channels or max(in_channels // 4, 16)
        self.normalize = normalize
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 3, padding=1, bias=False),
        )
        nn.init.zeros_(self.refine[-1].weight)

    def forward(self, f_skip: torch.Tensor, p_logits: torch.Tensor,
                return_attn: bool = False):
        p = torch.sigmoid(p_logits)
        a = (1.0 - p) * (4.0 * p * (1.0 - p))
        if self.normalize:
            a = a / (a.amax(dim=(2, 3), keepdim=True) + 1e-6)
        out = f_skip + self.refine(f_skip) * a
        if return_attn:
            return out, a
        return out
