# -*- coding: utf-8 -*-
"""
边界引导知识蒸馏 (Boundary-Guided Knowledge Distillation, BGKD)
=================================================================
对应执行包 §2.1 论文方法小节 "Boundary-Guided Knowledge Distillation"。
修复原审稿意见 P0-1：Gap 3 与 H4 假设无实验支撑。

组成:
  1) BoundaryWeightedImitationLoss —— 边界带加权中间层特征模仿 (L_fe)
  2) BoundaryWeightedLogitKD      —— 边界加权 logit 蒸馏 (L_kd, 温度缩放 KL)
  3) BGKDLoss                     —— 组合封装: L = L_seg + λ_fe·L_fe + λ_kd·L_kd

约定:
  * instance_label: (B,H,W) 或 (B,1,H,W), GT 实例标签图(每核一个整数 ID, 背景=0),
    全分辨率(如 256×256); ID 需 < 2^24 以保证 float32 精确比较
  * student_feats / teacher_feats: dict{stage_name: (B,C,h,w)}, 各 stage 空间尺寸可不同
    (teacher 特征/logit 自动 nearest 对齐到 student 网格; 通道不一致由 1×1 adapter 对齐)
  * teacher 前向必须在调用方用 torch.inference_mode() 包裹, 传入张量应为 detach 后的
  * 默认超参: band r=5px, T=4, γ=0.7, λ_fe=λ_kd=0.3 —— 仅在一折上调参后固定(论文需声明)
"""
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["boundary_band", "BoundaryWeightedImitationLoss",
           "BoundaryWeightedLogitKD", "BGKDLoss"]


def boundary_band(instance_label: torch.Tensor, band_px: int = 5) -> torch.Tensor:
    """由 GT 实例标签图提取边界带: B = D_r(M) − E_r^(i)(M)。

    M = 标签>0 的 union 掩膜; E_r^(i) 为"逐实例腐蚀"——像素仅当半径 r 邻域内
    全为自身实例标签时才算内部。与对 union 掩膜直接腐蚀相比, 该定义把
    实例间接触界面(拥挤组织中主要误差来源)与宽度 <2r 的整个小核一并纳入边界带。
    max/min-pool 实现(方形结构元), GPU 友好、无额外依赖。
    返回 {0,1} 权重图, 形状与 instance_label 相同。
    """
    lab = instance_label.squeeze(1) if instance_label.dim() == 4 else instance_label
    L = lab.float().unsqueeze(1)                            # (B,1,H,W)
    if band_px <= 0:
        return (L > 0).float()
    k = 2 * band_px + 1
    dil = F.max_pool2d((L > 0).float(), kernel_size=k, stride=1, padding=band_px)
    mx = F.max_pool2d(L, kernel_size=k, stride=1, padding=band_px)      # 窗口内最大标签
    mn = -F.max_pool2d(-L, kernel_size=k, stride=1, padding=band_px)    # 窗口内最小标签(含背景0)
    interior = ((mx == L) & (mn == L) & (L > 0)).float()    # 自身实例内部: 邻域仅含自身标签
    return torch.clamp(dil - interior, min=0.0)


def _match_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把 x 的空间尺寸 nearest 对齐到 ref（band 降采样到特征分辨率; teacher 特征/logit 对齐到 student 网格）。"""
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="nearest")
    return x


class BoundaryWeightedImitationLoss(nn.Module):
    """L_fe = Σ_l [ Σ_x B(x)·||f̂_s^l(x) − f̂_t^l(x)||₂² ] / [ Σ_x B(x)·C_l ]

    * f̂: 沿通道维 L2 归一化后的特征（解耦激活尺度, 稳定 AMP 训练）
    * teacher 特征先 nearest 对齐到 student 空间网格; 通道数不一致时再用
      bias-free 1×1 adapter 对齐（adapter 属 student, 参与训练）
    * 分母为"边界带像素数×通道数" → 模仿预算完全集中在边界带
    * 全背景标签图(无核)时安全返回图连接的 0
    """

    def __init__(self, student_channels: Dict[str, int],
                 teacher_channels: Dict[str, int],
                 stages: Sequence[str] = ("s2", "s3", "s4"),
                 band_px: int = 5):
        super().__init__()
        self.stages = list(stages)
        self.band_px = band_px
        self.adapters = nn.ModuleDict()
        for s in self.stages:
            if student_channels[s] != teacher_channels[s]:
                self.adapters[s] = nn.Conv2d(
                    student_channels[s], teacher_channels[s],
                    kernel_size=1, bias=False)

    def forward(self, student_feats: Dict[str, torch.Tensor],
                teacher_feats: Dict[str, torch.Tensor],
                instance_label: torch.Tensor) -> torch.Tensor:
        device = instance_label.device
        total = torch.zeros((), device=device)
        denom = torch.zeros((), device=device)
        for s in self.stages:
            fs, ft = student_feats[s], teacher_feats[s]
            ft = _match_spatial(ft, fs)                     # teacher→student 空间对齐(修复: 架构网格不一致)
            if s in self.adapters:
                fs = self.adapters[s](fs)
            fs_n = F.normalize(fs, dim=1, eps=1e-6)
            ft_n = F.normalize(ft, dim=1, eps=1e-6)
            diff2 = ((fs_n - ft_n) ** 2).sum(dim=1, keepdim=True)          # (B,1,h,w)
            band = _match_spatial(boundary_band(instance_label, self.band_px), fs)
            total = total + (band * diff2).sum()
            denom = denom + band.sum() * fs.shape[1]
        return total / (denom + 1e-6)


class BoundaryWeightedLogitKD(nn.Module):
    """L_kd = T² · mean_x [ w̃(x)·KL( softmax(z_s/T) ‖ softmax(z_t/T) ) ]

    w̃(x) = (1−γ) + γ·B̂(x): 背景保留 (1−γ) 保底权重（避免退化为"只学边界"）,
    边界带权重抬升至 1。
    """

    def __init__(self, temperature: float = 4.0, gamma: float = 0.7, band_px: int = 5):
        super().__init__()
        self.T = temperature
        self.gamma = gamma
        self.band_px = band_px

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                instance_label: torch.Tensor) -> torch.Tensor:
        teacher_logits = _match_spatial(teacher_logits, student_logits)  # 防 head 网格不一致
        logp_s = F.log_softmax(student_logits / self.T, dim=1)
        p_t = F.softmax(teacher_logits / self.T, dim=1)
        kl = F.kl_div(logp_s, p_t, reduction="none").sum(dim=1, keepdim=True) * (self.T ** 2)
        band = _match_spatial(boundary_band(instance_label, self.band_px), student_logits)
        w = (1.0 - self.gamma) + self.gamma * band
        return (w * kl).mean()


class BGKDLoss(nn.Module):
    """BGKD 组合封装。训练循环内用法:

        with torch.inference_mode():
            t_out = teacher(x)          # teacher 已 eval()
        loss_seg = criterion(s_out, y)
        loss_bd, parts = bgkd(s_out["feats"], t_out["feats"],
                              s_out["type_logits"], t_out["type_logits"], inst_label)
        (loss_seg + loss_bd).backward()             # parts 用于日志监控 l_fe/l_kd
        # inst_label: GT 实例标签图 (B,H,W) 或 (B,1,H,W), 每核一个整数 ID, 背景=0

    L_kd 通道语义: *_logits 必须为多通道类别 logits (KL 沿 dim=1 softmax)。
    单通道 logits 会使 softmax 恒为 1 → KL 恒 0 (无效); NP 头若需蒸馏,
    以 2 通道 [−z, z] 形式传入 (sigmoid 蒸馏的等价 KL)。
    """

    def __init__(self, student_channels: Dict[str, int],
                 teacher_channels: Dict[str, int],
                 stages: Sequence[str] = ("s2", "s3", "s4"),
                 band_px: int = 5, temperature: float = 4.0,
                 gamma: float = 0.7, lambda_fe: float = 0.3, lambda_kd: float = 0.3):
        super().__init__()
        self.imit = BoundaryWeightedImitationLoss(
            student_channels, teacher_channels, stages, band_px)
        self.kd = BoundaryWeightedLogitKD(temperature, gamma, band_px)
        self.lambda_fe = lambda_fe
        self.lambda_kd = lambda_kd

    def forward(self, student_feats, teacher_feats,
                student_logits, teacher_logits, instance_label):
        # teacher 侧张量通常产自 torch.inference_mode() (推理图), 不能被保存用于反传;
        # 入口统一 clone 为普通张量 (t8 teacher s2/s3/s4+logits 合计 ~40MB @B32, 开销可忽略)
        teacher_feats = {k: v.clone() for k, v in teacher_feats.items()}
        teacher_logits = teacher_logits.clone()
        l_fe = self.imit(student_feats, teacher_feats, instance_label)
        l_kd = self.kd(student_logits, teacher_logits, instance_label)
        loss = self.lambda_fe * l_fe + self.lambda_kd * l_kd
        return loss, {"l_fe": l_fe.detach(), "l_kd": l_kd.detach()}
