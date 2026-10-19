# -*- coding: utf-8 -*-
"""NuLite-T 适配冒烟测试 (W1 遗留项②) —— 可复跑, 独立于 smoke_test_w1 套件。

验证: 注册路径构建 / 前向 shapes / 与 LightUNet 同接口 / 复合损失兼容 /
BGKD teacher 约定 (feats s2/s3/s4) / 参数量打印。
首次运行会下载 fastvit_t8.apple_in1k 预训练权重 (timm, ~17MB)。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boundary_distill import BGKDLoss
from models import BoundaryCompositeLoss, LightUNet, build_model


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 注册路径构建 (train.py 同款调用)
    model = build_model("nulite_t", base=32, use_sobel_gate=False, use_rab=False)
    model = model.to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    assert model.backbone == "fastvit_t8", model.backbone

    # 2) 前向 shapes
    x = torch.rand(2, 3, 256, 256, device=dev)
    with torch.inference_mode():
        out = model(x)
    assert out["np_logits"].shape == (2, 1, 256, 256), out["np_logits"].shape
    assert out["hv"].shape == (2, 2, 256, 256), out["hv"].shape
    assert out["type_logits"].shape == (2, 7, 256, 256), out["type_logits"].shape
    for s, hw in (("s2", 64), ("s3", 32), ("s4", 16)):
        assert out["feats"][s].shape == (2, out["feats"][s].shape[1], hw, hw), s
    # no-label 通道初始恒 −8 → argmax 不会选 6
    assert torch.allclose(out["type_logits"][:, 6], torch.full_like(out["type_logits"][:, 6], -8.0))
    print(f"[1-2] 注册构建 + 前向 OK; 参数量 {n_par:.2f}M; "
          f"feats 通道 s2/s3/s4 = {out['feats']['s2'].shape[1]}/"
          f"{out['feats']['s3'].shape[1]}/{out['feats']['s4'].shape[1]}")

    # 3) 复合损失兼容 (与 train.py 同路径)
    model.train()
    batch = {
        "img": x,
        "np": (torch.rand(2, 1, 256, 256, device=dev) > 0.5).float(),
        "hv": torch.rand(2, 2, 256, 256, device=dev) * 2 - 1,
        "inst": torch.zeros(2, 256, 256, dtype=torch.long, device=dev),
        "type": torch.zeros(2, 256, 256, dtype=torch.long, device=dev),
    }
    batch["inst"][:, 64:192, 64:192] = 1
    batch["type"][:, 64:192, 64:192] = 3
    loss, parts = BoundaryCompositeLoss(use_boundary=True)(model(x), batch)
    assert torch.isfinite(loss), loss
    loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert g > 0, "无梯度"
    print(f"[3] 复合损失 OK: loss={float(loss):.4f} parts={ {k: round(float(v), 4) for k, v in parts.items()} }")

    # 4) BGKD teacher 约定: student=LightUNet (训练图内), teacher=NuLiteT (inference_mode, 真实用法)
    model.eval()
    stu = LightUNet().to(dev).eval()
    so = stu(x)
    with torch.inference_mode():                       # 模拟 W5 真实蒸馏的 teacher 调用
        to = model(x)
    bgkd = BGKDLoss(student_channels={"s2": 128, "s3": 256, "s4": 512},
                    teacher_channels={"s2": to["feats"]["s2"].shape[1],
                                      "s3": to["feats"]["s3"].shape[1],
                                      "s4": to["feats"]["s4"].shape[1]}).to(dev)
    l_bd, bd_parts = bgkd(so["feats"], to["feats"], so["type_logits"], to["type_logits"],
                          batch["inst"])   # L_kd 用 7 通道 type_logits (单通道 softmax 恒 1, KL 恒 0)
    assert torch.isfinite(l_bd), l_bd
    print(f"[4] BGKD 兼容 OK: l_bd={float(l_bd):.4f} parts={ {k: round(float(v), 4) for k, v in bd_parts.items()} }")

    print("[ALL-PASS] nulite_adapter 冒烟全部通过")


if __name__ == "__main__":
    main()
