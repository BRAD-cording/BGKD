# -*- coding: utf-8 -*-
"""4060 8GB 训练吞吐优化要点模板（片段, 按需并入你的训练脚本; 不可直接运行）"""
import numpy as np
import torch

torch.set_float32_matmul_precision("high")   # TF32
torch.backends.cudnn.benchmark = True        # 输入尺寸固定时开启


def seed_everything(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---- 模型与优化器（示意） ----
# model = MyNuSegModel().cuda().to(memory_format=torch.channels_last)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# ---- 训练循环（示意）: PanNuke 全量 uint8 已缓存内存(~1.5GB), 数据加载零瓶颈 ----
# for x, y, inst in loader:
#     x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
#     # Ada(4060) 建议 bfloat16, 无需 GradScaler:
#     with torch.autocast("cuda", dtype=torch.bfloat16):
#         logits, feats = model(x)
#         loss = criterion(logits, y, feats, inst)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad(set_to_none=True)

# ---- 显存紧张时（CellViT 基线 / teacher+student 同卡蒸馏）: batch 减半 + 梯度累积×2 ----
# loss = loss / ACC
# loss.backward()
# if step_idx % ACC == ACC - 1:
#     optimizer.step(); optimizer.zero_grad(set_to_none=True)

# ---- 蒸馏纪律: teacher 冻结 + eval + inference_mode, 特征/logit 全部 detach 后再进 loss ----
# ---- 断点续训: 每 epoch 保存 {model, opt, sched, epoch, rng_state}, tmux/nohup 中跑 ----
