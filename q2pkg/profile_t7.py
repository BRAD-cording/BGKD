# -*- coding: utf-8 -*-
"""T7 三方法 (cellvit_t256 / cellvit_sam256 / ddbunet) 独占 GPU 步时剖析。

前提: 必须在队列暂停 (无其他训练进程) 时运行, 否则 8GB 卡会被主链占满,
本进程分配走 CUDA Sysmem Fallback, 数字放大 ~100 倍 (2026-10-05 实证)。
输出: 各模型各 batch 的 fwd+bwd+opt.step 均值步时 + 峰值显存,
写入 C:\\runs\\t7_profile.json; own_baseline b32 作为已有 132.8s/epoch 的锚点。
随机初始化测速 (与权重无关), pretrained=False 跳过 ckpt 装载。
"""
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from models import build_model  # noqa: E402

DEV = "cuda"
OUT_JSON = Path(r"C:\runs\t7_profile.json")


def build(name: str):
    """build_model 包装: 三方法支持 pretrained=False; own_baseline 无该参。"""
    if name == "own_baseline":
        return build_model(name)
    try:
        return build_model(name, pretrained=False)
    except TypeError:
        return build_model(name)


def fake_loss(out: dict) -> torch.Tensor:
    """全分支求和 (等价于所有头都有梯度), loss 计算本身开销可忽略。"""
    s = None
    for v in out.values():
        if torch.is_tensor(v) and v.is_floating_point():
            s = v.float().mean() if s is None else s + v.float().mean()
    return s


def bench(name: str, b: int, n_warm: int = 3, n_time: int = 5) -> dict:
    for attempt in range(2):
        try:
            m = build(name).to(DEV).train()
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"model": name, "batch": b, "error": "OOM at build"}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.rand(b, 3, 256, 256, device=DEV)
    torch.cuda.reset_peak_memory_stats()
    # warmup (含 cudnn/cublas 选择)
    for _ in range(n_warm):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            loss = fake_loss(m(x))
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_time):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            loss = fake_loss(m(x))
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    step_sec = (time.perf_counter() - t0) / n_time
    free, total = torch.cuda.mem_get_info()
    res = {
        "model": name,
        "batch": b,
        "step_sec": round(step_sec, 4),
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu_total_used_gb": round((total - free) / 1e9, 2),
    }
    print(f"{name:16s} b{b:<3d} step={step_sec:.3f}s "
          f"peak={res['peak_alloc_gb']}GB used={res['gpu_total_used_gb']}GB", flush=True)
    del m, opt, x, loss
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    return res


def main():
    assert torch.cuda.is_available()
    plan = [
        ("own_baseline", 32),          # 锚点: 已知 132.8s/epoch (b32)
        ("cellvit_t256", 8),
        ("cellvit_t256", 16),
        ("cellvit_sam256", 4),
        ("cellvit_sam256", 8),
        ("cellvit_sam256", 12),
        ("ddbunet", 16),
        ("ddbunet", 32),
    ]
    results = []
    for name, b in plan:
        try:
            results.append(bench(name, b))
        except torch.cuda.OutOfMemoryError:
            print(f"{name} b{b}: OOM, 跳过", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"{name} b{b}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"已写入 {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
