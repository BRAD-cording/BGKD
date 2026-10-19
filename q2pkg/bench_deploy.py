# -*- coding: utf-8 -*-
"""P1-2 部署基准: RTX 4060 (8GB) 独占 GPU, b1 fp32 网络前向延迟
=============================================================
测量: params(M) / latency median+P95 (ms) / throughput (patch/s) / peak VRAM (GB)
协议: eval 模式, 256x256 输入, 20 次预热 + 100 次计时 (torch.cuda.synchronize 包夹);
      不含 watershed 解码与指标提取 (仅网络前向, 对应正文 §5 声明)。
前提: 独占 GPU (受控停链后运行, 见项目记忆 8GB 卡并行禁令)。
输出: C:\\runs\\deploy_bench.json + stdout 表格; tables/deployment.tex 由结果另行覆盖。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch

from models import build_model

WARMUP, ITERS = 20, 100
OUT_JSON = Path(r"C:\runs\deploy_bench.json")

# (表内行名, build_model 参数) — own_full = SG+RAB 全开架构 (蒸馏不影响推理图)
ROWS = [
    ("NuLite-T (FastViT-T8)", ("nulite_t", {})),
    ("Ours (full)", ("own_baseline", {"use_sobel_gate": True, "use_rab": True})),
    ("NuLite-S (official)", ("nulite_s", {})),
    ("DDBU-Net (repro.)", ("ddbunet", {})),
    ("HoVer-Net (reimpl.)", ("hovernet", {})),
    ("CellViT-256", ("cellvit_t256", {})),
    ("CellViT-256-SAM", ("cellvit_sam256", {})),
]


def bench(name, build_kw):
    mname, mkw = build_kw
    model = build_model(mname, **mkw).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.rand(1, 3, 256, 256, device="cuda")
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    vram_gb = torch.cuda.max_memory_allocated() / 2**30
    times.sort()
    med = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95) - 1]
    del model, x
    torch.cuda.empty_cache()
    rec = {"params_M": round(n_params, 2), "median_ms": round(med, 2),
           "p95_ms": round(p95, 2), "thrpt_pps": round(1000.0 / med, 1),
           "vram_gb": round(vram_gb, 2)}
    print(f"{name:24s} {rec['params_M']:7.2f}M  {rec['median_ms']:7.2f} ms "
          f"(P95 {rec['p95_ms']:7.2f})  {rec['thrpt_pps']:7.1f} patch/s  {rec['vram_gb']:.2f} GB",
          flush=True)
    return rec


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA"
    used = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]
    assert used < 1.5 * 2**30, f"GPU 非独占 (已用 {used/2**30:.2f} GB), 先受控停链"
    results = {}
    for name, kw in ROWS:
        results[name] = bench(name, kw)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUT_JSON}")
