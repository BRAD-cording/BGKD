#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""P1-2b: 部署表模型 → ONNX 导出 (Jetson Orin Nano Super TRT 链路第一步)
=======================================================================
与 bench_deploy.py 口径严格一致:
  * 同一 ROWS 方法集 (NuLite-T / Ours-full / NuLite-S / DDBU-Net / HoVer-Net /
    CellViT-256 / CellViT-256-SAM), build_model 随机初始化 (延迟/能耗是
    权重无关的架构属性, 4060 表即如此测得)
  * 静态图: 输入 1×3×256×256 [0,1]; 输出 = 解码所需最小集
      np_logits (1,1,256,256) + hv (1,2,256,256)   (type 头不参与部署指标)
  * opset 17, 无动态轴 (TRT 静态 engine 友好), fp32 导出 (精度交给 trtexec --fp16)

产出: C:\data\onnx_export\<key>.onnx + manifest.json
      (torch 版本 / opset / IO 形状 / onnxruntime 数值对拍 max|Δ|)
验证: onnx.checker + onnxruntime CPU 对拍 (max abs diff, torch CPU fp32 vs ORT fp32)

用法:
  python export_onnx.py            # 全部 7 个
  python export_onnx.py --only own_full nulite_t
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch

from models import build_model

OUT_DIR = Path(r"C:\data\onnx_export")
OPSET = 17

# key 与 bench_deploy.py ROWS 一一对应 (部署表行名在 manifest 里保留)
ROWS = {
    "nulite_t": ("NuLite-T (FastViT-T8)", ("nulite_t", {})),
    "own_full": ("Ours (full)", ("own_baseline", {"use_sobel_gate": True,
                                                  "use_rab": True})),
    "nulite_s": ("NuLite-S (official)", ("nulite_s", {})),
    "ddbunet": ("DDBU-Net (repro.)", ("ddbunet", {})),
    "hovernet": ("HoVer-Net (reimpl.)", ("hovernet", {})),
    "cellvit_t256": ("CellViT-256", ("cellvit_t256", {})),
    "cellvit_sam256": ("CellViT-256-SAM", ("cellvit_sam256", {})),
}


class OnnxWrap(torch.nn.Module):
    """管线模型的 forward 返回 dict (np_logits/hv/type_logits/feats/gate_w);
    legacy ONNX exporter 需要元组输出与 output_names 一一对应 → 只暴露部署所需两 heads。
    注意: 必须 eval() — 否则 exporter 按顶层 training=True 处理 (BN batch-stats/dropout)。"""

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.eval()

    def forward(self, x):
        out = self.m(x)
        return out["np_logits"], out["hv"]


def export_one(key: str, row_label: str, mname: str, mkw: dict) -> dict:
    torch.manual_seed(0)                              # 随机权重可复现
    model = build_model(mname, **mkw).eval()
    x = torch.rand(1, 3, 256, 256)
    # 参考前向必须在 export 之前算好 (export 的 tracing 会动 module 的 train/eval 状态)
    with torch.no_grad():
        ref_out = model(x)
        ref_np, ref_hv = ref_out["np_logits"].numpy(), ref_out["hv"].numpy()
        del ref_out
    onnx_p = OUT_DIR / f"{key}.onnx"
    t0 = time.time()
    with torch.no_grad():                             # inference_mode 张量进 trace 有已知坑
        torch.onnx.export(
            OnnxWrap(model), x, str(onnx_p), opset_version=OPSET,
            do_constant_folding=True,
            input_names=["input"], output_names=["np_logits", "hv"],
            dynamo=False)
    dt = time.time() - t0
    rec = {"key": key, "row": row_label, "model": mname, "arch_kwargs": mkw,
           "opset": OPSET, "export_sec": round(dt, 1),
           "input": [1, 3, 256, 256],
           "outputs": {"np_logits": [1, 1, 256, 256], "hv": [1, 2, 256, 256]}}
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    rec["params_M"] = round(n_params, 2)

    # ---- 验证 1: onnx.checker ----
    import onnx
    m = onnx.load(str(onnx_p))
    onnx.checker.check_model(m)
    rec["checker"] = "OK"

    # ---- 验证 2: onnxruntime CPU 数值对拍 ----
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
        o_np, o_hv = sess.run(None, {"input": x.numpy()})
        rec["parity_maxdiff_np"] = float(abs(ref_np - o_np).max())
        rec["parity_maxdiff_hv"] = float(abs(ref_hv - o_hv).max())
        rec["parity"] = ("OK" if max(rec["parity_maxdiff_np"],
                                     rec["parity_maxdiff_hv"]) < 1e-4 else "MISMATCH")
    except ImportError:
        rec["parity"] = "SKIPPED (onnxruntime 未安装)"
    print(f"[{key}] {row_label}: export {dt:.1f}s, checker OK, "
          f"parity={rec.get('parity')}"
          + (f" (np {rec['parity_maxdiff_np']:.2e} / hv {rec['parity_maxdiff_hv']:.2e})"
             if "parity_maxdiff_np" in rec else ""), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None,
                    help="仅导出指定 key (默认全部)")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = a.only or list(ROWS)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.exists() else {}
    results = {}
    for k in keys:
        if k not in ROWS:
            print(f"[skip] 未知 key: {k} (可选: {list(ROWS)})", flush=True)
            continue
        label, (mname, mkw) = ROWS[k]
        try:
            results[k] = export_one(k, label, mname, mkw)
        except Exception as e:
            results[k] = {"key": k, "row": label, "error":
                          f"{type(e).__name__}: {e}"}
            print(f"[{k}] 导出失败: {type(e).__name__}: {e}", flush=True)
    manifest.update(results)
    manifest["_meta"] = {"torch": torch.__version__, "opset": OPSET,
                         "note": "随机初始化权重 (与 bench_deploy.py 口径一致); "
                                 "TRT engine 需在 Orin 上 trtexec --fp16 构建"}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"\nmanifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
