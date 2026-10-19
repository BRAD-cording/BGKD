#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson (Nano / Orin Nano) 标准化延迟-能耗基准 —— 对应执行包审查表 E 区。
修复原审稿意见 P1-5（延迟测量协议不可信）与 P2-7（平台过时/能耗缺失）。

协议（论文部署节必须原文声明）:
  1) sudo nvpmodel -m 0 && sudo jetson_clocks --fan    # MAXN + 锁频
  2) 预热 warmup 次推理, 全部丢弃（排除 TRT lazy init / 内存分配抖动）
  3) 正式测量 iters 次(默认≥500), 逐次 host 计时并同步
  4) 报告 median / P95 / P99 / mean±std —— 结论不得只引用均值
  5) 记录测量期 CPU/GPU 温度（过热降频是 Nano 的知名问题）
  6) 功耗: 优先 tegrastats 解析 VDD_IN, 其次 INA3221(sysfs);
     能耗主口径 E = P̄×t̄(mean latency), median/P95 能耗作稳健参考 [1 W·ms = 1 mJ]

依赖: tensorrt + pycuda（JetPack 自带）。TrtRunner 按 TRT 版本自动探测 API:
      TRT>=8.5/10 (Orin, JetPack 5/6) 走 tensor address V3 接口;
      TRT 8.2 (Nano, JetPack 4.6) 走旧 binding 接口。设备侧无需改代码。

用法:
  python jetson_latency_bench.py --engine model_fp16.trt --warmup 50 --iters 500 \
      --out-json bench_fp16.json --out-csv bench_fp16_raw.csv
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import threading
import time

import numpy as np


# ---------------------------------------------------------------- 环境控制
def lock_clocks(do_lock: bool = True):
    if not do_lock:
        print("[warn] 未锁频! 正式论文数据不得在未锁频下采集。")
        return
    for cmd in (["sudo", "-n", "nvpmodel", "-m", "0"],
                ["sudo", "-n", "jetson_clocks", "--fan"]):
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            print(f"[ok] {' '.join(cmd)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {cmd} 执行失败({exc}); 请手动执行后重测")


def read_temps() -> dict:
    temps = {}
    for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            with open(os.path.join(zone, "type")) as f:
                name = f.read().strip()
            with open(os.path.join(zone, "temp")) as f:
                temps[name] = int(f.read().strip()) / 1000.0
        except Exception:  # noqa: BLE001  Jetson 部分 thermal zone 读数返回 EIO(TypeError)
            continue
    return temps


# ---------------------------------------------------------------- 功耗监测
def find_ina3221():
    for dev in sorted(glob.glob("/sys/bus/iio/devices/iio:device*")):
        try:
            with open(os.path.join(dev, "name")) as f:
                if "ina3221" in f.read().lower():
                    return dev
        except OSError:
            continue
    return None


def ina3221_power_w(dev):
    """通道1 通常为模块总输入(VDD_IN), 具体以载板原理图为准。"""
    try:
        with open(os.path.join(dev, "in_voltage1_input")) as f:
            v = float(f.read()) / 1000.0
        with open(os.path.join(dev, "in_current1_input")) as f:
            i = float(f.read()) / 1000.0
        return v * i
    except (OSError, ValueError):
        return None


class TegraStatsMonitor:
    """后台运行 tegrastats, 解析 VDD_IN 平均功耗(W)。"""

    _PAT = re.compile(r"VDD_IN (\d+)mW")

    def __init__(self):
        self.proc = None
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", "100"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except (FileNotFoundError, OSError):
            print("[warn] tegrastats 不可用, 回退 INA3221/无功耗记录")
            return
        self._thread = threading.Thread(target=self._tail, daemon=True)
        self._thread.start()

    def _tail(self):
        try:
            for line in self.proc.stdout:
                if self._stop.is_set():
                    break
                m = self._PAT.search(line)
                if m:
                    self.samples.append(int(m.group(1)) / 1000.0)  # mW → W
        except (ValueError, OSError):
            pass

    def stop(self):
        self._stop.set()
        if self.proc:
            self.proc.terminate()
        if self._thread:
            self._thread.join(timeout=2)
        return float(np.mean(self.samples)) if self.samples else None


# ---------------------------------------------------------------- TRT 推理
class TrtRunner:
    """单输入 / 多输出 batch=1 静态引擎(细胞核模型导出 NP/HV 两输出头)。
    API 自动探测:
      * TRT >= 8.5 / TRT 10 (Orin, JetPack 5/6): num_io_tensors / get_tensor_name /
        context.set_tensor_address / execute_async_v3
      * TRT 8.2 旧 binding API (Nano, JetPack 4.6): num_bindings / binding_is_input /
        execute_v2
    infer 返回 np 数组列表 (与 engine 输出顺序一致)。
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt          # 仅 Jetson 存在
        try:
            import pycuda.autoinit      # noqa: F401  pylint: disable=unused-import
            import pycuda.driver as cuda
        except ImportError:             # pycuda 编译失败 → ctypes libcudart 回退
            from jetson_ctypes_cuda import CtypesCuda
            cuda = CtypesCuda()
            print("[backend] pycuda 不可用, 使用 ctypes libcudart (同步默认流)")

        self._cuda = cuda
        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._v3 = hasattr(self.engine, "num_io_tensors")   # TRT>=8.5/10

        if self._v3:
            names = [self.engine.get_tensor_name(i)
                     for i in range(self.engine.num_io_tensors)]
            self.input_name = next(
                n for n in names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
            self.output_names = [n for n in names
                                 if self.engine.get_tensor_mode(n)
                                 == trt.TensorIOMode.OUTPUT]
            if not self.output_names:
                raise RuntimeError("no output tensor")
            self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
            self.in_dtype = self._np_dtype(self.input_name)
            self.h_in = cuda.pagelocked_empty(self.input_shape, self.in_dtype)
            self.d_in = cuda.mem_alloc(self.h_in.nbytes)
            self.context.set_tensor_address(self.input_name, int(self.d_in))
            self.h_outs, self.d_outs = [], []
            for n in self.output_names:
                dtype = self._np_dtype(n)
                h = cuda.pagelocked_empty(
                    tuple(self.engine.get_tensor_shape(n)), dtype)
                self.h_outs.append(h)
                self.d_outs.append(cuda.mem_alloc(h.nbytes))
                self.context.set_tensor_address(n, int(self.d_outs[-1]))
        else:                                # 旧 binding API (TRT 8.2)
            self.input_idx = self._find_binding(is_input=True)
            self.output_idxs = [i for i in range(self.engine.num_bindings)
                                if not self.engine.binding_is_input(i)]
            if not self.output_idxs:
                raise RuntimeError("no output binding")
            self.input_shape = tuple(self.engine.get_binding_shape(self.input_idx))
            self.in_dtype = self._np_dtype_legacy(self.input_idx)
            self.h_in = cuda.pagelocked_empty(self.input_shape, self.in_dtype)
            self.d_in = cuda.mem_alloc(self.h_in.nbytes)
            self.h_outs, self.d_outs = [], []
            for i in self.output_idxs:
                dtype = self._np_dtype_legacy(i)
                h = cuda.pagelocked_empty(
                    tuple(self.engine.get_binding_shape(i)), dtype)
                self.h_outs.append(h)
                self.d_outs.append(cuda.mem_alloc(h.nbytes))
            self.bindings = [0] * len(self.engine)
            self.bindings[self.input_idx] = int(self.d_in)
            for i, d in zip(self.output_idxs, self.d_outs):
                self.bindings[i] = int(d)

    def _np_dtype(self, name):
        trt = self._trt
        return (np.float16 if self.engine.get_tensor_dtype(name) == trt.DataType.HALF
                else np.float32)

    def _np_dtype_legacy(self, idx):
        trt = self._trt
        return (np.float16 if self.engine.get_binding_dtype(idx) == trt.DataType.HALF
                else np.float32)

    def _find_binding(self, is_input: bool) -> int:
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i) == is_input:
                return i
        raise RuntimeError("binding not found")

    def infer(self, np_input: np.ndarray):
        np.copyto(self.h_in, np_input.astype(self.in_dtype))
        self._cuda.memcpy_htod(self.d_in, self.h_in)
        if self._v3:
            self.context.execute_async_v3(stream_handle=0)
        else:
            self.context.execute_v2(self.bindings)
        for h, d in zip(self.h_outs, self.d_outs):
            self._cuda.memcpy_dtoh(h, d)
        return self.h_outs      # 多输出列表; 单输出引擎取 outputs[0]


# ---------------------------------------------------------------- 统计汇总（纯函数, 便于单测）
def summarize(lat_ms, power_w=None, temps_start=None, temps_end=None,
              warmup=50, iters=500) -> dict:
    lat = np.asarray(lat_ms, float)
    rep = {
        "n_warmup": warmup, "n_measured": int(lat.size),
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "mean_ms": float(lat.mean()), "std_ms": float(lat.std(ddof=1)),
        "min_ms": float(lat.min()), "max_ms": float(lat.max()),
    }
    if power_w is not None:
        rep["power_mean_w"] = float(power_w)
        rep["energy_mean_mJ"] = float(power_w * rep["mean_ms"])       # 主口径 E = P̄·t̄, W×ms = mJ
        rep["energy_median_mJ"] = float(power_w * rep["median_ms"])   # 稳健参考
        rep["energy_p95_mJ"] = float(power_w * rep["p95_ms"])
    if temps_start:
        rep["temps_start_c"] = temps_start
    if temps_end:
        rep["temps_end_c"] = temps_end
    rep["protocol_ok"] = bool(warmup >= 50 and iters >= 500)
    return rep


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--no-lock", action="store_true", help="仅调试用; 正式数据禁止")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="bench_summary.json")
    ap.add_argument("--out-csv", default="bench_raw.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lock_clocks(not args.no_lock)
    runner = TrtRunner(args.engine)
    dummy = rng.standard_normal(runner.input_shape)

    for _ in range(args.warmup):                      # 预热, 全部丢弃
        runner.infer(dummy)

    temps_start = read_temps()
    monitor = TegraStatsMonitor()
    monitor.start()
    lat = np.empty(args.iters, dtype=np.float64)
    for i in range(args.iters):
        t0 = time.perf_counter()
        runner.infer(dummy)
        lat[i] = (time.perf_counter() - t0) * 1000.0
    power = monitor.stop()
    temps_end = read_temps()

    rep = summarize(lat, power, temps_start, temps_end,
                    warmup=args.warmup, iters=args.iters)
    rep["engine"] = args.engine

    print(json.dumps(rep, indent=2, ensure_ascii=False))
    with open(args.out_json, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "latency_ms"])
        w.writerows(enumerate(lat.tolist(), 1))
    print(f"[saved] {args.out_json} / {args.out_csv}")


def trtexec_layer_profile(onnx_path: str, out_dir: str = "profile", fp16: bool = True):
    """逐层 profile: 证明边界模块延迟增量 <2ms（审查表 E4）。
    返回可执行的 trtexec 命令; supplementary 需附完整命令与输出 CSV。"""
    os.makedirs(out_dir, exist_ok=True)
    tag = "fp16" if fp16 else "fp32"
    cmd = [
        "trtexec", f"--onnx={onnx_path}",
        "--dumpProfile", f"--exportProfile={out_dir}/layer_profile_{tag}.csv",
        f"--exportOutput={out_dir}/outputs_{tag}.json",
        "--fp16" if fp16 else "--fp32",
        "--iterations=200", "--avgRuns=50",
    ]
    print(" ".join(cmd))
    return cmd


if __name__ == "__main__":
    main()
