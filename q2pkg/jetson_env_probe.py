# -*- coding: utf-8 -*-
"""Jetson 设备侧环境探测: numpy/tensorrt/pycuda 绑定 + trtexec 版本 + 温度。"""
import importlib
import platform

print("python", platform.python_version())
for m in ("numpy", "tensorrt", "pycuda"):
    try:
        mod = importlib.import_module(m)
        print(m, "OK", getattr(mod, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        print(m, "FAIL", type(e).__name__, str(e)[:80])
