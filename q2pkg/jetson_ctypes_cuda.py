# -*- coding: utf-8 -*-
"""libcudart ctypes shim —— pycuda 不可用时的零依赖 CUDA 内存/拷贝后端。

只实现 jetson_latency_bench.TrtRunner 用到的接口子集:
  mem_alloc(nbytes) / pagelocked_empty(shape, dtype) /
  memcpy_htod(dev, arr) / memcpy_dtoh(arr, dev)
全部走默认流同步语义 (cudaMemcpy 隐式同步), 与 pycuda 路径的
execute_async_v3(stream_handle=0) 时序一致, host 计时口径不变。
"""
import ctypes
import glob

import numpy as np


class CtypesCuda:
    H2D, D2H = 1, 2

    def __init__(self):
        cands = (sorted(glob.glob("/usr/local/cuda/lib64/libcudart.so*"))
                 + sorted(glob.glob("/usr/lib/aarch64-linux-gnu/libcudart.so*")))
        self.rt = None
        for c in cands:
            try:
                self.rt = ctypes.CDLL(c)
                break
            except OSError:
                continue
        if self.rt is None:
            raise RuntimeError("libcudart not found: " + "; ".join(cands))
        self.rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.c_size_t]
        self.rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.c_int]
        self.rt.cudaGetErrorString.restype = ctypes.c_char_p

    def _chk(self, rc):
        if rc != 0:
            msg = self.rt.cudaGetErrorString(rc)
            raise RuntimeError(f"CUDA error {rc}: {msg.decode(errors='replace')}")

    def mem_alloc(self, nbytes: int):
        p = ctypes.c_void_p()
        self._chk(self.rt.cudaMalloc(ctypes.byref(p), int(nbytes)))
        return p.value                                # int 设备地址

    @staticmethod
    def pagelocked_empty(shape, dtype):
        return np.empty(shape, dtype)                 # 同步拷贝无需锁页

    def memcpy_htod(self, dev, arr):
        arr = np.ascontiguousarray(arr)
        self._chk(self.rt.cudaMemcpy(ctypes.c_void_p(dev),
                                     arr.ctypes.data_as(ctypes.c_void_p),
                                     arr.nbytes, self.H2D))

    def memcpy_dtoh(self, arr, dev):
        self._chk(self.rt.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_void_p(dev),
                                     arr.nbytes, self.D2H))
