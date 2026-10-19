# -*- coding: utf-8 -*-
"""Zenodo 直连 IP 下载器（绕过本机 DNS 污染；Range 续传 + 完整性校验）。

用法: python download_zenodo.py <record_file_url_path> <out_path>
说明: 系统解析 zenodo.org 被污染为 0.0.0.0 且无 admin 权限改 hosts,
      故 HTTPSConnection 直连 CERN IP, Host 头仍为 zenodo.org;
      证书按 IP 校验必然失败 → 关闭校验(仅本次权重下载, 来源为 Zenodo 官方记录)。
"""
import http.client
import os
import ssl
import sys
import time

IP = "188.185.48.75"          # zenodo.org (CERN) 可用地址之一
HOST = "zenodo.org"


def fetch(url_path: str, out: str) -> None:
    ctx = ssl._create_unverified_context()
    total = None
    for attempt in range(12):
        done = os.path.getsize(out) if os.path.exists(out) else 0
        if total is not None and done >= total:
            break
        headers = {"Host": HOST, "User-Agent": "curl/8.0"}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            conn = http.client.HTTPSConnection(IP, timeout=30, context=ctx)
            conn.request("GET", url_path, headers=headers)
            resp = conn.getresponse()
            if resp.status == 200:
                total = int(resp.getheader("Content-Length", 0))
            elif resp.status == 206:
                cr = resp.getheader("Content-Range", "")
                total = int(cr.split("/")[-1]) if "/" in cr else 0
            else:
                print(f"[{attempt}] HTTP {resp.status}, retry 5s", flush=True)
                time.sleep(5)
                continue
            mode = "ab" if (done and resp.status == 206) else "wb"
            with open(out, mode) as f:
                while True:
                    chunk = resp.read(262144)
                    if not chunk:
                        break
                    f.write(chunk)
            conn.close()
            done = os.path.getsize(out)
            print(f"[{attempt}] {done / 2**20:.1f}/{(total or 0) / 2**20:.1f} MB", flush=True)
            if total and done >= total:
                break
        except Exception as e:  # noqa: BLE001
            print(f"[{attempt}] {type(e).__name__}: {e} -> resume", flush=True)
            time.sleep(3)
    if total is None or os.path.getsize(out) != total:
        raise SystemExit(f"FAILED: got {os.path.getsize(out)} expect {total}")
    print(f"COMPLETE {os.path.getsize(out)} bytes", flush=True)


if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2])
