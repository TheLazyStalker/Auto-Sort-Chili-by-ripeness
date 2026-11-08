#!/usr/bin/env python3
"""Termux check_env — kiểm tra môi trường thiết bị trước khi chạy.

In: Android version (getprop), ABI, CPU, Python version, RAM, storage free,
     các gói Python quan trọng (tflite_runtime, PIL, numpy).

Dùng:  python check_env.py
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def main():
    info = {}
    # Android
    info["android_version"] = run("getprop ro.build.version.release") or "n/a"
    info["android_sdk"] = run("getprop ro.build.version.sdk") or "n/a"
    info["device"] = run("getprop ro.product.model") or "n/a"
    # ABI / CPU
    info["abi"] = run("getprop ro.product.cpu.abi") or platform.machine()
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    # storage free (Termux home)
    try:
        du = shutil.disk_usage(os.path.expanduser("~"))
        info["storage_free_mb"] = du.free // (1024 * 1024)
    except Exception:
        info["storage_free_mb"] = None
    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_total_mb"] = kb // 1024
                    break
    except Exception:
        info["ram_total_mb"] = None
    # packages
    pkgs = {"numpy": None, "PIL": None, "tflite_runtime": None,
            "tensorflow": None}
    for name in pkgs:
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "installed")
        except Exception:
            pkgs[name] = None
    info["packages"] = pkgs
    print(json.dumps(info, indent=2, ensure_ascii=False))
    # Gợi ý
    if pkgs["tflite_runtime"] is None:
        print("\n[GỢI Ý] Cài runtime: pip install tflite-runtime")
    if pkgs["PIL"] is None:
        print("\n[GỢI Ý] Cài Pillow: pip install pillow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
