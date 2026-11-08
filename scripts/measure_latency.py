"""Task 4 — đo latency inference CPU (development machine) cho model .keras.

Dùng:
  python scripts/measure_latency.py [--config config.yaml] [--model <path>]
                                    [--warmup 5] [--runs 50]

Phương pháp:
- Load model (TF 2.17, CPU).
- Warm-up N lần (compile/optimize kernels) trước khi đo.
- Đo M lần inference trên 1 tensor ngẫu nhiên 224×224×3 uint8 [0,255]
  (model tự áp preprocessing V3 bên trong).
- Báo mean/median/P95 (ms). Đây là đo sơ bộ trên máy phát triển — KHÔNG phải
  điện thoại Android (mẫu đã test) / OV2640; không claim target ≤1s/pepper.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from src.config import load_config
from src.utils import ensure_dir

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 4 — CPU inference latency")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = cfg["paths"]["artifacts_task4"]
    ensure_dir(out_dir)

    model_path = args.model or os.path.join(out_dir, "best_chili.keras")
    if not os.path.isfile(model_path):
        print(f"[LỖI] Không tìm thấy model: {model_path}")
        return 2

    model = tf.keras.models.load_model(model_path)
    x = tf.random.uniform((1, 224, 224, 3), 0, 255, dtype=tf.float32)

    # Warm-up
    for _ in range(args.warmup):
        model.predict(x, verbose=0)

    times_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        model.predict(x, verbose=0)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms = np.array(times_ms)
    result = {
        "model": model_path,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "device": "CPU (development machine — Podman ai-workspace)",
        "input_shape": [1, 224, 224, 3],
        "time_ms_mean": float(times_ms.mean()),
        "time_ms_median": float(np.median(times_ms)),
        "time_ms_p95": float(np.percentile(times_ms, 95)),
        "time_ms_min": float(times_ms.min()),
        "time_ms_max": float(times_ms.max()),
        "note": "Preliminary only; not điện thoại Android (mẫu đã test) / OV2640. Target 1s/pepper not claimed.",
    }
    out = os.path.join(out_dir, "latency_cpu.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
