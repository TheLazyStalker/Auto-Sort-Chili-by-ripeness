"""Task 5 — Benchmark CPU (container): loading time, latency, size comparison.

Đo cho: Keras .keras, TFLite Float32, TFLite Float16.
Methodology:
- Warm-up 5 runs trước khi đo.
- 50 runs/lần, input 1×224×224×3 float32 [0,255].
- Loading time: tf.keras.models.load_model / tf.lite.Interpreter(model_path) + allocate.
Kết quả CPU container KHÔNG đại diện cho điện thoại Android (mẫu đã test) (Task 6).
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


def measure_keras(path: str, warmup: int = 5, runs: int = 50):
    t0 = time.perf_counter()
    model = tf.keras.models.load_model(path)
    load_s = time.perf_counter() - t0
    x = np.random.RandomState(1).uniform(0, 255, size=(1, 224, 224, 3)).astype(np.float32)
    for _ in range(warmup):
        model.predict(x, verbose=0)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model.predict(x, verbose=0)
        times.append((time.perf_counter() - t) * 1000.0)
    return {"load_time_s": load_s, "latency_ms": times}


def measure_tflite(path: str, warmup: int = 5, runs: int = 50):
    t0 = time.perf_counter()
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    load_s = time.perf_counter() - t0
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    x = np.random.RandomState(1).uniform(0, 255, size=(1, 224, 224, 3)).astype(np.float32)
    if in_d["dtype"] == np.float16:
        x = x.astype(np.float16)
    for _ in range(warmup):
        interp.set_tensor(in_d["index"], x)
        interp.invoke()
        _ = interp.get_tensor(out_d["index"])
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        interp.set_tensor(in_d["index"], x)
        interp.invoke()
        _ = interp.get_tensor(out_d["index"])
        times.append((time.perf_counter() - t) * 1000.0)
    return {"load_time_s": load_s, "latency_ms": times}


def summarize(times):
    t = np.array(times)
    return {
        "mean_ms": float(t.mean()),
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "min_ms": float(t.min()),
        "max_ms": float(t.max()),
        "n": int(len(t)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 5 — CPU benchmark (container)")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    task4_dir = cfg["paths"]["artifacts_task4"]
    out_dir = os.path.join(os.path.dirname(task4_dir), "task5")
    ensure_dir(out_dir)

    keras_path = os.path.join(task4_dir, "best_chili.keras")
    f32_path = os.path.join(out_dir, "chili_float32.tflite")
    f16_path = os.path.join(out_dir, "chili_float16.tflite")

    bench = {}
    bench["keras"] = measure_keras(keras_path, args.warmup, args.runs)
    bench["float32"] = measure_tflite(f32_path, args.warmup, args.runs)
    bench["float16"] = measure_tflite(f16_path, args.warmup, args.runs)

    sizes = {}
    for name, p in (("keras", keras_path), ("float32", f32_path), ("float16", f16_path)):
        sizes[name] = {"bytes": os.path.getsize(p), "mb": os.path.getsize(p) / 1e6}
    sizes["keras"]["reduction_vs_keras"] = 0.0
    sizes["float32"]["reduction_vs_keras"] = 1 - sizes["float32"]["bytes"] / sizes["keras"]["bytes"]
    sizes["float16"]["reduction_vs_keras"] = 1 - sizes["float16"]["bytes"] / sizes["keras"]["bytes"]

    result = {
        "device": "CPU (Podman ai-workspace) — NOT điện thoại Android (mẫu đã test)",
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "input_shape": [1, 224, 224, 3],
        "note": "Preliminary; device targets (≤1s/pepper) not claimed until Task 6 on hardware.",
        "models": {k: {"load_time_s": v["load_time_s"], "latency": summarize(v["latency_ms"])}
                   for k, v in bench.items()},
        "size_comparison": sizes,
    }
    with open(os.path.join(out_dir, "latency_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "model_size_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(sizes, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved: {os.path.join(out_dir, 'latency_benchmark.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
