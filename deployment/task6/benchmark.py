#!/usr/bin/env python3
"""Termux benchmark — đo latency TFLite trên thiết bị thật.

Cách dùng:
  python benchmark.py --model models/chili_float16.tflite --warmup 10 --runs 100
  python benchmark.py --model float32 --runs 100

Báo cáo: load_time_s, mean/median/p95/min/max (ms), n_runs.
Đầu vào dùng tensor ngẫu nhiên (1,224,224,3) float32 [0,255] — preprocessing
nằm ngoài vòng đo; riêng end-to-end đo trong predict.py.

Lưu ý: kết quả trên máy container KHÔNG đại diện cho điện thoại Android (mẫu đã test).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
    RUNTIME = "tflite_runtime"
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    RUNTIME = "tensorflow.lite"

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_SIZE = 224


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark TFLite (Task 6)")
    parser.add_argument("--model", default=os.path.join(HERE, "models",
                                                        "chili_float16.tflite"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    model_path = args.model
    if model_path in ("float16", "float32"):
        model_path = os.path.join(HERE, "models", f"chili_{model_path}.tflite")
    if not os.path.isfile(model_path):
        print(f"[LỖI] Không tìm thấy model: {model_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    load_time = time.perf_counter() - t0
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    x = np.random.RandomState(42).uniform(0, 255, size=(1, INPUT_SIZE, INPUT_SIZE, 3)).astype(np.float32)
    if in_d["dtype"] == np.float16:
        x = x.astype(np.float16)

    # warm-up
    for _ in range(args.warmup):
        interp.set_tensor(in_d["index"], x)
        interp.invoke()
        _ = interp.get_tensor(out_d["index"])

    times = []
    for _ in range(args.runs):
        t = time.perf_counter()
        interp.set_tensor(in_d["index"], x)
        interp.invoke()
        _ = interp.get_tensor(out_d["index"])
        times.append((time.perf_counter() - t) * 1000.0)

    ts = np.array(times)
    result = {
        "runtime": RUNTIME,
        "model": model_path,
        "model_bytes": os.path.getsize(model_path),
        "input_shape": [int(v) for v in in_d["shape"]],
        "input_dtype": str(in_d["dtype"]),
        "output_shape": [int(v) for v in out_d["shape"]],
        "output_dtype": str(out_d["dtype"]),
        "warmup_runs": args.warmup,
        "n_runs": len(ts),
        "load_time_s": load_time,
        "inference_ms_mean": float(ts.mean()),
        "inference_ms_median": float(np.median(ts)),
        "inference_ms_p95": float(np.percentile(ts, 95)),
        "inference_ms_min": float(ts.min()),
        "inference_ms_max": float(ts.max()),
        "device": "UNKNOWN (chạy trên thiết bị thật qua Termux)",
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Đã ghi: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
