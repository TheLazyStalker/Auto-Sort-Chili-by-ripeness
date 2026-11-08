"""Task 6 — Verification: deployment predict.py (Termux-style) vs container TFLite vs Keras.

Chạy trong container để xác nhận preprocessing và pipeline deploy khớp với
pipeline đã train/evaluate (Task 4/5). Dùng chính 92 ảnh dev validation.

So sánh:
- accuracy của predict.py trên dev (so với label)
- prediction agreement & max|Δprob| so với TFLite container
- prediction agreement so với Keras gốc

Lưu ý: đây là kiểm tra consistency, KHÔNG phải benchmark độc lập.
"""
from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.config import load_config
from src.task4 import create_dev_split, load_task3_manifest

# deployment predict module
from deployment.task6.predict import Predictor, DEFAULT_MODEL, load_input

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cfg = load_config()
    classes = cfg["project"]["classes"]
    manifest = load_task3_manifest(cfg)
    dev = create_dev_split(manifest, classes, seed=42)
    val = dev["validation"]
    paths = [i["path"] for i in val]
    y_true = np.array([int(i["label"]) for i in val])
    print(f"Dev validation: {len(paths)} ảnh")

    # Deployment predictor (float16) — dùng load_input như trên máy
    pred_deploy = Predictor(os.path.join(HERE, "models", "chili_float16.tflite"))
    # Container TFLite float16 — cùng model, SS interpreter
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=os.path.join(
        HERE, "models", "chili_float16.tflite"))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    # Keras
    keras = tf.keras.models.load_model(
        "artifacts/task4/best_chili.keras")

    acc_deploy = 0
    agree_tflite = 0
    agree_keras = 0
    max_diff_tflite = 0.0
    max_diff_keras = 0.0
    for p, y in zip(paths, y_true):
        tensor = load_input(p)  # cùng preprocessing deployment

        # deployment predictor
        prob_d, _ = pred_deploy.predict(tensor)
        pred_d = int(np.argmax(prob_d))
        acc_deploy += int(pred_d == y)

        # container interpreter (cùng tensor đầu vào)
        x = tensor
        if in_d["dtype"] == np.float16:
            x = tensor.astype(np.float16)
        interp.set_tensor(in_d["index"], x)
        interp.invoke()
        prob_c = np.asarray(interp.get_tensor(out_d["index"]), dtype=np.float32)[0]
        agree_tflite += int(np.argmax(prob_c) == pred_d)
        max_diff_tflite = max(max_diff_tflite, float(np.abs(prob_c - prob_d).max()))

        # keras (float32)
        prob_k = keras.predict(tensor, verbose=0)[0].astype(np.float32)
        agree_keras += int(np.argmax(prob_k) == pred_d)
        max_diff_keras = max(max_diff_keras, float(np.abs(prob_k - prob_d).max()))

    n = len(paths)
    result = {
        "n_dev": n,
        "deploy_accuracy": acc_deploy / n,
        "deploy_vs_container_tflite_agreement": agree_tflite / n,
        "deploy_vs_container_tflite_max_prob_diff": max_diff_tflite,
        "deploy_vs_keras_agreement": agree_keras / n,
        "deploy_vs_keras_max_prob_diff": max_diff_keras,
        "note": "consistency check only; not independent benchmark",
    }
    print(json.dumps(result, indent=2))
    with open(os.path.join(HERE, "parity_check.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
