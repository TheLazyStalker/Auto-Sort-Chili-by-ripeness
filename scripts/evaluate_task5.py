"""Task 5 — Evaluate & compare Keras vs TFLite Float32/Float16 trên dev manifest.

Dùng:
  python scripts/evaluate_task5.py [--config config.yaml] [--task4-dir <dir>]

So sánh cùng dev validation (Task 4: 80/20 image-level, seed 42, leaky):
- Keras gốc (artifacts/task4/best_chili.keras)
- TFLite Float32 (artifacts/task5/chili_float32.tflite)
- TFLite Float16 (artifacts/task5/chili_float16.tflite)

Output: evaluation_metrics.json + .csv, agreement/prob-diff vs Keras.
Cảnh báo: dev split leaky (cùng quả có thể ở cả 2 tập) — không phải benchmark độc lập.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from src.config import load_config, set_seeds
from src.task4 import create_dev_split, load_task3_manifest, make_dataset
from src.utils import ensure_dir

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def tflite_predict(path: str, inputs: np.ndarray, batch: int = 32) -> np.ndarray:
    """Predict qua TFLite Interpreter trên toàn bộ inputs (float32 [0,255]).

    TFLite có thể fix batch (shape[0]==1 khi convert); tôn trọng shape đó.
    """
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    fixed_batch = int(in_d["shape"][0])
    eff_batch = batch if fixed_batch < 0 else max(1, min(batch, fixed_batch))
    outs = []
    for i in range(0, len(inputs), eff_batch):
        x = inputs[i:i + eff_batch].astype(np.float32)
        if in_d["dtype"] == np.float16:
            x = x.astype(np.float16)
        interp.set_tensor(in_d["index"], x)
        interp.invoke()
        o = np.asarray(interp.get_tensor(out_d["index"]), dtype=np.float32)
        outs.append(o)
    return np.concatenate(outs, axis=0)


def full_metrics(y_true, y_pred, classes):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 precision_recall_fscore_support)
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]),
                 "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": cm,
        "class_support": {c: int(support[i]) for i, c in enumerate(classes)},
        "n_samples": int(len(y_true)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 5 — evaluate Keras vs TFLite")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--task4-dir", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["project"]["classes"]
    task4_dir = args.task4_dir or cfg["paths"]["artifacts_task4"]
    out_dir = os.path.join(os.path.dirname(task4_dir), "task5")
    ensure_dir(out_dir)

    keras_path = os.path.join(task4_dir, "best_chili.keras")
    f32_path = os.path.join(out_dir, "chili_float32.tflite")
    f16_path = os.path.join(out_dir, "chili_float16.tflite")
    for p in (keras_path, f32_path, f16_path):
        if not os.path.isfile(p):
            print(f"[LỖI] Thiếu {p}")
            return 2

    # Tái tạo dev val (seed 42) — trùng Task 4
    manifest = load_task3_manifest(cfg)
    dev = create_dev_split(manifest, classes, seed=42)
    val = dev["validation"]
    paths = [i["path"] for i in val]
    y_true = np.array([int(i["label"]) for i in val])
    print(f"Dev validation: {len(paths)} ảnh")

    # Load toàn bộ inputs (uint8→float32 [0,255])
    ds = make_dataset(paths, y_true.astype(np.int64).tolist(), 64, shuffle=False)
    xs, ys = [], []
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(y.numpy())
    X = np.concatenate(xs, axis=0).astype(np.float32)
    y_true = np.concatenate(ys, axis=0)

    set_seeds(cfg)
    keras_model = tf.keras.models.load_model(keras_path)
    p_keras = keras_model.predict(X, batch_size=32, verbose=0)
    p_f32 = tflite_predict(f32_path, X)
    p_f16 = tflite_predict(f16_path, X)

    pred_keras = np.argmax(p_keras, axis=1)
    pred_f32 = np.argmax(p_f32, axis=1)
    pred_f16 = np.argmax(p_f16, axis=1)

    results = {}
    for name, p, pred in (("keras", p_keras, pred_keras),
                           ("float32", p_f32, pred_f32),
                           ("float16", p_f16, pred_f16)):
        metrics = full_metrics(y_true, pred, classes)
        metrics["prediction_agreement_with_keras"] = float((pred == pred_keras).mean())
        diff = np.abs(p - p_keras)
        metrics["prob_abs_diff_mean_vs_keras"] = float(diff.mean())
        metrics["prob_abs_diff_max_vs_keras"] = float(diff.max())
        metrics["finite_output"] = bool(np.isfinite(p).all())
        metrics["prob_sum_approx_1"] = bool(np.allclose(p.sum(axis=1), 1.0, atol=1e-3))
        results[name] = metrics
        print(f"\n[{name}] accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
              f"agreement={metrics['prediction_agreement_with_keras']:.4f} "
              f"prob_diff_mean={metrics['prob_abs_diff_mean_vs_keras']:.2e} "
              f"prob_diff_max={metrics['prob_abs_diff_max_vs_keras']:.2e}")

    out = {
        "split": "dev_validation_80_20_leaky_seed42",
        "note": "DEVELOPMENT results — not independent benchmark; leakage risk because "
                "pepper identity of old images is unknown (same pepper may appear in both subsets)",
        "classes": classes,
        "models": results,
    }
    with open(os.path.join(out_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # CSV
    with open(os.path.join(out_dir, "evaluation_metrics.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "macro_f1", "agreement_vs_keras",
                    "prob_diff_mean", "prob_diff_max", "finite", "prob_sum_ok"])
        for name in ("keras", "float32", "float16"):
            m = results[name]
            w.writerow([name, f"{m['accuracy']:.6f}", f"{m['macro_f1']:.6f}",
                        f"{m['prediction_agreement_with_keras']:.6f}",
                        f"{m['prob_abs_diff_mean_vs_keras']:.6e}",
                        f"{m['prob_abs_diff_max_vs_keras']:.6e}",
                        int(m['finite_output']), int(m['prob_sum_approx_1'])])
    print(f"\nSaved: {os.path.join(out_dir, 'evaluation_metrics.json')} + .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
