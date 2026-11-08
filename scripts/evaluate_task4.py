"""Task 4 — đánh giá DEVELOPMENT model trên split validation (80/20 dev).

Dùng:
  python scripts/evaluate_task4.py [--config config.yaml] [--model <path>]

Output (artifacts/task4/):
- evaluation_dev.json: accuracy, per-class precision/recall/f1, macro f1, support.
- confusion_matrix.png / .json
- prediction_probs.csv (tùy chọn --save-probs)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from src.config import load_config, set_seeds
from src.task4 import load_task3_manifest, create_dev_split, make_dataset
from src.utils import ensure_dir, save_metrics_json

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 4 — development evaluation")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None)
    parser.add_argument("--save-probs", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["project"]["classes"]
    out_dir = cfg["paths"]["artifacts_task4"]
    ensure_dir(out_dir)

    model_path = args.model or os.path.join(out_dir, "best_chili.keras")
    if not os.path.isfile(model_path):
        print(f"[LỖI] Không tìm thấy model: {model_path}")
        return 2

    set_seeds(cfg)
    model = tf.keras.models.load_model(model_path)

    # Tái tạo dev split 80/20 (seed 42) — trùng với train
    manifest = load_task3_manifest(cfg)
    dev = create_dev_split(manifest, classes, seed=42)
    val = dev["validation"]
    if not val:
        print("[LỖI] Dev validation rỗng.")
        return 2

    paths = [i["path"] for i in val]
    labels = [int(i["label"]) for i in val]
    bs = int(cfg["data"].get("batch_size", 32))
    ds = make_dataset(paths, labels, bs, shuffle=False)

    y_prob = model.predict(ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.array(labels)

    from sklearn.metrics import (classification_report, confusion_matrix,
                                 precision_recall_fscore_support)
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0)
    accuracy = float((y_pred == y_true).mean())
    macro_f1 = float(np.mean(f1))

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]),
                 "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": cm,
        "class_support": {c: int(support[i]) for i, c in enumerate(classes)},
        "n_samples": int(len(y_true)),
        "split": "dev_validation_80_20_leaky",
    }
    metrics_path = os.path.join(out_dir, "evaluation_dev.json")
    save_metrics_json(metrics, metrics_path)

    # Confusion matrix PNG
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.array(cm), cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — DEVELOPMENT (leaky 80/20)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    cm_png = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(cm_png)
    plt.close(fig)
    save_metrics_json({"confusion_matrix": cm}, os.path.join(out_dir, "confusion_matrix.json"))

    if args.save_probs:
        import csv
        with open(os.path.join(out_dir, "dev_prediction_probs.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "true_label", "pred_label"] + classes)
            for p, t, pr, pred in zip(paths, labels, y_prob, y_pred):
                w.writerow([p, t, pred] + [f"{v:.6f}" for v in pr])

    print(f"Accuracy (DEV): {accuracy:.4f}")
    print(f"Macro F1 (DEV): {macro_f1:.4f}")
    print(f"Per-class: {metrics['per_class']}")
    print(f"Confusion matrix: {cm}")
    print(f"Saved: {metrics_path}, {cm_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
