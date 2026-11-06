"""Đánh giá model trên tập test: accuracy, precision/recall/F1 từng lớp,
confusion matrix (lưu PNG + JSON)."""
from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

from .config import load_config, set_seeds
from .dataset import load_manifest
from .model import build_model
from .utils import save_metrics_json


def evaluate_model(cfg: dict, model_path: str | None = None, split: str = "test") -> dict:
    set_seeds(cfg)
    manifest = load_manifest(cfg["paths"]["split_manifest"])
    if manifest is None or split not in manifest:
        raise RuntimeError(f"Không có manifest/split '{split}'. Chạy prepare_datasets() trước.")

    model_path = model_path or os.path.join(cfg["paths"]["saved_model"], "best_model.keras")
    if not os.path.exists(model_path):
        candidates = [
            os.path.join(cfg["paths"]["checkpoints"], "chili_final.keras"),
            os.path.join(cfg["paths"]["checkpoints"], "best_chili.keras"),
            model_path,
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy model đã huấn luyện tại {model_path}. "
            "Chạy `python -m src.train` trước (sau khi thêm ảnh vào dataset/raw)."
        )
    model = tf.keras.models.load_model(model_path)

    items = manifest[split]
    file_paths = [i["path"] for i in items]
    y_true = np.array([i["label"] for i in items])

    import tensorflow as tf

    def load_fn(fpath):
        # Giữ uint8 [0,255]; model tự preprocess_input ([0,255]→[-1,1]).
        img = tf.io.read_file(fpath)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        return img

    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    y_prob = model.predict(ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    from sklearn.metrics import (classification_report, confusion_matrix,
                                 precision_recall_fscore_support)

    classes = cfg["project"]["classes"]
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    accuracy = float((y_pred == y_true).mean()) if len(y_true) else 0.0

    metrics = {
        "split": split,
        "n_samples": int(len(y_true)),
        "accuracy": accuracy,
        "precision_macro": float(prec.mean()),
        "recall_macro": float(rec.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": {
            cls: {"precision": float(p), "recall": float(r), "f1": float(f)}
            for cls, p, r, f in zip(classes, prec, rec, f1)
        },
        "confusion_matrix": cm,
        "classification_report": report,
    }

    out_dir = cfg["paths"]["reports_metrics"]
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"evaluation_{split}.json")
    save_metrics_json(metrics, metrics_path)

    # Vẽ confusion matrix
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
    ax.set_title(f"Confusion matrix ({split})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    cm_path = os.path.join(cfg["paths"]["reports_figures"], f"confusion_matrix_{split}.png")
    fig.savefig(cm_path)
    plt.close(fig)

    print(f"Accuracy ({split}): {accuracy:.4f}")
    print(f"Đã lưu metrics: {metrics_path} | confusion matrix: {cm_path}")
    return metrics


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Đánh giá model phân loại độ chín ớt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="Đường dẫn .keras / SavedModel")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    try:
        evaluate_model(cfg, model_path=args.model, split=args.split)
    except (FileNotFoundError, RuntimeError) as exc:
        import sys
        print(f"[LỖI] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
