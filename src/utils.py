"""Hàm hỗ trợ: thống kê, checkpoint, class weights, vẽ biểu đồ huấn luyện."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

import numpy as np


def file_md5(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(file_list: list) -> dict:
    """Phát hiện file trùng theo nội dung (md5). Trả về {md5: [file,...]} chỉ gồm nhóm trùng."""
    by_hash: dict[str, list] = {}
    for p in file_list:
        try:
            by_hash.setdefault(file_md5(p), []).append(p)
        except OSError:
            continue
    return {k: v for k, v in by_hash.items() if len(v) > 1}


def count_per_class(class_to_files: dict, classes: list) -> dict:
    return {c: len(class_to_files.get(c, [])) for c in classes}


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[labels]


def compute_class_weight(labels: np.ndarray, num_classes: int) -> dict:
    """Class weight dạng balanced: n_samples / (n_classes * count_c)."""
    counts = Counter(np.asarray(labels).tolist())
    n = len(labels)
    return {i: float(n / (num_classes * counts.get(i, 1))) for i in range(num_classes)}


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_metrics_json(metrics: dict, path: str) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return path


def plot_training_curves(history, out_path: str) -> str:
    """Vẽ loss/accuracy train vs validation. history: keras History hoặc dict."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = history.history if hasattr(history, "history") else history
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in (
        (axes[0], "loss", "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ):
        if metric in hist:
            ax.plot(hist[metric], label=f"train {metric}")
            vmetric = f"val_{metric}"
            if vmetric in hist:
                ax.plot(hist[vmetric], label=f"val {metric}")
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
            ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
