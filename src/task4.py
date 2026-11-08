"""Task 4 core: dataset load, dev split 80/20, augmentation, hai phase, metrics.

Mọi output đều ghi vào ``artifacts/task4/`` (không đụng Task 3 manifest/raw).
Dev split là IMAGE-LEVEL (leaky) — chỉ để development/monitoring, KHÔNG phải
final benchmark; tách biệt với validation/test độc lập tương lai.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import tensorflow as tf

from .config import load_config, set_seeds
from .model import build_model, compile_model, freeze_backbone, get_callbacks, normalize_base_name, _find_backbone
from .utils import compute_class_weight, ensure_dir

SPLIT_NAMES = ("train", "validation", "test")
DEV_SPLIT_RATIO = (0.8, 0.2)


class Task4DataError(RuntimeError):
    """Dữ liệu Task 4 không hợp lệ — dừng an toàn, không train."""


# ---------------------------------------------------------------- dataset ----

def load_task3_manifest(cfg: dict) -> dict:
    """Đọc split_manifest_task3.json (chỉ train là có dữ liệu thật)."""
    path = cfg["paths"]["split_manifest_task3"]
    if not os.path.isfile(path):
        raise Task4DataError(f"Thiếu manifest Task 3: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_manifest_class_counts(manifest: dict, classes: list) -> dict:
    """Xác minh số lượng/lớp + tính đầy đủ label. Trả về counts."""
    available = {c: 0 for c in classes}
    for item in manifest.get("train", []):
        label = int(item["label"])
        if not (0 <= label < len(classes)):
            raise Task4DataError(f"Label ngoài phạm vi: {label}")
        available[classes[label]] += 1
    return available


def create_dev_split(manifest: dict, classes: list, seed: int = 42,
                     ratio=(0.8, 0.2)) -> dict:
    """Tạo split 80/20 image-level, stratified theo lớp, seed cố định.

    KHÔNG group-aware (identity không rõ → leaky có chủ đích, chỉ để dev).
    Trả về {train: [...], val: [...]} theo manifest (items giữ nguyên dict).
    """
    rng = random.Random(seed)
    train_items, val_items = [], []
    per_class = {c: [] for c in classes}
    for item in manifest.get("train", []):
        per_class[classes[int(item["label"])]].append(item)

    for cls in classes:
        items = per_class[cls]
        rng.shuffle(items)
        n_val = int(round(len(items) * ratio[1]))
        val_items.extend(items[:n_val])
        train_items.extend(items[n_val:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return {"train": train_items, "validation": val_items}


# -------------------------------------------------------------- dataset TF ----

def load_derivative_image(path: str):
    """Đọc derivative PNG → uint8 [0,255] (TF). Không thêm preprocessing khác."""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    return tf.cast(img, tf.uint8)


def make_dataset(paths, labels, batch_size: int, shuffle: bool = False,
                seed: int = 42):
    """tf.data.Dataset từ list path + label (int)."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: (load_derivative_image(p), l),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048, seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------- augmentation ----

def build_augmentation(seed: int = 42):
    """Augmentation nhẹ, KHÔNG đổi hue/saturation (màu là tín hiệu chính)."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        tf.keras.layers.RandomRotation(0.1, fill_mode="constant", fill_value=255.0, seed=seed),
        tf.keras.layers.RandomZoom(0.1, fill_mode="constant", fill_value=255.0, seed=seed),
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=255.0, seed=seed),
        tf.keras.layers.RandomBrightness(0.05, value_range=(0.0, 255.0), seed=seed),
    ], name="augmentation_leaky_dev")


def apply_augmentation(ds, aug):
    """Aug chỉ áp cho train; map qua từng batch (prefetch giữ nguyên).

    Cast sang float32 [0,255] trước khi augment (Keras preprocessing layers yêu
    cầu float); model V3 vẫn nhận float [0,255] → Rescaling nhúng.
    """
    def _aug(x, y):
        x = tf.cast(x, tf.float32)
        return aug(x, training=True), y
    return ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)


# ------------------------------------------------------------- training ----

def get_phase_config(cfg: dict, phase: str) -> dict:
    t = cfg.get("training", {})
    if phase == "a":
        return {
            "epochs": int(t.get("stage1_epochs", 30)),
            "lr": float(t.get("stage1_lr", 0.001)),
            "patience": int(t.get("early_stopping_patience", 7)),
            "reduce_patience": int(t.get("reduce_lr_patience", 4)),
            "reduce_factor": float(t.get("reduce_lr_factor", 0.5)),
            "min_lr": float(t.get("reduce_lr_min_lr", 1e-6)),
            "fine_tune_layers": 0,
        }
    return {
        "epochs": int(t.get("stage2_epochs", 20)),
        "lr": float(t.get("stage2_lr", 1e-4)),
        "patience": int(t.get("early_stopping_patience", 7)),
        "reduce_patience": int(t.get("reduce_lr_patience", 3)),
        "reduce_factor": float(t.get("reduce_lr_factor", 0.5)),
        "min_lr": float(t.get("reduce_lr_min_lr", 1e-7)),
        "fine_tune_layers": int(cfg["model"].get(
            "fine_tune_layers_v3small",
            cfg["model"].get("fine_tune_layers", 20))),
    }


def make_callbacks_for_phase(ckpt_path: str, phase_cfg: dict, verbose: bool = True):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=phase_cfg["patience"],
            restore_best_weights=True, verbose=1 if verbose else 0),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1 if verbose else 0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=phase_cfg["reduce_factor"],
            patience=phase_cfg["reduce_patience"],
            min_lr=phase_cfg["min_lr"], verbose=1 if verbose else 0),
    ]


def train_two_phases(cfg: dict, model, train_ds, val_ds, out_dir: str,
                     use_class_weight: bool = True, verbose: bool = True) -> dict:
    """Chạy Phase A (freeze) + Phase B (fine-tune). Ghi best .keras cho từng phase."""
    os.makedirs(out_dir, exist_ok=True)

    # Class weights (nếu có) từ labels train
    class_weight = None
    if use_class_weight:
        labels = np.array([tf.cast(y, tf.int64).numpy() for _, y in train_ds.unbatch()])
        class_weight = compute_class_weight(labels, int(cfg["project"]["num_classes"]))

    history = {"phase_a": None, "phase_b": None}
    ckpt_a = os.path.join(out_dir, "best_phase_a.keras")
    ckpt_b = os.path.join(out_dir, "best_chili.keras")

    # ---------- Phase A: freeze backbone, train head ----------
    pa = get_phase_config(cfg, "a")
    print(f"\n=== Phase A (freeze v{cfg['model'].get('base','mobilenetv2')}) "
          f"epochs={pa['epochs']} lr={pa['lr']} ===")
    freeze_backbone(model, fine_tune_layers=0)
    compile_model(model, learning_rate=pa["lr"], loss="sparse_categorical_crossentropy")
    callbacks_a = make_callbacks_for_phase(ckpt_a, pa, verbose=verbose)
    hist_a = model.fit(
        train_ds, validation_data=val_ds,
        epochs=pa["epochs"], callbacks=callbacks_a, shuffle=False,
        class_weight=class_weight, verbose=1 if verbose else 0)
    history["phase_a"] = hist_a.history
    print(f"  Phase A best val_accuracy: {max(hist_a.history['val_accuracy']):.4f}")

    # ---------- Phase B: fine-tune phần cuối backbone ----------
    pb = get_phase_config(cfg, "b")
    print(f"\n=== Phase B (fine-tune {pb['fine_tune_layers']} layer cuối) "
          f"epochs={pb['epochs']} lr={pb['lr']} ===")
    freeze_backbone(model, fine_tune_layers=pb["fine_tune_layers"])
    compile_model(model, learning_rate=pb["lr"], loss="sparse_categorical_crossentropy")
    callbacks_b = make_callbacks_for_phase(ckpt_b, pb, verbose=verbose)
    hist_b = model.fit(
        train_ds, validation_data=val_ds,
        epochs=pb["epochs"], callbacks=callbacks_b, shuffle=False,
        class_weight=class_weight, verbose=1 if verbose else 0)
    history["phase_b"] = hist_b.history
    print(f"  Phase B best val_accuracy: {max(hist_b.history['val_accuracy']):.4f}")

    return {"history": history, "class_weight": class_weight,
            "checkpoint_a": ckpt_a, "checkpoint_b": ckpt_b,
            "best_epoch_a": int(np.argmax(hist_a.history["val_accuracy"])),
            "best_epoch_b": int(np.argmax(hist_b.history["val_accuracy"]))}


# ------------------------------------------------------------- metrics ----

def plot_curves(history: dict, out_path: str):
    """Vẽ loss/accuracy train vs val cho từng phase (gộp 2 subplot)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for phase, hist in history.items():
        if not hist:
            continue
        for ax, metric, title in ((axes[0], "loss", "Loss"),
                                  (axes[1], "accuracy", "Accuracy")):
            ax.plot(hist[metric], label=f"{phase} train {metric}")
            v = f"val_{metric}"
            if v in hist:
                ax.plot(hist[v], label=f"{phase} val {metric}")
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
