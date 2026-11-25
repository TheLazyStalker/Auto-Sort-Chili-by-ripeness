"""Training pipeline chung (multi-dataset) — dùng DatasetConfig.

Tái sử dụng:
- src/task4.py: build_augmentation, apply_augmentation, make_dataset,
  train_two_phases, plot_curves, save_json, verifier.
- src/model.py: build_model(classes=..., num_classes=...).
- src/datasets.py: validate_dataset (chặn train khi dataset rỗng/thiếu lớp).

EXPERIMENT: mỗi lần train tạo experiments/<dataset_id>/run_NNN/ và ghi:
  config.yaml, class_mapping.json, dataset_manifest.json, training_history.json,
  metrics.json, model/best_chili.keras
"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np
import tensorflow as tf

from .config import set_seeds
from .datasets import DatasetConfig, validate_dataset
from .experiment import create_run, write_run_artifacts
from .model import build_model, freeze_backbone, compile_model
from .task4 import (apply_augmentation, build_augmentation, make_dataset,
                    plot_curves, train_two_phases)
from .utils import ensure_dir


class TrainingError(RuntimeError):
    """Lỗi training dừng an toàn."""


def build_training_config(dataset_cfg: DatasetConfig, train_cfg: dict,
                          model_cfg: dict) -> dict:
    """Gộp cấu hình thực tế của một run (để lưu vào experiment)."""
    return {
        "dataset": {
            "dataset_id": dataset_cfg.dataset_id,
            "dataset_version": dataset_cfg.dataset_version,
            "classes": list(dataset_cfg.classes),
            "num_classes": dataset_cfg.num_classes,
            "image_size": dataset_cfg.image_size,
            "color_mode": dataset_cfg.color_mode,
            "preprocessing_pipeline": dataset_cfg.preprocessing_pipeline,
            "preprocessing_version": dataset_cfg.preprocessing_version,
            "split_strategy": dataset_cfg.split_strategy,
            "split_ratios": list(dataset_cfg.split_ratios),
            "split_seed": dataset_cfg.split_seed,
        },
        "training": dict(train_cfg),
        "model": dict(model_cfg),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _labels_from_manifest(items: list[dict]) -> np.ndarray:
    return np.array([int(i["label"]) for i in items])


def train_experiment(dataset_cfg: DatasetConfig, train_cfg: dict,
                     model_cfg: dict, manifest: dict, seed: int = 42,
                     verbose: bool = True, experiments_root: str | None = None) -> dict:
    """Chạy training cho một dataset (manifest đã có). Trả về đường dẫn run.

    Nếu dataset rỗng/thiếu lớp → TrainingError (không train).
    """
    # Bảo vệ: validate trước khi train (empty/thiếu lớp → dừng)
    v = validate_dataset(dataset_cfg)
    if not v["ok"]:
        raise TrainingError(
            f"Dataset '{dataset_cfg.dataset_id}' không hợp lệ để train:\n"
            + "\n".join(v["errors"]))

    cfg = {"data": {"image_size": dataset_cfg.image_size},
           "model": model_cfg,
           "project": {"classes": list(dataset_cfg.classes),
                       "num_classes": dataset_cfg.num_classes},
           "training": train_cfg}

    set_seeds(cfg)
    run_dir = create_run(dataset_cfg.dataset_id, root=experiments_root)
    os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)

    train_items = manifest.get("train", [])
    val_items = manifest.get("validation", [])
    if not train_items:
        raise TrainingError(f"Dataset '{dataset_cfg.dataset_id}': manifest không có train.")

    bs = int(train_cfg.get("batch_size", 32))
    train_paths = [i["path"] for i in train_items]
    train_labels = [int(i["label"]) for i in train_items]
    val_paths = [i["path"] for i in val_items]
    val_labels = [int(i["label"]) for i in val_items]

    train_ds = make_dataset(train_paths, train_labels, bs, shuffle=True, seed=seed)
    train_ds = apply_augmentation(train_ds, build_augmentation(seed=seed))
    val_ds = make_dataset(val_paths, val_labels, bs, shuffle=False) if val_items else None

    model = build_model(cfg, trainable_backbone=False,
                        classes=list(dataset_cfg.classes),
                        num_classes=dataset_cfg.num_classes)

    result = train_two_phases(cfg, model, train_ds, val_ds, run_dir,
                              use_class_weight=(train_cfg.get("class_weight", "balanced") != "null"),
                              num_classes=dataset_cfg.num_classes)
    history = result["history"]

    # Lưu model + history + config
    model.save(os.path.join(run_dir, "model", "best_chili.keras"))

    merged = {}
    for phase, h in history.items():
        if h is None:
            continue
        for k, v in h.items():
            merged[f"{phase}_{k}"] = v
    curve_path = os.path.join(run_dir, "training_curves.png")
    plot_curves(history, curve_path)

    manifest_artifact = {
        split: [{"path": it["path"], "label": it["label"], "class": dataset_cfg.classes[int(it["label"])],
                 "group_id": it.get("group_id", ""), "image_id": it.get("image_id", "")}
                for it in items]
        for split, items in manifest.items()
    }

    real_cfg = build_training_config(dataset_cfg, train_cfg, model_cfg)
    va_a = history["phase_a"].get("val_accuracy")
    va_b = history["phase_b"].get("val_accuracy")
    metrics = {
        "phaseA_best_val_acc": float(max(va_a)) if va_a else None,
        "phaseB_best_val_acc": float(max(va_b)) if va_b else None,
        "best_epoch_a": result["best_epoch_a"],
        "best_epoch_b": result["best_epoch_b"],
    }
    written = write_run_artifacts(
        run_dir, config=real_cfg,
        class_mapping=dataset_cfg.class_mapping(),
        manifest=manifest_artifact, history=merged, metrics=metrics)

    summary = {
        "run_dir": run_dir,
        "dataset_id": dataset_cfg.dataset_id,
        "classes": list(dataset_cfg.classes),
        "num_classes": dataset_cfg.num_classes,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "phaseA_best_val_acc": metrics["phaseA_best_val_acc"],
        "phaseB_best_val_acc": metrics["phaseB_best_val_acc"],
        "model_path": os.path.join(run_dir, "model", "best_chili.keras"),
        "history_path": os.path.join(run_dir, "training_history.json"),
        "curves_path": curve_path,
        "written": written,
    }
    if verbose:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
