"""Quản lý cấu hình config.yaml cho dự án chili-ripeness-ai."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

try:
    import numpy as np
except ImportError:  # numpy luôn có cùng TensorFlow, phòng khi import nhẹ
    np = None

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("Cần PyYAML: pip install pyyaml") from exc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Đọc config.yaml thành dict. Đường dẫn tương đối tính từ gốc dự án."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Chuẩn hóa các đường dẫn thành tuyệt đối
    for key in ("dataset_raw", "dataset_train", "dataset_validation", "dataset_test",
                "checkpoints", "saved_model", "tflite", "reports_figures",
                "reports_metrics", "split_manifest", "groups_file"):
        p = cfg.get("paths", {}).get(key)
        if p:
            cfg["paths"][key] = os.path.join(PROJECT_ROOT, p)
    return cfg


def get_classes(cfg: dict) -> list:
    return list(cfg["project"]["classes"])


def get_class_mapping(cfg: dict) -> dict:
    return {cls: i for i, cls in enumerate(get_classes(cfg))}


@dataclass
class SeedSetting:
    seed: int
    np_seed: int
    python_seed: int


def resolve_seed(cfg: dict) -> SeedSetting:
    s = int(cfg.get("training", {}).get("seed", 42))
    return SeedSetting(seed=s, np_seed=s, python_seed=s)


def set_seeds(cfg: dict) -> SeedSetting:
    """Cố định random seed cho Python / NumPy / TensorFlow (nếu có)."""
    ss = resolve_seed(cfg)
    random.seed(ss.python_seed)
    if np is not None:
        np.random.seed(ss.np_seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(ss.seed)
    except Exception:
        pass
    return ss


def get_split_ratios(cfg: dict) -> tuple[float, float, float]:
    ratios = cfg["data"]["split_ratios"]
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6, f"Tỷ lệ split không hợp lệ: {ratios}"
    return tuple(float(r) for r in ratios)
