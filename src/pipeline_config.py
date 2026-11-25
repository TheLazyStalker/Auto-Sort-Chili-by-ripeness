"""Config trung tâm của pipeline — đọc `pipeline_config.json` (ngoài mã nguồn).

Mọi bước (center → raw → derivative → train → export) đọc tham số từ đây, nên
sửa file JSON là toàn bộ quá trình thay đổi theo. Không hardcode tham số ở nơi khác.
"""
from __future__ import annotations

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "pipeline_config.json")

DEFAULTS: dict = {
    "paths": {
        "datasets_root": "datasets",
        "experiments_root": "experiments",
        "staging_root": ".pipeline_staging",
        "center_tool_dir": "tools/make-object-to-center",
    },
    "dataset": {"default_id": "chili_quality_v2", "classes": []},
    "centering": {
        "enabled": True, "python": "python3", "script": "crop_chili.py",
        "size": 1024, "background_bgr": [255, 255, 255], "fruit_frac": 0.5,
        "pad": 0.12, "ext": ".jpg", "suffix": "", "timeout_s": 1800,
    },
    "preprocessing": {
        "pipeline": "crop_v1", "output_size": 224, "pad_frac": 0.2,
        "threshold": 30.0, "border_frac": 0.05, "min_area_frac": 0.005,
        "max_area_frac": 0.6, "dilate_px": 2,
    },
    "split": {"ratios": [0.7, 0.15, 0.15], "seed": 42},
    "training": {
        "seed": 42, "batch_size": 32, "class_weight": "balanced",
        "stage1_epochs": 30, "stage1_lr": 0.001, "stage2_epochs": 20,
        "stage2_lr": 0.0001, "early_stopping_patience": 7,
        "reduce_lr_patience": 4, "reduce_lr_factor": 0.5, "reduce_lr_min_lr": 1e-06,
    },
    "model": {
        "name": "mobilenetv3small", "base": "mobilenetv3small", "weights": "imagenet",
        "include_top": False, "include_preprocessing": True, "dropout": 0.3,
        "fine_tune_layers": 20, "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
    },
}


class PipelineConfigError(RuntimeError):
    """pipeline_config.json không hợp lệ."""


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_pipeline_config(path: str | None = None) -> dict:
    """Đọc pipeline_config.json, hợp nhất với DEFAULTS và validate."""
    p = path or DEFAULT_CONFIG_PATH
    raw = {}
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
    cfg = _deep_merge(DEFAULTS, raw)
    validate_pipeline_config(cfg, source=p)
    # resolve đường dẫn tương đối → tuyệt đối (tính từ PROJECT_ROOT)
    for key in ("datasets_root", "experiments_root", "staging_root", "center_tool_dir"):
        v = cfg["paths"].get(key)
        if v and not os.path.isabs(v):
            cfg["paths"][key] = os.path.join(PROJECT_ROOT, v)
    cfg["_source"] = p
    return cfg


def validate_pipeline_config(cfg: dict, source: str = "") -> None:
    ratios = cfg.get("split", {}).get("ratios", [])
    if len(ratios) != 3 or abs(sum(float(r) for r in ratios) - 1.0) > 1e-6:
        raise PipelineConfigError(
            f"split.ratios phải gồm 3 giá trị tổng ≈1 (nguồn: {source}): {ratios}")
    bg = cfg.get("centering", {}).get("background_bgr")
    if not (isinstance(bg, list) and len(bg) == 3 and all(0 <= int(v) <= 255 for v in bg)):
        raise PipelineConfigError(
            f"centering.background_bgr phải là [B,G,R] 0..255 (nguồn: {source}): {bg}")
    if int(cfg.get("centering", {}).get("size", 0)) <= 0:
        raise PipelineConfigError(f"centering.size phải > 0 (nguồn: {source})")
    classes = cfg.get("dataset", {}).get("classes") or []
    if classes and len(set(classes)) != len(classes):
        raise PipelineConfigError(f"dataset.classes bị trùng (nguồn: {source}): {classes}")


# ---- helpers: chuyển config trung tâm → tham số cho từng bước ----

def crop_params(cfg: dict) -> dict:
    """Tham số truyền cho crop_v1 (src/crop.py)."""
    p = cfg["preprocessing"]
    return {
        "pad_frac": float(p["pad_frac"]),
        "output_size": int(p["output_size"]),
        "threshold": float(p["threshold"]),
        "border_frac": float(p["border_frac"]),
        "min_area_frac": float(p["min_area_frac"]),
        "max_area_frac": float(p["max_area_frac"]),
        "dilate_px": int(p["dilate_px"]),
    }


def center_params(cfg: dict) -> dict:
    """Tham số CLI cho make-object-to-center/crop_chili.py."""
    c = cfg["centering"]
    bg = ",".join(str(int(v)) for v in c["background_bgr"])
    return {
        "python": c["python"], "script": c["script"], "size": int(c["size"]),
        "bg": bg, "fruit_frac": float(c["fruit_frac"]), "pad": float(c["pad"]),
        "ext": c["ext"], "suffix": c["suffix"], "enabled": bool(c["enabled"]),
        "timeout_s": int(c.get("timeout_s", 1800)),
        "tool_dir": cfg["paths"]["center_tool_dir"],
    }


def training_config(cfg: dict) -> dict:
    return dict(cfg["training"])


def model_config(cfg: dict) -> dict:
    return dict(cfg["model"])


def split_config(cfg: dict) -> dict:
    return {"ratios": [float(r) for r in cfg["split"]["ratios"]],
            "seed": int(cfg["split"]["seed"])}
