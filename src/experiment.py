"""Experiment manager — mỗi lần train là một run riêng, không ghi đè.

Cấu trúc:
    experiments/<dataset_id>/run_<NNN>/
        config.yaml              # cấu hình thực tế đã dùng (dataset+training+model)
        class_mapping.json
        dataset_manifest.json    # ảnh dùng trong lần train này
        training_history.json
        metrics.json
        evaluation/
        model/
"""
from __future__ import annotations

import json
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")


class ExperimentError(RuntimeError):
    """Lỗi quản lý experiment."""


def experiments_root(root: str | None = None) -> str:
    return root or DEFAULT_EXPERIMENTS_ROOT


def list_runs(dataset_id: str, root: str | None = None) -> list[str]:
    base = os.path.join(experiments_root(root), dataset_id)
    if not os.path.isdir(base):
        return []
    runs = []
    for name in sorted(os.listdir(base)):
        if name.startswith("run_") and os.path.isdir(os.path.join(base, name)):
            runs.append(name)
    return runs


def latest_run(dataset_id: str, root: str | None = None) -> str | None:
    runs = list_runs(dataset_id, root)
    return runs[-1] if runs else None


def next_run_number(dataset_id: str, root: str | None = None) -> int:
    nums = []
    for name in list_runs(dataset_id, root):
        try:
            nums.append(int(name.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
    return (max(nums) + 1) if nums else 1


def create_run(dataset_id: str, root: str | None = None) -> str:
    """Tạo thư mục run kế tiếp (run_001, run_002...). Không bao giờ ghi đè."""
    n = next_run_number(dataset_id, root)
    run_dir = os.path.join(experiments_root(root), dataset_id, f"run_{n:03d}")
    os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "evaluation"), exist_ok=True)
    return run_dir


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_yaml(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def write_run_artifacts(run_dir: str, *, config: dict, class_mapping: dict,
                        manifest: dict, history: dict | None = None,
                        metrics: dict | None = None) -> list[str]:
    """Ghi toàn bộ artifact của một run; trả về danh sách file đã ghi."""
    written = []
    p = os.path.join(run_dir, "config.yaml")
    save_yaml(config, p)
    written.append(p)
    p = os.path.join(run_dir, "class_mapping.json")
    save_json(class_mapping, p)
    written.append(p)
    p = os.path.join(run_dir, "dataset_manifest.json")
    save_json(manifest, p)
    written.append(p)
    if history is not None:
        p = os.path.join(run_dir, "training_history.json")
        save_json(history, p)
        written.append(p)
    if metrics is not None:
        p = os.path.join(run_dir, "metrics.json")
        save_json(metrics, p)
        written.append(p)
    return written


def load_run_metadata(run_dir: str) -> dict:
    """Đọc config + class_mapping từ một run (để evaluation/prediction dùng)."""
    out = {}
    for key, fname in (("config", "config.yaml"), ("class_mapping", "class_mapping.json"),
                       ("manifest", "dataset_manifest.json")):
        p = os.path.join(run_dir, fname)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                out[key] = (yaml.safe_load(f) if fname.endswith(".yaml") else json.load(f))
    return out
