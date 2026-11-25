"""Tests multi-dataset framework (Phase 3) — datasets registry, experiment, train pipeline."""
import json
import os

import numpy as np
import pytest

from src.datasets import (DatasetConfigError, load_dataset_config, list_dataset_ids,
                          scan_dataset, validate_dataset)
from src.experiment import create_run, list_runs, latest_run, write_run_artifacts, load_run_metadata
from src.model import build_model

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.config import load_config

try:
    from PIL import Image
except ImportError:
    pytest.skip("Pillow not available", allow_module_level=True)


def _cfg():
    return load_config()


# ---------------- registry ----------------

def test_list_dataset_ids_finds_both():
    ids = list_dataset_ids()
    assert "legacy_ripeness_v1" in ids
    assert "chili_quality_v2" in ids


def test_load_legacy_config():
    d = load_dataset_config("legacy_ripeness_v1")
    assert d.classes == ["unripe", "medium_ripe", "ripe"]
    assert d.num_classes == 3
    assert d.class_mapping() == {"unripe": 0, "medium_ripe": 1, "ripe": 2}
    # path trỏ về dataset/ cũ (giữ nguyên)
    assert d.raw_dir.endswith("dataset/raw")


def test_load_v2_config():
    d = load_dataset_config("chili_quality_v2")
    assert d.classes == ["ripe", "unripe", "damaged"]
    assert d.num_classes == 3
    assert "medium_ripe" not in d.classes
    assert d.class_mapping() == {"ripe": 0, "unripe": 1, "damaged": 2}


def test_class_mappings_independent():
    d1 = load_dataset_config("legacy_ripeness_v1")
    d2 = load_dataset_config("chili_quality_v2")
    assert d1.class_mapping() != d2.class_mapping()


def test_load_missing_dataset_raises():
    with pytest.raises(DatasetConfigError, match="Không tìm thấy"):
        load_dataset_config("does_not_exist_xyz")


def test_scan_legacy_counts_match_original():
    d = load_dataset_config("legacy_ripeness_v1")
    scan = scan_dataset(d)
    assert scan["total"] == 473
    assert scan["counts"] == {"unripe": 197, "medium_ripe": 142, "ripe": 134}
    assert scan["empty_classes"] == []


def _write_tmp_dataset(root, dataset_id, classes, require_nonempty=True, make_dirs=True):
    """Tạo dataset_config.yaml tạm trong root/<id>/ (cho test độc lập dữ liệu thật)."""
    os.makedirs(os.path.join(root, dataset_id), exist_ok=True)
    cfg = {
        "dataset_id": dataset_id,
        "dataset_version": 1,
        "classes": list(classes),
        "paths": {
            "raw": os.path.join(root, dataset_id, "raw"),
            "manifests": os.path.join(root, dataset_id, "manifests"),
            "derived": os.path.join(root, dataset_id, "derived"),
        },
        "input": {"image_size": 224, "color_mode": "RGB"},
        "preprocessing": {"pipeline": "crop_v1", "version": "v1"},
        "split": {"strategy": "group_stratified", "ratios": [0.7, 0.15, 0.15], "seed": 42},
        "labels": {"source": "directory", "allow_pending_review": True},
        "validation": {"require_nonempty": require_nonempty},
    }
    import yaml
    with open(os.path.join(root, dataset_id, "dataset_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    if make_dirs:
        for c in classes:
            os.makedirs(os.path.join(root, dataset_id, "raw", c), exist_ok=True)
    return os.path.join(root, dataset_id)


def test_validate_empty_dataset_ok_when_not_required(tmp_path):
    root = str(tmp_path)
    _write_tmp_dataset(root, "ds_optional", ["a", "b"], require_nonempty=False)
    d = load_dataset_config("ds_optional", root=root)
    v = validate_dataset(d)
    assert v["ok"] is True
    assert any("chưa đủ lớp" in w or "TRỐNG" in w for w in v["warnings"])


def test_validate_empty_dataset_errors_when_required(tmp_path):
    root = str(tmp_path)
    _write_tmp_dataset(root, "ds_required", ["a", "b"], require_nonempty=True)
    d = load_dataset_config("ds_required", root=root)
    v = validate_dataset(d)
    assert v["ok"] is False
    assert any("thiếu lớp" in e or "TRỐNG" in e for e in v["errors"])


def test_train_script_rejects_empty_dataset(tmp_path):
    """Dataset trống (require_nonempty) → train.py dừng an toàn (exit 2)."""
    import json
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root = str(tmp_path)
    _write_tmp_dataset(root, "ds_empty", ["a", "b"], require_nonempty=True)
    pcfg_path = os.path.join(root, "pipeline_config.json")
    with open(pcfg_path, "w", encoding="utf-8") as f:
        json.dump({"paths": {"datasets_root": root,
                             "experiments_root": os.path.join(root, "experiments"),
                             "staging_root": os.path.join(root, "staging"),
                             "center_tool_dir": os.path.join(root, "tool")}}, f)
    from scripts.train import main as train_main
    with pytest.raises(SystemExit) as ei:
        train_main(["--dataset", "ds_empty", "--dry-run",
                    "--pipeline-config", pcfg_path])
    assert ei.value.code == 2


def test_build_manifest_rejects_missing_derivatives(tmp_path):
    """Dataset có ảnh raw nhưng chưa có derivative → TrainingError rõ ràng."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.train import build_manifest
    from src.training import TrainingError
    from src.datasets import DatasetConfig
    fake = DatasetConfig(dataset_id="no_der", classes=["x", "y"], root=str(tmp_path),
                         raw_dir=str(tmp_path), derived_dir=str(tmp_path / "none"),
                         manifests_dir=str(tmp_path))
    with pytest.raises(TrainingError, match="chưa có derivative"):
        build_manifest(fake, {"counts": {}, "files": {}}, 42)


def test_alias_resolution_legacy():
    from src.datasets import resolve_class_dir
    d = load_dataset_config("legacy_ripeness_v1")
    # medium-ripe (dấu gạch ngang) nếu có → resolve qua alias; ở đây dùng medium_ripe
    assert os.path.isdir(resolve_class_dir(d, "medium_ripe"))


# ---------------- experiment ----------------

def test_create_run_increments_and_no_overwrite(tmp_path):
    root = str(tmp_path)
    r1 = create_run("chili_quality_v2", root=root)
    r2 = create_run("chili_quality_v2", root=root)
    assert r1 != r2
    assert r1.endswith("run_001") and r2.endswith("run_002")
    assert os.path.isdir(os.path.join(r1, "model"))
    assert list_runs("chili_quality_v2", root=root) == ["run_001", "run_002"]


def test_write_and_read_run_artifacts(tmp_path):
    run = create_run("chili_quality_v2", root=str(tmp_path))
    write_run_artifacts(run,
                        config={"seed": 42, "dataset": "chili_quality_v2"},
                        class_mapping={"ripe": 0, "unripe": 1, "damaged": 2},
                        manifest={"train": [], "validation": []},
                        history={"loss": [0.1]})
    meta = load_run_metadata(run)
    assert meta["class_mapping"] == {"ripe": 0, "unripe": 1, "damaged": 2}
    assert meta["config"]["seed"] == 42


def test_latest_run(tmp_path):
    create_run("chili_quality_v2", root=str(tmp_path))
    create_run("chili_quality_v2", root=str(tmp_path))
    assert latest_run("chili_quality_v2", root=str(tmp_path)).endswith("run_002")


# ---------------- model output size ----------------

def test_model_output_shape_from_dataset_classes():
    cfg = _cfg()
    # V2: 3 lớp khác nhau
    d2 = load_dataset_config("chili_quality_v2")
    m2 = build_model(cfg, classes=d2.classes, num_classes=d2.num_classes)
    assert m2.output_shape == (None, 3)
    assert m2.classes_ == ["ripe", "unripe", "damaged"]
    # legacy: 3 lớp cũ
    d1 = load_dataset_config("legacy_ripeness_v1")
    m1 = build_model(cfg, classes=d1.classes, num_classes=d1.num_classes)
    assert m1.output_shape == (None, 3)
    assert m1.classes_ == ["unripe", "medium_ripe", "ripe"]


def test_model_output_size_generic_num_classes():
    cfg = _cfg()
    m = build_model(cfg, classes=["a", "b"], num_classes=2)
    assert m.output_shape == (None, 2)


def test_model_backward_compat_default():
    cfg = _cfg()
    m = build_model(cfg)
    assert m.output_shape == (None, 3)  # fallback về project config


# ---------------- pipeline init (không train thật) ----------------

def test_train_pipeline_init_legacy_derivatives():
    """legacy có derivative → manifest được build từ derivative, không crash."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.train import build_manifest
    from src.datasets import load_dataset_config, scan_dataset
    d = load_dataset_config("legacy_ripeness_v1")
    scan = scan_dataset(d)
    m = build_manifest(d, scan, seed=42)
    assert len(m["train"]) > 0
    # legacy hiện chỉ có derivative train (provisional Task 3) → val/test có thể rỗng
    # mọi path phải là derivative PNG tồn tại
    for split in ("train", "validation", "test"):
        for it in m[split]:
            assert os.path.isfile(it["path"])
            assert it["path"].endswith(".png")
            assert it["label"] in (0, 1, 2)


def test_training_pipeline_builds_datasets_for_v2_like(tmp_path):
    """Framework generic: dataset 3 lớp khác (ripe/unripe/damaged) có thể init pipeline.

    Dùng dataset tạm trong tmp_path (fixture, không phải dataset thật).
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.training import train_experiment
    from src.datasets import DatasetConfig
    from src.experiment import list_runs

    # Dataset tạm: 3 lớp, mỗi lớp 3 ảnh PNG 224 (fixture)
    root = str(tmp_path / "tmpds")
    raw = os.path.join(root, "raw")
    classes = ["ripe", "unripe", "damaged"]
    paths = []
    for i, cls in enumerate(classes):
        d = os.path.join(raw, cls)
        os.makedirs(d, exist_ok=True)
        for k in range(3):
            p = os.path.join(d, f"{cls}_{k}.png")
            Image.new("RGB", (224, 224), (40 + i * 60, 60, 80)).save(p)
            paths.append({"path": p, "label": i})

    dcfg = DatasetConfig(
        dataset_id="tmpds",
        classes=classes,
        root=root, raw_dir=raw,
        derived_dir=os.path.join(root, "derived"),
        manifests_dir=os.path.join(root, "manifests"),
    )
    manifest = {"train": paths, "validation": [], "test": []}
    train_cfg = {"batch_size": 3, "class_weight": "null",
                 "stage1_epochs": 1, "stage2_epochs": 1,
                 "early_stopping_patience": 10, "reduce_lr_patience": 10}
    model_cfg = {"base": "mobilenetv3small", "weights": None,
                 "dropout": 0.0, "fine_tune_layers": 1}
    summary = train_experiment(dcfg, train_cfg, model_cfg, manifest, seed=42, verbose=False,
                               experiments_root=os.path.join(root, "experiments"))
    assert summary["num_classes"] == 3
    assert summary["classes"] == ["ripe", "unripe", "damaged"]
    assert os.path.isfile(summary["model_path"])
    assert list_runs("tmpds", root=os.path.join(root, "experiments"))  # experiment đã tạo
