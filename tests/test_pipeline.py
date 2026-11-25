"""Tests cho pipeline trung tâm: pipeline_config.json + centering wiring."""
import json
import os

import pytest

from src.centering import build_command, copy_into_raw, list_images
from src.pipeline_config import (DEFAULT_CONFIG_PATH, PipelineConfigError, center_params,
                                 crop_params, load_pipeline_config, model_config,
                                 split_config, training_config)


def _write_config(tmp_path, **overrides):
    cfg = json.load(open(DEFAULT_CONFIG_PATH))
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    p = os.path.join(str(tmp_path), "pipeline_config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return p


def test_default_config_loads():
    cfg = load_pipeline_config()
    assert os.path.isabs(cfg["paths"]["datasets_root"])
    assert cfg["dataset"]["classes"] == ["ripe", "unripe", "damaged"]
    assert cfg["centering"]["background_bgr"] == [235, 235, 220]
    assert cfg["preprocessing"]["pipeline"] == "crop_v1"
    assert cfg["split"]["ratios"] == [0.7, 0.15, 0.15]


def test_config_change_propagates_to_centering(tmp_path):
    """Đổi màu nền trong JSON → đổi tham số CLI của tool (không hardcode)."""
    p = _write_config(tmp_path, centering={"background_bgr": [255, 255, 255], "size": 512})
    cfg = load_pipeline_config(p)
    params = center_params(cfg)
    assert params["bg"] == "255,255,255"
    assert params["size"] == 512
    cmd = build_command("/in", "/out", params)
    assert "255,255,255" in cmd
    assert "512" in cmd


def test_config_change_propagates_to_crop(tmp_path):
    p = _write_config(tmp_path, preprocessing={"threshold": 40.0, "pad_frac": 0.35})
    cfg = load_pipeline_config(p)
    cp = crop_params(cfg)
    assert cp["threshold"] == 40.0 and cp["pad_frac"] == 0.35


def test_config_change_propagates_to_training(tmp_path):
    p = _write_config(tmp_path, training={"stage1_epochs": 3, "batch_size": 8})
    cfg = load_pipeline_config(p)
    t = training_config(cfg)
    assert t["stage1_epochs"] == 3 and t["batch_size"] == 8
    assert model_config(cfg)["base"] == "mobilenetv3small"
    assert split_config(cfg)["seed"] == 42


def test_invalid_split_ratios_rejected(tmp_path):
    p = _write_config(tmp_path, split={"ratios": [0.5, 0.1, 0.1]})
    with pytest.raises(PipelineConfigError, match="split.ratios"):
        load_pipeline_config(p)


def test_invalid_background_rejected(tmp_path):
    p = _write_config(tmp_path, centering={"background_bgr": [300, 0, 0]})
    with pytest.raises(PipelineConfigError, match="background_bgr"):
        load_pipeline_config(p)


def test_missing_config_falls_back_to_defaults(tmp_path):
    cfg = load_pipeline_config(os.path.join(str(tmp_path), "nope.json"))
    assert cfg["centering"]["size"] > 0  # defaults vẫn dùng được


def test_centering_copy_never_overwrites(tmp_path):
    src = tmp_path / "centered"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"new")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.jpg").write_bytes(b"old")
    res = copy_into_raw(str(src), str(raw))
    assert res["copied"] == []
    assert res["skipped_existing"] == ["a.jpg"]
    assert (raw / "a.jpg").read_bytes() == b"old"  # không ghi đè


def test_list_images_filters_extensions(tmp_path):
    d = tmp_path
    for name in ("a.jpg", "b.JPEG", "c.png", "notes.txt"):
        (d / name).write_bytes(b"x")
    names = [os.path.basename(p).lower() for p in list_images(str(d))]
    assert names == ["a.jpg", "b.jpeg", "c.png"]