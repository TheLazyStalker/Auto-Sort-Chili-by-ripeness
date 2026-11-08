"""Tests Task 4 — MobileNetV3Small, preprocessing, dev split, artifacts."""
import os

import numpy as np
import pytest
import tensorflow as tf

from src.config import load_config, set_seeds
from src.model import (build_base_model, build_model, compile_model, freeze_backbone,
                       normalize_base_name)
from src.task4 import (create_dev_split, load_task3_manifest, make_dataset,
                       verify_manifest_class_counts, build_augmentation)


def _cfg_v3():
    cfg = load_config()
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["base"] = "mobilenetv3small"
    return cfg


def test_normalize_base_name():
    assert normalize_base_name("mobilenetv3small") == "mobilenetv3small"
    assert normalize_base_name("v3") == "mobilenetv3small"
    assert normalize_base_name("mobilenetv2") == "mobilenetv2"
    with pytest.raises(ValueError):
        normalize_base_name("resnet50")


def test_v3_backbone_has_internal_rescaling():
    cfg = _cfg_v3()
    base = build_base_model(cfg, trainable=False)
    # include_preprocessing=True → layer thứ 2 phải là Rescaling
    rescaling = [l for l in base.layers if isinstance(l, tf.keras.layers.Rescaling)]
    assert rescaling, [l.name for l in base.layers[:3]]
    assert rescaling[0].scale is not None


def test_v3_model_output_shape_and_softmax():
    cfg = _cfg_v3()
    set_seeds(cfg)
    model = build_model(cfg)
    x = tf.random.uniform((3, 224, 224, 3), 0, 255, seed=42)  # [0,255]
    y = model.predict(x, verbose=0)
    assert y.shape == (3, 3)
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5)
    assert (y >= 0).all() and (y <= 1).all()
    assert model.base_name == "mobilenetv3small"


def test_v3_model_handles_uint8_input():
    cfg = _cfg_v3()
    model = build_model(cfg)
    x_u8 = tf.cast(tf.random.uniform((2, 224, 224, 3), 0, 255, seed=7), tf.uint8)
    y = model.predict(x_u8, verbose=0)
    assert y.shape == (2, 3)


def test_freeze_backbone_v3():
    cfg = _cfg_v3()
    model = build_model(cfg)
    freeze_backbone(model, fine_tune_layers=0)
    base = model.layers[1]
    assert base.trainable is False
    freeze_backbone(model, fine_tune_layers=5)
    assert base.trainable is True
    trainable_names = [l.name for l in base.layers if l.trainable]
    assert len(trainable_names) == 5


def test_class_mapping_order():
    cfg = load_config()
    mapping = {c: i for i, c in enumerate(cfg["project"]["classes"])}
    assert mapping == {"unripe": 0, "medium_ripe": 1, "ripe": 2}


def test_dev_split_80_20_stratified():
    manifest = {"train": []}
    classes = ["unripe", "medium_ripe", "ripe"]
    for i, cls in enumerate(classes):
        for k in range(50):
            manifest["train"].append({"path": f"/x/{cls}/{k}.png", "label": i})
    dev = create_dev_split(manifest, classes, seed=42)
    assert len(dev["train"]) == 120
    assert len(dev["validation"]) == 30
    # stratified: 10 val mỗi lớp (20% của 50)
    for i, cls in enumerate(classes):
        nv = sum(1 for it in dev["validation"] if it["label"] == i)
        assert nv == 10, (cls, nv)


def test_dev_split_deterministic():
    manifest = {"train": [{"path": f"/x/{i}.png", "label": i % 3} for i in range(60)]}
    classes = ["unripe", "medium_ripe", "ripe"]
    a = create_dev_split(manifest, classes, seed=42)
    b = create_dev_split(manifest, classes, seed=42)
    assert [i["path"] for i in a["train"]] == [i["path"] for i in b["train"]]
    assert [i["path"] for i in a["validation"]] == [i["path"] for i in b["validation"]]


def test_verify_manifest_class_counts():
    manifest = {"train": [
        {"path": "/x/1.png", "label": 0},
        {"path": "/x/2.png", "label": 1},
        {"path": "/x/3.png", "label": 0},
        {"path": "/x/4.png", "label": 2},
    ]}
    counts = verify_manifest_class_counts(manifest, ["unripe", "medium_ripe", "ripe"])
    assert counts == {"unripe": 2, "medium_ripe": 1, "ripe": 1}


def test_make_dataset_shapes():
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        paths, labels = [], []
        for i in range(4):
            p = os.path.join(td, f"i{i}.png")
            Image.new("RGB", (224, 224), (i * 40, 100, 100)).save(p)
            paths.append(p)
            labels.append(i % 3)
        ds = make_dataset(paths, labels, batch_size=2, shuffle=True, seed=1)
        for x, y in ds.take(1):
            assert x.shape == (2, 224, 224, 3)
            assert x.dtype == tf.uint8
            assert y.shape == (2,)
            assert y.dtype in (tf.int32, tf.int64)


def test_augmentation_no_hue_saturation():
    aug = build_augmentation(seed=42)
    names = [type(l).__name__ for l in aug.layers]
    assert not any("Hue" in n or "Saturation" in n for n in names)


def test_v3_saved_model_roundtrip(tmp_path):
    cfg = _cfg_v3()
    model = build_model(cfg)
    p = str(tmp_path / "m.keras")
    model.save(p)
    loaded = tf.keras.models.load_model(p)
    x = tf.random.uniform((1, 224, 224, 3), 0, 255, seed=1)
    np.testing.assert_allclose(loaded.predict(x, verbose=0),
                               model.predict(x, verbose=0), atol=1e-6)
