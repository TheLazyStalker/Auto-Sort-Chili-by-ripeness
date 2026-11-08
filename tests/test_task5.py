"""Tests Task 5 — TFLite conversion (Float32/Float16) & evaluation utilities."""
import os

import numpy as np
import pytest
import tensorflow as tf

from src.config import load_config
from src.task4 import create_dev_split, load_task3_manifest
from scripts.convert_task5 import convert_from_savedmodel, validate_tflite_bytes
from scripts.evaluate_task5 import tflite_predict, full_metrics

CLASSES = ["unripe", "medium_ripe", "ripe"]


def _build_toy_model():
    """Model nhỏ 2 conv → dense 3 classes (không cần ImageNet)."""
    inp = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(4, 3, activation="relu")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def _save_tflite(model, tmp_path, dtype):
    saved = str(tmp_path / "saved")
    model.export(saved)
    data = convert_from_savedmodel(saved, dtype)
    p = str(tmp_path / f"m_{dtype}.tflite")
    with open(p, "wb") as f:
        f.write(data)
    return p, data


def test_convert_float32_and_float16(tmp_path):
    model = _build_toy_model()
    for dtype in ("float32", "float16"):
        _p, data = _save_tflite(model, tmp_path, dtype)
        assert len(data) > 0
        checks = validate_tflite_bytes(data, input_shape=(1, 32, 32, 3),
                                      num_classes=3,
                                      dummy_input=np.zeros((1, 32, 32, 3), np.float32))
        assert checks["input_is_224_rgb"] is True
        assert checks["output_is_3_classes"] is True
        assert checks["inference_ok"] is True
        assert checks["finite"] is True
        assert checks["prob_sum_approx_1"] is True
        assert checks["input_quantization"] is None  # float32/16 không quant hóa

def test_convert_rejects_int8(tmp_path):
    model = _build_toy_model()
    saved = str(tmp_path / "saved")
    model.export(saved)
    with pytest.raises(ValueError, match="float32/float16"):
        convert_from_savedmodel(saved, "int8")


def test_tflite_predict_matches_keras(tmp_path):
    model = _build_toy_model()
    # train nhẹ để output có ý nghĩa
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 255, size=(32, 32, 32, 3)).astype(np.float32)
    y = rng.randint(0, 3, size=(32,))
    model.fit(X, y, epochs=2, verbose=0)
    p, _ = _save_tflite(model, tmp_path, "float32")

    keras_out = model.predict(X, verbose=0)
    tflite_out = tflite_predict(p, X, batch=8)
    assert keras_out.shape == tflite_out.shape
    # tolerance rộng vì conv nhỏ/float32 chuyển đổi có thể sai khác nhỏ
    assert np.abs(keras_out - tflite_out).max() < 0.05
    assert (np.argmax(keras_out, axis=1) == np.argmax(tflite_out, axis=1)).mean() >= 0.9


def test_full_metrics_shape():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 0, 2])
    m = full_metrics(y_true, y_pred, CLASSES)
    assert m["accuracy"] == pytest.approx(7 / 9)
    assert set(m["per_class"]) == set(CLASSES)
    assert m["confusion_matrix"] == [[4, 0, 0], [0, 2, 1], [0, 1, 1]]


def test_dev_split_manifest_loadable():
    cfg = load_config()
    manifest = load_task3_manifest(cfg)
    dev = create_dev_split(manifest, CLASSES, seed=42)
    assert len(dev["train"]) + len(dev["validation"]) == 463
    assert len(dev["validation"]) == 92
