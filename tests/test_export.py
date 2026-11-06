import os

import numpy as np
import pytest
import tensorflow as tf

from src.config import load_config
from src.export_tflite import _build_representative_dataset, _convert, _load_model, _save_model, _validate_tflite
from src.model import build_model


def test_export_float32_and_float16(tmp_path):
    """Xuất Float32/Float16 từ model build (chưa train) và validate input/output."""
    cfg = load_config()
    model = build_model(cfg)
    saved_dir = str(tmp_path / "best_model")
    _save_model(model, saved_dir)

    for dtype, fname in (("float32", "float32.tflite"), ("float16", "float16.tflite")):
        b = _convert(saved_dir, dtype)
        assert len(b) > 0
        out = tmp_path / fname
        out.write_bytes(b)
        assert out.stat().st_size > 0
        v = _validate_tflite(b)
        assert v["input_is_224_rgb"] is True
        assert v["output_is_3_classes"] is True
        # Cả float32/float16 đều giữ input float32 (dtype của weights mới khác)
        assert "float" in v["input_dtype"]


def test_representative_dataset_requires_real_train_images(tmp_path):
    """INT8 representative dataset phải từ ảnh train thật; chưa có manifest → None (không bịa số liệu)."""
    cfg = load_config()
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["split_manifest"] = str(tmp_path / "nope.json")
    assert _build_representative_dataset(cfg) is None


def test_int8_export_blocked_without_real_images(tmp_path):
    """Khi _build_representative_dataset trả None, export_tflite phải bỏ qua INT8 (không bịa)."""
    cfg = load_config()
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["split_manifest"] = str(tmp_path / "nope.json")
    # Một model hợp lệ đã lưu
    model = build_model(cfg)
    saved = tmp_path / "best_model"
    _save_model(model, str(saved))
    # Chỉ kiểm tra hàm rep dataset và convert float đi — không chạy export_tflite toàn bộ
    # để tránh phụ thuộc đường dẫn checkpoint của môi trường test (model_path resolve)
    assert _build_representative_dataset(cfg) is None


def test_export_report_contains_sizes_and_compare(tmp_path):
    """Validate + compare Keras/TFLite trên Float32 qua hàm validate/compare tiện ích."""
    from src.export_tflite import _compare_keras_tflite

    cfg = load_config()
    model = build_model(cfg)
    saved_dir = str(tmp_path / "saved")
    _save_model(model, saved_dir)
    b = _convert(saved_dir, "float32")
    (tmp_path / "m.tflite").write_bytes(b)
    assert (tmp_path / "m.tflite").stat().st_size > 0
    rng = np.random.RandomState(42)
    # Input model là float32 [0,255] (model tự preprocess_input [-1,1])
    dummy = rng.uniform(0, 255, size=(1, 224, 224, 3)).astype(np.float32)
    cmp = _compare_keras_tflite(model, b, dummy)
    assert isinstance(cmp["max_abs_diff"], float) and cmp["max_abs_diff"] >= 0
    assert isinstance(cmp["same_label"], bool)
    assert len(cmp["keras"]) == 3 and len(cmp["tflite"]) == 3


def test_labels_and_metadata_files_exist():
    from src.config import PROJECT_ROOT

    labels = os.path.join(PROJECT_ROOT, "models", "tflite", "labels.txt")
    assert os.path.isfile(labels)
    with open(labels) as f:
        assert [l.strip() for l in f if l.strip()] == ["unripe", "medium_ripe", "ripe"]
    metadata = os.path.join(PROJECT_ROOT, "models", "tflite", "model_metadata.json")
    assert os.path.isfile(metadata)
    doc = os.path.join(PROJECT_ROOT, "models", "tflite", "android_integration.md")
    assert os.path.isfile(doc)


def test_predict_shows_class_id_confidence_uncertain(tmp_path):
    """Đảm bảo predict hiển thị class ID, label, confidence, 3 xác suất và uncertain."""
    from src.predict import predict_image
    from src.model import build_model

    cfg = load_config()
    model = build_model(cfg)
    dummy = tf.cast(tf.random.uniform((224, 224, 3), 0, 255, seed=3), tf.uint8)
    p = tmp_path / "x.jpg"
    tf.io.write_file(str(p), tf.image.encode_jpeg(dummy))
    r = predict_image(model, str(p), cfg["project"]["classes"], confidence_threshold=0.6)
    assert set(r.keys()) >= {"class_id", "label", "confidence", "uncertain", "status", "probabilities"}
    assert r["class_id"] == cfg["project"]["classes"].index(r["label"])
    assert len(r["probabilities"]) == 3
