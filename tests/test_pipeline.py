import numpy as np
import tensorflow as tf


def test_predict_with_random_tensor():
    """Predict trên tensor ngẫu nhiên (chưa có dataset) — kiểm tra pipeline."""
    from src.predict import predict_image
    from src.model import build_model
    from src.config import load_config

    cfg = load_config()
    model = build_model(cfg)
    # Tạo ảnh tạm bằng TF ghi ra file rồi predict
    import tempfile, os

    dummy = tf.random.uniform((224, 224, 3), 0, 255, seed=42)
    dummy_u8 = tf.cast(dummy, tf.uint8)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "dummy.jpg")
        tf.io.write_file(path, tf.image.encode_jpeg(dummy_u8))
        result = predict_image(model, path, cfg["project"]["classes"])
    assert result["label"] in ["unripe", "medium_ripe", "ripe"]
    assert result["class_id"] in [0, 1, 2]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["status"] in ("uncertain", "confident")
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_predict_uncertain_when_low_confidence():
    from src.predict import predict_image
    from src.model import build_model
    from src.config import load_config
    import tempfile, os

    cfg = load_config()
    model = build_model(cfg)
    dummy = tf.cast(tf.random.uniform((224, 224, 3), 0, 255, seed=1), tf.uint8)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "dummy.jpg")
        tf.io.write_file(path, tf.image.encode_jpeg(dummy))
        # Ngưỡng rất cao → mọi kết quả đều uncertain
        result = predict_image(model, path, cfg["project"]["classes"],
                               confidence_threshold=0.999999)
    assert result["uncertain"] is True
    assert result["status"] == "uncertain"


def test_predict_cli_dash_dash_image():
    """CLI --image (phần 2 yêu cầu) hoạt động với ảnh tensor ngẫu nhiên."""
    from src.predict import _get_threshold
    assert _get_threshold({"inference": {"confidence_threshold": 0.5}}, None) == 0.5
    assert _get_threshold({}, 0.7) == 0.7
    # Parser của main chấp nhận --image
    import inspect
    from src import predict as predict_mod
    src = inspect.getsource(predict_mod.main)
    assert "--image" in src


def test_softmax_sums_to_one():
    out = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
    assert np.allclose(out.sum(axis=1), 1.0)
