"""Smoke test model: khởi tạo MobileNetV2 + dự đoán tensor ngẫu nhiên (chưa cần dataset)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, set_seeds  # noqa: E402
from src.model import build_model, freeze_backbone  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    set_seeds(cfg)
    print("Đang khởi tạo MobileNetV2 (pretrained ImageNet)...")
    model = build_model(cfg)
    freeze_backbone(model, fine_tune_layers=0)
    model.summary()
    # Tên file weights ImageNet thực tế của MobileNetV2 trong Keras cache
    # (include_top=False → dùng bản ..._1.0_224_no_top.h5).
    weights_path = os.path.expanduser(
        "~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    )
    print("Pretrained weights downloaded:", os.path.exists(weights_path), f"({weights_path})")

    print("\nDự đoán tensor ngẫu nhiên (2, 224, 224, 3)...")
    dummy = tf.random.uniform((2, 224, 224, 3), 0.0, 1.0, seed=cfg["training"]["seed"])
    pred = model.predict(dummy, verbose=0)
    print("Output shape:", pred.shape)
    print("Output (softmax):", np.round(pred, 4))
    assert pred.shape == (2, 3), "Output phải có 3 lớp"
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-5), "Softmax phải tổng bằng 1"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
