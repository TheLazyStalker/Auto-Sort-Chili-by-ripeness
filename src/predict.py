"""Dự đoán ảnh đơn: đọc ảnh → tiền xử lý → softmax → nhãn + xác suất từng lớp.

Hỗ trợ:
- CLI: python -m src.predict --image path/to/image.jpg [--model ...] [--threshold 0.6]
- Hiển thị class ID, tên lớp, confidence và xác suất cả ba lớp.
- Trạng thái `uncertain` khi confidence dưới ngưỡng (mặc định config: inference.confidence_threshold).
"""
from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

from .config import get_classes, load_config

CLASSES_FALLBACK = ["unripe", "medium_ripe", "ripe"]


def load_image_tensor(image_path: str, image_size: int = 224):
    """Đọc ảnh → uint8 tensor 224x224x3 trong [0, 255].

    Model tự preprocess_input ([0,255]→[-1,1]), KHÔNG rescale trước.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    return tf.expand_dims(img, axis=0)


def predict_image(model, image_path: str, classes: list | None = None,
                  image_size: int = 224, confidence_threshold: float = 0.6) -> dict:
    """Dự đoán một ảnh. Trả về dict {class_id, label, confidence, uncertain, probabilities}."""
    classes = classes or getattr(model, "classes_", None) or CLASSES_FALLBACK
    tensor = load_image_tensor(image_path, image_size)
    prob = model.predict(tensor, verbose=0)[0]
    prob = np.asarray(prob, dtype=float)
    # Đảm bảo softmax (tổng ~1)
    prob = prob / prob.sum() if prob.sum() > 0 else prob
    best = int(np.argmax(prob))
    confidence = float(prob[best])
    uncertain = confidence < float(confidence_threshold)
    return {
        "path": image_path,
        "class_id": best,
        "label": classes[best],
        "confidence": confidence,
        "uncertain": uncertain,
        "status": "uncertain" if uncertain else "confident",
        "probabilities": {cls: float(p) for cls, p in zip(classes, prob)},
    }


def predict_batch(model, image_paths: list, classes: list | None = None,
                  image_size: int = 224, confidence_threshold: float = 0.6) -> list:
    return [predict_image(model, p, classes, image_size, confidence_threshold) for p in image_paths]


def load_model_for_predict(cfg: dict, model_path: str | None = None):
    model_path = model_path or os.path.join(cfg["paths"]["saved_model"], "best_model.keras")
    if not os.path.exists(model_path):
        for c in [os.path.join(cfg["paths"]["checkpoints"], "chili_final.keras"),
                  os.path.join(cfg["paths"]["checkpoints"], "best_chili.keras")]:
            if os.path.exists(c):
                model_path = c
                break
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy model đã huấn luyện tại {model_path}. "
            "Chạy `python -m src.train` trước (sau khi thêm ảnh vào dataset/raw)."
        )
    model = tf.keras.models.load_model(model_path)
    return model


def _get_threshold(cfg: dict, override: float | None) -> float:
    if override is not None:
        return float(override)
    return float(cfg.get("inference", {}).get("confidence_threshold", 0.6))


def main(argv=None) -> None:
    import argparse, json

    parser = argparse.ArgumentParser(description="Dự đoán độ chín ớt từ ảnh")
    parser.add_argument("--image", default=None, help="Đường dẫn ảnh (.jpg/.jpeg/.png)")
    parser.add_argument("positional_image", nargs="?", default=None,
                          help="Đường dẫn ảnh (phần 1 dùng positional)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--threshold", type=float, default=None,
                          help="Ngưỡng confidence; dưới ngưỡng → uncertain")
    args = parser.parse_args(argv)

    image_path = args.image or args.positional_image
    if not image_path:
        parser.error("Cần chỉ định --image path/to/image.jpg (hoặc truyền đường dẫn trực tiếp)")

    cfg = load_config(args.config) if args.config else load_config()
    try:
        model = load_model_for_predict(cfg, args.model)
    except FileNotFoundError as exc:
        import sys
        print(f"[LỖI] {exc}", file=sys.stderr)
        sys.exit(2)
    threshold = _get_threshold(cfg, args.threshold)
    result = predict_image(model, image_path, get_classes(cfg),
                           confidence_threshold=threshold)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nclass={result['class_id']} label={result['label']} "
          f"confidence={result['confidence']:.4f} status={result['status']}")


if __name__ == "__main__":
    main()
