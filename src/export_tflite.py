"""Xuất mô hình sang TensorFlow Lite + SavedModel cho Android.

Xuất 3 loại (khi có model đã huấn luyện):
- Float32:  chili_ripeness.tflite        (fallback tương thích mọi thiết bị)
- Float16:  chili_ripeness_float16.tflite (GPU delegates, nhỏ hơn Float32)
- INT8:     chili_ripeness_int8.tflite   (full integer, representative dataset từ ảnh train thực tế)

Kiểm chứng: input RGB 224x224, tensor shape, datatype, output 3 lớp,
so sánh Keras vs TFLite (max diff trên tập hợp/ảnh ngẫu nhiên khi chưa có dataset,
trên ảnh train thật khi đã có dataset) và lưu báo cáo vào models/tflite/.

Lưu ý MediaPipe: model TFLite thô KHÔNG tự tương thích MediaPipe Tasks Vision
Image Classifier — cần thêm label/metadata qua công cụ chính thức (metadata writer).
File tiện ích được sinh ra cùng lúc: labels.txt + model_metadata.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
import tensorflow as tf

from .config import load_config, set_seeds
from .dataset import load_manifest


def _resolve_model_path(cfg: dict, model_path: str | None) -> str:
    model_path = model_path or os.path.join(cfg["paths"]["saved_model"], "best_model.keras")
    if not os.path.exists(model_path):
        for c in [os.path.join(cfg["paths"]["checkpoints"], "chili_final.keras"),
                  os.path.join(cfg["paths"]["checkpoints"], "best_chili.keras"),
                  os.path.join(cfg["paths"]["saved_model"], "best_model")]:
            if os.path.exists(c):
                model_path = c
                break
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy model đã huấn luyện tại {model_path}. "
            "Chạy `python -m src.train` trước (sau khi thêm ảnh vào dataset/raw)."
        )
    return model_path


def _load_model(path: str):
    if os.path.isdir(path):
        return tf.saved_model.load(path)
    return tf.keras.models.load_model(path)


def _save_model(model, saved_dir: str) -> str:
    os.makedirs(saved_dir, exist_ok=True)
    if hasattr(model, "export"):
        model.export(saved_dir)
    elif hasattr(model, "save"):
        model.save(saved_dir, save_format="tf")
    return saved_dir


def _convert(saved_dir: str, dtype: str, representative_ds=None) -> bytes:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    if dtype == "float32":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif dtype == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter._experimental_new_converter = True
    elif dtype == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_ds
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    else:
        raise ValueError(f"dtype không hỗ trợ: {dtype}")
    return converter.convert()


def _build_representative_dataset(cfg: dict, max_samples: int = 500):
    """Representative dataset từ ảnh TRAIN THẬT trong manifest (không dùng ảnh test).

    Trả về None nếu chưa có manifest/ảnh train — khi đó INT8 sẽ bị bỏ qua
    và ghi rõ lý do trong báo cáo (không dùng ảnh ngẫu nhiên để calibration).
    """
    manifest = load_manifest(cfg["paths"]["split_manifest"])
    if manifest is None or "train" not in manifest:
        return None
    train_items = manifest["train"]
    if not train_items:
        return None
    image_size = int(cfg["data"]["image_size"])
    file_paths = [i["path"] for i in train_items][:max_samples]

    def generator():
        # Giữ uint8 [0,255] khớp input model (model tự preprocess_input [-1,1]).
        for fpath in file_paths:
            img = tf.io.read_file(fpath)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, [image_size, image_size])
            yield [tf.expand_dims(img, axis=0)]

    return generator


def _validate_tflite(tflite_bytes: bytes, expected_shape=(1, 224, 224, 3), num_classes=3,
                     dummy_input: np.ndarray | None = None) -> dict:
    """Kiểm chứng TFLite: input/output shape, dtype, inference chạy được."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    input_shape = list(in_detail["shape"])
    output_shape = list(out_detail["shape"])
    checks = {
        "input_shape": [int(v) for v in input_shape],
        "input_dtype": str(in_detail["dtype"]),
        "output_shape": [int(v) for v in output_shape],
        "output_dtype": str(out_detail["dtype"]),
        "input_is_224_rgb": bool(input_shape == list(expected_shape)),
        "output_is_3_classes": bool(int(output_shape[-1]) == int(num_classes)),
    }
    if dummy_input is not None:
        interpreter.set_tensor(in_detail["index"], dummy_input)
        interpreter.invoke()
        out = interpreter.get_tensor(out_detail["index"])
        checks["inference_output_shape"] = list(out.shape)
        checks["inference_ok"] = True
    return checks


def _compare_keras_tflite(model, tflite_bytes: bytes, dummy: np.ndarray) -> dict:
    """So sánh dự đoán Keras vs TFLite trên tensor ngẫu nhiên (chưa cần dataset)."""
    keras_out = np.asarray(model.predict(dummy, verbose=0), dtype=float)[0]
    # TFLite chạy trên CPU với input float32
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    x = dummy.astype(np.float32)
    if in_detail["dtype"] == np.uint8:
        x = np.clip(x * 255, 0, 255).astype(np.uint8)
    elif in_detail["dtype"] == np.float16:
        x = x.astype(np.float16)
    interp.set_tensor(in_detail["index"], x)
    interp.invoke()
    tflite_out = np.asarray(interp.get_tensor(out_detail["index"]), dtype=float)[0]
    if tflite_out.dtype == np.uint8 or out_detail["dtype"] == np.uint8:
        tflite_out = tflite_out / 255.0
    return {
        "keras": keras_out.tolist(),
        "tflite": tflite_out.tolist(),
        "max_abs_diff": float(np.max(np.abs(keras_out - tflite_out))),
        "same_label": bool(int(np.argmax(keras_out)) == int(np.argmax(tflite_out))),
    }


def export_tflite(cfg: dict, model_path: str | None = None) -> dict:
    set_seeds(cfg)
    e = cfg.get("export", {})
    image_size = int(cfg["data"]["image_size"])
    paths = cfg["paths"]
    classes = cfg["project"]["classes"]
    max_samples = int(e.get("representative_samples", 500))

    model_path = _resolve_model_path(cfg, model_path)
    model = _load_model(model_path)

    saved_dir = _save_model(model, os.path.join(paths["saved_model"], "best_model"))
    os.makedirs(paths["tflite"], exist_ok=True)

    # Dummy input cho validate/compare: float32 [0,255] khớp input model
    # (model tự preprocess_input [0,255] -> [-1,1]). Tensor ngẫu nhiên của test.
    rng = np.random.RandomState(int(cfg.get("training", {}).get("seed", 42)))
    dummy = rng.uniform(0, 255, size=(1, image_size, image_size, 3)).astype(np.float32)

    # Labels + metadata cho Android/MediaPipe
    labels_path = os.path.join(paths["tflite"], "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    metadata_path = os.path.join(paths["tflite"], "model_metadata.json")
    metadata = {
        "model": "chili_ripeness_mobilenetv2",
        "input": {"shape": [1, image_size, image_size, 3], "type": "RGB",
                  "dtype": "float32 [0,255] (model tự preprocess_input [0,255]->[-1,1]) | uint8 for INT8",
                  "normalization": "Resize 224x224 (uint8 RGB [0,255]); model tự preprocess_input"},
        "output": {"shape": [1, len(classes)], "type": "softmax",
                   "labels": classes},
        "note": ("TFLite thô KHÔNG tự tương thích MediaPipe Tasks Vision; "
                 "cần metadata writer để nhúng labels/associates nếu dùng MediaPipe."),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    results: dict = {
        "saved_model": saved_dir,
        "labels": labels_path,
        "metadata": metadata_path,
        "models": {},
    }

    # 1) Float32
    float32_bytes = _convert(saved_dir, "float32")
    float32_path = os.path.join(paths["tflite"], e.get("tflite_filename", "chili_ripeness.tflite"))
    with open(float32_path, "wb") as f:
        f.write(float32_bytes)
    results["models"]["float32"] = {
        "path": float32_path, "size_kb": len(float32_bytes) / 1024,
        "validate": _validate_tflite(float32_bytes, dummy_input=dummy),
        "compare": _compare_keras_tflite(model, float32_bytes, dummy),
    }

    # 2) Float16
    try:
        float16_bytes = _convert(saved_dir, "float16")
        float16_path = os.path.join(paths["tflite"], e.get("float16_filename", "chili_ripeness_float16.tflite"))
        with open(float16_path, "wb") as f:
            f.write(float16_bytes)
        # Với float16 cần so sánh riêng; dummy float32 có thể sai dtype nên validate không truyền dummy
        results["models"]["float16"] = {
            "path": float16_path, "size_kb": len(float16_bytes) / 1024,
            "validate": _validate_tflite(float16_bytes),
        }
    except Exception as exc:
        results["models"]["float16_error"] = str(exc)

    # 3) INT8 — chỉ khi có ảnh train thật
    rep_ds = _build_representative_dataset(cfg, max_samples=max_samples)
    if rep_ds is None:
        results["int8"] = ("skipped (no train manifest/real train images — "
                          "INT8 cần ảnh train thực tế để calibration; không dùng ảnh ngẫu nhiên)")
        print("INT8: bỏ qua (chưa có ảnh train thực tế — cần calibration bằng ảnh train thật).")
    else:
        try:
            int8_bytes = _convert(saved_dir, "int8", rep_ds)
            int8_path = os.path.join(paths["tflite"], e.get("int8_filename", "chili_ripeness_int8.tflite"))
            with open(int8_path, "wb") as f:
                f.write(int8_bytes)
            results["models"]["int8"] = {
                "path": int8_path, "size_kb": len(int8_bytes) / 1024,
                "validate": _validate_tflite(int8_bytes),
            }
        except Exception as exc:
            results["models"]["int8_error"] = str(exc)

    # Lưu báo cáo tổng
    report_path = os.path.join(paths["tflite"], "export_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    results["report"] = report_path

    for name, info in results.get("models", {}).items():
        if isinstance(info, dict) and "size_kb" in info:
            print(f"{name}: {info['path']} ({info['size_kb']:.1f} KB)")
            if "validate" in info:
                print(f"  validate: {info['validate']}")
    print(f"Báo cáo export: {report_path}")
    return results


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Xuất mô hình sang TFLite")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    try:
        export_tflite(cfg, model_path=args.model)
    except FileNotFoundError as exc:
        import sys
        print(f"[CHƯA CÓ MODEL] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
