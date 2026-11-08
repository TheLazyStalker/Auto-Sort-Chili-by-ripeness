"""Task 5 — Convert best_chili.keras → TFLite Float32 + Float16 (không INT8).

Quyết định Task 5: loại bỏ hoàn toàn INT8. Chỉ tạo:
- chili_float32.tflite (bản đối chiếu)
- chili_float16.tflite (bản tối ưu chính cho điện thoại Android (mẫu đã test))

Quy trình:
1. Load best_chili.keras (Task 4).
2. Export SavedModel tạm (không đụng .keras gốc).
3. Convert Float32 (optimize DEFAULT) và Float16 (supported_types=[tf.float16]).
4. Validate TFLite (input/output shape, dtype, inference chạy, sum≈1, finite).
5. Ghi artifacts/task5/ (conversion_config.json, class_mapping.json).

Preprocessing thực tế (đã xác minh từ model): MobileNetV3Small nhúng
Rescaling(scale=1/128, offset=-1) → input float32 RGB [0,255] → [-1,1] bên trong.
KHÔNG thêm preprocessing ngoài; không double-scale.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from src.config import load_config
from src.utils import ensure_dir

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

TASK4_MODEL = "best_chili.keras"
F32_NAME = "chili_float32.tflite"
F16_NAME = "chili_float16.tflite"


def load_task4_model(out_dir: str):
    path = os.path.join(out_dir, TASK4_MODEL)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy {path} — chạy Task 4 trước.")
    return tf.keras.models.load_model(path), path


def save_savedmodel_tmp(model, tmp_dir: str) -> str:
    os.makedirs(tmp_dir, exist_ok=True)
    saved_dir = os.path.join(tmp_dir, "saved_model")
    model.export(saved_dir)
    return saved_dir


def convert_from_savedmodel(saved_dir: str, dtype: str) -> bytes:
    """Convert SavedModel → TFLite.

    - ``float32``: KHÔNG set optimizations → float32 thuần (bản đối chiếu).
      Lưu ý: ``Optimize.DEFAULT`` không có representative dataset sẽ áp dynamic-range
      quantization (weights → int8), làm lệch kết quả — Task 5 loại bỏ INT8 nên
      bản float32 phải giữ nguyên trọng số float32.
    - ``float16``: ``Optimize.DEFAULT`` + ``supported_types=[tf.float16]`` → trọng số
      float16, activations float32.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    if dtype == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif dtype == "float32":
        pass  # float32 thuần: không bật optimization nào
    else:
        raise ValueError(f"Chỉ hỗ trợ float32/float16, nhận '{dtype}'")
    return converter.convert()


def validate_tflite_bytes(tflite_bytes: bytes, input_shape=(1, 224, 224, 3),
                          num_classes: int = 3, dummy_input=None) -> dict:
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    checks = {
        "input_shape": [int(v) for v in in_d["shape"]],
        "input_dtype": str(in_d["dtype"]),
        "output_shape": [int(v) for v in out_d["shape"]],
        "output_dtype": str(out_d["dtype"]),
        "input_is_224_rgb": list(in_d["shape"]) == list(input_shape),
        "output_is_3_classes": int(out_d["shape"][-1]) == num_classes,
    }
    # Quantization params nếu có (float32/16 thường None)
    checks["input_quantization"] = (
        {"scale": float(in_d["quantization_parameters"]["scales"][0]),
         "zero_point": int(in_d["quantization_parameters"]["zero_points"][0])}
        if len(in_d["quantization_parameters"]["scales"]) else None)
    if dummy_input is not None:
        x = dummy_input.astype(np.float32)
        if in_d["dtype"] == np.float16:
            x = x.astype(np.float16)
        interp.set_tensor(in_d["index"], x)
        interp.invoke()
        out = interp.get_tensor(out_d["index"])
        out = np.asarray(out, dtype=np.float32)
        checks["inference_output_shape"] = list(out.shape)
        checks["inference_ok"] = True
        checks["finite"] = bool(np.isfinite(out).all())
        checks["prob_sum_approx_1"] = bool(abs(float(out.sum()) - 1.0) < 1e-3)
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 5 — TFLite Float32/Float16 conversion")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--task4-dir", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    task4_dir = args.task4_dir or cfg["paths"]["artifacts_task4"]
    out_dir = os.path.join(os.path.dirname(task4_dir), "task5")
    ensure_dir(out_dir)

    model, keras_path = load_task4_model(task4_dir)
    print(f"Loaded Keras: {keras_path}")
    print(f"Input: {model.input_shape} | Output: {model.output_shape}")

    tmp_dir = os.path.join(out_dir, "_tmp_saved_model")
    saved_dir = save_savedmodel_tmp(model, tmp_dir)
    print(f"SavedModel tạm: {saved_dir}")

    dummy = np.random.RandomState(42).uniform(0, 255, size=(1, 224, 224, 3)).astype(np.float32)

    conversion = {"float32": {}, "float16": {}}
    for dtype, fname in (("float32", F32_NAME), ("float16", F16_NAME)):
        t0 = time.perf_counter()
        data = convert_from_savedmodel(saved_dir, dtype)
        dt = time.perf_counter() - t0
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "wb") as f:
            f.write(data)
        checks = validate_tflite_bytes(data, dummy_input=dummy)
        conversion[dtype] = {
            "path": out_path,
            "size_bytes": len(data),
            "convert_time_s": dt,
            "validate": checks,
        }
        print(f"{dtype}: {fname} | {len(data)/1e6:.3f} MB | convert {dt:.2f}s")
        print(f"  validate: {checks}")

    # config + class mapping
    classes = cfg["project"]["classes"]
    class_mapping = {c: i for i, c in enumerate(classes)}
    conversion_config = {
        "source": keras_path,
        "source_keras_params": int(model.count_params()),
        "decision": "INT8 fully removed (Task 5 §1); only Float32 + Float16",
        "float32": {"optimizations": "none (pure float32)", "note": "benchmark reference; "
                    "NO INT8/dynamic-range quantization (Task 5 decision)"},
        "float16": {"optimizations": "DEFAULT", "supported_types": ["tf.float16"],
                    "note": "primary deployment variant for điện thoại Android (mẫu đã test)"},
        "preprocessing_actual": "MobileNetV3Small include_preprocessing=True: "
                                "Rescaling(scale=0.0078431, offset=-1.0) inside model; "
                                "input float32 RGB [0,255]; no extra preprocessing; no double-scale",
        "input_shape": [1, 224, 224, 3],
        "output_classes": classes,
        "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "conversion_config.json"), "w", encoding="utf-8") as f:
        json.dump(conversion_config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)

    # dọn SavedModel tạm (không xóa artifact khác)
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nArtifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
