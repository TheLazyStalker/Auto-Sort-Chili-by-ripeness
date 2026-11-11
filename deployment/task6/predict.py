#!/usr/bin/env python3
"""Termux CLI — predict ripeness cho ảnh ớt (crop_v1 derivatives).

Cách dùng:
  python predict.py image.jpg
  python predict.py --model models/chili_float16.tflite image.jpg
  python predict.py --dir images/ --out results.csv
  python predict.py --model float32 --dir images/ --out results.csv

Input mong đợi: ảnh đã crop_v1 (224×224×3 RGB [0,255]) hoặc raw có nền đồng
nhất → script áp dụng cùng crop_v1 như pipeline gốc (PIL/NumPy thuần).

Preprocessing:
- Đọc RGB [0,255], resize 224×224 (nếu chưa), KHÔNG chia 255, KHÔNG đổi RGB→BGR,
  KHÔNG thêm normalization khác. Model MobileNetV3Small có Rescaling(1/128, -1)
  nhúng bên trong (verify từ Task 4/5).

Output mỗi ảnh: filename, class, xác suất 3 lớp, preprocessing_ms, inference_ms,
total_ms. --dir + --out ghi CSV.

Runtime: ưu tiên ``tflite_runtime`` (Termux), fallback ``tensorflow.lite``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

# ---- Runtime TFLite (Termux: tflite_runtime; container: tensorflow.lite) ----
try:
    from tflite_runtime.interpreter import Interpreter
    RUNTIME = "tflite_runtime"
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        RUNTIME = "tensorflow.lite"
    except ImportError:
        Interpreter = None
        RUNTIME = None

try:
    from PIL import Image
except ImportError:
    Image = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "chili_float16.tflite")
CLASS_MAPPING = os.path.join(HERE, "models", "class_mapping.json")

CLASSES = ["unripe", "medium_ripe", "ripe"]
INPUT_SIZE = 224
# Nếu ảnh đã là derivative 224×224 (được xuất từ pipeline) → bỏ qua crop.
ALREADY_CROPPED_SIZE = (224, 224)

# fast_crop: morphology numpy + union-find (nhanh hơn nhiều so với PIL rankfilter/DFS)
from fast_crop import crop_v1 as fast_crop_v1  # noqa: E402


def load_classes(mapping_path=CLASS_MAPPING):
    import json
    if os.path.isfile(mapping_path):
        with open(mapping_path, encoding="utf-8") as f:
            mp = json.load(f)
        # class_mapping.json: {label: index}
        return [k for k, _ in sorted(mp.items(), key=lambda kv: kv[1])]
    return CLASSES


def load_input(path, decode_max=1024):
    """Đọc ảnh → float32 (1,224,224,3) [0,255].

    - Nếu ảnh đã 224×224 (derivative crop_v1) → không crop lại (nhanh ~1-2 ms).
    - Ảnh lớn: dùng ``Image.draft`` để JPEG decode ở tỉ lệ nhỏ (nhanh hơn nhiều
      và ít RAM), rồi chạy fast_crop_v1 (morphology numpy + union-find).
    """
    if Image is None:
        raise RuntimeError("Cần Pillow (pip install pillow)")
    with Image.open(path) as im:
        # JPEG: decode giảm tỉ lệ (no-op cho PNG/ảnh nhỏ). Không đổi kết quả vì
        # bước cuối luôn resize về 224.
        if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg"):
            try:
                im.draft("RGB", (decode_max, decode_max))
            except Exception:
                pass
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if arr.shape[:2] != ALREADY_CROPPED_SIZE:
        arr = fast_crop_v1(arr)
    tensor = arr.astype(np.float32)
    return np.expand_dims(tensor, axis=0)


class Predictor:
    def __init__(self, model_path=DEFAULT_MODEL):
        if Interpreter is None:
            raise RuntimeError("Không tìm thấy TFLite runtime. Cài "
                               "tflite_runtime (Termux) hoặc tensorflow.")
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.in_d = self.interp.get_input_details()[0]
        self.out_d = self.interp.get_output_details()[0]
        # Validate shape/dtype
        in_shape = list(self.in_d["shape"])
        if in_shape != [1, INPUT_SIZE, INPUT_SIZE, 3]:
            raise RuntimeError(f"Input shape bất ngờ: {in_shape}")
        self.classes = load_classes()

    def predict(self, tensor):
        t0 = time.perf_counter()
        x = tensor
        if self.in_d["dtype"] == np.float16:
            x = tensor.astype(np.float16)
        self.interp.set_tensor(self.in_d["index"], x)
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_d["index"])
        prob = np.asarray(out, dtype=np.float32)[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return prob, infer_ms

    def predict_file(self, path):
        t0 = time.perf_counter()
        tensor = load_input(path)
        prep_ms = (time.perf_counter() - t0) * 1000.0
        prob, infer_ms = self.predict(tensor)
        cls_idx = int(np.argmax(prob))
        return {
            "filename": os.path.basename(path),
            "class": self.classes[cls_idx],
            "class_id": cls_idx,
            "probabilities": {c: float(prob[i]) for i, c in enumerate(self.classes)},
            "preprocessing_ms": prep_ms,
            "inference_ms": infer_ms,
            "total_ms": prep_ms + infer_ms,
        }

    def predict_bytes(self, data: bytes, q: int = 90):
        """Đọc ảnh từ bytes (VD ESP32/OV2640 stream) → kết quả predict.

        Trả về dict giống predict_file (filename='<bytes>').
        """
        import io
        t0 = time.perf_counter()
        with Image.open(io.BytesIO(data)) as im:
            if os.path.splitext(getattr(im, "filename", "") or "")[1].lower() in (".jpg", ".jpeg") or True:
                try:
                    im.draft("RGB", (1024, 1024))
                except Exception:
                    pass
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if arr.shape[:2] != ALREADY_CROPPED_SIZE:
            arr = fast_crop_v1(arr)
        prep_ms = (time.perf_counter() - t0) * 1000.0
        prob, infer_ms = self.predict(arr.astype(np.float32)[None, ...])
        cls_idx = int(np.argmax(prob))
        return {
            "filename": "<stream>",
            "class": self.classes[cls_idx],
            "class_id": cls_idx,
            "probabilities": {c: float(prob[i]) for i, c in enumerate(self.classes)},
            "preprocessing_ms": prep_ms,
            "inference_ms": infer_ms,
            "total_ms": prep_ms + infer_ms,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Termux CLI predict ớt (Task 6)")
    parser.add_argument("image", nargs="?", help="ảnh đơn")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="đường dẫn .tflite hoặc 'float32'/'float16'")
    parser.add_argument("--dir", help="thư mục nhiều ảnh")
    parser.add_argument("--out", help="CSV output (với --dir)")
    parser.add_argument("--stdin", action="store_true",
                        help="đọc JPEG từ stdin (luồng ESP32/OV2640)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    model_path = args.model
    if model_path in ("float16", "float32"):
        model_path = os.path.join(HERE, "models", f"chili_{model_path}.tflite")
    if not os.path.isfile(model_path):
        print(f"[LỖI] Không tìm thấy model: {model_path}", file=sys.stderr)
        return 2

    try:
        predictor = Predictor(model_path)
    except RuntimeError as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Runtime: {RUNTIME} | model: {model_path}")

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"[LỖI] Không có thư mục: {args.dir}", file=sys.stderr)
            return 2
        rows = []
        for fname in sorted(os.listdir(args.dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            p = os.path.join(args.dir, fname)
            try:
                r = predictor.predict_file(p)
            except Exception as exc:
                r = {"filename": fname, "class": "ERROR", "class_id": -1,
                     "probabilities": {}, "preprocessing_ms": -1,
                     "inference_ms": -1, "total_ms": -1, "error": str(exc)}
            rows.append(r)
            if not args.quiet:
                print(f"{r['filename']}: {r['class']} (total {r['total_ms']:.1f}ms)")
        if args.out:
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["filename", "class", "class_id", "p_unripe",
                            "p_medium_ripe", "p_ripe", "preprocessing_ms",
                            "inference_ms", "total_ms", "error"])
                for r in rows:
                    pr = r["probabilities"]
                    w.writerow([r["filename"], r["class"], r["class_id"],
                                pr.get("unripe", ""), pr.get("medium_ripe", ""),
                                pr.get("ripe", ""), f"{r['preprocessing_ms']:.2f}",
                                f"{r['inference_ms']:.2f}", f"{r['total_ms']:.2f}",
                                r.get("error", "")])
            print(f"Đã ghi CSV: {args.out}")
        return 0

    if args.stdin:
        data = sys.stdin.buffer.read()
        r = predictor.predict_bytes(data)
        print(json_dumps(r))
        return 0

    if not args.image:
        parser.print_help()
        return 1
    r = predictor.predict_file(args.image)
    print(json_dumps(r))
    return 0


def json_dumps(obj):
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
