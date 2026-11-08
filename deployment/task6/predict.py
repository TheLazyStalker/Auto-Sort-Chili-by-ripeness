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
# crop_v1: giữ nguyên tham số từ config.yaml (threshold 30, pad 0.20, ...)
CROP_THRESHOLD = 30.0
CROP_PAD_FRAC = 0.20
CROP_BORDER_FRAC = 0.05
CROP_MIN_AREA_FRAC = 0.005
CROP_MAX_AREA_FRAC = 0.60
CROP_DILATE_PX = 2
# Nếu ảnh đã là derivative 224×224 (được xuất từ pipeline) → bỏ qua crop.
ALREADY_CROPPED_SIZE = (224, 224)


def load_classes(mapping_path=CLASS_MAPPING):
    import json
    if os.path.isfile(mapping_path):
        with open(mapping_path, encoding="utf-8") as f:
            mp = json.load(f)
        # class_mapping.json: {label: index}
        return [k for k, _ in sorted(mp.items(), key=lambda kv: kv[1])]
    return CLASSES


def estimate_background(arr, border_frac=CROP_BORDER_FRAC):
    h, w = arr.shape[:2]
    band = max(1, int(round(min(h, w) * border_frac)))
    parts = [arr[:band].reshape(-1, 3), arr[-band:].reshape(-1, 3),
             arr[:, :band].reshape(-1, 3), arr[:, -band:].reshape(-1, 3)]
    return np.median(np.concatenate(parts, axis=0), axis=0).astype(np.float32)


def clean_mask(mask, dilate_px=CROP_DILATE_PX):
    """Mở/đóng + largest connected component + dilation (PIL thuần)."""
    if Image is None:
        raise RuntimeError("Cần Pillow (pip install pillow)")
    from PIL import ImageFilter
    m = Image.fromarray((mask.astype(np.uint8) * 255))
    # open 3x3
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    # close 5x5
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    arr = np.asarray(m) > 127

    # largest connected component (DFS stack)
    h, w = arr.shape
    out = np.zeros_like(arr)
    visited = np.zeros_like(arr)
    best = (0, None)
    for y in range(h):
        for x in range(w):
            if arr[y, x] and not visited[y, x]:
                stack = [(y, x)]
                comp = []
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    comp.append((cy, cx))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and arr[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(comp) > best[0]:
                    best = (len(comp), comp)
    if best[1]:
        for y, x in best[1]:
            out[y, x] = True
    if dilate_px > 0:
        out = np.asarray(Image.fromarray((out.astype(np.uint8) * 255)).filter(
            ImageFilter.MaxFilter(2 * dilate_px + 1))) > 127
    return out


def crop_v1(arr, threshold=CROP_THRESHOLD, pad_frac=CROP_PAD_FRAC):
    """Crop quả khỏi nền (cùng quy trình như task trước); trả về 224×224 RGB uint8."""
    h, w = arr.shape[:2]
    bg = estimate_background(arr)
    diff = arr.astype(np.float32) - bg.reshape(1, 1, 3)
    mask = np.sqrt(np.sum(diff * diff, axis=-1)) > float(threshold)
    mask = clean_mask(mask)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("empty_mask: không tìm thấy foreground (ảnh nền trơn?)")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    pad = float(pad_frac) * max(bw, bh)
    side = max(1, int(round(max(bw, bh) + 2.0 * pad)))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sx0, sy0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    bg_u8 = np.clip(np.rint(bg), 0, 255).astype(np.uint8)
    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[...] = bg_u8
    ix0, iy0 = max(0, sx0), max(0, sy0)
    ix1, iy1 = min(w, sx0 + side), min(h, sy0 + side)
    if ix1 > ix0 and iy1 > iy0:
        canvas[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = arr[iy0:iy1, ix0:ix1]
    img = Image.fromarray(canvas).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_input(path):
    """Đọc ảnh → float32 (1,224,224,3) [0,255].

    - Nếu ảnh đã 224×224 (derivative crop_v1) → không crop lại.
    - Ngược lại chạy crop_v1 (cùng hàm pipeline gốc).
    """
    if Image is None:
        raise RuntimeError("Cần Pillow (pip install pillow)")
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if arr.shape[:2] != ALREADY_CROPPED_SIZE:
        arr = crop_v1(arr)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Termux CLI predict ớt (Task 6)")
    parser.add_argument("image", nargs="?", help="ảnh đơn")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="đường dẫn .tflite hoặc 'float32'/'float16'")
    parser.add_argument("--dir", help="thư mục nhiều ảnh")
    parser.add_argument("--out", help="CSV output (với --dir)")
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
