"""Tests Task 6 — deployment package (predict CLI, benchmark, env check)."""
import csv
import os
import subprocess
import sys

import numpy as np
import pytest

DEPLOY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "deployment", "task6")
sys.path.insert(0, DEPLOY)

from predict import Predictor, load_classes, load_input  # noqa: E402
from benchmark import main as benchmark_main  # noqa: E402

try:
    from PIL import Image
except ImportError:
    pytest.skip("Pillow not available", allow_module_level=True)

MODEL16 = os.path.join(DEPLOY, "models", "chili_float16.tflite")
MODEL32 = os.path.join(DEPLOY, "models", "chili_float32.tflite")
CLASS_MAP = os.path.join(DEPLOY, "models", "class_mapping.json")


def _make_img(path, size=(224, 224), color=(200, 60, 60)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_class_mapping_order():
    assert load_classes(CLASS_MAP) == ["unripe", "medium_ripe", "ripe"]


@pytest.mark.parametrize("model", [MODEL16, MODEL32])
def test_predictor_loads(model):
    p = Predictor(model)
    assert p.interp is not None
    assert list(p.in_d["shape"]) == [1, 224, 224, 3]


def test_single_image_predict(tmp_path):
    p = os.path.join(str(tmp_path), "a.png")
    _make_img(p)
    predictor = Predictor(MODEL16)
    r = predictor.predict_file(p)
    assert r["class"] in ["unripe", "medium_ripe", "ripe"]
    assert r["class_id"] in (0, 1, 2)
    assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-3
    assert r["preprocessing_ms"] >= 0 and r["inference_ms"] >= 0
    assert r["total_ms"] >= r["inference_ms"]


def test_directory_csv_output(tmp_path):
    d = str(tmp_path / "imgs")
    os.makedirs(d)
    for i in range(3):
        _make_img(os.path.join(d, f"img{i}.jpg"))
    out_csv = str(tmp_path / "out.csv")
    from predict import main as predict_main
    rc = predict_main(["--dir", d, "--out", out_csv, "--quiet",
                       "--model", MODEL16])
    assert rc == 0
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for r in rows:
        assert r["class"] in ["unripe", "medium_ripe", "ripe"]
        assert r["total_ms"] != ""


def test_invalid_image_handled_in_dir(tmp_path):
    d = str(tmp_path / "imgs")
    os.makedirs(d)
    _make_img(os.path.join(d, "ok.png"))
    with open(os.path.join(d, "broken.jpg"), "wb") as f:
        f.write(b"not an image")
    from predict import main as predict_main
    out_csv = str(tmp_path / "out.csv")
    rc = predict_main(["--dir", d, "--out", out_csv, "--quiet",
                       "--model", MODEL16])
    assert rc == 0
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    broken = [r for r in rows if r["filename"] == "broken.jpg"][0]
    assert broken["class"] == "ERROR"


def test_crop_v1_on_large_image(tmp_path):
    """Ảnh 1024×1024 (raw-style) với foreground → được crop về 224×224 bởi load_input."""
    p = os.path.join(str(tmp_path), "big.png")
    arr = np.full((1024, 1024, 3), 220, dtype=np.uint8)  # nền sáng
    arr[400:700, 400:700] = (40, 200, 40)  # "quả" xanh trung tâm
    Image.fromarray(arr).save(p)
    tensor = load_input(p)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 255.0


def test_benchmark_runs(tmp_path, capsys):
    out = str(tmp_path / "b.json")
    rc = benchmark_main(["--model", MODEL16, "--warmup", "2", "--runs", "5",
                         "--out", out])
    assert rc == 0
    import json
    with open(out) as f:
        b = json.load(f)
    assert b["n_runs"] == 5
    assert b["inference_ms_mean"] > 0


# ---------------- fast_crop (Task 6 perf) ----------------

def _scene(size, pepper_box):
    import numpy as np
    arr = np.full((size[1], size[0], 3), 220, dtype=np.uint8)
    x0, y0, x1, y1 = pepper_box
    arr[y0:y1, x0:x1] = (40, 200, 40)
    return arr


def test_fast_crop_output_shape_and_bbox():
    from fast_crop import crop_v1, foreground_bbox
    arr = _scene((1024, 1024), (400, 400, 700, 700))
    out = crop_v1(arr)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    bbox = foreground_bbox(arr)
    assert bbox is not None
    x0, y0, x1, y1, _ = bbox
    # bbox xấp xỉ vùng quả (dung sai do downscale)
    assert abs(x0 - 400) <= 40 and abs(y0 - 400) <= 40
    assert abs(x1 - 700) <= 40 and abs(y1 - 700) <= 40


def test_fast_crop_empty_mask_raises():
    import numpy as np
    from fast_crop import crop_v1
    flat = np.full((512, 512, 3), 200, dtype=np.uint8)
    with pytest.raises(ValueError, match="empty_mask"):
        crop_v1(flat)


def test_fast_crop_preserves_class_vs_pil_crop():
    """fast_crop phải cho cùng class dự đoán như crop gốc (src/crop.py)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src"))
    from crop import crop_v1 as slow_crop
    from fast_crop import crop_v1 as fast_crop

    arr = _scene((1024, 1024), (380, 420, 690, 700))
    slow = slow_crop(arr).image
    fast = fast_crop(arr)
    predictor = Predictor(MODEL16)
    p_slow = predictor.predict(slow.astype(np.float32)[None, ...])[0]
    p_fast = predictor.predict(fast.astype(np.float32)[None, ...])[0]
    assert int(np.argmax(p_slow)) == int(np.argmax(p_fast))


def test_predict_bytes_matches_file(tmp_path):
    """predict_bytes(JPEG) phải khớp predict_file trên cùng ảnh."""
    p = os.path.join(str(tmp_path), "img.jpg")
    _make_img(p, size=(224, 224), color=(200, 60, 60))
    predictor = Predictor(MODEL16)
    r_file = predictor.predict_file(p)
    with open(p, "rb") as f:
        data = f.read()
    r_bytes = predictor.predict_bytes(data)
    assert r_bytes["class_id"] == r_file["class_id"]
    assert abs(sum(r_bytes["probabilities"].values()) - 1.0) < 1e-3


def test_predict_bytes_large_image(tmp_path):
    """Ảnh lớn (OV2640-like) qua predict_bytes → crop + predict OK."""
    p = os.path.join(str(tmp_path), "big.jpg")
    arr = _scene((1600, 1200), (600, 400, 1000, 800))
    Image.fromarray(arr).save(p, quality=90)
    predictor = Predictor(MODEL16)
    with open(p, "rb") as f:
        r = predictor.predict_bytes(f.read())
    assert r["class_id"] in (0, 1, 2)
    assert r["total_ms"] > 0
