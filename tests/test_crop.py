"""Tests cho src/crop.py — crop_v1 (Phase 1)."""
import numpy as np
import pytest

from src.crop import (
    DEFAULT_OUTPUT_SIZE,
    crop_v1,
    estimate_background,
    foreground_mask,
    load_image_rgb,
)

try:
    from PIL import Image
except ImportError:
    pytest.skip("Pillow not available", allow_module_level=True)


def _img(size=64, bg=(255, 255, 255), pepper=None, pepper_color=(40, 40, 40)):
    """Sinh ảnh RGB synthetic: nền đồng nhất + quả (hình chữ nhật) ở giữa."""
    arr = np.empty((size, size, 3), dtype=np.uint8)
    arr[...] = np.asarray(bg, dtype=np.uint8)
    if pepper is not None:
        x0, y0, x1, y1 = pepper
        arr[y0:y1, x0:x1] = np.asarray(pepper_color, dtype=np.uint8)
    return arr


def test_estimate_background_median_of_border():
    """Nền đồng nhất → median viền = màu nền."""
    img = _img(bg=(220, 235, 235))
    bg = estimate_background(img)
    assert tuple(int(round(v)) for v in bg) == (220, 235, 235)


def test_foreground_mask_threshold():
    img = _img(pepper=(20, 20, 44, 44))
    bg = estimate_background(img)
    mask = foreground_mask(img, bg, threshold=30.0)
    assert mask.shape == img.shape[:2]
    assert mask[32, 32]  # bên trong quả
    assert not mask[0, 0]  # nền


def test_crop_v1_deterministic():
    img = _img(pepper=(20, 20, 44, 44))
    r1 = crop_v1(img)
    r2 = crop_v1(img)
    assert r1.status == "ok"
    assert r1.image is not None and r2.image is not None
    assert np.array_equal(r1.image, r2.image)


def test_crop_v1_output_shape_and_dtype():
    img = _img(pepper=(20, 20, 44, 44))
    r = crop_v1(img)
    assert r.image is not None
    assert r.image.shape == (DEFAULT_OUTPUT_SIZE, DEFAULT_OUTPUT_SIZE, 3)
    assert r.image.dtype == np.uint8
    assert r.image.min() >= 0 and r.image.max() <= 255


def test_crop_v1_bbox_centred_with_padding():
    img = _img(pepper=(20, 20, 44, 44))
    r = crop_v1(img, pad_frac=0.20)
    x0, y0, x1, y1 = r.bbox
    # Bbox bao đúng quả (sau morphology có thể mở rộng chút do dilation 2px)
    assert x0 >= 20 - 4 and y0 >= 20 - 4
    assert x1 <= 44 + 4 and y1 <= 44 + 4
    # Crop square 224 → sau resize, quả nằm ở giữa ảnh
    assert r.image is not None
    assert r.crop_box is not None
    cx = (r.crop_box[0] + r.crop_box[2]) / 2
    cy = (r.crop_box[1] + r.crop_box[3]) / 2
    assert abs(cx - 32) <= 6 and abs(cy - 32) <= 6


def test_crop_v1_empty_mask_failed():
    img = _img()  # toàn nền
    r = crop_v1(img)
    assert r.status == "failed"
    assert r.image is None
    assert r.reason == "empty_mask"


def test_crop_v1_mask_too_small_needs_review():
    img = _img(pepper=(28, 28, 36, 36))  # quả 8×8 (1.56% khung)
    r = crop_v1(img, min_area_frac=0.05)
    assert r.status == "needs_review"
    assert "too_small" in r.reason
    assert r.image is not None  # vẫn xuất ảnh để QA


def test_crop_v1_mask_too_large_needs_review():
    arr = np.full((64, 64, 3), 255, dtype=np.uint8)  # nền trắng
    arr[2:62, 2:62] = (40, 40, 40)  # quả lớn 60×60 (~88% khung)
    r = crop_v1(arr, min_area_frac=0.005, max_area_frac=0.60)
    assert r.status == "needs_review"
    assert "too_large" in r.reason


def test_crop_v1_accepts_background_ref():
    img = _img(pepper=(20, 20, 44, 44))
    r_ref = crop_v1(img, background_ref=(255, 255, 255))
    r_auto = crop_v1(img)
    assert r_ref.status == "ok"
    assert r_ref.background_rgb == (255, 255, 255)
    assert r_auto.status == "ok"


def test_load_image_rgb_returns_uint8(tmp_path):
    p = tmp_path / "img.png"
    Image.fromarray(_img()).save(p)
    arr = load_image_rgb(str(p))
    assert arr.dtype == np.uint8
    assert arr.shape == (64, 64, 3)
