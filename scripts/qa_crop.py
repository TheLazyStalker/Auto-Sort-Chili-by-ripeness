"""QA crop_v1: overlay mask/bbox, kiểm tra kích thước/mode/tính xác định.

Dùng:
  python scripts/qa_crop.py [--config config.yaml] [--sample 30] [--seed 42]

Tạo:
- reports/figures/crop_qa/overlay_grid_samples.png — các ảnh mẫu: raw / mask / bbox.
- reports/figures/crop_qa/overlay_{class}.png — contact sheet overlay theo lớp.
- reports/metrics/crop_qa.json — số liệu QA: size/mode/determinism/status.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.crop import CROP_VERSION, crop_v1, load_image_rgb  # noqa: E402
from src.dataset import scan_raw_dataset  # noqa: E402

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def render_overlay(raw, result, thumb: int = 224):
    """Tạo ảnh QA: raw + mask + bbox (cùng tỷ lệ, 224×224)."""
    h, w = raw.shape[:2]
    scale = thumb / max(h, w)
    th = max(1, int(round(h * scale)))
    tw = max(1, int(round(w * scale)))

    from PIL import Image, ImageDraw
    raw_img = Image.fromarray(raw).resize((tw, th), Image.Resampling.BILINEAR)
    mask_img = Image.fromarray((result.mask.astype(np.uint8) * 255)).resize(
        (tw, th), Image.Resampling.NEAREST)
    overlay = raw_img.convert("RGB").copy()
    d = ImageDraw.Draw(overlay, "RGBA")
    mask_rgba = mask_img.convert("RGBA")
    red = Image.new("RGBA", mask_rgba.size, (255, 0, 0, 90))
    overlay = Image.alpha_composite(overlay.convert("RGBA"), Image.alpha_composite(mask_rgba, red))
    if result.bbox:
        bbox = tuple(int(round(v * scale)) for v in result.bbox)
        d2 = ImageDraw.Draw(overlay)
        d2.rectangle([bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1], outline=(0, 255, 0), width=3)
    return np.asarray(overlay.convert("RGB"), dtype=np.uint8)


def make_grid(images, cell=224, cols=None, pad: int = 8, bg=(0, 0, 0)):
    """Xếp các ảnh (list numpy array RGB) thành lưới; tự chọn số cột."""
    from PIL import Image
    total = len(images)
    if total == 0:
        return None
    cols = cols or min(total, 6)
    import math
    rows = math.ceil(total / cols)
    grid = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + pad) + pad), bg)
    for idx, arr in enumerate(images):
        r, c = divmod(idx, cols)
        img = Image.fromarray(arr).resize((cell, cell), Image.Resampling.BILINEAR)
        grid.paste(img, (pad + c * (cell + pad), pad + r * (cell + pad)))
    return grid


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="QA crop_v1")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--sample", type=int, default=30, help="Số ảnh mẫu cho grid QA")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = cfg["paths"]
    crop = cfg["crop"]
    classes = cfg["project"]["classes"]

    summary = scan_raw_dataset(paths["dataset_raw"], classes)
    raw_files = sorted(p for fs in summary.files.values() for p in fs)

    count = dict(Counter())
    size_fail: list[str] = []
    mode_fail: list[str] = []
    status_counts: Counter = Counter()
    reasons: Counter = Counter()

    samples: dict[str, list] = {c: [] for c in classes}

    rng = np.random.default_rng(args.seed)
    for cls in classes:
        files = summary.files.get(cls, [])
        pool = sorted(files)
        rng.shuffle(pool)
        picked = pool[: max(1, args.sample // len(classes) + 1)]
        for src_path in picked:
            result = crop_v1(
                load_image_rgb(src_path),
                threshold=float(crop["threshold"]),
                pad_frac=float(crop["pad_frac"]),
                output_size=int(crop["output_size"]),
                border_frac=float(crop["border_frac"]),
                min_area_frac=float(crop["min_area_frac"]),
                max_area_frac=float(crop["max_area_frac"]),
                dilate_px=int(crop["dilate_px"]),
            )
            status_counts[result.status] += 1
            if result.reason:
                reasons[result.reason] += 1
            if result.image is not None:
                if result.image.shape != (int(crop["output_size"]), int(crop["output_size"]), 3):
                    size_fail.append(os.path.basename(src_path))
                from PIL import Image
                img = Image.fromarray(result.image)
                if img.mode != "RGB":
                    mode_fail.append(os.path.basename(src_path))
            raw = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.uint8)
            samples[cls].append(render_overlay(raw, result))

    # Determinism: chạy lại 1 ảnh, so sánh pixel
    det_img = load_image_rgb(raw_files[0])
    a = crop_v1(det_img, threshold=float(crop["threshold"]), pad_frac=float(crop["pad_frac"]),
                 output_size=int(crop["output_size"]), border_frac=float(crop["border_frac"]),
                 min_area_frac=float(crop["min_area_frac"]),
                 max_area_frac=float(crop["max_area_frac"]),
                 dilate_px=int(crop["dilate_px"]))
    b = crop_v1(det_img, threshold=float(crop["threshold"]), pad_frac=float(crop["pad_frac"]),
                 output_size=int(crop["output_size"]), border_frac=float(crop["border_frac"]),
                 min_area_frac=float(crop["min_area_frac"]),
                 max_area_frac=float(crop["max_area_frac"]),
                 dilate_px=int(crop["dilate_px"]))
    deterministic = a.image is not None and b.image is not None and bool(np.array_equal(a.image, b.image))

    out_fig = os.path.join(paths["reports_figures"], "crop_qa")
    os.makedirs(out_fig, exist_ok=True)

    # Grid tổng hợp: tất cả mẫu
    all_cells = []
    for cls in classes:
        all_cells += samples[cls]
    grid_all = make_grid(all_cells, cell=224)
    if grid_all:
        grid_all.save(os.path.join(out_fig, "overlay_grid_samples.png"))

    # Grid riêng theo lớp
    for cls, arrs in samples.items():
        img = make_grid(arrs, cell=224)
        if img:
            img.save(os.path.join(out_fig, f"overlay_{cls}.png"))

    report = {
        "crop_version": CROP_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "sample_per_class": {c: len(samples[c]) for c in classes},
        "all_images": len(raw_files),
        "status_counts": dict(status_counts),
        "reasons": dict(reasons),
        "size_fail": size_fail,
        "mode_fail": mode_fail,
        "deterministic": deterministic,
        "output_dir": out_fig,
        "ok": not size_fail and not mode_fail and deterministic,
    }
    report_path = os.path.join(paths["reports_metrics"], "crop_qa.json")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Status counts: {dict(status_counts)}")
    print(f"Reasons: {dict(reasons)}")
    print(f"Size fail: {len(size_fail)}; Mode fail: {len(mode_fail)}; Deterministic: {deterministic}")
    print(f"QA report: {report_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
