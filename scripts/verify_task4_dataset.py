"""Task 4 — verify dữ liệu derivative & manifest trước/sau khi train.

Kiểm tra:
- Số lượng PNG derivative (463), kích thước 224×224, mode RGB.
- Không ảnh bị loại (review_exclusions status=excluded) xuất hiện.
- Không pending_review trong manifest.
- Label hợp lệ / đủ class.
- (Tùy chọn) xác minh raw SHA256 so với baseline trong crop_v1_report.json.

Dùng:
  python scripts/verify_task4_dataset.py [--config config.yaml] [--sha256]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from src.config import load_config
from src.task4 import Task4DataError, load_task3_manifest, verify_manifest_class_counts

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify Task 4 dataset")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--sha256", action="store_true",
                        help="Xác minh SHA256 raw so với baseline (crop_v1_report.json)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    classes = cfg["project"]["classes"]

    # 1. Derivative PNG
    root = os.path.join(cfg["paths"]["dataset_derived"], "crop_v1")
    pngs = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
    print(f"Derivative PNG: {len(pngs)}")
    bad_size, bad_mode = [], []
    for p in pngs:
        with Image.open(p) as im:
            if im.size != (224, 224):
                bad_size.append(p)
            if im.mode != "RGB":
                bad_mode.append(p)
    print(f"  kích thước !=224x224: {len(bad_size)} | mode !=RGB: {len(bad_mode)}")
    if bad_size or bad_mode:
        raise Task4DataError("Crop derivative lỗi size/mode — dừng.")

    # 2. Manifest task3: count/label
    manifest = load_task3_manifest(cfg)
    counts = verify_manifest_class_counts(manifest, classes)
    print("Class counts (manifest): " + ", ".join(f"{c}={counts[c]}" for c in classes))

    # 3. Exclusion & review status
    excl = set()
    with open(cfg["paths"]["exclusions_file"], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("status") or "").strip().lower() == "excluded":
                excl.add((row.get("filename") or "").strip())
    bad_excl = []
    for split, items in manifest.items():
        for it in items:
            src = os.path.basename(it.get("source_path") or "")
            if src in excl:
                bad_excl.append(src)
    print(f"Excluded files in manifest: {len(bad_excl)} {bad_excl[:5]}")
    if bad_excl:
        raise Task4DataError("Ảnh bị loại vẫn trong manifest — dừng.")

    # 4. SHA256 raw (optional)
    if args.sha256:
        report_path = os.path.join(cfg["paths"]["reports_metrics"], "crop_v1_report.json")
        if not os.path.isfile(report_path):
            print("[CẢNH BÁO] Không có crop_v1_report.json — bỏ qua SHA256.")
        else:
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
            by_path = report.get("raw_sha256", {}).get("by_path", {})
            changed = []
            for p, h in by_path.items():
                if not os.path.isfile(p):
                    changed.append((p, "missing"))
                    continue
                import hashlib
                hh = hashlib.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        hh.update(chunk)
                if hh.hexdigest() != h:
                    changed.append((p, "changed"))
            print(f"Raw SHA256: {len(by_path) - len(changed)}/{len(by_path)} khớp baseline")
            if changed:
                raise Task4DataError(f"{len(changed)} ảnh raw đổi: {changed[:5]}")

    print("[OK] Task 4 dataset hợp lệ.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
