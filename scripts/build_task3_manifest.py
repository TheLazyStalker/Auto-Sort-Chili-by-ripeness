"""Xây dựng manifest Task 3 từ metadata crop_v1 (provisional khi val/test rỗng).

Dùng:
  python scripts/build_task3_manifest.py [--config config.yaml] [--require-all-splits]

Quy tắc (đã chốt Q1–Q7, task3_data_plan.md §8):
- Chỉ đưa derivative có crop_status=ok vào manifest.
- Ảnh bị loại (review_exclusions.csv) không bao giờ vào manifest.
- Ảnh cũ (không group_id) → train.
- Ảnh mới có group_id → split theo group_splits.csv; thiếu gán → dừng an toàn.
- Mặc định cho phép validation/test rỗng (manifest tạm thời Phase 1); dùng
  --require-all-splits khi có dữ liệu mới đầy đủ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.task3 import (
    SPLIT_NAMES,
    Task3ManifestError,
    build_task3_manifest,
    excluded_filenames,
    load_capture_log,
    load_derived_metadata,
    load_group_splits,
    write_task3_manifest,
)  # noqa: E402
from scripts.label_review import load_label_overrides  # noqa: E402

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Xây dựng manifest Task 3")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--require-all-splits", action="store_true",
                        help="Bắt buộc đủ 3 lớp ở cả 3 tập (khi có dữ liệu mới)")
    parser.add_argument("--allow-needs-review", action="store_true",
                        help="Cho phép đưa derivative needs_review vào manifest (mặc định: không)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = cfg["paths"]
    crop = cfg["crop"]
    classes = cfg["project"]["classes"]

    derived_root = os.path.join(paths["dataset_derived"], crop["version"])
    metadata_path = os.path.join(derived_root, "metadata.csv")
    manifest_path = paths["split_manifest_task3"]
    meta_path = os.path.join(os.path.dirname(manifest_path), "split_manifest_task3_meta.json")

    exclusions = excluded_filenames(paths["exclusions_file"])
    group_splits = load_group_splits(paths["group_splits"])
    rows = load_derived_metadata(metadata_path)

    # review_status từ capture_log: chỉ mọi dòng có filename; status rỗng → bỏ qua.
    review_statuses = {}
    for name, row in load_capture_log(paths["capture_log"]).items():
        status = (row.get("review_status") or "").strip()
        if status:
            review_statuses[name] = status

    # label_overrides: filename → new_label
    label_overrides = {}
    for row in load_label_overrides(paths["label_overrides"]):
        fname = (row.get("filename") or "").strip()
        new_label = (row.get("new_label") or "").strip()
        if fname and new_label:
            label_overrides[fname] = new_label

    print(f"Metadata derivative: {metadata_path} ({len(rows)} dòng)")
    print(f"Exclusions: {len(exclusions)} ảnh bị loại")
    print(f"Group splits: {len(group_splits)} group đã gán")
    print(f"Review status (từ capture_log): {len(review_statuses)} ảnh")
    print(f"Label overrides: {len(label_overrides)}")

    try:
        result = build_task3_manifest(
            rows,
            classes,
            exclusions=exclusions,
            group_splits=group_splits,
            allow_needs_review=args.allow_needs_review,
            check_files=True,
            require_all_splits=args.require_all_splits,
            review_statuses=review_statuses,
            label_overrides=label_overrides,
        )
    except Task3ManifestError as exc:
        print(f"[LỖI] {exc}")
        return 2

    manifest = result["manifest"]
    verification = result["verification"]

    meta = write_task3_manifest(
        manifest,
        manifest_path,
        meta_path,
        crop_version=crop["version"],
        exclusions_path=paths["exclusions_file"],
        extra={
            "metadata_csv_sha256": _file_sha256(metadata_path) if os.path.isfile(metadata_path) else "",
            "verification": verification,
            "skipped": result["skipped"],
        },
    )

    print(f"\nManifest: {manifest_path}")
    for split in SPLIT_NAMES:
        print(f"  {split:10s}: {len(manifest.get(split, []))} ảnh")
    print(f"  totals: {verification['totals']}")
    print(f"  unique_groups: {verification['unique_groups']}")
    print(f"  provisional: {verification['provisional']}")
    print(f"  missing_class_split: {verification['missing_class_split']}")
    print(f"  excluded_present: {verification['excluded_present']}")
    print(f"  overlap_paths: {verification['overlap_paths']}")
    print(f"  group_leaks: {verification['group_leaks']}")
    print(f"\nMetadata manifest: {meta_path}")
    print(f"  manifest_sha256: {meta['manifest_sha256']}")
    print(f"  counts: {meta['counts']}")

    if not verification["ok"]:
        print("\n[CẢNH BÁO] Xác minh chưa đạt (xem chi tiết bên trên).")
        return 1
    print("\n[OK] Manifest hợp lệ theo xác minh Task 3.")
    return 0


def _file_sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
