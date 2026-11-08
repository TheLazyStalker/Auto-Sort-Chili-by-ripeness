"""Sinh derivative crop_v1 cho các ảnh raw được chấp nhận (Phase 1).

Dùng:
  python scripts/make_crop_derivatives.py [--config config.yaml]

Quy tắc (đã chốt Q1–Q7):
- Raw luôn giữ nguyên; script chỉ ĐỌC dataset/raw/** và GHỊ derivative vào
  dataset/derived/crop_v1/{train,validation,test,_unassigned}/<class>/.
- Ảnh trong ``review_exclusions.csv`` (status đang bị loại) hoàn toàn không được
  tạo derivative — ghi vào báo cáo để review.
- Ảnh cũ (không có group_id): split=train.
- Ảnh mới có group_id: split theo ``group_splits.csv``; nếu group chưa được gán
  split → xếp vào _unassigned (metadata để trống split → manifest sẽ dừng an toàn
  cho đến khi group được gán).
- Cùng hàm ``crop_v1`` từ src/crop.py; không thêm Rescaling(1/255).
- Xác minh SHA256 của mọi ảnh raw trước và sau khi sinh derivative.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.crop import CROP_VERSION, crop_v1, load_image_rgb  # noqa: E402
from src.dataset import scan_raw_dataset  # noqa: E402
from src.task3 import excluded_filenames, load_capture_log, load_group_splits  # noqa: E402

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

METADATA_COLUMNS = [
    "source_path", "source_sha256", "crop_version", "image_id",
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    "pad_frac", "output_path", "output_sha256",
    "width", "height", "label", "group_id", "split",
    "crop_status", "background_rgb", "threshold", "notes",
]


def sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw_sha256(raw_root: str, derived_root: str, metrics_dir: str,
                      classes: list) -> int:
    """Xác minh SHA256 raw so với baseline trong crop_v1_report.json (nếu có)."""
    import json

    report_path = os.path.join(metrics_dir, "crop_v1_report.json")
    if not os.path.isfile(report_path):
        print("[LỖI] Không tìm thấy crop_v1_report.json — chạy generation đầy đủ trước.")
        return 2
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    baseline = report.get("raw_sha256", {})
    by_path = baseline.get("by_path") or {}
    if not by_path:
        print("[LỖI] Report không có baseline SHA256 per-file (by_path). "
              "Chạy lại generation với --no-write để ghi nhận baseline.")
        return 2

    summary = scan_raw_dataset(raw_root, classes)
    raw_files = sorted(p for fs in summary.files.values() for p in fs)
    changed = [p for p in raw_files if sha256_file(p) != by_path.get(p)]
    print(f"Xác minh SHA256 raw: {len(raw_files)} ảnh, "
          f"{len(raw_files) - len(changed)} khớp baseline")
    if changed:
        print(f"[LỖI] {len(changed)} ảnh raw bị thay đổi so với baseline:")
        for p in changed[:20]:
            print(f"  - {p}")
        return 1
    print("[OK] Raw không đổi so với baseline đã ghi nhận.")
    return 0


def write_csv(path: str, columns: list, rows: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sinh derivative crop_v1 từ raw")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--no-write", action="store_true",
                        help="Chỉ tính toán và báo cáo, không ghi derivative (khô chạy)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Chỉ xác minh SHA256 raw so với baseline trong report hiện có")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = cfg["paths"]
    crop = cfg["crop"]
    classes = cfg["project"]["classes"]

    raw_root = paths["dataset_raw"]
    derived_root = os.path.join(paths["dataset_derived"], crop["version"])
    exclusions_path = paths["exclusions_file"]
    capture_log_path = paths["capture_log"]
    group_splits_path = paths["group_splits"]

    if args.verify_only:
        return verify_raw_sha256(raw_root, derived_root, paths["reports_metrics"], classes)

    print(f"Raw: {raw_root}")
    print(f"Derived: {derived_root}")

    summary = scan_raw_dataset(raw_root, classes)
    print(f"Quét raw: {summary.total} ảnh "
          f"({', '.join(f'{c}={summary.counts[c]}' for c in classes)})")
    if summary.corrupt:
        print(f"[LỖI] {len(summary.corrupt)} ảnh lỗi/không đọc được — dừng an toàn.")
        for p in summary.corrupt[:10]:
            print(f"  - {p}")
        return 2
    if summary.duplicates:
        print(f"[CẢNH BÁO] {len(summary.duplicates)} nhóm ảnh trùng nội dung (md5):")
        for h, ps in list(summary.duplicates.items())[:5]:
            print(f"  - md5={h[:8]}... → {len(ps)} file: {ps[:3]}")

    excluded = excluded_filenames(exclusions_path)
    capture_log = load_capture_log(capture_log_path)
    group_splits = load_group_splits(group_splits_path)

    # ---- Baseline SHA256 của toàn bộ raw trước khi sinh ----
    raw_files = sorted(p for fs in summary.files.values() for p in fs)
    print(f"Ghi nhận SHA256 baseline của {len(raw_files)} ảnh raw ...")
    baseline = {p: sha256_file(p) for p in raw_files}
    print("  OK")

    print(f"Ảnh bị loại (không tạo derivative): {len(excluded)}")
    for name in sorted(excluded):
        print(f"  - {name}")

    rows: list[dict] = []
    status_counts: dict[str, int] = {}
    excluded_found: list[str] = []
    failed_rows: list[dict] = []
    unassigned_groups: list[dict] = []

    for cls in classes:
        for src_path in summary.files.get(cls, []):
            fname = os.path.basename(src_path)
            if fname in excluded:
                excluded_found.append(fname)
                continue

            log_row = capture_log.get(fname, {})
            image_id = (log_row.get("image_id") or "").strip()
            group_id = (log_row.get("group_id") or "").strip()
            if group_id:
                split = (group_splits.get(group_id) or "").strip()
                if not split:
                    split_dir = "_unassigned"
                    unassigned_groups.append({"filename": fname, "group_id": group_id, "class": cls})
                else:
                    split_dir = split
            else:
                split = "train"
                split_dir = "train"

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
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

            stem = os.path.splitext(fname)[0]
            out_rel = os.path.join(split_dir, cls, f"{stem}.png")
            out_path = os.path.join(derived_root, out_rel)
            out_sha = ""

            if result.image is not None:
                if not args.no_write:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    from PIL import Image
                    Image.fromarray(result.image, mode="RGB").save(out_path, format="PNG")
                out_sha = sha256_file(out_path) if not args.no_write else ""

            if result.status == "failed":
                failed_rows.append({
                    "filename": fname, "class": cls, "group_id": group_id,
                    "reason": result.reason, "source_path": src_path,
                })

            rows.append({
                "source_path": src_path,
                "source_sha256": baseline[src_path],
                "crop_version": crop["version"],
                "image_id": image_id,
                "bbox_x0": result.bbox[0] if result.bbox else "",
                "bbox_y0": result.bbox[1] if result.bbox else "",
                "bbox_x1": result.bbox[2] if result.bbox else "",
                "bbox_y1": result.bbox[3] if result.bbox else "",
                "pad_frac": f"{result.pad_frac:.6f}",
                "output_path": out_path if result.image is not None else "",
                "output_sha256": out_sha,
                "width": result.image.shape[1] if result.image is not None else "",
                "height": result.image.shape[0] if result.image is not None else "",
                "label": cls,
                "group_id": group_id,
                "split": split,
                "crop_status": result.status,
                "background_rgb": ",".join(str(v) for v in result.background_rgb),
                "threshold": f"{result.threshold:.1f}",
                "notes": result.reason,
            })

    # ---- Xác minh lại SHA256 raw sau khi sinh ----
    after = {p: sha256_file(p) for p in raw_files}
    changed = [p for p in raw_files if after[p] != baseline[p]]
    print(f"\nXác minh SHA256 raw sau khi sinh: {len(raw_files) - len(changed)}/{len(raw_files)} khớp")
    if changed:
        print(f"[LỖI] {len(changed)} ảnh raw bị thay đổi:")
        for p in changed[:10]:
            print(f"  - {p}")
        return 2

    metadata_path = os.path.join(derived_root, "metadata.csv")
    manifest_path = os.path.join(derived_root, "manifest.json")
    if not args.no_write:
        write_csv(metadata_path, METADATA_COLUMNS, rows)

    included = len(rows)
    generated = sum(1 for r in rows if r["output_path"])
    report = {
        "crop_version": CROP_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "raw": {
            "total": summary.total,
            "included": included,
            "excluded": len(excluded_found),
            "corrupt": len(summary.corrupt),
            "duplicate_groups": len(summary.duplicates),
        },
        "status_counts": status_counts,
        "per_class": {
            c: {
                "included": sum(1 for r in rows if r["label"] == c),
                "generated": sum(1 for r in rows if r["label"] == c and r["output_path"]),
            }
            for c in classes
        },
        "generated": generated,
        "failed": failed_rows,
        "unassigned_groups": unassigned_groups,
        "excluded_list": sorted(excluded_found),
        "raw_sha256": {
            "checked": len(raw_files),
            "all_match": not changed,
            "changed": changed,
            "by_path": baseline,
        },
        "metadata_csv": metadata_path,
        "manifest_json": manifest_path,
    }

    report_path = os.path.join(paths["reports_metrics"], "crop_v1_report.json")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if not args.no_write:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "crop_version": CROP_VERSION,
                "generated_at": report["generated_at"],
                "raw_total": summary.total,
                "included": included,
                "excluded": len(excluded_found),
                "status_counts": status_counts,
                "generated": generated,
                "metadata_csv": "metadata.csv",
            }, f, indent=2, ensure_ascii=False)

    print(f"\nDerivative đã sinh: {generated}/{included}")
    print("Status: " + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
    print(f"Báo cáo: {report_path}")

    if excluded_found:
        print(f"[OK] {len(excluded_found)} ảnh bị loại không tạo derivative.")
    if unassigned_groups:
        print(f"[CẢNH BÁO] {len(unassigned_groups)} ảnh mới có group chưa gán split "
              "→ _unassigned (dừng manifest cho đến khi gán).")
    if failed_rows:
        print(f"[CẢNH BÁO] {len(failed_rows)} ảnh crop failed (không có derivative):")
        for fr in failed_rows[:10]:
            print(f"  - {fr['filename']} ({fr['class']}): {fr['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
