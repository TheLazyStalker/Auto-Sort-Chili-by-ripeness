"""Import ảnh mới từ staging vào dataset/raw/<class>/ và ghi capture_log.

Dùng:
  python scripts/import_new_images.py --source <dir> --session-id <id> \
      [--camera OV2640] [--review-status accepted] [--notes "..."] [--dry-run]

Quy tắc:
- Nguồn phải có thư mục con theo lớp: unripe/, medium_ripe/, ripe/.
- Mỗi ảnh được validate trước khi copy: decode được, >0 px, không trùng md5
  với raw hiện có hoặc trong cùng batch.
- Không ghi đè file raw đã tồn tại (filename trùng → lỗi, dừng toàn bộ).
- Sinh image_id ổn định riêng (image_id KHÔNG phụ thuộc filename).
- Ghi thêm dòng vào capture_log.csv (mọi status phải thuộc từ vựng đã chốt).
- Không sửa/xóa/ghi đè bất kỳ ảnh raw hiện có.
- Không dùng ảnh ngoài nguồn được chỉ định.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.dataset import scan_raw_dataset  # noqa: E402
from src.task3 import (  # noqa: E402
    CAPTURE_LOG_COLUMNS,
    Task3ManifestError,
    _read_csv,
    write_csv_rows,
)
from src.utils import file_md5  # noqa: E402

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")
IMAGE_ID_PREFIX = "T3"


def list_source_files(source_dir: str) -> list[str]:
    """Tất cả file ảnh trong source_dir (đệ quy)."""
    found = []
    for root, _dirs, files in os.walk(source_dir):
        for fname in sorted(files):
            if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                found.append(os.path.join(root, fname))
    return found


def group_by_class(source_dir: str, classes: list) -> dict[str, list[str]]:
    """Phân loại file theo thư mục con trùng tên lớp."""
    out = {c: [] for c in classes}
    for p in list_source_files(source_dir):
        rel = os.path.relpath(p, source_dir)
        parts = rel.split(os.sep)
        if len(parts) < 2:
            continue
        folder = parts[0]
        if folder in classes:
            out[folder].append(p)
        else:
            raise Task3ManifestError(
                f"File '{rel}' nằm ngoài thư mục lớp đã biết {classes}. "
                "Sắp xếp staging theo: unripe/ medium_ripe/ ripe/.")
    return out


def validate_image(path: str) -> None:
    """Decode được và có kích thước > 0."""
    from PIL import Image
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if img.width <= 0 or img.height <= 0:
                raise ValueError("kích thước <= 0")
    except Exception as exc:
        raise Task3ManifestError(f"Ảnh không hợp lệ {path}: {exc}") from exc


def next_image_id(existing_ids: set[str], captured_at: datetime) -> str:
    """Sinh image_id duy nhất: T3_YYYYMMDD_NNNN."""
    prefix = f"{IMAGE_ID_PREFIX}_{captured_at:%Y%m%d}_"
    n = 1
    while f"{prefix}{n:04d}" in existing_ids:
        n += 1
    return f"{prefix}{n:04d}"


def check_duplicates(files: list[str], existing_md5: set[str]) -> list[str]:
    """Trả về danh sách các file có md5 trùng nhau hoặc trùng raw hiện có."""
    seen: dict[str, str] = {}
    dupes = []
    for p in files:
        h = file_md5(p)
        if h in existing_md5 or h in seen:
            dupes.append(p)
        else:
            seen[h] = p
    return dupes


def import_images(
    source_dir: str,
    classes: list,
    raw_root: str,
    session_id: str,
    camera: str,
    review_status: str,
    capture_log_path: str,
    notes: str = "",
    dry_run: bool = False,
    captured_at: datetime | None = None,
) -> dict:
    """Import toàn bộ batch. Trả về summary.

    Validate TOÀN BỘ trước khi copy (atomic): không để batch lỗi giữa chừng.
    """
    from src.task3 import validate_review_status
    review_status = validate_review_status(review_status)

    captured_at = captured_at or datetime.now()
    files_by_class = group_by_class(source_dir, classes)
    all_files = [p for ps in files_by_class.values() for p in ps]

    if not all_files:
        raise Task3ManifestError("Không tìm thấy ảnh nào trong nguồn.")

    for p in all_files:
        validate_image(p)

    existing = scan_raw_dataset(raw_root, classes)
    existing_md5 = set()
    for ps in existing.files.values():
        for p in ps:
            existing_md5.add(file_md5(p))

    dupes = check_duplicates(all_files, existing_md5)
    if dupes:
        raise Task3ManifestError(
            f"{len(dupes)} ảnh trùng md5 (trong batch hoặc với raw hiện có): "
            + ", ".join(os.path.basename(d) for d in dupes[:10]))

    # Kiểm tra filename trùng raw hiện có
    existing_names = {os.path.basename(p) for ps in existing.files.values() for p in ps}
    batch_names = [os.path.basename(p) for p in all_files]
    collisions = [n for n in batch_names if n in existing_names]
    if collisions:
        raise Task3ManifestError(
            f"Filename đã tồn tại trong raw (không ghi đè): {collisions[:10]}")

    # Đọc capture_log hiện có để lấy image_id đã dùng
    existing_rows = _read_csv(capture_log_path)
    existing_ids = {(r.get("image_id") or "").strip() for r in existing_rows}

    new_rows = []
    seq = 0
    planned = []
    for cls in classes:
        for src_path in files_by_class[cls]:
            seq += 1
            iid = next_image_id(existing_ids, captured_at)
            existing_ids.add(iid)
            fname = os.path.basename(src_path)
            dst = os.path.join(raw_root, cls, fname)
            new_rows.append({
                "image_id": iid,
                "filename": fname,
                "group_id": "",  # gán sau khi chụp/import (xem guide)
                "class_label": cls,
                "split": "",
                "session_id": session_id,
                "captured_at": captured_at.isoformat(timespec="seconds"),
                "camera": camera,
                "lighting": "", "background": "", "wb_setting": "",
                "exposure": "", "gain": "", "distance_cm": "",
                "orientation": "",
                "review_status": review_status,
                "provisional_label": cls,
                "annotator": "",
                "notes": notes,
            })
            planned.append({"src": src_path, "dst": dst, "class": cls, "image_id": iid})

    if dry_run:
        return {
            "dry_run": True,
            "planned": len(planned),
            "rows": new_rows,
            "capture_log_rows": len(existing_rows) + len(new_rows),
        }

    # Copy (sau khi đã validate toàn bộ)
    for item in planned:
        os.makedirs(os.path.dirname(item["dst"]), exist_ok=True)
        shutil.copy2(item["src"], item["dst"])

    # Append capture_log
    merged = existing_rows + new_rows
    write_csv_rows(capture_log_path, CAPTURE_LOG_COLUMNS, merged)

    return {
        "dry_run": False,
        "imported": len(planned),
        "by_class": {c: sum(1 for p in planned if p["class"] == c) for c in classes},
        "capture_log_rows": len(merged),
        "first_image_id": planned[0]["image_id"] if planned else "",
        "last_image_id": planned[-1]["image_id"] if planned else "",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Import ảnh mới vào raw + capture_log")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source", required=True, help="Thư mục staging (chứa unripe/ medium_ripe/ ripe/)")
    parser.add_argument("--session-id", required=True, help="Mã phiên chụp (VD T3_20261108_01)")
    parser.add_argument("--camera", default="OV2640", help="Camera dùng chụp (mặc định OV2640)")
    parser.add_argument("--review-status", default="accepted",
                        choices=["accepted", "pending_review", "excluded"])
    parser.add_argument("--notes", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["project"]["classes"]

    try:
        summary = import_images(
            source_dir=args.source,
            classes=classes,
            raw_root=cfg["paths"]["dataset_raw"],
            session_id=args.session_id,
            camera=args.camera,
            review_status=args.review_status,
            capture_log_path=cfg["paths"]["capture_log"],
            notes=args.notes,
            dry_run=args.dry_run,
        )
    except Task3ManifestError as exc:
        print(f"[LỖI] {exc}")
        return 2

    if summary["dry_run"]:
        print(f"[DRY RUN] Sẽ import {summary['planned']} ảnh; "
              f"capture_log sẽ có {summary['capture_log_rows']} dòng.")
    else:
        print(f"Đã import {summary['imported']} ảnh: " + ", ".join(
            f"{c}={n}" for c, n in summary["by_class"].items()))
        print(f"capture_log: {summary['capture_log_rows']} dòng "
              f"(image_id {summary['first_image_id']} … {summary['last_image_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
