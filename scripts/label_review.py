"""Cập nhật nhãn/trạng thái review cho ảnh (capture_log + label_overrides).

Dùng:
  python scripts/label_review.py --set-status --filename <name> --status pending_review \
      [--reason "..."] [--annotator A] [--dry-run]
  python scripts/label_review.py --override-label --filename <name> --label ripe \
      [--reason "..."] [--annotator A] [--approved-by user] [--dry-run]
  python scripts/label_review.py --show [--filename <name>]

Quy tắc:
- Không tự sửa nhãn: chỉ ghi qua ``label_overrides.csv`` (filename,new_label,reason,
  approved_by,date) — manifest builder sẽ áp dụng override khi build.
- Trạng thái review chỉ chấp nhận: accepted / pending_review / excluded (đã chốt).
- Không xóa/sửa ảnh raw; không đổi nhãn thư mục raw.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.task3 import (  # noqa: E402
    CAPTURE_LOG_COLUMNS,
    REVIEW_STATUSES,
    Task3ManifestError,
    _read_csv,
    validate_review_status,
    write_csv_rows,
)

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

LABEL_OVERRIDES_COLUMNS = ["filename", "new_label", "reason", "approved_by", "date"]


def load_label_overrides(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    return _read_csv(path)


def set_status(capture_log_path: str, filename: str, status: str,
               reason: str, annotator: str, dry_run: bool = False) -> dict:
    status = validate_review_status(status)
    rows = _read_csv(capture_log_path)
    by_name = {r.get("filename", "").strip(): r for r in rows}
    row = by_name.get(filename)
    if row is None:
        raise Task3ManifestError(
            f"Không tìm thấy '{filename}' trong capture_log — import trước đã.")
    old = row.get("review_status", "")
    if old == status:
        return {"changed": False, "filename": filename, "status": status}
    row["review_status"] = status
    if reason:
        row["notes"] = (row.get("notes", "") or "").strip(" ")
        row["notes"] = (row["notes"] + f" [{reason}]").strip()
    if annotator:
        row["annotator"] = annotator
    if not dry_run:
        # Giữ nguyên thứ tự dòng
        write_csv_rows(capture_log_path, CAPTURE_LOG_COLUMNS, rows)
    return {"changed": True, "filename": filename, "status": status,
            "old_status": old}


def override_label(overrides_path: str, filename: str, label: str,
                   reason: str, approved_by: str, dry_run: bool = False) -> dict:
    now = datetime.datetime.now().isoformat(timespec="seconds")
    rows = load_label_overrides(overrides_path)
    existing = [r for r in rows if r.get("filename", "").strip() == filename]
    if existing:
        # Cập nhật override cũ (chỉ 1 override cho phép)
        row = existing[0]
        row["new_label"] = label
        row["reason"] = reason
        row["approved_by"] = approved_by
        row["date"] = now
    else:
        rows.append({
            "filename": filename, "new_label": label,
            "reason": reason, "approved_by": approved_by, "date": now,
        })
    if not dry_run:
        write_csv_rows(overrides_path, LABEL_OVERRIDES_COLUMNS, rows)
    return {"changed": True, "filename": filename, "label": label,
            "approved_by": approved_by, "date": now}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cập nhật nhãn/status review")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--set-status", choices=list(REVIEW_STATUSES))
    parser.add_argument("--override-label")
    parser.add_argument("--filename", required=True)
    parser.add_argument("--reason", default="")
    parser.add_argument("--annotator", default="")
    parser.add_argument("--approved-by", default="owner")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cap_path = cfg["paths"]["capture_log"]
    ov_path = cfg["paths"]["label_overrides"]

    if args.show:
        rows = _read_csv(cap_path)
        for r in rows:
            if args.filename and r.get("filename", "").strip() != args.filename:
                continue
            print(r)
        if args.filename:
            for o in load_label_overrides(ov_path):
                if o.get("filename", "").strip() == args.filename:
                    print("override:", o)
        return 0

    try:
        if args.set_status:
            res = set_status(cap_path, args.filename, args.set_status,
                             args.reason, args.annotator, args.dry_run)
            print(f"[{'DRY RUN' if args.dry_run else 'OK'}] status → {res['status']} "
                  f"({res['filename']})")
        elif args.override_label:
            res = override_label(ov_path, args.filename, args.override_label,
                                 args.reason, args.approved_by, args.dry_run)
            print(f"[{'DRY RUN' if args.dry_run else 'OK'}] label override → {res['label']} "
                  f"({res['filename']}, approved={res['approved_by']})")
        else:
            parser.error("cần --set-status hoặc --override-label hoặc --show")
    except Task3ManifestError as exc:
        print(f"[LỖI] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
