"""Gán split 50/50 (theo group, stratified theo lớp) cho quả mới.

Dùng:
  python scripts/split_new_groups.py [--config config.yaml] [--seed 42] [--dry-run]

Nguồn group:
- ``capture_log.csv``: dòng có ``group_id`` (rỗng → group không hợp lệ).
- Chỉ các group CHƯA có trong ``group_splits.csv`` được gán (không ghi đè).
- Nhãn lớp lấy từ ``class_label`` (provisional).

Quy tắc:
- Không bao giờ gán cùng một group cho 2 split (đảm bảo bởi assign_group_splits).
- Không thêm group mới vào train (473 ảnh cũ vẫn là nguồn train duy nhất).
- Nếu <10 group/lớp ở một split → cảnh báo nổi bật; không claim robustness.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.task3 import (  # noqa: E402
    SPLIT_NAMES,
    Task3ManifestError,
    _read_csv,
    assign_group_splits,
    load_group_splits,
    write_csv_rows,
)

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def collect_unassigned_groups(capture_log_path: str, existing: dict[str, str],
                              classes: list) -> dict[str, list[str]]:
    """{class: [group_id, ...]} chỉ gồm group chưa có trong group_splits."""
    rows = _read_csv(capture_log_path)
    groups_by_class: dict[str, list[str]] = {c: [] for c in classes}
    for row in rows:
        gid = (row.get("group_id") or "").strip()
        cls = (row.get("class_label") or "").strip()
        if not gid:
            continue
        if cls not in classes:
            raise Task3ManifestError(
                f"class_label không hợp lệ trong capture_log: '{cls}' "
                f"(group {gid}). Classes: {classes}")
        if gid in existing:
            continue  # đã gán — không đụng
        groups_by_class[cls].append(gid)
    return groups_by_class


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Gán split 50/50 theo group cho quả mới")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = cfg["paths"]
    classes = cfg["project"]["classes"]

    existing = load_group_splits(paths["group_splits"])
    pending = collect_unassigned_groups(paths["capture_log"], existing, classes)
    total = sum(len(v) for v in pending.values())

    print(f"Group đã gán: {len(existing)}; group mới chờ gán: {total}")
    if total == 0:
        print("[OK] Không có group mới cần gán.")
        return 0

    assignment, stats = assign_group_splits(pending, seed=args.seed)
    for cls, info in stats["by_class"].items():
        print(f"  {cls}: {info['groups']} group → validation {info['validation']} / "
              f"test {info['test']}")
    for w in stats["warnings"]:
        print(f"[CẢNH BÁO] {w}")

    if args.dry_run:
        print(f"[DRY RUN] Sẽ gán {len(assignment)} group "
              f"(validation {sum(1 for s in assignment.values() if s == 'validation')}, "
              f"test {sum(1 for s in assignment.values() if s == 'test')})")
        return 0

    rows = [{"group_id": gid, "split": split} for gid, split in existing.items()]
    rows += [{"group_id": gid, "split": split} for gid, split in assignment.items()]
    write_csv_rows(paths["group_splits"], ["group_id", "split"], rows)
    print(f"Đã ghi {len(rows)} dòng vào {paths['group_splits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
