"""Tiện ích Task 3: exclusion list, capture log, group split và manifest.

Mục tiêu:
- Không bao giờ đưa ảnh bị loại (``review_exclusions.csv``) vào train/val/test.
- Mọi ảnh mới phải có ``group_id`` và group đó phải được gán đúng một split.
- Ảnh cũ (473 ảnh, không có mapping quả) chỉ được nằm ở ``train``.
- Manifest cuối phải không trùng file, không rò rỉ group, và có đủ 3 lớp.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import math
import os

SPLIT_NAMES = ("train", "validation", "test")

# Từ vựng trạng thái review (đã chốt Task 3 §1.A) — chỉ chấp nhận đúng 3 giá trị.
REVIEW_STATUS_ACCEPTED = "accepted"
REVIEW_STATUS_PENDING_REVIEW = "pending_review"
REVIEW_STATUS_EXCLUDED = "excluded"
REVIEW_STATUSES = (REVIEW_STATUS_ACCEPTED, REVIEW_STATUS_PENDING_REVIEW, REVIEW_STATUS_EXCLUDED)

# Trạng thái được đưa vào manifest (không bị loại). Chỉ ``accepted`` (và rỗng =
# không có dòng review) được giữ; ``pending_review`` và ``excluded`` đều bị loại.
KEEP_STATUSES = {REVIEW_STATUS_ACCEPTED, ""}


def validate_review_status(status) -> str:
    """Kiểm tra trạng thái review; trả về chuẩn hóa lowercase.

    Raise ``Task3ManifestError`` nếu trống hoặc không nằm trong từ vựng — không
    âm thầm chấp nhận giá trị lạ.
    """
    s = str(status or "").strip().lower()
    if s not in REVIEW_STATUSES:
        raise Task3ManifestError(
            f"Trạng thái review không hợp lệ: '{status}'. "
            f"Chỉ chấp nhận {REVIEW_STATUSES}.")
    return s


class Task3ManifestError(RuntimeError):
    """Manifest Task 3 không hợp lệ — dừng an toàn, không train."""


def _read_csv(path: str) -> list[dict]:
    """Đọc CSV, bỏ dòng trống và dòng bắt đầu bằng '#' (cho phép chú thích)."""
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def load_exclusions(path: str) -> dict[str, dict]:
    """Đọc ``review_exclusions.csv`` → {filename: row}.

    Validate trạng thái review theo từ vựng; status lạ → raise.
    """
    rows = _read_csv(path)
    out: dict[str, dict] = {}
    for row in rows:
        name = (row.get("filename") or "").strip()
        if not name:
            continue
        # Validate status ngay khi đọc; rỗng → lỗi (mọi dòng review phải ghi rõ).
        validate_review_status(row.get("status"))
        out[name] = row
    return out


def is_excluded_row(row: dict) -> bool:
    """True nếu row đang bị loại (status không nằm trong KEEP_STATUSES)."""
    status = validate_review_status(row.get("status"))
    return status not in KEEP_STATUSES


def excluded_filenames(path: str) -> set[str]:
    """Tập filename đang bị loại khỏi train/validation/test."""
    return {name for name, row in load_exclusions(path).items() if is_excluded_row(row)}


def load_capture_log(path: str) -> dict[str, dict]:
    """Đọc ``capture_log.csv`` → {filename: row}.

    Nếu dòng có ``review_status`` không rỗng → validate theo từ vựng.
    """
    rows = _read_csv(path)
    out: dict[str, dict] = {}
    for row in rows:
        name = (row.get("filename") or "").strip()
        if not name:
            continue
        status = (row.get("review_status") or "").strip()
        if status:
            validate_review_status(status)
        out[name] = row
    return out


def load_group_splits(path: str) -> dict[str, str]:
    """Đọc ``group_splits.csv`` (group_id,split) → {group_id: split}.

    Raise nếu split không hợp lệ hoặc cùng group bị gán hai split khác nhau.
    """
    rows = _read_csv(path)
    out: dict[str, str] = {}
    for row in rows:
        gid = (row.get("group_id") or "").strip()
        split = (row.get("split") or "").strip()
        if not gid:
            continue
        if split not in SPLIT_NAMES:
            raise Task3ManifestError(
                f"group '{gid}' có split không hợp lệ: '{split}'. "
                f"Chỉ chấp nhận {SPLIT_NAMES}.")
        if gid in out and out[gid] != split:
            raise Task3ManifestError(
                f"group '{gid}' bị gán hai split khác nhau: '{out[gid]}' và '{split}'.")
        out[gid] = split
    return out


CAPTURE_LOG_COLUMNS = [
    "image_id", "filename", "group_id", "class_label", "split",
    "session_id", "captured_at", "camera",
    "lighting", "background", "wb_setting", "exposure", "gain",
    "distance_cm", "orientation",
    "review_status", "provisional_label", "annotator", "notes",
]


def write_csv_rows(path: str, columns: list, rows: list[dict]) -> None:
    """Ghi CSV (tạo thư mục nếu cần). Toàn bộ dòng phải có đủ key."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def load_derived_metadata(path: str) -> list[dict]:
    """Đọc ``dataset/derived/<crop_version>/metadata.csv``."""
    return _read_csv(path)


def assign_group_splits(
    groups_by_class: dict[str, list[str]],
    seed: int = 42,
    validation_ratio: float = 0.5,
) -> tuple[dict[str, str], dict]:
    """Chia group mới theo quả (group-aware, stratified theo lớp) → 50/50 val/test.

    - ``groups_by_class``: {class: [group_id, ...]} — mỗi group chỉ xuất hiện ở
      đúng một lớp.
    - Mỗi lớp: shuffle ổn định theo ``seed``, chọn ceil(n*ratio) group vào
      validation, phần còn lại vào test. Mọi ảnh của một group luôn cùng split.
    - Trả về ``({group_id: split}, stats)``; ``stats`` gồm số lượng group/ảnh
      không có (chỉ group) theo lớp và các cảnh báo chia nhỏ.

    Không có group nào bị chia; không có ảnh nào được chia riêng.
    """
    import random

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    stats: dict[str, dict] = {"by_class": {}, "warnings": []}

    for cls, groups in groups_by_class.items():
        groups = list(dict.fromkeys(groups))  # bỏ trùng, giữ thứ tự
        rng.shuffle(groups)
        n = len(groups)
        val_n = math.ceil(n * float(validation_ratio)) if n else 0
        for i, gid in enumerate(groups):
            assignment[gid] = "validation" if i < val_n else "test"
        stats["by_class"][cls] = {"groups": n, "validation": val_n, "test": n - val_n}
        if n > 0 and (val_n < 10 or (n - val_n) < 10):
            stats["warnings"].append(
                f"{cls}: {n} group → validation {val_n} / test {n - val_n} "
                f"(<10 group ở một split — evaluation không robust về mặt thống kê)")
        if n == 0:
            stats["warnings"].append(f"{cls}: không có group mới")

    return assignment, stats


def _normalise_label(value, classes: list) -> int:
    text = str(value or "").strip()
    if text in classes:
        return classes.index(text)
    try:
        idx = int(text)
    except (TypeError, ValueError):
        raise Task3ManifestError(f"Nhãn không hợp lệ: '{value}'. Classes: {classes}") from None
    if idx < 0 or idx >= len(classes):
        raise Task3ManifestError(f"Nhãn index ngoài phạm vi: {idx} (classes={classes})")
    return idx


def verify_task3_manifest(
    manifest: dict,
    classes: list,
    require_all_splits: bool = False,
    exclusions: set[str] | None = None,
) -> dict:
    """Xác minh manifest Task 3: trùng file, rò rỉ group, thiếu lớp, exclusion.

    ``require_all_splits=False`` cho phép validation/test rỗng (manifest tạm thời
    ở Phase 1). Train luôn phải có đủ 3 lớp.
    """
    exclusions = exclusions or set()
    counts = {s: {c: 0 for c in classes} for s in SPLIT_NAMES}
    unique_groups = {s: {c: set() for c in classes} for s in SPLIT_NAMES}
    split_of_path: dict[str, set] = {}
    split_of_group: dict[str, set] = {}
    excluded_present: list[str] = []

    for split in SPLIT_NAMES:
        for item in manifest.get(split, []):
            path = item["path"]
            label = int(item["label"])
            cls = classes[label]
            counts[split][cls] += 1
            split_of_path.setdefault(path, set()).add(split)
            gid = (item.get("group_id") or "").strip()
            if gid:
                split_of_group.setdefault(gid, set()).add(split)
                unique_groups[split][cls].add(gid)
            source = os.path.basename(item.get("source_path") or "")
            if source and source in exclusions:
                excluded_present.append(source)

    overlap = {p: sorted(s) for p, s in split_of_path.items() if len(s) > 1}
    group_leaks = {g: sorted(s) for g, s in split_of_group.items() if len(s) > 1}
    missing = [
        f"{cls}/{split}"
        for split in SPLIT_NAMES
        for cls in classes
        if counts[split][cls] == 0
    ]
    train_missing = [f"{cls}/train" for cls in classes if counts["train"][cls] == 0]

    ok = not overlap and not group_leaks and not train_missing and not excluded_present
    if require_all_splits and missing:
        ok = False

    return {
        "ok": ok,
        "counts": counts,
        "totals": {s: sum(counts[s].values()) for s in SPLIT_NAMES},
        "unique_groups": {
            s: {c: len(unique_groups[s][c]) for c in classes} for s in SPLIT_NAMES
        },
        "overlap_paths": overlap,
        "group_leaks": group_leaks,
        "missing_class_split": missing,
        "excluded_present": excluded_present,
        "provisional": not (manifest.get("validation") and manifest.get("test")),
    }


def build_task3_manifest(
    derived_rows: list[dict],
    classes: list,
    exclusions: set[str] | None = None,
    group_splits: dict[str, str] | None = None,
    allow_needs_review: bool = False,
    check_files: bool = False,
    require_all_splits: bool = False,
    review_statuses: dict[str, str] | None = None,
    label_overrides: dict[str, str] | None = None,
) -> dict:
    """Ghép metadata derivative thành manifest Task 3.

    Quy tắc:
    - Row có source filename trong ``exclusions`` → bỏ.
    - ``crop_status != ok`` → bỏ (trừ khi ``allow_needs_review=True``).
    - Nếu ``review_statuses`` được cung cấp (filename → review_status): status
      khác ``accepted`` → bỏ (pending_review/excluded KHÔNG vào manifest).
    - ``label_overrides`` (filename → new_label) được áp dụng trước khi
      chuẩn hóa nhãn — không bao giờ tự sửa nhãn gốc.
    - Ảnh có ``group_id``: group phải có trong ``group_splits`` và split trong
      metadata phải khớp (hoặc để trống để lấy từ ``group_splits``).
    - Ảnh không có ``group_id`` (ảnh cũ) → chỉ được ở ``train``.
    """
    exclusions = exclusions or set()
    group_splits = group_splits or {}
    review_statuses = review_statuses or {}
    label_overrides = label_overrides or {}

    manifest: dict[str, list] = {s: [] for s in SPLIT_NAMES}
    skipped = {"excluded": [], "crop_status": [], "missing_output": [],
               "review_status": []}

    for row in derived_rows:
        output_path = (row.get("output_path") or "").strip()
        source_path = (row.get("source_path") or "").strip()
        source_name = os.path.basename(source_path) if source_path else os.path.basename(output_path)

        if source_name in exclusions:
            skipped["excluded"].append(source_name)
            continue

        if source_name in review_statuses:
            rstatus = review_statuses[source_name]
            if rstatus != REVIEW_STATUS_ACCEPTED:
                skipped["review_status"].append((source_name, rstatus))
                continue

        crop_status = (row.get("crop_status") or "ok").strip() or "ok"
        if crop_status != "ok" and not allow_needs_review:
            skipped["crop_status"].append((source_name, crop_status))
            continue
        if not output_path:
            skipped["missing_output"].append(source_name)
            continue

        if check_files and not os.path.isfile(output_path):
            raise Task3ManifestError(f"Thiếu file derivative: {output_path}")

        raw_label = row.get("label") or row.get("class")
        override = label_overrides.get(source_name)
        if override is not None:
            raw_label = override
        label = _normalise_label(raw_label, classes)
        group_id = (row.get("group_id") or "").strip()
        image_id = (row.get("image_id") or "").strip()
        split = (row.get("split") or "").strip()

        if group_id:
            if group_id not in group_splits:
                raise Task3ManifestError(
                    f"Group '{group_id}' ({source_name}) chưa được gán split trong "
                    "group_splits.csv — không thể đưa vào manifest.")
            expected = group_splits[group_id]
            if split and split != expected:
                raise Task3ManifestError(
                    f"Group '{group_id}' ({source_name}) có split '{split}' trong metadata "
                    f"nhưng group_splits.csv gán '{expected}'.")
            split = expected
        else:
            if split and split != "train":
                raise Task3ManifestError(
                    f"Ảnh cũ không có group_id ({source_name}) phải ở 'train', "
                    f"nhưng metadata ghi '{split}'.")
            split = "train"

        if split not in SPLIT_NAMES:
            raise Task3ManifestError(f"Split không hợp lệ cho {source_name}: '{split}'")

        manifest[split].append({
            "path": os.path.abspath(output_path),
            "label": label,
            "class": classes[label],
            "group_id": group_id,
            "image_id": image_id,
            "source_path": os.path.abspath(source_path) if source_path else "",
            "crop_version": (row.get("crop_version") or "").strip(),
        })

    verification = verify_task3_manifest(
        manifest, classes, require_all_splits=require_all_splits, exclusions=exclusions)

    if verification["overlap_paths"]:
        raise Task3ManifestError(
            f"{len(verification['overlap_paths'])} file xuất hiện ở nhiều split.")
    if verification["group_leaks"]:
        raise Task3ManifestError(
            f"{len(verification['group_leaks'])} group rò rỉ qua nhiều split.")
    if verification["excluded_present"]:
        raise Task3ManifestError(
            f"{len(verification['excluded_present'])} ảnh bị loại vẫn có trong manifest.")
    if verification["counts"]["train"] and any(
            verification["counts"]["train"][c] == 0 for c in classes):
        raise Task3ManifestError(
            "Train thiếu lớp: "
            + ", ".join(f"{c}/train" for c in classes if verification["counts"]["train"][c] == 0))
    if require_all_splits and verification["missing_class_split"]:
        raise Task3ManifestError(
            "Thiếu lớp/tập: " + ", ".join(verification["missing_class_split"]))

    return {"manifest": manifest, "verification": verification, "skipped": skipped}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha256(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_task3_manifest(
    manifest: dict,
    manifest_path: str,
    meta_path: str,
    crop_version: str,
    exclusions_path: str = "",
    extra: dict | None = None,
) -> dict:
    """Ghi manifest + metadata; trả về metadata (gồm hash test/manifest)."""
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    meta = {
        "crop_version": crop_version,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "manifest_path": os.path.abspath(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "exclusions_file": os.path.abspath(exclusions_path) if exclusions_path else "",
        "exclusions_sha256": sha256_file(exclusions_path) if exclusions_path and os.path.isfile(exclusions_path) else "",
        "provisional": not (manifest.get("validation") and manifest.get("test")),
        "test_manifest_sha256": _canonical_sha256(manifest.get("test", [])),
        "counts": {s: len(manifest.get(s, [])) for s in SPLIT_NAMES},
    }
    if extra:
        meta.update(extra)

    os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta
