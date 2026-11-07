"""Dataset loader: quét ảnh, kiểm tra lỗi/trùng, chia train/val/test stratified.

- Hỗ trợ JPG/JPEG/PNG, kiểm tra ảnh lỗi/trùng và đếm theo lớp.
- Tỉ lệ 70/15/15, stratified theo lớp, seed cố định; hỗ trợ group ID
  để mọi ảnh của cùng một quả ớt luôn nằm trong một tập (tránh rò rỉ dữ liệu).
- Không xóa ảnh gốc; không chia lại khi manifest đã tồn tại (trừ khi force=True).
- Nếu dataset trống: thông báo rõ ràng và dừng an toàn (raise DatasetEmptyError).
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass, field

import numpy as np

from .utils import ensure_dir

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")
SPLIT_NAMES = ("train", "validation", "test")

# Tên thư mục raw thay thế được chấp nhận cho một lớp chuẩn.
# Yêu cầu dự án: thư mục `medium-ripe` phải ánh xạ sang nhãn chuẩn `medium_ripe`
# mà KHÔNG đổi tên thư mục raw. Vì vậy ta chấp nhận cả hai cách viết.
CLASS_DIR_ALIASES = {
    "medium_ripe": ("medium_ripe", "medium-ripe"),
}


class DatasetEmptyError(RuntimeError):
    """Dataset trống / thiếu lớp — dừng pipeline một cách an toàn."""


@dataclass
class DatasetSummary:
    counts: dict
    files: dict
    total: int
    duplicates: dict = field(default_factory=dict)
    corrupt: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "counts": self.counts,
            "files": {k: list(v) for k, v in self.files.items()},
            "duplicates": self.duplicates,
            "corrupt": self.corrupt,
        }


def _is_supported_image(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_EXTENSIONS)


def resolve_class_dir(raw_root: str, cls: str) -> str | None:
    """Trả về thư mục raw thực tế cho lớp `cls`, hỗ trợ tên thay thế.

    Ví dụ `medium_ripe` khớp cả `medium_ripe/` và `medium-ripe/`.
    Trả về None nếu không có thư mục nào tồn tại.
    """
    candidates = CLASS_DIR_ALIASES.get(cls, (cls,))
    for name in candidates:
        candidate = os.path.join(raw_root, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def _validate_image(path: str) -> bool:
    """Kiểm tra ảnh có mở được bằng Pillow và có kích thước > 0 không."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.width > 0 and img.height > 0
    except Exception:
        return False


def _content_hash(path: str) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def scan_raw_dataset(raw_root: str, classes: list) -> DatasetSummary:
    """Quét dataset/raw/<class>/, kiểm tra ảnh lỗi + ảnh trùng, đếm theo lớp."""
    files: dict[str, list] = {}
    corrupt: list[str] = []
    hash_to_files: dict[str, list] = {}

    for cls in classes:
        class_dir = resolve_class_dir(raw_root, cls)
        collected = []
        if class_dir is not None:
            for fname in sorted(os.listdir(class_dir)):
                fpath = os.path.join(class_dir, fname)
                if not os.path.isfile(fpath) or not _is_supported_image(fpath):
                    continue
                if not _validate_image(fpath):
                    corrupt.append(fpath)
                    continue
                collected.append(fpath)
                h = _content_hash(fpath)
                if h:
                    hash_to_files.setdefault(h, []).append(fpath)
        files[cls] = collected

    duplicates = {h: ps for h, ps in hash_to_files.items() if len(ps) > 1}
    counts = {c: len(files.get(c, [])) for c in classes}
    return DatasetSummary(
        counts=counts, files=files, total=sum(counts.values()),
        duplicates=duplicates, corrupt=corrupt,
    )


def generate_groups_template(raw_root: str, classes: list, out_path: str | None = None) -> str:
    """Sinh template dataset/groups.csv từ các ảnh hiện có trong dataset/raw/.

    - Liệt kê mọi ảnh hợp lệ (filename) với cột group_id để trống.
    - Người dùng điền group_id cho từng quả ớt (mọi ảnh cùng quả cùng mã).
    - Nếu out_path=None → không ghi file, chỉ trả về nội dung CSV.
    - Không tạo ảnh; chỉ đọc cấu trúc hiện có.
    """
    summary = scan_raw_dataset(raw_root, classes)
    lines = ["filename,group_id"]
    for cls in classes:
        for p in summary.files.get(cls, []):
            lines.append(f"{os.path.basename(p)},")
    content = "\n".join(lines) + "\n"
    if out_path:
        ensure_dir(os.path.dirname(os.path.abspath(out_path)))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        return out_path
    return content


def load_groups_file(groups_path: str) -> dict:
    """Đọc CSV filename,group_id → {filename: group_id}. Trả về {} nếu không có file.

    Bỏ qua dòng trống và dòng bắt đầu bằng '#' (file template có thể chứa chú thích).
    """
    groups: dict[str, str] = {}
    if not groups_path or not os.path.isfile(groups_path):
        return groups
    with open(groups_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return groups
    reader = csv.DictReader(lines)
    if not reader.fieldnames or "filename" not in reader.fieldnames:
        return groups
    for row in reader:
        name = (row.get("filename") or "").strip()
        gid = (row.get("group_id") or "").strip()
        if name and gid:
            groups[name] = gid
    return groups


def _largest_remainder(ratios: tuple[float, float, float], total: int) -> list[int]:
    """Phân bổ số ảnh theo tỉ lệ bằng largest-remainder (không dùng round()).

    Đảm bảo tổng target = total và mỗi target >= 0.
    """
    raw = [r * total for r in ratios]
    floor = [int(x) for x in raw]
    remainder = [x - int(x) for x in raw]
    remaining = total - sum(floor)
    order = sorted(range(3), key=lambda i: remainder[i], reverse=True)
    for k in range(remaining):
        floor[order[k % 3]] += 1
    return floor


def stratified_group_split(
    class_to_files: dict,
    classes: list,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    groups: dict | None = None,
) -> dict:
    """Chia 70/15/15 stratified theo lớp với seed cố định.

    - Nếu có group ID: mọi ảnh cùng group luôn nằm trong cùng một tập,
      chia theo đơn vị group (chia gần đúng tỉ lệ theo số ảnh).
    - Cố gắng đảm bảo mỗi tập (train/validation/test) không rỗng cho từng lớp
      khi có đủ ảnh/group (>= 3). Nếu không đủ → cảnh báo rõ ràng.
    - Trả về {split_name: [(filepath, class_label), ...]}.
    """
    import warnings

    groups = groups or {}
    rng = random.Random(seed)
    splits: dict[str, list] = {name: [] for name in SPLIT_NAMES}
    emitted_warning = False

    for label, cls in enumerate(classes):
        file_paths = sorted(class_to_files.get(cls, []))
        if not file_paths:
            continue
        # Gom các file theo group (file không có group = group riêng của chính nó)
        by_group: dict[str, list] = {}
        for p in file_paths:
            gid = groups.get(os.path.basename(p), f"__single_{os.path.basename(p)}")
            by_group.setdefault(gid, []).append(p)
        group_items = list(by_group.items())
        rng.shuffle(group_items)

        total = len(file_paths)
        num_groups = len(group_items)
        targets = _largest_remainder(ratios, total)

        bucket = [[], [], []]
        assigned = [0, 0, 0]

        # Giai đoạn 0: đảm bảo mỗi split có ít nhất 1 group (nếu có >= 3 group)
        guarantee = min(num_groups, 3)
        for i in range(guarantee):
            _gid, gfiles = group_items[i]
            bucket[i].extend((p, label) for p in gfiles)
            assigned[i] += len(gfiles)

        # Giai đoạn 1: phân bổ các group còn lại vào split đang thiếu hụt nhất
        for _gid, gfiles in group_items[guarantee:]:
            deficits = [targets[i] - assigned[i] for i in range(3)]
            best = int(np.argmax(deficits))
            bucket[best].extend((p, label) for p in gfiles)
            assigned[best] += len(gfiles)

        # Cảnh báo khi không thể chia lớp này thành 3 tập không rỗng
        if (total < 3 or num_groups < 3) and not emitted_warning:
            warnings.warn(
                f"Lớp '{cls}': có {total} ảnh / {num_groups} group — không đủ để "
                "chia thành đủ 3 tập (train/validation/test) không rỗng cho lớp này. "
                "Kết quả có thể có validation/test rỗng; nên bổ sung thêm ảnh thật.",
                stacklevel=2,
            )
            emitted_warning = True

        for i in range(3):
            splits[SPLIT_NAMES[i]].extend(bucket[i])

    # Xáo trộn cuối của mỗi tập cho nhất quán
    for name in SPLIT_NAMES:
        rng.shuffle(splits[name])
    return splits


def write_scan_report(summary: DatasetSummary, report_path: str, classes: list | None = None) -> str:
    """Ghi báo cáo quét dataset ra JSON (thống kê, lỗi, trùng, danh sách file).

    Dùng để theo dõi trạng thái dataset trước và sau khi thu thập ảnh.
    """
    classes = classes or list(summary.counts)
    ensure_dir(os.path.dirname(os.path.abspath(report_path)))
    report = summary.to_dict()
    report["classes"] = classes
    report["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report_path


def save_split_manifest(splits: dict, manifest_path: str, classes: list) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    manifest = {
        split: [{"path": p, "label": int(lbl), "class": classes[lbl]} for p, lbl in items]
        for split, items in splits.items()
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path


def load_manifest(manifest_path: str) -> dict | None:
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_split(splits: dict, classes: list, groups: dict | None = None) -> dict:
    """Xác minh một kết quả chia tập: trùng lặp, group leakage, tỷ lệ theo lớp.

    Trả về dict tổng hợp (không raise) để script in báo cáo trung thực.
    """
    groups = groups or {}
    split_of: dict[str, set] = {}
    per_split_counts: dict[str, dict] = {}
    for name in SPLIT_NAMES:
        items = splits.get(name, [])
        counts = {c: 0 for c in classes}
        for path, label in items:
            split_of.setdefault(path, set()).add(name)
            cls = classes[label] if isinstance(label, int) and label < len(classes) else str(label)
            counts[cls] = counts.get(cls, 0) + 1
        per_split_counts[name] = counts

    overlap = {p: sorted(s) for p, s in split_of.items() if len(s) > 1}

    group_splits: dict[str, set] = {}
    for name in SPLIT_NAMES:
        for path, _label in splits.get(name, []):
            gid = groups.get(os.path.basename(path))
            if gid:
                group_splits.setdefault(gid, set()).add(name)
    group_leaks = {g: sorted(s) for g, s in group_splits.items() if len(s) > 1}

    total = sum(len(splits.get(n, [])) for n in SPLIT_NAMES)
    totals = {n: len(splits.get(n, [])) for n in SPLIT_NAMES}
    ratios = {n: (round(totals[n] / total, 4) if total else 0.0) for n in SPLIT_NAMES}

    missing = [
        f"{cls}/{name}"
        for name in SPLIT_NAMES
        for cls in classes
        if per_split_counts[name].get(cls, 0) == 0
    ]

    return {
        "total": total,
        "totals": totals,
        "ratios": ratios,
        "per_split_counts": per_split_counts,
        "overlap_paths": overlap,
        "group_leaks": group_leaks,
        "empty_class_split": missing,
        "groups_used": len(group_splits),
    }


def _parse_img_timestamp(filename: str):
    """Đọc timestamp từ tên IMG_YYYYMMDD_HHMMSS.jpg → datetime | None."""
    import re

    m = re.match(r"IMG_(\d{8})_(\d{6})", os.path.basename(filename))
    if not m:
        return None
    try:
        return datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def build_burst_groups(class_to_files: dict, max_gap_s: float = 10.0,
                       prefix: str = "burst") -> dict:
    """Sinh group ID HEURISTIC theo cụm timestamp (chụp burst liên tiếp).

    CẢNH BÁO: đây chỉ là giả định "các ảnh cách nhau <= max_gap_s có thể cùng
    một quả ớt", KHÔNG phải xác minh. Dùng để thử nghiệm split an toàn hơn,
    không thay thế việc gán group_id thủ công.

    Trả về {filename: group_id}.
    """
    groups: dict[str, str] = {}
    for cls, paths in class_to_files.items():
        stamped = []
        for p in paths:
            ts = _parse_img_timestamp(p)
            if ts is not None:
                stamped.append((ts, p))
        stamped.sort(key=lambda x: x[0])
        idx = 0
        prev = None
        for ts, p in stamped:
            if prev is None or (ts - prev).total_seconds() > max_gap_s:
                idx += 1
            groups[os.path.basename(p)] = f"{prefix}_{cls}_{idx:03d}"
            prev = ts
        for p in paths:
            name = os.path.basename(p)
            if name not in groups:
                groups[name] = f"{prefix}_{cls}_na_{name}"
    return groups


def find_burst_leaks(splits: dict, max_gap_s: float = 5.0) -> list:
    """Ước lượng rò rỉ do chụp burst: cặp ảnh chụp cách nhau <= max_gap_s
    nhưng nằm ở hai tập khác nhau (khả năng cao cùng một quả ớt).

    Đây chỉ là ƯỚC LƯỢNG theo timestamp, không khẳng định cùng quả.
    Trả về list dict {a, b, gap_s, split_a, split_b}.
    """
    stamped = []
    for name in SPLIT_NAMES:
        for path, _label in splits.get(name, []):
            ts = _parse_img_timestamp(path)
            if ts is not None:
                stamped.append((ts, path, name))
    stamped.sort(key=lambda x: x[0])

    leaks = []
    for i in range(1, len(stamped)):
        prev_ts, prev_path, prev_split = stamped[i - 1]
        cur_ts, cur_path, cur_split = stamped[i]
        gap = (cur_ts - prev_ts).total_seconds()
        if gap <= max_gap_s and prev_split != cur_split:
            leaks.append({
                "a": prev_path,
                "b": cur_path,
                "gap_s": gap,
                "split_a": prev_split,
                "split_b": cur_split,
            })
    return leaks


def materialize_split(splits: dict, split_dirs: dict, copy: bool = True) -> None:
    """Copy ảnh vào train/validation/test (KHÔNG xóa ảnh gốc)."""
    for split, items in splits.items():
        dest = split_dirs.get(split)
        if not dest:
            continue
        os.makedirs(dest, exist_ok=True)
        for fpath, _label in items:
            target = os.path.join(dest, os.path.basename(fpath))
            if os.path.abspath(fpath) == os.path.abspath(target):
                continue
            if copy:
                shutil.copy2(fpath, target)
            else:  # move=False mặc định; nhánh này không dùng để xóa gốc
                raise ValueError("move không được hỗ trợ để bảo vệ ảnh gốc")


def prepare_datasets(
    cfg: dict,
    force_resplit: bool = False,
    copy_to_split_dirs: bool = True,
    classes: list | None = None,
    seed: int | None = None,
) -> dict:
    """Pipeline điểm vào: quét → kiểm tra → chia → lưu manifest → copy.

    - Không chia lại nếu manifest đã tồn tại (trừ khi force_resplit=True).
    - Dataset trống/thiếu lớp → raise DatasetEmptyError (dừng an toàn).
    """
    from .config import get_split_ratios

    paths = cfg["paths"]
    classes = classes or cfg["project"]["classes"]
    seed = seed if seed is not None else int(cfg.get("training", {}).get("seed", 42))

    summary = scan_raw_dataset(paths["dataset_raw"], classes)
    empty_classes = [c for c in classes if summary.counts[c] == 0]
    if summary.total == 0:
        raise DatasetEmptyError(
            f"Dataset trống: không tìm thấy ảnh {SUPPORTED_EXTENSIONS} "
            f"trong {paths['dataset_raw']}/{{unripe,medium_ripe,ripe}}. "
            "Hãy đặt ảnh vào dataset/raw/<class>/ rồi chạy lại."
        )
    if empty_classes:
        raise DatasetEmptyError(
            f"Thiếu ảnh ở lớp: {', '.join(empty_classes)}. "
            "Mỗi lớp cần ít nhất 1 ảnh hợp lệ trong dataset/raw/<class>/."
        )

    manifest_path = paths["split_manifest"]
    existing = None if force_resplit else load_manifest(manifest_path)
    if existing is not None:
        print(f"Đã có manifest {manifest_path} — bỏ qua chia lại (dùng force=True để chia lại).")
        splits = {
            split: [(item["path"], item["label"]) for item in existing[split]]
            for split in SPLIT_NAMES
        }
    else:
        groups = load_groups_file(cfg["paths"].get("groups_file", ""))
        splits = stratified_group_split(
            summary.files, classes,
            ratios=get_split_ratios(cfg), seed=seed, groups=groups,
        )
        save_split_manifest(splits, manifest_path, classes)
        print(f"Đã lưu manifest chia tập: {manifest_path}")

    if copy_to_split_dirs:
        materialize_split(
            splits,
            {
                "train": paths["dataset_train"],
                "validation": paths["dataset_validation"],
                "test": paths["dataset_test"],
            },
            copy=True,
        )

    result = summary.to_dict()
    result["splits"] = {k: len(v) for k, v in splits.items()}
    result["manifest"] = manifest_path
    return result


def _decode_image_safe(fpath: str, image_size: int):
    """Đọc ảnh → tensor uint8 [0,255] kích thước (image_size, image_size, 3).

    Dùng expand_animations=False + set_shape để giữ static shape trong graph mode
    (tránh lỗi "images contains no shape" khi `tf.image.resize` chạy trong
    tf.data / AutoGraph với `tf.image.decode_image`).
    """
    import tensorflow as tf

    img = tf.io.read_file(fpath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_size, image_size])
    return img


def _make_empty_ds(cfg: dict, split: str):
    """Tạo một tf.data.Dataset rỗng (0 phần tử) cho split rỗng, có cấu trúc khớp."""
    import tensorflow as tf

    image_size = int(cfg["data"]["image_size"])
    batch = int(cfg["data"]["batch_size"])
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.int64))
    )
    ds = ds.map(
        lambda fpath, label: (
            tf.zeros([image_size, image_size, 3], dtype=tf.uint8),
            tf.cast(label, tf.int64),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def build_tf_datasets(cfg: dict, manifest_path: str | None = None, augment: bool = True):
    """Tạo tf.data.Dataset cho train/val/test từ manifest (lazy import TF).

    - Với split rỗng trong manifest: trả về dataset rỗng có cấu trúc khớp
      (KHÔNG crash), kèm cảnh báo rõ ràng.
    """
    import tensorflow as tf

    from .config import get_split_ratios

    paths = cfg["paths"]
    classes = cfg["project"]["classes"]
    image_size = int(cfg["data"]["image_size"])
    batch = int(cfg["data"]["batch_size"])
    manifest_path = manifest_path or paths["split_manifest"]
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise DatasetEmptyError(f"Chưa có manifest chia tập tại {manifest_path}. Chạy prepare_datasets() trước.")

    warnings_issued = {"train": False, "validation": False, "test": False}

    def make_ds(split: str, with_aug: bool):
        items = manifest.get(split, [])
        if not items:
            if not warnings_issued[split]:
                print(f"[CẢNH BÁO] Split '{split}' rỗng trong manifest — bỏ qua "
                      f"(dataset có thể quá nhỏ để chia đủ 3 tập).")
                warnings_issued[split] = True
            return _make_empty_ds(cfg, split)

        file_paths = [i["path"] for i in items]
        labels = [i["label"] for i in items]

        def load_and_preprocess(fpath, label):
            img = _decode_image_safe(fpath, image_size)
            # Giữ uint8 [0,255]; MobileNetV2 preprocess_input bên trong model sẽ map [-1,1].
            return img, tf.cast(label, tf.int64)

        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if with_aug:
            aug = build_augmentation(cfg)
            ds = ds.map(lambda im, lb: (aug(im, training=True), lb),
                        num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(len(items) or 1) if split == "train" else ds
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    return {
        "train": make_ds("train", augment),
        "validation": make_ds("validation", False),
        "test": make_ds("test", False),
    }


def build_augmentation(cfg: dict):
    """Data augmentation NHẸ — không đổi hue/saturation mạnh (màu là đặc trưng chín)."""
    import tensorflow as tf

    a = cfg.get("augmentation", {})
    layers = []
    if a.get("horizontal_flip", True):
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    if a.get("vertical_flip", False):
        layers.append(tf.keras.layers.RandomFlip("vertical"))
    layers.append(tf.keras.layers.RandomRotation(float(a.get("rotation_factor", 0.1))))
    layers.append(tf.keras.layers.RandomZoom(float(a.get("zoom_factor", 0.1))))
    layers.append(tf.keras.layers.RandomTranslation(
        float(a.get("translation_factor", 0.05)),
        float(a.get("translation_factor", 0.05)),
    ))
    bf = float(a.get("brightness_factor", 0.0))
    if bf > 0:
        layers.append(tf.keras.layers.RandomBrightness(bf))
    cf = float(a.get("contrast_factor", 0.0))
    if cf > 0:
        layers.append(tf.keras.layers.RandomContrast(cf))
    # Hue/saturation giữ nguyên theo yêu cầu (mặc định 0.0, không áp dụng).
    return tf.keras.Sequential(layers, name="augmentation")
