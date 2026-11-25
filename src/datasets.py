"""Dataset registry — quản lý nhiều bộ dữ liệu độc lập (multi-dataset).

- Mỗi dataset có ``dataset_config.yaml`` trong ``datasets/<dataset_id>/``.
- Mọi thông tin (classes, thứ tự lớp, paths, preprocessing, split) đọc từ config;
  KHÔNG hardcode tên lớp trong logic chung.
- Classes rỗng/trùng → raise; dataset trống → trả warnings, không tự train.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")

DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)


class DatasetConfigError(RuntimeError):
    """Cấu hình dataset không hợp lệ — dừng an toàn."""


def _resolve(path: str, base: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base, path))


@dataclass
class DatasetConfig:
    dataset_id: str
    dataset_version: int = 1
    task_type: str = "image_classification"
    classes: list[str] = field(default_factory=list)
    image_size: int = 224
    color_mode: str = "RGB"
    preprocessing_pipeline: str = "crop_v1"
    preprocessing_version: str = "v1"
    split_strategy: str = "group_stratified"
    split_ratios: tuple = field(default_factory=lambda: DEFAULT_SPLIT_RATIOS)
    split_seed: int = 42
    labels_source: str = "directory"
    allow_pending_review: bool = True
    aliases: dict[str, list[str]] = field(default_factory=dict)
    require_nonempty: bool = True
    # resolved paths (tuyệt đối)
    root: str = ""
    raw_dir: str = ""
    derived_dir: str = ""
    manifests_dir: str = ""
    exclusions_path: str = ""
    capture_log_path: str = ""
    group_splits_path: str = ""
    label_overrides_path: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def class_mapping(self) -> dict[str, int]:
        """{tên lớp: index} — thứ tự = thứ tự trong config."""
        return {c: i for i, c in enumerate(self.classes)}


def list_dataset_ids(root: str | None = None) -> list[str]:
    base = root or DEFAULT_DATASETS_ROOT
    out = []
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            if os.path.isfile(os.path.join(base, name, "dataset_config.yaml")):
                out.append(name)
    return out


def load_dataset_config(dataset_id: str, root: str | None = None) -> DatasetConfig:
    base = root or DEFAULT_DATASETS_ROOT
    cfg_path = os.path.join(base, dataset_id, "dataset_config.yaml")
    if not os.path.isfile(cfg_path):
        raise DatasetConfigError(
            f"Không tìm thấy dataset_config.yaml cho '{dataset_id}' "
            f"(tìm tại {cfg_path}). Có sẵn: {list_dataset_ids(base)}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    classes = [str(c) for c in (raw.get("classes") or [])]
    if not classes:
        raise DatasetConfigError(
            f"dataset '{dataset_id}': classes rỗng — phải khai báo danh sách lớp.")
    if len(set(classes)) != len(classes):
        raise DatasetConfigError(
            f"dataset '{dataset_id}': classes bị trùng lặp: {classes}")

    ds_root = os.path.join(base, dataset_id)
    p = raw.get("paths") or {}
    split = raw.get("split") or {}
    ratios = tuple(float(x) for x in split.get("ratios", list(DEFAULT_SPLIT_RATIOS)))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise DatasetConfigError(
            f"dataset '{dataset_id}': split.ratios phải 3 giá trị, tổng ≈1: {ratios}")

    cfg = DatasetConfig(
        dataset_id=dataset_id,
        dataset_version=int(raw.get("dataset_version", 1)),
        task_type=raw.get("task_type", "image_classification"),
        classes=classes,
        image_size=int((raw.get("input") or {}).get("image_size", 224)),
        color_mode=(raw.get("input") or {}).get("color_mode", "RGB"),
        preprocessing_pipeline=(raw.get("preprocessing") or {}).get("pipeline", "crop_v1"),
        preprocessing_version=(raw.get("preprocessing") or {}).get("version", "v1"),
        split_strategy=split.get("strategy", "group_stratified"),
        split_ratios=ratios,
        split_seed=int(split.get("seed", 42)),
        labels_source=(raw.get("labels") or {}).get("source", "directory"),
        allow_pending_review=bool((raw.get("labels") or {}).get("allow_pending_review", True)),
        aliases={k: list(v) for k, v in ((raw.get("labels") or {}).get("aliases") or {}).items()},
        require_nonempty=bool((raw.get("validation") or {}).get("require_nonempty", True)),
        root=ds_root,
        raw_dir=_resolve(p.get("raw", f"datasets/{dataset_id}/raw"), PROJECT_ROOT),
        derived_dir=_resolve(p.get("derived", f"datasets/{dataset_id}/derived"), PROJECT_ROOT),
        manifests_dir=_resolve(p.get("manifests", f"datasets/{dataset_id}/manifests"), PROJECT_ROOT),
        exclusions_path=_resolve(p.get("exclusions", ""), PROJECT_ROOT),
        capture_log_path=_resolve(p.get("capture_log", ""), PROJECT_ROOT),
        group_splits_path=_resolve(p.get("group_splits", ""), PROJECT_ROOT),
        label_overrides_path=_resolve(p.get("label_overrides", ""), PROJECT_ROOT),
    )
    return cfg


# ---------------------------------------------------------------- scan ----

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def resolve_class_dir(cfg: DatasetConfig, cls: str) -> str | None:
    """Thư mục raw thực tế cho lớp, hỗ trợ alias từ config (VD medium-ripe)."""
    candidates = (cls,) + tuple(cfg.aliases.get(cls, ()))
    for name in candidates:
        cand = os.path.join(cfg.raw_dir, name)
        if os.path.isdir(cand):
            return cand
    return None


def _content_hash(path: str) -> str | None:
    import hashlib
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _validate_image(path: str) -> bool:
    from PIL import Image
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.width > 0 and img.height > 0
    except Exception:
        return False


def scan_dataset(cfg: DatasetConfig) -> dict:
    """Quét raw theo classes trong config.

    Trả về {counts, files, total, corrupt, duplicates, empty_classes, warnings}.
    Không raise khi dataset trống — trả empty_classes để caller quyết định.
    """
    files: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    corrupt: list[str] = []
    hash_to_files: dict[str, list[str]] = {}
    empty_classes: list[str] = []
    warnings: list[str] = []

    for cls in cfg.classes:
        class_dir = resolve_class_dir(cfg, cls)
        collected = []
        if class_dir is not None:
            for fname in sorted(os.listdir(class_dir)):
                fpath = os.path.join(class_dir, fname)
                if not os.path.isfile(fpath) or not fpath.lower().endswith(SUPPORTED_EXTENSIONS):
                    continue
                if not _validate_image(fpath):
                    corrupt.append(fpath)
                    continue
                collected.append(fpath)
                h = _content_hash(fpath)
                if h:
                    hash_to_files.setdefault(h, []).append(fpath)
        files[cls] = collected
        counts[cls] = len(collected)
        if not collected:
            empty_classes.append(cls)
            warnings.append(f"Lớp '{cls}' KHÔNG có ảnh (thư mục rỗng/missing).")

    duplicates = {h: ps for h, ps in hash_to_files.items() if len(ps) > 1}
    for h, ps in duplicates.items():
        warnings.append(f"{len(ps)} ảnh trùng nội dung (md5 {h[:8]}): "
                        + ", ".join(os.path.basename(p) for p in ps[:5]))

    # Mất cân bằng: lớp có ít hơn 30% so với lớp đông nhất
    mx = max(counts.values(), default=0)
    for cls, n in counts.items():
        if mx > 0 and n > 0 and n / mx < 0.3:
            warnings.append(f"Lớp '{cls}' mất cân bằng: {n} ảnh vs lớp đông nhất {mx}.")

    return {
        "counts": counts,
        "files": files,
        "total": sum(counts.values()),
        "corrupt": corrupt,
        "duplicates": duplicates,
        "empty_classes": empty_classes,
        "warnings": warnings,
    }


def validate_dataset(cfg: DatasetConfig) -> dict:
    """Kiểm tra dataset trước khi train. Trả {ok, errors, warnings, scan}.

    - corrupt > 0 → error (ảnh lỗi).
    - empty class và require_nonempty → error (đủ 3 lớp bắt buộc).
    """
    scan = scan_dataset(cfg)
    errors = []
    if scan["corrupt"]:
        errors.append(f"{len(scan['corrupt'])} ảnh lỗi/không đọc được (xem corrupt).")
    if scan["empty_classes"]:
        msg = "thư mục lớp trống/missing: " + ", ".join(scan["empty_classes"])
        if cfg.require_nonempty:
            errors.append(f"Dataset '{cfg.dataset_id}' thiếu lớp: {msg}")
        else:
            scan["warnings"].append(
                f"Dataset '{cfg.dataset_id}' chưa đủ lớp ({msg}) — sẽ dừng train cho đến khi có ảnh.")
    if scan["total"] == 0 and cfg.require_nonempty:
        errors.append(f"Dataset '{cfg.dataset_id}' TRỐNG (0 ảnh) — không thể train.")
    return {"ok": not errors, "errors": errors,
            "warnings": scan["warnings"], "scan": scan}
