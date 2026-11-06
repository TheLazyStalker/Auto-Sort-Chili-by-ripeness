import os

import numpy as np
import pytest

from src.config import load_config
from src.dataset import (
    DatasetEmptyError,
    load_groups_file,
    prepare_datasets,
    scan_raw_dataset,
    stratified_group_split,
)
from src.utils import compute_class_weight, count_per_class

try:
    from PIL import Image
except ImportError:
    pytest.skip("Pillow not available", allow_module_level=True)


def _make_image(path, color=(255, 0, 0), size=(64, 64)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")


def test_scan_empty_raw_raises_dataset_empty(tmp_path):
    cfg = load_config()
    raw = tmp_path / "raw"
    for c in ["unripe", "medium_ripe", "ripe"]:
        (raw / c).mkdir(parents=True)
    with pytest.raises(DatasetEmptyError):
        prepare_datasets(_patch_paths(cfg, raw), copy_to_split_dirs=False)


def _patch_paths(cfg, raw_dir):
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["dataset_raw"] = str(raw_dir)
    cfg["paths"]["split_manifest"] = str(raw_dir / "split_manifest.json")
    cfg["paths"]["dataset_train"] = str(raw_dir / "train")
    cfg["paths"]["dataset_validation"] = str(raw_dir / "validation")
    cfg["paths"]["dataset_test"] = str(raw_dir / "test")
    cfg["paths"]["groups_file"] = ""
    return cfg


def test_scan_counts_and_detects_duplicates(tmp_path):
    raw = tmp_path / "raw"
    _make_image(str(raw / "unripe" / "a.jpg"), (255, 0, 0))
    _make_image(str(raw / "unripe" / "b.jpg"), (255, 0, 0))  # trùng nội dung với a.jpg
    _make_image(str(raw / "ripe" / "c.png"), (0, 255, 0))
    (raw / "medium_ripe").mkdir(parents=True, exist_ok=True)

    summary = scan_raw_dataset(str(raw), ["unripe", "medium_ripe", "ripe"])
    cfg = load_config()
    assert count_per_class(summary.files, ["unripe", "medium_ripe", "ripe"]) == {
        "unripe": 2, "medium_ripe": 0, "ripe": 1}
    assert len(summary.duplicates) == 1  # a.jpg == b.jpg


def test_supported_extensions():
    assert True  # jpg/jpeg/png được liệt kê trong config và dataset SUPPORTED_EXTENSIONS
    from src.dataset import SUPPORTED_EXTENSIONS

    assert set(SUPPORTED_EXTENSIONS) == {".jpg", ".jpeg", ".png"}


def test_stratified_split_ratios_and_no_overlap():
    cfg = load_config()
    classes = ["unripe", "medium_ripe", "ripe"]
    class_to_files = {
        c: [f"{c}_{i}.jpg" for i in range(100)] for c in classes
    }
    splits = stratified_group_split(class_to_files, classes, ratios=(0.70, 0.15, 0.15), seed=42)
    assert len(splits["train"]) == 210
    assert len(splits["validation"]) == 45
    assert len(splits["test"]) == 45
    # Không giao nhau
    paths = [set(p for p, _ in splits[s]) for s in ("train", "validation", "test")]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    # Mỗi lớp được stratified: 100 ảnh × 3 lớp = 300 → train 210, val/test mỗi 45
    expected = {"train": 210, "validation": 45, "test": 45}
    for s in splits:
        assert len(splits[s]) == expected[s]


def test_group_id_keeps_same_pepper_in_one_split():
    classes = ["unripe", "medium_ripe", "ripe"]
    class_to_files = {
        "unripe": [f"unripe_{i}.jpg" for i in range(20)],
        "medium_ripe": [],
        "ripe": [f"ripe_{i}.jpg" for i in range(20)],
    }
    # 5 quả ớt, mỗi quả 4 ảnh → group phải giữ nguyên trong 1 tập
    groups = {}
    for i in range(5):
        for j in range(4):
            groups[f"unripe_{i*4+j}.jpg"] = f"pepper_{i}"
    splits = stratified_group_split(class_to_files, ["unripe", "ripe"],
                                     ratios=(0.70, 0.15, 0.15), seed=42, groups=groups)
    # Kiểm tra: cùng group không xuất hiện ở 2 tập khác nhau
    split_of = {}
    for name, items in splits.items():
        for p, _ in items:
            g = groups.get(p)
            if g:
                assert split_of.setdefault(g, name) == name, f"Rò rỉ group {g}"


def test_prepare_datasets_writes_manifest_and_copies(tmp_path):
    cfg = load_config()
    raw = tmp_path / "raw"
    n = 20
    for c in ["unripe", "medium_ripe", "ripe"]:
        for i in range(n):
            _make_image(str(raw / c / f"{c}_{i}.jpg"))
    cfg = _patch_paths(cfg, raw)
    result = prepare_datasets(cfg, copy_to_split_dirs=True)
    assert result["splits"] == {"train": 42, "validation": 9, "test": 9}
    assert os.path.isfile(cfg["paths"]["split_manifest"])
    # Không xóa ảnh gốc
    assert os.path.isfile(str(raw / "unripe" / "unripe_0.jpg"))
    # Không chia lại lần 2
    result2 = prepare_datasets(cfg, copy_to_split_dirs=False)
    assert result2["splits"] == result["splits"]


def test_load_groups_file(tmp_path):
    csv_path = tmp_path / "groups.csv"
    with open(csv_path, "w") as f:
        f.write("filename,group_id\nimg1.jpg,pepper_1\nimg2.jpg,pepper_2\n")
    assert load_groups_file(str(csv_path)) == {"img1.jpg": "pepper_1", "img2.jpg": "pepper_2"}


def test_compute_class_weight_balanced():
    w = compute_class_weight(np.array([0, 0, 0, 1, 2]), 3)
    assert w[0] < w[1] and w[0] < w[2]
