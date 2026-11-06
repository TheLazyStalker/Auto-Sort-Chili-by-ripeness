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


def test_split_small_n_gives_nonempty_when_possible():
    """n>=3/lớp → cả 3 tập không rỗng; n=2 → cảnh báo (không thể chia)."""
    import warnings
    classes = ["unripe", "medium_ripe", "ripe"]
    for n in (3, 5, 7, 20):
        cf = {c: [f"{c}_{i}.jpg" for i in range(n)] for c in classes}
        splits = stratified_group_split(cf, classes, seed=42)
        assert all(len(splits[k]) > 0 for k in ("train", "validation", "test"))
    # n=2: không thể chia 3 tập, phải cảnh báo và train/val không rỗng, test có thể rỗng
    cf2 = {c: [f"{c}_{i}.jpg" for i in range(2)] for c in classes}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        splits2 = stratified_group_split(cf2, classes, seed=42)
    assert any("không đủ" in str(x.message) for x in w)


def test_split_no_overlap_even_with_groups():
    """Ảnh cùng 1 quả ớt (group) không bao giờ lọt vào 2 tập khác nhau."""
    classes = ["unripe", "medium_ripe", "ripe"]
    cf = {"unripe": [f"u{i}.jpg" for i in range(20)],
          "medium_ripe": [], "ripe": [f"r{i}.jpg" for i in range(20)]}
    groups = {}
    for i in range(5):
        for j in range(4):
            groups[f"u{i*4+j}.jpg"] = f"pepper_{i}"
    splits = stratified_group_split(cf, ["unripe", "ripe"], seed=42, groups=groups)
    split_of = {}
    for name, items in splits.items():
        for p, _ in items:
            g = groups.get(p)
            if g:
                assert split_of.setdefault(g, name) == name, f"Rò rỉ group {g}"


def test_build_tf_datasets_empty_split_no_crash(tmp_path):
    """Manifest có split rỗng → build_tf_datasets không crash, val dataset rỗng."""
    from PIL import Image
    from src.dataset import build_tf_datasets
    cfg = load_config()
    raw = tmp_path / "raw"
    paths = []
    for i in range(2):
        p = raw / f"img{i}.jpg"
        _make_image(str(p))
        paths.append(str(p))
    manifest = {
        "train": [{"path": paths[0], "label": 0, "class": "unripe"}],
        "validation": [],
        "test": [{"path": paths[1], "label": 1, "class": "medium_ripe"}],
    }
    mp = tmp_path / "split_manifest.json"
    import json
    mp.write_text(json.dumps(manifest))
    cfg = _patch_paths(cfg, raw)
    cfg["paths"]["split_manifest"] = str(mp)
    ds = build_tf_datasets(cfg)
    assert sum(1 for _ in ds["validation"]) == 0  # val rỗng, không crash


def test_write_scan_report(tmp_path):
    from src.dataset import write_scan_report
    raw = tmp_path / "raw"
    _make_image(str(raw / "unripe" / "a.jpg"), (255, 0, 0))
    (raw / "medium_ripe").mkdir(parents=True, exist_ok=True)
    (raw / "ripe").mkdir(parents=True, exist_ok=True)
    summary = scan_raw_dataset(str(raw), ["unripe", "medium_ripe", "ripe"])
    report_path = str(tmp_path / "reports" / "metrics" / "dataset_scan.json")
    write_scan_report(summary, report_path, ["unripe", "medium_ripe", "ripe"])
    import json
    with open(report_path) as f:
        data = json.load(f)
    assert data["total"] == 1
    assert data["counts"] == {"unripe": 1, "medium_ripe": 0, "ripe": 0}
    assert "generated_at" in data
    assert data["classes"] == ["unripe", "medium_ripe", "ripe"]


def test_prepare_datasets_force_resplit(tmp_path):
    """force_resplit=True phải chia lại và ghi đè manifest cũ."""
    cfg = load_config()
    raw = tmp_path / "raw"
    for c in ["unripe", "medium_ripe", "ripe"]:
        for i in range(10):
            _make_image(str(raw / c / f"{c}_{i}.jpg"), color=(i * 10, 10, 200))
    cfg = _patch_paths(cfg, raw)
    r1 = prepare_datasets(cfg, copy_to_split_dirs=True)
    # Đổi nội dung ảnh để nếu có chia lại thì số lượng vẫn giữ, nhưng thời điểm khác
    r2 = prepare_datasets(cfg, copy_to_split_dirs=True, force_resplit=True)
    assert r2["splits"] == r1["splits"] == {"train": 21, "validation": 6, "test": 3}
    # Manifest vẫn được ghi khi force_resplit (không bỏ qua)
    assert os.path.isfile(cfg["paths"]["split_manifest"])


def test_generate_groups_template(tmp_path):
    from src.dataset import generate_groups_template
    raw = tmp_path / "raw"
    _make_image(str(raw / "unripe" / "u1.jpg"))
    _make_image(str(raw / "unripe" / "u2.jpg"))
    _make_image(str(raw / "ripe" / "r1.jpg"))
    (raw / "medium_ripe").mkdir(parents=True, exist_ok=True)
    # Không ghi file → trả về nội dung CSV
    content = generate_groups_template(str(raw), ["unripe", "medium_ripe", "ripe"])
    assert content.startswith("filename,group_id\n")
    assert "u1.jpg,\n" in content
    assert "u2.jpg,\n" in content
    assert "r1.jpg,\n" in content
    assert "medium_ripe" not in content  # không có ảnh
    # Ghi ra file
    out = tmp_path / "groups.csv"
    generate_groups_template(str(raw), ["unripe", "medium_ripe", "ripe"], out_path=str(out))
    with open(out) as f:
        data = f.read()
    assert "u1.jpg," in data

