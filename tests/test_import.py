"""Tests cho scripts/import_new_images.py và scripts/label_review.py (Task 3)."""
import os

import pytest

from src.task3 import Task3ManifestError

try:
    from PIL import Image
except ImportError:
    pytest.skip("Pillow not available", allow_module_level=True)

from src.task3 import _read_csv  # noqa: E402
from scripts.import_new_images import (  # noqa: E402
    import_images,
    list_source_files,
    next_image_id,
)
from scripts.label_review import (  # noqa: E402
    LABEL_OVERRIDES_COLUMNS,
    load_label_overrides,
    set_status,
    override_label,
)
from src.task3 import CAPTURE_LOG_COLUMNS  # noqa: E402


def _make_image(path, color=(120, 200, 30), size=(64, 64)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")


def _make_batch(root, n_per_class=2):
    """Tạo staging: root/{unripe,medium_ripe,ripe}/img_XX.jpg (màu duy nhất/ảnh)."""
    classes = ["unripe", "medium_ripe", "ripe"]
    paths = []
    idx = 0
    for cls in classes:
        for i in range(n_per_class):
            p = os.path.join(root, cls, f"img_{cls}_{i:02d}.jpg")
            _make_image(p, color=(10 + idx * 7, 200, 30))
            paths.append(p)
            idx += 1
    return paths


def _make_capture_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(CAPTURE_LOG_COLUMNS) + "\n")


def test_list_source_files_collects_only_images(tmp_path):
    batch = _make_batch(str(tmp_path / "src"))
    files = list_source_files(str(tmp_path / "src"))
    assert len(files) == 6
    assert all(f in files for f in batch)


def test_import_dry_run_plans_without_writing(tmp_path):
    src = str(tmp_path / "src")
    _make_batch(src)
    raw = str(tmp_path / "raw")
    cap = str(tmp_path / "capture_log.csv")
    _make_capture_log(cap)
    summary = import_images(
        source_dir=src, classes=["unripe", "medium_ripe", "ripe"],
        raw_root=raw, session_id="S1", camera="OV2640",
        review_status="accepted", capture_log_path=cap, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["planned"] == 6
    assert not os.path.exists(os.path.join(raw, "unripe"))
    assert _read_csv(cap) == []


def test_import_copies_and_records_capture_log(tmp_path):
    src = str(tmp_path / "src")
    _make_batch(src)
    raw = str(tmp_path / "raw")
    cap = str(tmp_path / "capture_log.csv")
    _make_capture_log(cap)
    summary = import_images(
        source_dir=src, classes=["unripe", "medium_ripe", "ripe"],
        raw_root=raw, session_id="S1", camera="OV2640",
        review_status="accepted", capture_log_path=cap)
    assert summary["imported"] == 6
    assert summary["by_class"] == {"unripe": 2, "medium_ripe": 2, "ripe": 2}
    for cls in ["unripe", "medium_ripe", "ripe"]:
        assert len(os.listdir(os.path.join(raw, cls))) == 2
    rows = _read_csv(cap)
    assert len(rows) == 6
    ids = {r["image_id"] for r in rows}
    assert len(ids) == 6  # image_id duy nhất
    for r in rows:
        assert r["review_status"] == "accepted"
        assert r["session_id"] == "S1"
        assert r["camera"] == "OV2640"


def test_import_filename_collision_raises(tmp_path):
    src = str(tmp_path / "src")
    _make_batch(src)
    raw = str(tmp_path / "raw")
    cap = str(tmp_path / "capture_log.csv")
    _make_capture_log(cap)
    _make_image(os.path.join(raw, "unripe", "img_unripe_00.jpg"))
    with pytest.raises(Task3ManifestError, match="không ghi đè"):
        import_images(source_dir=src, classes=["unripe", "medium_ripe", "ripe"],
                      raw_root=raw, session_id="S1", camera="OV2640",
                      review_status="accepted", capture_log_path=cap)


def test_import_duplicate_within_batch_raises(tmp_path):
    src = str(tmp_path / "src")
    _make_image(os.path.join(src, "unripe", "a.jpg"))
    _make_image(os.path.join(src, "unripe", "b.jpg"), color=(10, 10, 10))
    # b trùng a
    import shutil
    shutil.copy2(os.path.join(src, "unripe", "a.jpg"),
                  os.path.join(src, "unripe", "c.jpg"))
    raw = str(tmp_path / "raw")
    cap = str(tmp_path / "capture_log.csv")
    _make_capture_log(cap)
    with pytest.raises(Task3ManifestError, match="trùng md5"):
        import_images(source_dir=src, classes=["unripe", "medium_ripe", "ripe"],
                      raw_root=raw, session_id="S1", camera="OV2640",
                      review_status="accepted", capture_log_path=cap)


def test_next_image_id_unique_sequence():
    ids = set()
    from datetime import datetime
    dt = datetime(2026, 11, 8, 12, 0, 0)
    first = next_image_id(ids, dt)
    ids.add(first)
    second = next_image_id(ids, dt)
    assert first != second
    assert first.startswith("T3_20261108_")
    assert ".jpg" not in first


def test_import_invalid_status_raises(tmp_path):
    src = str(tmp_path / "src")
    _make_batch(src)
    cap = str(tmp_path / "capture_log.csv")
    _make_capture_log(cap)
    with pytest.raises(Task3ManifestError, match="không hợp lệ"):
        import_images(source_dir=src, classes=["unripe", "medium_ripe", "ripe"],
                      raw_root=str(tmp_path / "raw"), session_id="S1",
                      camera="OV2640", review_status="bogus",
                      capture_log_path=cap)


def test_label_review_set_status_updates_csv(tmp_path):
    src = str(tmp_path / "src")
    _make_batch(src)
    raw = str(tmp_path / "raw")
    cap = str(tmp_path / "capture_log.csv")
    _make_capture_log(cap)
    import_images(source_dir=src, classes=["unripe", "medium_ripe", "ripe"],
                  raw_root=raw, session_id="S1", camera="OV2640",
                  review_status="accepted", capture_log_path=cap)
    res = set_status(cap, "img_unripe_00.jpg", "pending_review",
                     reason="mờ", annotator="A1")
    assert res["changed"] is True
    rows = _read_csv(cap)
    row = next(r for r in rows if r["filename"] == "img_unripe_00.jpg")
    assert row["review_status"] == "pending_review"
    assert "mờ" in row["notes"]
    assert row["annotator"] == "A1"


def test_label_review_set_status_unknown_filename_raises(tmp_path):
    cap = str(tmp_path / "capture_log.csv")
    _make_capture_log(cap)
    with pytest.raises(Task3ManifestError, match="tìm thấy"):
        set_status(cap, "nope.jpg", "accepted", reason="", annotator="")


def test_label_review_override_label_appends(tmp_path):
    ov = str(tmp_path / "label_overrides.csv")
    res = override_label(ov, "img_unripe_00.jpg", "ripe",
                         reason="review", approved_by="owner")
    assert res["changed"] is True
    rows = load_label_overrides(ov)
    assert len(rows) == 1
    assert rows[0]["filename"] == "img_unripe_00.jpg"
    assert rows[0]["new_label"] == "ripe"
