"""Tests cho src/task3.py — exclusion, group splits, manifest Task 3 (Phase 1)."""
import csv
import os

import pytest

from src.task3 import (
    SPLIT_NAMES,
    REVIEW_STATUS_ACCEPTED,
    REVIEW_STATUS_PENDING_REVIEW,
    Task3ManifestError,
    assign_group_splits,
    build_task3_manifest,
    excluded_filenames,
    is_excluded_row,
    load_capture_log,
    load_exclusions,
    load_group_splits,
    verify_task3_manifest,
    write_task3_manifest,
)


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _derived_row(**kw):
    row = {
        "source_path": "/raw/unripe/a.jpg",
        "output_path": "/derived/train/unripe/a.png",
        "crop_status": "ok",
        "label": "unripe",
        "group_id": "",
        "split": "train",
        "crop_version": "crop_v1",
    }
    row.update(kw)
    return row


def _manifest_rows(n=6, base=1):
    classes = ["unripe", "medium_ripe", "ripe"]
    rows = []
    i = 0
    for _ in range(n):
        cls = classes[i % 3]
        rows.append(_derived_row(
            source_path=f"/raw/{cls}/i{base + i}.jpg",
            output_path=f"/derived/train/{cls}/i{base + i}.png",
            label=cls, group_id="", split="train",
        ))
        i += 1
    return rows


# ---------------- exclusions ----------------

def test_load_exclusions_and_excluded_filenames(tmp_path):
    p = tmp_path / "excl.csv"
    _write_csv(p, [
        {"filename": "bad.jpg", "status": "excluded", "reviewed": "no"},
        {"filename": "ok.jpg", "status": "accepted", "reviewed": "yes"},
    ])
    ex = load_exclusions(str(p))
    assert set(ex) == {"bad.jpg", "ok.jpg"}
    assert is_excluded_row(ex["bad.jpg"])
    assert not is_excluded_row(ex["ok.jpg"])
    assert excluded_filenames(str(p)) == {"bad.jpg"}


def test_exclusions_file_missing_returns_empty(tmp_path):
    assert load_exclusions(str(tmp_path / "absent.csv")) == {}


def test_unknown_status_raises_in_exclusions(tmp_path):
    p = tmp_path / "excl.csv"
    _write_csv(p, [{"filename": "x.jpg", "status": "wat"}])
    with pytest.raises(Task3ManifestError, match="không hợp lệ"):
        load_exclusions(str(p))


def test_pending_review_is_excluded(tmp_path):
    p = tmp_path / "excl.csv"
    _write_csv(p, [{"filename": "x.jpg", "status": "pending_review"}])
    assert excluded_filenames(str(p)) == {"x.jpg"}


def test_load_capture_log_validates_review_status(tmp_path):
    p = tmp_path / "cap.csv"
    _write_csv(p, [{"filename": "a.jpg", "review_status": "bogus"}])
    with pytest.raises(Task3ManifestError, match="không hợp lệ"):
        load_capture_log(str(p))


# ---------------- group splits ----------------

def test_load_group_splits_ok(tmp_path):
    p = tmp_path / "groups.csv"
    _write_csv(p, [
        {"group_id": "g1", "split": "validation"},
        {"group_id": "g2", "split": "test"},
    ])
    assert load_group_splits(str(p)) == {"g1": "validation", "g2": "test"}


def test_load_group_splits_invalid_split_raises(tmp_path):
    p = tmp_path / "groups.csv"
    _write_csv(p, [{"group_id": "g1", "split": "bogus"}])
    with pytest.raises(Task3ManifestError, match="không hợp lệ"):
        load_group_splits(str(p))


def test_load_group_splits_conflict_raises(tmp_path):
    p = tmp_path / "groups.csv"
    _write_csv(p, [
        {"group_id": "g1", "split": "validation"},
        {"group_id": "g1", "split": "test"},
    ])
    with pytest.raises(Task3ManifestError, match="hai split khác nhau"):
        load_group_splits(str(p))


# ---------------- build_task3_manifest ----------------

def test_assign_group_splits_50_50_no_group_split():
    groups = {"unripe": [f"g{i}" for i in range(20)],
              "medium_ripe": [f"m{i}" for i in range(20)],
              "ripe": [f"r{i}" for i in range(20)]}
    assignment, stats = assign_group_splits(groups, seed=42)
    assert len(assignment) == 60
    assert set(assignment.values()) <= {"validation", "test"}
    # 50/50 mỗi lớp
    for cls in groups:
        assert stats["by_class"][cls]["validation"] == 10
        assert stats["by_class"][cls]["test"] == 10
    # Không có group nào bị chia (mỗi group 1 split)
    assert len(set(assignment.values())) <= 2
    assert stats["warnings"] == []


def test_assign_group_splits_warns_small_splits():
    groups = {"unripe": [f"g{i}" for i in range(12)]}
    _assignment, stats = assign_group_splits(groups, seed=1)
    assert any("không robust" in w for w in stats["warnings"])


def test_assign_group_splits_deterministic():
    groups = {"unripe": [f"g{i}" for i in range(40)]}
    a1, _ = assign_group_splits(groups, seed=42)
    a2, _ = assign_group_splits(groups, seed=42)
    assert a1 == a2


def test_build_manifest_respects_review_statuses():
    rows = []
    for i, cls in enumerate(["unripe", "medium_ripe", "ripe"]):
        rows.append(_derived_row(
            source_path=f"/raw/{cls}/i{i}.jpg",
            output_path=f"/derived/train/{cls}/i{i}.png",
            label=cls))
    # i1 (medium_ripe) ở pending_review → bị bỏ khỏi manifest
    statuses = {f"i{i}.jpg": REVIEW_STATUS_ACCEPTED for i in range(3)}
    statuses["i1.jpg"] = REVIEW_STATUS_PENDING_REVIEW
    with pytest.raises(Task3ManifestError, match="Train thiếu lớp"):
        build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"],
                             review_statuses=statuses)


def test_build_manifest_applies_label_overrides():
    rows = [_derived_row(
        source_path="/raw/unripe/u1.jpg", output_path="/derived/train/unripe/u1.png",
        label="unripe"),
        _derived_row(
        source_path="/raw/unripe/u2.jpg", output_path="/derived/train/unripe/u2.png",
        label="unripe"),
        _derived_row(
        source_path="/raw/medium_ripe/m.jpg", output_path="/derived/train/medium_ripe/m.png",
        label="medium_ripe"),
        _derived_row(
        source_path="/raw/ripe/r.jpg", output_path="/derived/train/ripe/r.png",
        label="ripe")]
    overrides = {"u1.jpg": "ripe"}  # override unripe → ripe
    res = build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"],
                               label_overrides=overrides)
    train = res["manifest"]["train"]
    u1 = [i for i in train if i["source_path"].endswith("u1.jpg")][0]
    assert u1["label"] == 2
    assert u1["class"] == "ripe"
    assert res["verification"]["counts"]["train"] == {
        "unripe": 1, "medium_ripe": 1, "ripe": 2}


def test_build_manifest_old_images_train_only():
    rows = _manifest_rows()
    res = build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"])
    m = res["manifest"]
    assert len(m["train"]) == 6
    assert m["validation"] == [] and m["test"] == []
    assert res["verification"]["totals"]["train"] == 6
    assert res["verification"]["provisional"] is True


def test_build_manifest_excludes_excluded_sources():
    rows = _manifest_rows()
    rows.append(_derived_row(
        source_path="/raw/unripe/bad.jpg", output_path="/derived/train/unripe/bad.png"))
    res = build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"],
                               exclusions={"bad.jpg"})
    assert res["skipped"]["excluded"] == ["bad.jpg"]
    assert len(res["manifest"]["train"]) == 6


def test_build_manifest_skips_needs_review_by_default():
    rows = _manifest_rows()
    rows.append(_derived_row(
        source_path="/raw/unripe/rev.jpg", output_path="/derived/train/unripe/rev.png",
        crop_status="needs_review"))
    res = build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"])
    assert ("rev.jpg", "needs_review") in res["skipped"]["crop_status"]
    assert len(res["manifest"]["train"]) == 6


def test_build_manifest_allows_needs_review_when_flagged():
    rows = _manifest_rows()
    rows.append(_derived_row(
        source_path="/raw/unripe/rev.jpg", output_path="/derived/train/unripe/rev.png",
        crop_status="needs_review"))
    res = build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"],
                               allow_needs_review=True)
    assert len(res["manifest"]["train"]) == 7


def test_build_manifest_new_group_without_split_raises():
    rows = _manifest_rows()
    rows.append(_derived_row(
        source_path="/raw/unripe/new.jpg", output_path="/derived/train/unripe/new.png",
        group_id="g_new", split=""))
    with pytest.raises(Task3ManifestError, match="chưa được gán split"):
        build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"],
                             group_splits={})


def test_build_manifest_new_group_split_mismatch_raises():
    rows = _manifest_rows()
    rows.append(_derived_row(
        source_path="/raw/unripe/new.jpg", output_path="/derived/train/unripe/new.png",
        group_id="g1", split="test"))
    with pytest.raises(Task3ManifestError, match="nhưng group_splits.csv gán"):
        build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"],
                             group_splits={"g1": "validation"})


def test_build_manifest_new_group_inherits_split():
    rows = _manifest_rows(base=10)
    rows.append(_derived_row(
        source_path="/raw/ripe/p.jpg", output_path="/derived/test/ripe/p.png",
        label="ripe", group_id="g_test", split=""))
    res = build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"],
                               group_splits={"g_test": "test"})
    assert len(res["manifest"]["test"]) == 1
    assert res["manifest"]["test"][0]["group_id"] == "g_test"


def test_build_manifest_requires_all_three_classes_in_train():
    rows = [_derived_row(
        source_path=f"/raw/unripe/i{i}.jpg",
        output_path=f"/derived/train/unripe/i{i}.png",
        label="unripe", group_id="", split="train",
    ) for i in range(2)]  # chỉ unripe
    with pytest.raises(Task3ManifestError, match="Train thiếu lớp"):
        build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"])


def test_build_manifest_all_classes_in_train_passes():
    rows = []
    for i, cls in enumerate(["unripe", "medium_ripe", "ripe"]):
        rows.append(_derived_row(
            source_path=f"/raw/{cls}/i{i}.jpg",
            output_path=f"/derived/train/{cls}/i{i}.png",
            label=cls))
    res = build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"])
    assert res["verification"]["ok"] is True
    assert res["verification"]["counts"]["train"] == {"unripe": 1, "medium_ripe": 1, "ripe": 1}


def test_build_manifest_file_overlap_raises():
    rows = _manifest_rows(base=1)
    # Cùng output_path với dòng train đầu tiên nhưng được gán sang validation
    rows.append(_derived_row(
        source_path="/raw/unripe/i1.jpg", output_path="/derived/train/unripe/i1.png",
        group_id="gv", split="validation"))
    with pytest.raises(Task3ManifestError, match="nhiều split"):
        build_task3_manifest(rows, ["unripe", "medium_ripe", "ripe"],
                             group_splits={"gv": "validation"})

# ---------------- verify ---------------

def test_verify_detects_excluded_present():
    manifest = {"train": [{"path": "/x/a.png", "label": 0, "class": "unripe",
                           "group_id": "", "source_path": "/raw/unripe/bad.jpg"}],
                "validation": [], "test": []}
    v = verify_task3_manifest(manifest, ["unripe", "medium_ripe", "ripe"],
                              exclusions={"bad.jpg"})
    assert v["excluded_present"] == ["bad.jpg"]
    assert v["ok"] is False


def test_verify_detects_group_leak():
    manifest = {"train": [{"path": "/x/a.png", "label": 0, "class": "unripe",
                           "group_id": "g1", "source_path": "/raw/unripe/a.jpg"}],
                "validation": [{"path": "/x/b.png", "label": 0, "class": "unripe",
                                "group_id": "g1", "source_path": "/raw/unripe/b.jpg"}],
                "test": []}
    v = verify_task3_manifest(manifest, ["unripe", "medium_ripe", "ripe"])
    assert v["group_leaks"] == {"g1": ["train", "validation"]}
    assert v["ok"] is False


def test_verify_provisional_flag():
    manifest = {"train": [{"path": "/x/a.png", "label": 0, "class": "unripe",
                           "group_id": "", "source_path": "/raw/unripe/a.jpg"}],
                "validation": [], "test": []}
    v = verify_task3_manifest(manifest, ["unripe", "medium_ripe", "ripe"],)
    assert v["provisional"] is True

# ---------------- write ---------------

def test_write_task3_manifest_records_hashes(tmp_path):
    manifest = {"train": [{"path": "/x/a.png", "label": 0, "class": "unripe",
                           "group_id": "", "source_path": "/raw/unripe/a.jpg"}],
                "validation": [], "test": []}
    manifest_path = str(tmp_path / "split_manifest_task3.json")
    meta_path = str(tmp_path / "meta.json")
    meta = write_task3_manifest(manifest, manifest_path, meta_path,
                                crop_version="crop_v1")
    assert os.path.isfile(manifest_path)
    assert len(meta["manifest_sha256"]) == 64
    assert meta["crop_version"] == "crop_v1"
    assert meta["provisional"] is True
    assert meta["counts"] == {"train": 1, "validation": 0, "test": 0}


def test_write_task3_manifest_meta_extra(tmp_path):
    manifest = {"train": [], "validation": [], "test": []}
    meta_path = str(tmp_path / "meta.json")
    meta = write_task3_manifest(manifest, str(tmp_path / "m.json"), meta_path,
                                crop_version="crop_v1", extra={"k": "v"})
    assert meta["k"] == "v"
