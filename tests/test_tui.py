"""Tests cho lớp phủ TUI (scripts/tui.py) — chỉ kiểm tra dựng lệnh, không cần tty."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tui import ACTIONS, TOGGLES, build_argv, default_state  # noqa: E402


def _state(**kw):
    st = default_state()
    st["dataset"] = "chili_quality_v2"
    st.update(kw)
    return st


def test_every_action_builds_argv():
    st = _state(source="/tmp/imgs")
    for key in ACTIONS:
        argv = build_argv(key, st)
        assert isinstance(argv, list) and argv, key
        assert ".py" in " ".join(argv) or "-m" in argv


def test_dataset_required():
    st = {"dataset": "", "source": "", "class_name": "",
          "skip_center": False, "skip_train": False, "export": True}
    with pytest.raises(ValueError, match="dataset"):
        build_argv("train", st)


def test_unknown_action():
    with pytest.raises(KeyError):
        build_argv("nope", _state())


def test_toggles_map_to_flags():
    st = _state(source="/tmp/imgs", skip_center=True, skip_train=True, export=False)
    argv = build_argv("pipeline", st)
    assert "--skip-center" in argv
    assert "--skip-train" in argv
    assert "--export" not in argv


def test_pipeline_export_toggle_on():
    st = _state(source="/tmp/imgs", skip_center=False, skip_train=False, export=True)
    argv = build_argv("pipeline", st)
    assert "--export" in argv and "--skip-center" not in argv


def test_quick_pipeline_never_trains():
    st = _state(source="/tmp/imgs", skip_train=False, export=True)
    argv = build_argv("pipelines", st)
    assert "--skip-train" in argv and "--export" not in argv


def test_eval_split_flags():
    assert "--split" in build_argv("eval_test", _state())
    argv = build_argv("eval_test", _state())
    assert argv[argv.index("--split") + 1] == "test"


def test_verify_uses_dry_run():
    argv = build_argv("verify", _state())
    assert "--dry-run" in argv and "--dataset" in argv


def test_toggle_defaults():
    st = default_state()
    assert set(st) >= set(TOGGLES)
    assert st["export"] is True