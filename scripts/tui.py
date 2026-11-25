#!/usr/bin/env python3
"""Lớp phủ TUI (terminal UI) — nút bấm cho các lệnh, công tắc cho tham số.

CHỈ LÀ LỚP PHỦ: mỗi "nút" tương ứng đúng một lệnh CLI của project. Gỡ file này
đi thì mọi lệnh vẫn dùng bình thường. Không chứa logic xử lý ảnh/train.

Dùng:
  python scripts/tui.py            # giao diện tương tác
  python scripts/tui.py --list     # in danh sách nút/lệnh (không cần tty)
  python scripts/tui.py --run verify --dataset chili_quality_v2   # chạy 1 nút

Điều khiển: ↑/↓ hoặc j/k di chuyển · Enter chạy · t bật/tắt công tắc · q thoát
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PY = sys.executable


# ---------------------------------------------------------------- actions ----
# Mỗi entry: label, argv builder, có cần dataset không
ACTIONS = {
    "verify": {
        "label": "Kiểm tra dữ liệu (dry-run)",
        "argv": lambda s: [PY, "scripts/train.py", "--dataset", s["dataset"], "--dry-run"],
        "needs_dataset": True,
    },
    "derivatives": {
        "label": "Sinh derivative crop_v1",
        "argv": lambda s: [PY, "scripts/make_dataset_derivatives.py",
                           "--dataset", s["dataset"]],
        "needs_dataset": True,
    },
    "train": {
        "label": "Train (experiment mới)",
        "argv": lambda s: [PY, "scripts/train.py", "--dataset", s["dataset"]],
        "needs_dataset": True,
    },
    "eval_test": {
        "label": "Đánh giá split TEST",
        "argv": lambda s: [PY, "scripts/evaluate.py", "--dataset", s["dataset"],
                           "--split", "test", "--save-probs"],
        "needs_dataset": True,
    },
    "eval_val": {
        "label": "Đánh giá split VALIDATION",
        "argv": lambda s: [PY, "scripts/evaluate.py", "--dataset", s["dataset"],
                           "--split", "validation"],
        "needs_dataset": True,
    },
    "export": {
        "label": "Export TFLite (F32 + F16)",
        "argv": lambda s: [PY, "scripts/export.py", "--dataset", s["dataset"]],
        "needs_dataset": True,
    },
    "pipeline": {
        "label": "PIPELINE: center → raw → derivative → train",
        "argv": lambda s: _pipeline_argv(s),
        "needs_dataset": True,
    },
    "pipelines": {
        "label": "PIPELINE nhanh trên nguồn mẫu (kiểm thử)",
        "argv": lambda s: _pipeline_argv(s, force_skip_train=True),
        "needs_dataset": True,
    },
    "tests": {
        "label": "Chạy test suite",
        "argv": lambda s: [PY, "-m", "pytest", "-q"],
        "needs_dataset": False,
    },
}

TOGGLES = {
    "skip_center": {"label": "bỏ qua centering", "default": False},
    "skip_train": {"label": "bỏ qua train", "default": False},
    "export": {"label": "export sau train", "default": True},
}


def _pipeline_argv(state: dict, force_skip_train: bool = False) -> list[str]:
    src = state.get("source") or ""
    argv = [PY, "scripts/pipeline.py", "--dataset", state["dataset"], "--source", src]
    if state.get("class_name"):
        argv += ["--class", state["class_name"]]
    if state.get("skip_center"):
        argv.append("--skip-center")
    if force_skip_train or state.get("skip_train"):
        argv.append("--skip-train")
    if state.get("export") and not (force_skip_train or state.get("skip_train")):
        argv.append("--export")
    return argv


def build_argv(action: str, state: dict) -> list[str]:
    """Sinh argv cho một nút (hàm thuần — test được không cần curses)."""
    if action not in ACTIONS:
        raise KeyError(f"Không có nút '{action}'. Có: {sorted(ACTIONS)}")
    if ACTIONS[action]["needs_dataset"] and not state.get("dataset"):
        raise ValueError("Cần chọn dataset trước.")
    return ACTIONS[action]["argv"](state)


def default_state() -> dict:
    st = {"dataset": "", "source": "", "class_name": ""}
    st.update({k: v["default"] for k, v in TOGGLES.items()})
    return st


def pick_default_dataset() -> str:
    try:
        from src.datasets import list_dataset_ids
        ids = list_dataset_ids()
        if ids:
            return os.environ.get("TUI_DATASET", ids[0])
    except Exception:
        pass
    return os.environ.get("TUI_DATASET", "")


def print_list(state: dict) -> None:
    print("Nút:")
    for key, a in ACTIONS.items():
        try:
            argv = " ".join(build_argv(key, state))
        except Exception:
            argv = "(cần chọn dataset)"
        print(f"  {key:12s} {a['label']}\n               $ {argv}")
    print("\nCông tắc:")
    for k, t in TOGGLES.items():
        print(f"  {k:12s} {t['label']} = {state.get(k)}")


# -------------------------------------------------------------------- TUI ----

def run_command(argv: list[str], cwd: str = PROJECT_ROOT) -> int:
    """Chạy lệnh con; trả exit code."""
    print("\n$ " + " ".join(argv) + "\n", flush=True)
    try:
        return subprocess.call(argv, cwd=cwd)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[LỖI] {exc}")
        return 1


def _cycle_dataset(state: dict) -> None:
    from src.datasets import list_dataset_ids
    ids = list_dataset_ids()
    if not ids:
        return
    cur = state.get("dataset")
    state["dataset"] = ids[(ids.index(cur) + 1) % len(ids)] if cur in ids else ids[0]


def _cycle_class(state: dict) -> None:
    from src.datasets import load_dataset_config
    try:
        classes = load_dataset_config(state["dataset"]).classes if state["dataset"] else []
    except Exception:
        classes = []
    if not classes:
        return
    cur = state.get("class_name")
    state["class_name"] = classes[(classes.index(cur) + 1) % len(classes)] if cur in classes else classes[0]


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def run_tui(state: dict) -> int:
    import curses

    items = list(ACTIONS.items())

    def draw(scr, sel: int) -> None:
        scr.erase()
        h, w = scr.getmaxyx()
        title = " CHILI PIPELINE — TUI "
        scr.addstr(0, 0, title.center(w - 1, "─")[: w - 1], curses.A_BOLD)
        scr.addstr(1, 2, f"dataset: {state['dataset'] or '(chưa chọn)'}   "
                          f"source: {state['source'] or '(chưa nhập)'}   "
                          f"class: {state['class_name'] or '(auto)'}"[: w - 3])
        toggles = "  ".join(f"[{k}{'✓' if state.get(k) else '·'}]" for k in TOGGLES)
        scr.addstr(2, 2, toggles[: w - 3], curses.A_DIM)
        scr.addstr(3, 2, "↑/↓ chọn · Enter chạy · t đổi công tắc · d đổi dataset · "
                         "c đổi class · s nhập source · q thoát"[: w - 3], curses.A_DIM)

        row = 5
        for i, (key, a) in enumerate(items):
            label = f"  {a['label']}"
            if i == sel:
                scr.addstr(row + i, 1, label.ljust(w - 3)[: w - 3], curses.A_REVERSE)
            else:
                scr.addstr(row + i, 1, label[: w - 3])
        scr.refresh()

    def loop(scr) -> int:
        curses.curs_set(0)
        sel = 0
        while True:
            draw(scr, sel)
            ch = scr.getch()
            if ch in (ord("q"), 27):
                return 0
            if ch in (curses.KEY_UP, ord("k")):
                sel = (sel - 1) % len(items)
            elif ch in (curses.KEY_DOWN, ord("j")):
                sel = (sel + 1) % len(items)
            elif ch == ord("t"):
                keys = list(TOGGLES)
                # xoay công tắc đầu tiên chưa bật, hoặc tắt hết nếu đang bật hết
                idx = next((i for i, k in enumerate(keys) if not state.get(k)), None)
                if idx is None:
                    for k in keys:
                        state[k] = False
                else:
                    state[keys[idx]] = True
            elif ch == ord("d"):
                _cycle_dataset(state)
            elif ch == ord("c"):
                _cycle_class(state)
            elif ch == ord("s"):
                curses.endwin()
                state["source"] = _ask("Đường dẫn thư mục ảnh nguồn: ")
                curses.reset_prog_mode()
            elif ch in (curses.KEY_ENTER, 10, 13):
                key, _a = items[sel]
                if key == "pipeline" and not state.get("source"):
                    curses.endwin()
                    state["source"] = _ask("Đường dẫn thư mục ảnh nguồn: ")
                    curses.reset_prog_mode()
                    if not state["source"]:
                        continue
                try:
                    argv = build_argv(key, state)
                except Exception as exc:  # noqa: BLE001
                    curses.endwin()
                    print(f"[LỖI] {exc}")
                    _ask("\n[Enter] để quay lại...")
                    curses.reset_prog_mode()
                    continue
                curses.def_prog_mode()
                curses.endwin()
                rc = run_command(argv)
                print(f"\n[exit {rc}]")
                _ask("[Enter] để quay lại TUI...")
                curses.reset_prog_mode()
                scr.clear()

    return curses.wrapper(loop)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TUI overlay cho pipeline chili")
    ap.add_argument("--list", action="store_true", help="in danh sách nút/lệnh")
    ap.add_argument("--run", default=None, help="chạy 1 nút rồi thoát")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--class", dest="class_name", default=None)
    ap.add_argument("--skip-center", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args(argv)

    st = default_state()
    st["dataset"] = args.dataset or pick_default_dataset()
    st["source"] = args.source or ""
    st["class_name"] = args.class_name or ""
    if args.skip_center:
        st["skip_center"] = True
    if args.skip_train:
        st["skip_train"] = True
    if args.no_export:
        st["export"] = False

    if args.list:
        print_list(st)
        return 0
    if args.run:
        try:
            argv = build_argv(args.run, st)
        except (KeyError, ValueError) as exc:
            print(f"[LỖI] {exc}", file=sys.stderr)
            return 2
        return run_command(argv)
    try:
        return run_tui(st)
    except Exception as exc:  # noqa: BLE001 - thiếu TERM / không phải tty
        print(f"[LỖI] Không khởi động được TUI: {exc}", file=sys.stderr)
        print("Gợi ý: chạy trong terminal thật (export TERM=xterm), hoặc dùng "
              "`--list` để xem lệnh và chạy trực tiếp.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())