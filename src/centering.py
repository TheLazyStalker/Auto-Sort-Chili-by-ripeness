"""Chạy make-object-to-center (đã vendor trong tools/) với tham số từ pipeline_config.

Không sửa source của tool; chỉ gọi CLI với tham số lấy từ `pipeline_config.json`
⇒ đổi config (VD màu nền) là bước centering thay đổi theo.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

from .pipeline_config import center_params

SUPPORTED_IN = (".jpg", ".jpeg", ".png")


class CenteringError(RuntimeError):
    """Lỗi bước centering."""


def list_images(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, f) for f in os.listdir(directory)
                  if f.lower().endswith(SUPPORTED_IN))


def build_command(input_dir: str, output_dir: str, params: dict) -> list[str]:
    script = os.path.join(params["tool_dir"], params["script"])
    if not os.path.isfile(script):
        raise CenteringError(f"Không tìm thấy tool: {script}")
    return [
        params["python"], script,
        "--input", input_dir, "--output", output_dir,
        "--size", str(params["size"]),
        "--bg", params["bg"],
        "--fruit-frac", str(params["fruit_frac"]),
        "--pad", str(params["pad"]),
        "--ext", params["ext"],
        "--suffix", params["suffix"],
    ]


def run_centering(input_dir: str, output_dir: str, pipeline_cfg: dict,
                  verbose: bool = True) -> dict:
    """Chạy centering cho toàn bộ ảnh trong input_dir → output_dir.

    Trả về {ok, n_input, n_output, failed, command, stdout_tail, report}.
    """
    params = center_params(pipeline_cfg)
    if not params["enabled"]:
        raise CenteringError("centering.enabled = false trong pipeline_config.json")

    inputs = list_images(input_dir)
    if not inputs:
        raise CenteringError(f"Không có ảnh trong {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    cmd = build_command(input_dir, output_dir, params)
    if verbose:
        print("  $ " + " ".join(cmd))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=params["timeout_s"])
    except subprocess.TimeoutExpired as exc:
        raise CenteringError(f"Centering timeout sau {params['timeout_s']}s") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise CenteringError(f"Centering thất bại (exit {proc.returncode}):\n{tail}")

    outs = list_images(output_dir)
    missing = [os.path.basename(p) for p in inputs
               if not os.path.isfile(os.path.join(
                   output_dir, os.path.splitext(os.path.basename(p))[0] + params["ext"]))]
    result = {
        "ok": len(outs) > 0,
        "n_input": len(inputs),
        "n_output": len(outs),
        "missing": missing,
        "command": cmd,
        "stdout_tail": (proc.stdout or "")[-1500:],
    }
    if verbose:
        print(f"  centering: {len(inputs)} vào → {len(outs)} ra"
              + (f" (thiếu {len(missing)})" if missing else ""))
    return result


def copy_into_raw(centered_dir: str, raw_class_dir: str, ext: str = ".jpg") -> dict:
    """Copy ảnh đã center vào raw/<class>/ (KHÔNG bao giờ di chuyển/ghi đè)."""
    os.makedirs(raw_class_dir, exist_ok=True)
    copied, skipped = [], []
    for p in list_images(centered_dir):
        dst = os.path.join(raw_class_dir, os.path.basename(p))
        if os.path.exists(dst):
            skipped.append(os.path.basename(p))
            continue
        shutil.copy2(p, dst)
        copied.append(os.path.basename(p))
    return {"copied": copied, "skipped_existing": skipped}
