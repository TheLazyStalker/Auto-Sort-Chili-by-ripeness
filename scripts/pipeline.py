"""PIPELINE HOÀN CHỈNH: ảnh raw → make-object-to-center → dataset raw → derivative → train → (export).

Dùng:
  # 1 lớp (toàn bộ ảnh trong thư mục thuộc 1 lớp)
  python scripts/pipeline.py --source /path/to/images --class damaged

  # nhiều lớp (thư mục con tên = tên lớp của dataset)
  python scripts/pipeline.py --source /path/to/have_class_subdirs

  # nguồn tải trên mạng (zip)
  python scripts/pipeline.py --source-url https://example.com/imgs.zip --class damaged

  # kiểm thử nhanh, không train
  python scripts/pipeline.py --source /path/to/images --class damaged --skip-train

Nguyên tắc:
- NGUỒN KHÔNG BAO GIỜ BỊ DI CHUYỂN/SỬA — mặc định được COPY vào staging (--no-copy-source để đọc trực tiếp).
- Mọi tham số (màu nền centering, crop, split, training, model) đọc từ
  `pipeline_config.json`; sửa file đó là cả quy trình đổi theo.
- Không ghi đè run/model cũ: mỗi lần train tạo experiment mới.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from src.centering import copy_into_raw, list_images, run_centering
from src.datasets import load_dataset_config, scan_dataset
from src.pipeline_config import (crop_params, load_pipeline_config, model_config,
                                 split_config, training_config)
from src.training import train_experiment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPORTED_IN = (".jpg", ".jpeg", ".png")


class PipelineError(RuntimeError):
    pass


# ------------------------------------------------------------------ source ----

def download_zip(url: str, dest_dir: str) -> str:
    """Tải zip từ URL và giải nén vào dest_dir; trả về thư mục chứa ảnh."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "_download.zip")
    print(f"  tải: {url}")
    urllib.request.urlretrieve(url, zip_path)
    extract_dir = os.path.join(dest_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    os.remove(zip_path)
    # nếu zip bọc trong 1 thư mục duy nhất → đi xuống
    entries = [e for e in os.listdir(extract_dir) if not e.startswith("__MACOSX")]
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
        return os.path.join(extract_dir, entries[0])
    return extract_dir


def acquire_source(args, staging: str) -> str:
    if args.source_url:
        return download_zip(args.source_url, os.path.join(staging, "source"))
    src = os.path.abspath(args.source)
    if not os.path.isdir(src):
        raise PipelineError(f"Nguồn không tồn tại: {src}")
    if args.no_copy_source:
        return src
    dst = os.path.join(staging, "source")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    return dst


def route_classes(src_dir: str, classes: list[str], class_name: str | None) -> dict:
    """Xác định {class: thư_mục_ảnh}. Ưu tiên thư mục con tên lớp."""
    subdirs = {d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))}
    matched = {c: os.path.join(src_dir, c) for c in classes if c in subdirs}
    if matched and not class_name:
        return matched
    if class_name:
        if class_name not in classes:
            raise PipelineError(f"--class '{class_name}' không thuộc {classes}")
        return {class_name: src_dir}
    n = len(list_images(src_dir))
    raise PipelineError(
        f"Không suy ra được lớp: {src_dir} có {n} ảnh nhưng không có thư mục con "
        f"trùng tên lớp {classes}. Hãy truyền --class <tên_lớp>.")


# -------------------------------------------------------------- validation ----

def validate_centered(centered_dir: str, expect_size: int, bg_bgr: list[int]) -> dict:
    """Kiểm tra ảnh đã center: đọc được, đúng kích thước, nền đồng nhất, có object."""
    import numpy as np
    bg = np.array([bg_bgr[2], bg_bgr[1], bg_bgr[0]], dtype=np.float32)  # → RGB
    bad_size, bad_read, no_object, edge_touch, areas = [], [], [], 0, []
    for p in list_images(centered_dir):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.size != (expect_size, expect_size):
                    bad_size.append(os.path.basename(p))
                    continue
                arr = np.asarray(im, dtype=np.float32)
        except Exception:
            bad_read.append(os.path.basename(p))
            continue
        d = np.sqrt(((arr - bg) ** 2).sum(-1))
        mask = d > 30
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            no_object.append(os.path.basename(p))
            continue
        areas.append(float(mask.mean()))
        if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= expect_size - 2 or ys.max() >= expect_size - 2:
            edge_touch += 1
    return {
        "n": len(list_images(centered_dir)),
        "bad_read": bad_read, "bad_size": bad_size, "no_object": no_object,
        "edge_touch": edge_touch,
        "area_frac_min": min(areas) if areas else None,
        "area_frac_max": max(areas) if areas else None,
        "ok": not (bad_read or bad_size or no_object),
    }


# -------------------------------------------------------------------- main ----

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pipeline hoàn chỉnh: raw → center → dataset → train")
    ap.add_argument("--dataset", default=None, help="dataset_id (mặc định theo pipeline_config)")
    ap.add_argument("--source", default=None, help="thư mục ảnh nguồn (sẽ được COPY, không di chuyển)")
    ap.add_argument("--source-url", default=None, help="URL zip chứa ảnh nguồn")
    ap.add_argument("--class", dest="class_name", default=None,
                    help="tên lớp cho toàn bộ ảnh nguồn (khi nguồn không có thư mục con theo lớp)")
    ap.add_argument("--pipeline-config", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-copy-source", action="store_true")
    ap.add_argument("--skip-center", action="store_true")
    ap.add_argument("--skip-derivatives", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--export", action="store_true", help="export TFLite sau khi train")
    ap.add_argument("--limit-per-class", type=int, default=0, help="giới hạn ảnh/lớp (kiểm thử)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.source and not args.source_url:
        ap.error("cần --source <dir> hoặc --source-url <zip-url>")

    pcfg = load_pipeline_config(args.pipeline_config)
    dataset_id = args.dataset or pcfg["dataset"]["default_id"]
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    staging = os.path.join(pcfg["paths"]["staging_root"], f"{dataset_id}_{run_id}")
    os.makedirs(staging, exist_ok=True)

    print("=" * 70)
    print(f"PIPELINE | dataset={dataset_id} | run_id={run_id}")
    print(f"pipeline_config: {pcfg['_source']}")
    print(f"staging: {staging}")
    print("=" * 70)

    report = {"dataset_id": dataset_id, "run_id": run_id,
              "pipeline_config": pcfg["_source"],
              "started_at": datetime.datetime.now().isoformat(timespec="seconds")}

    # --- dataset config ---
    ds = load_dataset_config(dataset_id, root=pcfg["paths"]["datasets_root"])
    ds.preprocessing_pipeline = pcfg["preprocessing"]["pipeline"]
    ds.split_ratios = tuple(split_config(pcfg)["ratios"])
    ds.split_seed = int(split_config(pcfg)["seed"])
    print(f"[1] Dataset classes: {ds.classes} | raw: {ds.raw_dir}")

    # --- source ---
    print("[2] Chuẩn bị nguồn (copy, không di chuyển)...")
    src_dir = acquire_source(args, staging)
    report["source"] = {"original": args.source or args.source_url, "staged": src_dir}
    routing = route_classes(src_dir, ds.classes, args.class_name)
    report["routing"] = {c: d for c, d in routing.items()}
    print(f"    nguồn → lớp: { {c: len(list_images(d)) for c, d in routing.items()} }")

    if args.dry_run:
        print("[DRY RUN] dừng trước khi centering.")
        return 0

    # --- centering ---
    centered_dirs = {}
    if args.skip_center:
        print("[3] Bỏ qua centering (--skip-center); dùng ảnh nguồn trực tiếp.")
        centered_dirs = routing
    else:
        print("[3] make-object-to-center (tham số từ pipeline_config.json)...")
        for cls, d in routing.items():
            out = os.path.join(staging, "centered", cls)
            res = run_centering(d, out, pcfg)
            centered_dirs[cls] = out
            report.setdefault("centering", {})[cls] = {
                "n_input": res["n_input"], "n_output": res["n_output"],
                "missing": res["missing"], "command": " ".join(res["command"]),
            }
    if not centered_dirs:
        raise PipelineError("Không có lớp nào để xử lý.")

    # --- validate ---
    print("[4] Kiểm tra ảnh sau center...")
    size = int(pcfg["centering"]["size"]) if not args.skip_center else None
    validation = {}
    for cls, d in centered_dirs.items():
        if size:
            v = validate_centered(d, size, pcfg["centering"]["background_bgr"])
        else:
            v = {"n": len(list_images(d)), "ok": True}
        validation[cls] = v
        print(f"    {cls}: n={v['n']} ok={v['ok']}"
              + (f" bad_read={len(v.get('bad_read', []))} bad_size={len(v.get('bad_size', []))}"
                 f" no_object={len(v.get('no_object', []))} edge_touch={v.get('edge_touch')}" if size else ""))
        if not v["ok"]:
            raise PipelineError(f"Ảnh sau center của lớp '{cls}' không hợp lệ: {v}")
    report["validation"] = validation

    # --- vào raw ---
    print("[5] Copy vào dataset raw (không ghi đè)...")
    report["raw_import"] = {}
    for cls, d in centered_dirs.items():
        dst = os.path.join(ds.raw_dir, cls)
        res = copy_into_raw(d, dst)
        report["raw_import"][cls] = {"copied": len(res["copied"]),
                                     "skipped_existing": len(res["skipped_existing"])}
        print(f"    {cls}: +{len(res['copied'])} (bỏ qua trùng {len(res['skipped_existing'])})")

    # --- derivative ---
    if args.skip_derivatives:
        print("[6] Bỏ qua sinh derivative (--skip-derivatives).")
    else:
        print("[6] Sinh derivative crop_v1...")
        from scripts.make_dataset_derivatives import generate_derivatives
        report["derivatives"] = generate_derivatives(
            ds, crop_params(pcfg), limit=args.limit_per_class)

    # --- train ---
    if args.skip_train:
        print("[7] Bỏ qua train (--skip-train).")
    else:
        print("[7] Train (experiment mới, không ghi đè)...")
        scan = scan_dataset(ds)
        if scan["empty_classes"]:
            raise PipelineError(f"Thiếu lớp: {scan['empty_classes']} — không train.")
        from scripts.train import build_manifest
        manifest = build_manifest(ds, scan, ds.split_seed)
        print(f"    split: train={len(manifest['train'])} "
              f"val={len(manifest['validation'])} test={len(manifest['test'])}")
        summary = train_experiment(ds, training_config(pcfg), model_config(pcfg),
                                   manifest, seed=ds.split_seed, verbose=False,
                                   experiments_root=pcfg["paths"]["experiments_root"])
        report["training"] = {k: summary[k] for k in
                              ("run_dir", "model_path", "phaseA_best_val_acc",
                               "phaseB_best_val_acc", "classes", "num_classes")}
        print(f"    run: {summary['run_dir']}")
        print(f"    val_acc: A={summary['phaseA_best_val_acc']} B={summary['phaseB_best_val_acc']}")

        if args.export:
            print("[8] Export TFLite (F32+F16)...")
            import subprocess
            run_name = os.path.basename(summary["run_dir"])
            export_cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "export.py"),
                          "--dataset", dataset_id, "--run", run_name,
                          "--pipeline-config", pcfg["_source"]]
            r = subprocess.run(export_cmd, capture_output=True, text=True)
            report["export"] = {"ok": r.returncode == 0,
                                "tail": (r.stdout or r.stderr)[-800:]}
            print(f"    export ok={r.returncode == 0}")

    # --- report ---
    report["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    out = os.path.join(staging, "pipeline_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("=" * 70)
    print(f"XONG. Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
