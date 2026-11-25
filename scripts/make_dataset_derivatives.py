"""Sinh derivative crop_v1 cho MỘT dataset bất kỳ (multi-dataset aware).

Dùng:
  python scripts/make_dataset_derivatives.py --dataset chili_quality_v2

Đọc `datasets/<dataset_id>/dataset_config.yaml` + `pipeline_config.json` (tham số
crop), quét `raw/<class>/`, áp dụng `crop_v1` (src/crop.py — canonical pipeline), ghi:
  <derived>/<pipeline>/<class>/<stem>.png
  <derived>/<pipeline>/metadata.csv
  <derived>/<pipeline>/report.json

Không sửa raw. Không tự đổi nhãn. Bỏ qua ảnh trong exclusion list (nếu dataset khai
báo `paths.exclusions`). Ghi rõ crop_status (ok/needs_review/failed).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from src.crop import crop_v1, load_image_rgb
from src.datasets import DatasetConfigError, load_dataset_config, scan_dataset

METADATA_COLUMNS = [
    "source_path", "source_sha256", "crop_version", "class", "label",
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "pad_frac",
    "output_path", "output_sha256", "width", "height",
    "crop_status", "background_rgb", "threshold", "split", "notes",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_excluded(path: str) -> set:
    if not path or not os.path.isfile(path):
        return set()
    out = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("status") or "").strip().lower() == "excluded":
                out.add((row.get("filename") or "").strip())
    return out


def generate_derivatives(ds, crop_params: dict, excluded: set | None = None,
                         limit: int = 0, verbose: bool = True) -> dict:
    """Sinh derivative cho dataset `ds`; trả về report dict."""
    excluded = excluded if excluded is not None else load_excluded(ds.exclusions_path)
    scan = scan_dataset(ds)
    der_root = os.path.join(ds.derived_dir, ds.preprocessing_pipeline)

    rows, status_counts, skipped = [], {}, 0
    for cls in ds.classes:
        out_dir = os.path.join(der_root, cls)
        os.makedirs(out_dir, exist_ok=True)
        files = scan["files"].get(cls, [])
        if limit:
            files = files[:limit]
        for src in files:
            fname = os.path.basename(src)
            if fname in excluded:
                skipped += 1
                continue
            res = crop_v1(load_image_rgb(src), **crop_params)
            status_counts[res.status] = status_counts.get(res.status, 0) + 1
            out_path, out_sha = "", ""
            if res.image is not None:
                out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".png")
                Image.fromarray(res.image, mode="RGB").save(out_path, format="PNG")
                out_sha = sha256_file(out_path)
            rows.append({
                "source_path": src, "source_sha256": sha256_file(src),
                "crop_version": ds.preprocessing_pipeline,
                "class": cls, "label": ds.classes.index(cls),
                "bbox_x0": res.bbox[0] if res.bbox else "",
                "bbox_y0": res.bbox[1] if res.bbox else "",
                "bbox_x1": res.bbox[2] if res.bbox else "",
                "bbox_y1": res.bbox[3] if res.bbox else "",
                "pad_frac": f"{res.pad_frac:.4f}",
                "output_path": out_path, "output_sha256": out_sha,
                "width": res.image.shape[1] if res.image is not None else "",
                "height": res.image.shape[0] if res.image is not None else "",
                "crop_status": res.status,
                "background_rgb": ",".join(str(v) for v in res.background_rgb),
                "threshold": f"{res.threshold:.1f}",
                "split": "", "notes": res.reason,
            })

    os.makedirs(der_root, exist_ok=True)
    meta_path = os.path.join(der_root, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    report = {
        "dataset_id": ds.dataset_id,
        "pipeline": ds.preprocessing_pipeline,
        "crop_params": crop_params,
        "raw_counts": scan["counts"],
        "processed": len(rows),
        "skipped_excluded": skipped,
        "status_counts": status_counts,
        "per_class": {c: sum(1 for r in rows if r["class"] == c) for c in ds.classes},
        "metadata_csv": meta_path,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    rep_path = os.path.join(der_root, "report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"  derivative: {len(rows)} ảnh | status={status_counts}")
        print(f"  per class: {report['per_class']} (bỏ qua {skipped} excluded)")
        if status_counts.get("failed"):
            print(f"  [CẢNH BÁO] {status_counts['failed']} ảnh crop failed")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sinh derivative crop_v1 cho dataset")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pipeline-config", default=None,
                    help="pipeline_config.json (nguồn tham số crop; mặc định gốc project)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N ảnh/lớp (debug)")
    args = ap.parse_args(argv)

    from src.pipeline_config import crop_params, load_pipeline_config
    pcfg = load_pipeline_config(args.pipeline_config)
    cparams = crop_params(pcfg)

    try:
        ds = load_dataset_config(args.dataset)
    except DatasetConfigError as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        return 2
    ds.preprocessing_pipeline = pcfg["preprocessing"]["pipeline"]

    print(f"Dataset: {ds.dataset_id} | classes={ds.classes}")
    print(f"pipeline_config: {pcfg['_source']}")
    print(f"Crop params: {cparams}")
    print(f"Raw: {ds.raw_dir}\nDerived: {os.path.join(ds.derived_dir, ds.preprocessing_pipeline)}")
    report = generate_derivatives(ds, cparams, limit=args.limit)
    print(f"\nmetadata: {report['metadata_csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
