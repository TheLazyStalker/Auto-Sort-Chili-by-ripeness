"""Export TFLite (Float32 + Float16, KHÔNG INT8) cho một experiment của framework.

Dùng:
  python scripts/export.py --dataset chili_quality_v2                 # run mới nhất
  python scripts/export.py --dataset chili_quality_v2 --run run_001
  python scripts/export.py --dataset chili_quality_v2 --out exports/my_model

Mặc định ghi vào `<run_dir>/export/` — KHÔNG đụng model/export cũ
(artifacts/task5, deployment/task6/models, models/ giữ nguyên).

Đọc class mapping TỪ RUN (không hardcode lớp).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import yaml

from src.experiment import list_runs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")


def resolve_run(dataset_id: str, run: str | None, root: str | None = None) -> str:
    runs = list_runs(dataset_id, root=root)
    if not runs:
        raise SystemExit(f"[LỖI] Chưa có run nào cho dataset '{dataset_id}'.")
    if not run or run == "latest":
        run = runs[-1]
    if run not in runs:
        raise SystemExit(f"[LỖI] Không có run '{run}'. Có: {runs}")
    return os.path.join(root or EXPERIMENTS_ROOT, dataset_id, run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export TFLite cho experiment (multi-dataset)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", default="latest")
    ap.add_argument("--out", default=None, help="thư mục đích (mặc định <run>/export)")
    ap.add_argument("--pipeline-config", default=None,
                    help="pipeline_config.json (nguồn experiments_root)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from src.pipeline_config import load_pipeline_config
    pcfg = load_pipeline_config(args.pipeline_config)
    run_dir = resolve_run(args.dataset, args.run, root=pcfg["paths"]["experiments_root"])
    out_dir = args.out or os.path.join(run_dir, "export")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Run: {run_dir}\nOut: {out_dir}")

    with open(os.path.join(run_dir, "class_mapping.json"), encoding="utf-8") as f:
        class_mapping = json.load(f)
    classes = [k for k, _ in sorted(class_mapping.items(), key=lambda kv: int(kv[1]))]

    image_size = 224
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            image_size = int((yaml.safe_load(f) or {}).get("dataset", {}).get("image_size", 224))

    model_path = os.path.join(run_dir, "model", "best_chili.keras")
    if not os.path.isfile(model_path):
        raise SystemExit(f"[LỖI] Không tìm thấy model: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"Classes: {classes} | image_size={image_size} | params={model.count_params()}")

    # Reuse converter helpers (SavedModel tạm → convert) — không đụng models/
    from scripts.convert_task5 import convert_from_savedmodel, validate_tflite_bytes

    tmp_dir = os.path.join(out_dir, "_tmp_saved_model")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    saved_dir = os.path.join(tmp_dir, "saved_model")
    model.export(saved_dir)

    import numpy as np

    dummy = np.random.RandomState(args.seed).uniform(
        0, 255, size=(1, image_size, image_size, 3)).astype(np.float32)

    results = {"source_model": model_path, "classes": classes,
               "preprocessing": "crop_v1 (RGB [0,255] → model nội bộ Rescaling 1/128, -1)",
               "int8": "removed by decision (Task 5)", "models": {}}
    for dtype, fname in (("float32", "chili_float32.tflite"),
                         ("float16", "chili_float16.tflite")):
        data = convert_from_savedmodel(saved_dir, dtype)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "wb") as f:
            f.write(data)
        checks = validate_tflite_bytes(data, input_shape=(1, image_size, image_size, 3),
                                       num_classes=len(classes), dummy_input=dummy)
        results["models"][dtype] = {"path": out_path, "size_bytes": len(data),
                                    "validate": checks}
        print(f"{dtype}: {fname} | {len(data)/1e6:.3f} MB | ok={checks.get('inference_ok')} "
              f"finite={checks.get('finite')} sum1={checks.get('prob_sum_approx_1')}")

    # labels + metadata + class mapping cho deployment
    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for c in classes:
            f.write(f"{c}\n")
    with open(os.path.join(out_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset_id": args.dataset, "run_dir": run_dir,
            "input": {"shape": [1, image_size, image_size, 3], "type": "RGB",
                      "dtype": "float32 [0,255]"},
            "output": {"shape": [1, len(classes)], "type": "softmax", "labels": classes},
            "preprocessing": "crop_v1; KHÔNG chia 255; model tự Rescaling(1/128, -1)",
        }, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "export_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\nĐã export vào: {out_dir}")
    print("Model/export CŨ không bị thay đổi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
