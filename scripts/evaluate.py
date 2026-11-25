"""Đánh giá một experiment (run) của framework multi-dataset.

Dùng:
  python scripts/evaluate.py --dataset chili_quality_v2                 # run mới nhất, split test
  python scripts/evaluate.py --dataset chili_quality_v2 --run run_001 --split validation
  python scripts/evaluate.py --dataset chili_quality_v2 --save-probs

Đọc class mapping + manifest TỪ RUN (không hardcode lớp). Lưu:
  experiments/<dataset_id>/<run>/evaluation/<split>_metrics.json
  experiments/<dataset_id>/<run>/evaluation/confusion_matrix_<split>.png
  (tùy chọn) <split>_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from src.config import set_seeds
from src.experiment import list_runs
from src.task4 import make_dataset

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
    ap = argparse.ArgumentParser(description="Đánh giá experiment (multi-dataset)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", default="latest")
    ap.add_argument("--pipeline-config", default=None)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-probs", action="store_true")
    args = ap.parse_args(argv)

    from src.pipeline_config import load_pipeline_config
    pcfg = load_pipeline_config(args.pipeline_config)
    run_dir = resolve_run(args.dataset, args.run, root=pcfg["paths"]["experiments_root"])
    print(f"Run: {run_dir}")

    with open(os.path.join(run_dir, "class_mapping.json"), encoding="utf-8") as f:
        class_mapping = json.load(f)
    classes = [k for k, _ in sorted(class_mapping.items(), key=lambda kv: int(kv[1]))]
    with open(os.path.join(run_dir, "dataset_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    items = manifest.get(args.split, [])
    if not items:
        raise SystemExit(f"[LỖI] Split '{args.split}' rỗng trong run này.")
    paths = [i["path"] for i in items]
    y_true = np.array([int(i["label"]) for i in items])
    print(f"Split {args.split}: {len(items)} ảnh | classes={classes}")

    model_path = os.path.join(run_dir, "model", "best_chili.keras")
    if not os.path.isfile(model_path):
        raise SystemExit(f"[LỖI] Không tìm thấy model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    set_seeds({"training": {"seed": 42}})
    ds = make_dataset(paths, y_true.astype(np.int64).tolist(), args.batch_size, shuffle=False)

    # Đo thời gian inference
    t0 = time.perf_counter()
    y_prob = model.predict(ds, verbose=0)
    infer_total = time.perf_counter() - t0
    y_pred = np.argmax(y_prob, axis=1)

    from sklearn.metrics import (classification_report, confusion_matrix,
                                 precision_recall_fscore_support)
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    accuracy = float((y_pred == y_true).mean())
    macro_f1 = float(np.mean(f1))

    per_image_ms = infer_total / max(1, len(paths)) * 1000.0
    metrics = {
        "dataset_id": args.dataset,
        "run_dir": run_dir,
        "split": args.split,
        "classes": classes,
        "n_samples": int(len(y_true)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]),
                          "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": cm,
        "class_support": {c: int(support[i]) for i, c in enumerate(classes)},
        "inference_total_s": infer_total,
        "inference_ms_per_image_batched": per_image_ms,
        "note": ("DEVELOPMENT split (image-level; pepper identity unknown → có thể leak). "
                 "Không phải benchmark độc lập."),
    }

    out_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"{args.split}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.array(cm), cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {args.dataset} ({args.split})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    cm_path = os.path.join(out_dir, f"confusion_matrix_{args.split}.png")
    fig.savefig(cm_path); plt.close(fig)

    if args.save_probs:
        with open(os.path.join(out_dir, f"{args.split}_predictions.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "true_label", "true_class", "pred_label", "pred_class"]
                       + [f"p_{c}" for c in classes])
            for p, t, pr, pd_ in zip(paths, y_true, y_prob, y_pred):
                w.writerow([p, int(t), classes[int(t)], int(pd_), classes[int(pd_)]]
                           + [f"{v:.6f}" for v in pr])

    print(f"Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")
    for c in classes:
        m = metrics["per_class"][c]
        print(f"  {c:9s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")
    print(f"Confusion: {cm}")
    print(f"Saved: {metrics_path}\n       {cm_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
