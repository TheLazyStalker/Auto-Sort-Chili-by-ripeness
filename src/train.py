"""Pipeline huấn luyện 2 giai đoạn (freeze backbone → fine-tune các layer cuối).

Dùng: python -m src.train [--config config.yaml] [--force-resplit]
Nếu dataset trống → thông báo rõ ràng và dừng an toàn (không crash khó hiểu).
"""
from __future__ import annotations

import os
import sys

import tensorflow as tf

from .config import load_config, set_seeds
from .dataset import DatasetEmptyError, build_tf_datasets, prepare_datasets
from .model import build_model, compile_model, freeze_backbone, get_callbacks
from .utils import compute_class_weight, ensure_dir, plot_training_curves, save_metrics_json


def run_training(cfg: dict, force_resplit: bool = False) -> dict:
    set_seeds(cfg)
    print("Kiểm tra & chuẩn bị dataset...")
    try:
        summary = prepare_datasets(cfg, force_resplit=force_resplit)
    except DatasetEmptyError as exc:
        print(f"[ĐỪNG HUẤN LUYỆN] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"Thống kê lớp (raw): {summary['counts']}")
    print(f"Tổng ảnh hợp lệ: {summary['total']} | trùng nội dung: {len(summary['duplicates'])} "
          f"| ảnh lỗi: {len(summary['corrupt'])}")
    print(f"Chia tập sẵn: {summary['splits']}")

    datasets = build_tf_datasets(cfg)
    train_ds, val_ds = datasets["train"], datasets["validation"]

    # Class weights từ manifest tập train
    from .dataset import load_manifest

    manifest = load_manifest(cfg["paths"]["split_manifest"])
    train_labels = [i["label"] for i in manifest["train"]]
    import numpy as np

    class_weight = compute_class_weight(np.array(train_labels), int(cfg["project"]["num_classes"]))
    print(f"Class weights: {class_weight}")

    t = cfg.get("training", {})
    model = build_model(cfg, trainable_backbone=False)
    freeze_backbone(model, fine_tune_layers=0)
    compile_model(model, learning_rate=float(t.get("stage1_lr", 0.001)))

    print("\n=== Giai đoạn 1: freeze backbone, train head ===")
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=int(t.get("stage1_epochs", 10)),
        callbacks=get_callbacks(cfg), class_weight=class_weight, verbose=2,
    )

    # Giai đoạn 2: fine-tune
    print("\n=== Giai đoạn 2: fine-tune các layer cuối ===")
    freeze_backbone(model, fine_tune_layers=int(cfg["model"].get("fine_tune_layers", 30)))
    compile_model(model, learning_rate=float(t.get("stage2_lr", 0.0001)))
    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=int(t.get("stage2_epochs", 20)),
        callbacks=get_callbacks(cfg), class_weight=class_weight, verbose=2,
    )

    # Gộp history để vẽ
    merged = {k: list(hist1.history[k]) + list(hist2.history[k]) for k in hist1.history}
    merged.update({k: v for k, v in hist2.history.items() if k not in merged})
    for k, v in hist2.history.items():
        if k in hist1.history and isinstance(v, list):
            merged[k] = list(hist1.history[k]) + list(v)

    ensure_dir(cfg["paths"]["checkpoints"])
    ensure_dir(cfg["paths"]["reports_figures"])
    ensure_dir(cfg["paths"]["reports_metrics"])

    final_path = os.path.join(cfg["paths"]["checkpoints"], "chili_final.keras")
    model.save(final_path)
    print(f"Đã lưu model cuối: {final_path}")

    curve_path = os.path.join(cfg["paths"]["reports_figures"], "training_curves.png")
    plot_training_curves(merged, curve_path)
    print(f"Đã lưu đường cong huấn luyện: {curve_path}")

    metrics_path = os.path.join(cfg["paths"]["reports_metrics"], "training_summary.json")
    save_metrics_json({
        "stage1_epochs": int(t.get("stage1_epochs", 10)),
        "stage2_epochs": int(t.get("stage2_epochs", 20)),
        "class_weight": {str(k): v for k, v in class_weight.items()},
        "final_loss": float(hist2.history.get("loss", [0])[-1]),
        "final_accuracy": float(hist2.history.get("accuracy", [0])[-1]),
        "final_val_accuracy": float(hist2.history.get("val_accuracy", [0])[-1]),
        "history": merged,
    }, metrics_path)
    print(f"Đã lưu tóm tắt: {metrics_path}")
    return {"model_path": final_path, "curves": curve_path, "metrics": metrics_path}


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Huấn luyện MobileNetV2 phân loại độ chín ớt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-resplit", action="store_true", help="Chia lại tập ngay cả khi đã có manifest")
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    run_training(cfg, force_resplit=args.force_resplit)


if __name__ == "__main__":
    main()
