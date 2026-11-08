"""Task 4 — huấn luyện MobileNetV3Small (development) trên derivative crop_v1.

Dùng:
  python scripts/train_task4.py [--config config.yaml] [--seed 42]
                                [--no-class-weight] [--epochs-a 30] [--epochs-b 20]
                                [--device /cpu:0]

Quy trình:
1. Đọc manifest Task 3 (train-only; 463 derivative) và verify counts/label.
2. Tạo dev split 80/20 (image-level, stratified, seed cố định) — LEAKY, chỉ dev.
3. Build MobileNetV3Small (ImageNet, include_preprocessing=True).
4. Phase A: freeze backbone → train head (EarlyStopping + ModelCheckpoint).
5. Phase B: fine-tune N layer cuối với lr thấp.
6. Ghi artifact JSON (config, class mapping, split counts, history, best epochs).

KHÔNG: train final benchmark, không TFLite, không chỉnh sửa raw/Task 3 manifest.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from src.config import load_config, set_seeds
from src.task4 import (
    Task4DataError,
    apply_augmentation,
    build_augmentation,
    create_dev_split,
    load_task3_manifest,
    make_dataset,
    plot_curves,
    save_json,
    train_two_phases,
    verify_manifest_class_counts,
)
from src.model import build_model, normalize_base_name
from src.utils import ensure_dir

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 4 — train MobileNetV3Small (development)")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--epochs-a", type=int, default=None)
    parser.add_argument("--epochs-b", type=int, default=None)
    parser.add_argument("--device", default="/cpu:0")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["project"]["classes"]
    base_name = normalize_base_name(cfg["model"].get("base", "mobilenetv3small"))
    print(f"Base model: {base_name} | classes: {classes} | seed: {args.seed}")

    # ---------- 1. Dataset verification ----------
    manifest = load_task3_manifest(cfg)
    train_items = manifest.get("train", [])
    print(f"Manifest Task 3 train: {len(train_items)} ảnh")
    available = verify_manifest_class_counts(manifest, classes)
    print("Class counts (include): " + ", ".join(f"{c}={available[c]}" for c in classes))
    if any(available[c] == 0 for c in classes):
        raise Task4DataError("Thiếu lớp trong dữ liệu — dừng.")

    # ---------- 2. Dev split 80/20 ----------
    dev = create_dev_split(manifest, classes, seed=args.seed)
    n_train, n_val = len(dev["train"]), len(dev["validation"])
    print(f"Dev split 80/20 (leaky, image-level): train={n_train} val={n_val}")
    for c in classes:
        nc = sum(1 for i in dev["train"] if classes[int(i["label"])] == c)
        nv = sum(1 for i in dev["validation"] if classes[int(i["label"])] == c)
        print(f"  {c}: train={nc} val={nv}")

    # ---------- 3. tf.data ----------
    bs = int(cfg["data"].get("batch_size", 32))
    train_paths = [i["path"] for i in dev["train"]]
    train_labels = [int(i["label"]) for i in dev["train"]]
    val_paths = [i["path"] for i in dev["validation"]]
    val_labels = [int(i["label"]) for i in dev["validation"]]

    train_ds = make_dataset(train_paths, train_labels, bs, shuffle=True, seed=args.seed)
    val_ds = make_dataset(val_paths, val_labels, bs, shuffle=False)
    aug = build_augmentation(seed=args.seed)
    train_ds = apply_augmentation(train_ds, aug)

    # ---------- 4. Model + training ----------
    set_seeds(cfg)
    with tf.device(args.device):
        model = build_model(cfg, trainable_backbone=False)

    out_dir = cfg["paths"]["artifacts_task4"]
    ensure_dir(out_dir)

    use_cw = not args.no_class_weight and cfg.get("training", {}).get("class_weight", "balanced") != "null"
    if args.epochs_a is not None:
        cfg.setdefault("training", {})["stage1_epochs"] = args.epochs_a
    if args.epochs_b is not None:
        cfg.setdefault("training", {})["stage2_epochs"] = args.epochs_b

    result = train_two_phases(cfg, model, train_ds, val_ds, out_dir,
                              use_class_weight=use_cw)

    # ---------- 5. Artifacts ----------
    hist = result["history"]
    merged = {}
    for phase, h in hist.items():
        if h is None:
            continue
        for k, v in h.items():
            key = f"{phase}_{k}"
            merged[key] = v

    curve_path = os.path.join(out_dir, "training_curves.png")
    plot_curves(hist, curve_path)

    model.save(os.path.join(out_dir, "best_chili.keras"))
    print(f"Đã lưu best model (phase B): {os.path.join(out_dir, 'best_chili.keras')}")

    # config artifact
    cfg_artifact = {
        "base": base_name,
        "classes": classes,
        "num_classes": len(classes),
        "seed": args.seed,
        "batch_size": bs,
        "epochs_a": args.epochs_a or cfg["training"].get("stage1_epochs", 30),
        "epochs_b": args.epochs_b or cfg["training"].get("stage2_epochs", 20),
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "class_weight": use_cw,
        "augmentation": "hflip_vflip_rot0.1_zoom0.1_trans0.05_bright0.05 (no hue/sat)",
        "fine_tune_layers": cfg["model"].get(
            "fine_tune_layers_v3small", cfg["model"].get("fine_tune_layers", 20)),
        "manifest_source": "dataset/split_manifest_task3.json",
        "dev_split_ratio": "80/20 image-level (leaky, seed 42)",
        "preprocessing": "MobileNetV3Small include_preprocessing=True (Rescaling 1/128 offset -1); input RGB [0,255]",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "best_epoch_a": result["best_epoch_a"],
        "best_epoch_b": result["best_epoch_b"],
    }
    save_json(cfg_artifact, os.path.join(out_dir, "training_config.json"))
    save_json({"class_mapping": {c: i for i, c in enumerate(classes)}},
              os.path.join(out_dir, "class_mapping.json"))
    save_json({
        "phaseA_best_val_acc": max(hist["phase_a"]["val_accuracy"]),
        "phaseB_best_val_acc": max(hist["phase_b"]["val_accuracy"]),
        "phaseA_best_epoch": result["best_epoch_a"],
        "phaseB_best_epoch": result["best_epoch_b"],
        "final_val_acc": hist["phase_b"]["val_accuracy"][-1],
        "final_val_loss": hist["phase_b"]["val_loss"][-1],
    }, os.path.join(out_dir, "training_history_summary.json"))
    save_json(merged, os.path.join(out_dir, "training_history.json"))

    # dev manifest as separate artifact
    dev_artifact = {
        k: [{"path": i["path"], "label": i["label"], "class": classes[int(i["label"])],
             "group_id": i.get("group_id", ""), "image_id": i.get("image_id", ""),
             "source_path": i.get("source_path", "")} for i in v]
        for k, v in dev.items()
    }
    save_json(dev_artifact, os.path.join(out_dir, "dev_split_manifest.json"))

    print("\n===== TRAINING COMPLETE =====")
    print(f"Phase A best val_acc: {max(hist['phase_a']['val_accuracy']):.4f} "
          f"(epoch {result['best_epoch_a']})")
    print(f"Phase B best val_acc: {max(hist['phase_b']['val_accuracy']):.4f} "
          f"(epoch {result['best_epoch_b']})")
    print(f"Artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
