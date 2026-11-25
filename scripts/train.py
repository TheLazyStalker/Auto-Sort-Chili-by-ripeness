"""Multi-dataset training CLI.

Dùng:
  python scripts/train.py --dataset chili_quality_v2
  python scripts/train.py --dataset legacy_ripeness_v1
  python scripts/train.py --dataset chili_quality_v2 --config configs/training/default.yaml
  python scripts/train.py --dataset chili_quality_v2 --model-config configs/models/mobilenetv3small.yaml

Mỗi lần train tạo experiment: experiments/<dataset_id>/run_NNN/ (không ghi đè).
Dataset trống/thiếu lớp → dừng an toàn (không train).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.datasets import DatasetConfigError, list_dataset_ids, load_dataset_config, validate_dataset
from src.training import TrainingError, train_experiment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_CONFIG = os.path.join(PROJECT_ROOT, "configs", "training", "default.yaml")
DEFAULT_MODEL_CONFIG = os.path.join(PROJECT_ROOT, "configs", "models", "mobilenetv3small.yaml")


def load_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_dataset(dataset_id: str, datasets_root: str | None = None) -> tuple[object, dict]:
    """Trả (dataset_cfg, scan) — dừng rõ nếu dataset không tồn tại."""
    try:
        cfg = load_dataset_config(dataset_id, root=datasets_root)
    except DatasetConfigError as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        sys.exit(2)
    v = validate_dataset(cfg)
    for w in v["warnings"]:
        print(f"[CẢNH BÁO] {w}")
    if not v["ok"]:
        print("[LỖI] Dataset không hợp lệ để train:", file=sys.stderr)
        for e in v["errors"]:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(2)
    return cfg, v["scan"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train multi-dataset (framework)")
    parser.add_argument("--dataset", required=True, help="dataset_id (VD chili_quality_v2)")
    parser.add_argument("--config", default=None,
                        help="training config yaml (override pipeline_config.json)")
    parser.add_argument("--model-config", default=None,
                        help="model config yaml (override pipeline_config.json)")
    parser.add_argument("--pipeline-config", default=None,
                        help="pipeline_config.json (mặc định: gốc project)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="chỉ kiểm tra dataset + manifest, không train")
    args = parser.parse_args(argv)

    from src.pipeline_config import load_pipeline_config
    pcfg = load_pipeline_config(args.pipeline_config)
    train_cfg = dict(pcfg["training"])
    model_cfg = dict(pcfg["model"])
    # YAML (nếu được truyền rõ) override JSON
    if args.config:
        train_cfg.update(load_yaml(args.config))
    if args.model_config:
        model_cfg.update(load_yaml(args.model_config))
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    seed = int(train_cfg.get("seed", 42))

    print(f"pipeline_config: {pcfg['_source']}")
    print(f"Dataset IDs có sẵn: {list_dataset_ids()}")
    dataset_cfg, scan = resolve_dataset(args.dataset, pcfg["paths"]["datasets_root"])
    # split ratios/seed lấy từ pipeline_config.json (một nguồn duy nhất)
    dataset_cfg.split_ratios = tuple(float(r) for r in pcfg["split"]["ratios"])
    dataset_cfg.split_seed = int(pcfg["split"]["seed"])
    print(f"Dataset: {args.dataset} | classes={dataset_cfg.classes} "
          f"| num_classes={dataset_cfg.num_classes} | total={scan['total']} ảnh")
    print(f"Split ratios={dataset_cfg.split_ratios} seed={dataset_cfg.split_seed}")

    # Manifest: nếu chưa có split manifest cho dataset này, build từ scan (split 70/15/15)
    try:
        manifest = build_manifest(dataset_cfg, scan, dataset_cfg.split_seed)
    except TrainingError as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"[DRY RUN] Sẽ tạo run mới dưới experiments/{dataset_cfg.dataset_id}/; "
              f"train={len(manifest['train'])} val={len(manifest['validation'])} "
              f"test={len(manifest['test'])}")
        return 0

    try:
        summary = train_experiment(dataset_cfg, train_cfg, model_cfg, manifest,
                                   seed=seed,
                                   experiments_root=pcfg["paths"]["experiments_root"])
    except TrainingError as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        return 2

    print(f"\nDone. Run: {summary['run_dir']}")
    print(f"  model: {summary['model_path']}")
    print(f"  history: {summary['history_path']}")
    return 0


def build_manifest(dataset_cfg, scan: dict, seed: int) -> dict:
    """Tạo split manifest (train/validation/test) cho dataset.

    Thứ tự ưu tiên:
    1. `<derived>/<pipeline>/metadata.csv` (chuẩn framework): chỉ lấy crop_status
       == "ok"; nếu cột `split` có giá trị thì tôn trọng, nếu trống thì chia
       stratified theo `split_ratios`.
    2. Layout cũ `<derived>/<pipeline>/<split>/<class>/*.png` (legacy Task 3).
    3. Layout `<derived>/<pipeline>/<class>/*.png` → chia stratified.

    Không tự crop raw (không đổi preprocessing pipeline ngoài ý muốn).
    """
    import csv as _csv

    classes = dataset_cfg.classes
    der_root = os.path.join(dataset_cfg.derived_dir, dataset_cfg.preprocessing_pipeline)
    splits = ("train", "validation", "test")

    # ---- 1) metadata.csv ----
    meta_path = os.path.join(der_root, "metadata.csv")
    if os.path.isfile(meta_path):
        by_class: dict[str, list] = {c: [] for c in classes}
        explicit: dict[str, list] = {s: [] for s in splits}
        seen_explicit = False
        with open(meta_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if (row.get("crop_status") or "ok").strip() != "ok":
                    continue
                p = (row.get("output_path") or "").strip()
                if not p or not os.path.isfile(p):
                    continue
                cls = (row.get("class") or "").strip()
                if cls not in classes:
                    try:
                        cls = classes[int(row.get("label"))]
                    except (TypeError, ValueError, IndexError):
                        continue
                sp = (row.get("split") or "").strip()
                if sp in splits:
                    explicit[sp].append({"path": p, "label": classes.index(cls)})
                    seen_explicit = True
                else:
                    by_class[cls].append({"path": p, "label": classes.index(cls)})
        if seen_explicit and not any(by_class.values()):
            return explicit
        if any(by_class.values()):
            return _stratified_split(by_class, dataset_cfg, seed)

    # ---- 2) split-subdir layout (legacy) ----
    der = {s: [] for s in splits}
    has_split_dirs = False
    for split in splits:
        for cls in classes:
            d = os.path.join(der_root, split, cls)
            if os.path.isdir(d):
                has_split_dirs = True
                for fname in sorted(os.listdir(d)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        der[split].append({"path": os.path.join(d, fname),
                                           "label": classes.index(cls)})
    if has_split_dirs:
        return der

    # ---- 3) class-only layout ----
    by_class = {c: [] for c in classes}
    for cls in classes:
        d = os.path.join(der_root, cls)
        if os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    by_class[cls].append({"path": os.path.join(d, fname),
                                          "label": classes.index(cls)})
    if any(by_class.values()):
        return _stratified_split(by_class, dataset_cfg, seed)

    raise TrainingError(
        f"Dataset '{dataset_cfg.dataset_id}' chưa có derivative tại {der_root}. "
        "Chạy: python scripts/make_dataset_derivatives.py --dataset "
        f"{dataset_cfg.dataset_id}")


def _stratified_split(by_class: dict, dataset_cfg, seed: int) -> dict:
    """Chia stratified theo lớp với tỷ lệ từ config (mặc định 70/15/15)."""
    import random

    rng = random.Random(seed)
    r1, r2 = dataset_cfg.split_ratios[0], dataset_cfg.split_ratios[1]
    out = {"train": [], "validation": [], "test": []}
    for _cls, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_tr = int(round(n * r1))
        n_va = int(round(n * r2))
        out["train"] += items[:n_tr]
        out["validation"] += items[n_tr:n_tr + n_va]
        out["test"] += items[n_tr + n_va:]
    for s in out:
        rng.shuffle(out[s])
    return out


if __name__ == "__main__":
    sys.exit(main())
