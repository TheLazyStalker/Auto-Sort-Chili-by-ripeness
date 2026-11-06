"""Script kiểm tra dataset: quét raw, thống kê từng lớp, ảnh lỗi/trùng.

Dùng: python scripts/check_dataset.py [--config config.yaml]
Không cần có ảnh thật — báo rõ ràng khi dataset trống.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.dataset import DatasetEmptyError, prepare_datasets, scan_raw_dataset  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Kiểm tra dataset ớt")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--prepare", action="store_true", help="Tạo manifest + chia tập nếu dataset đủ")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["project"]["classes"]
    summary = scan_raw_dataset(cfg["paths"]["dataset_raw"], classes)

    print(f"Thư mục raw: {cfg['paths']['dataset_raw']}")
    print("\nThống kê theo lớp:")
    for c in classes:
        print(f"  {c:12s}: {summary.counts[c]} ảnh")
    print(f"  {'TOTAL':12s}: {summary.total} ảnh")
    print(f"\nẢnh lỗi/thiệt nhận (decode thất bại): {len(summary.corrupt)}")
    for p in summary.corrupt[:10]:
        print(f"  - {p}")
    print(f"Ảnh trùng nội dung (md5): {len(summary.duplicates)} nhóm")
    for h, ps in list(summary.duplicates.items())[:5]:
        print(f"  md5={h[:8]}... → {len(ps)} file: {ps[:3]}")

    if summary.total == 0:
        print("\n[CANH BÁO] Dataset trống — hãy đặt ảnh vào dataset/raw/<class>/.")
        sys.exit(0)

    if args.prepare:
        try:
            result = prepare_datasets(cfg, copy_to_split_dirs=True)
            print(f"\nChuẩn bị dataset xong: chia tập {result['splits']}")
            print(f"Manifest: {result['manifest']}")
        except DatasetEmptyError as exc:
            print(f"\n[LỖI] {exc}")
            sys.exit(2)


if __name__ == "__main__":
    main()
