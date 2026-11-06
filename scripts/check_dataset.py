"""Script kiểm tra dataset: quét raw, thống kê từng lớp, ảnh lỗi/trùng.

Dùng:
  python scripts/check_dataset.py [--config config.yaml]
  python scripts/check_dataset.py --prepare            # tạo manifest + chia tập
  python scripts/check_dataset.py --prepare --force-resplit  # chia lại từ đầu
  python scripts/check_dataset.py --report            # ghi báo cáo JSON
Không cần có ảnh thật — báo rõ ràng khi dataset trống.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    DatasetEmptyError,
    prepare_datasets,
    scan_raw_dataset,
    write_scan_report,
    generate_groups_template,
)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

# Số ảnh tối thiểu khuyến nghị cho mỗi lớp (để chia đủ 3 tập có ý nghĩa).
RECOMMENDED_MIN_PER_CLASS = 30


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Kiểm tra dataset ớt")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--prepare", action="store_true", help="Tạo manifest + chia tập nếu dataset đủ")
    parser.add_argument("--force-resplit", action="store_true",
                        help="Bỏ qua manifest hiện có và chia lại từ đầu")
    parser.add_argument("--report", metavar="PATH", nargs="?", const="",
                        help="Ghi báo cáo quét dataset ra JSON (mặc định: reports/metrics/dataset_scan.json)")
    parser.add_argument("--groups-template", metavar="PATH", nargs="?", const="",
                        help="Sinh template dataset/groups.csv từ các ảnh hiện có (group_id trống)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["project"]["classes"]
    summary = scan_raw_dataset(cfg["paths"]["dataset_raw"], classes)

    print(f"Thư mục raw: {cfg['paths']['dataset_raw']}")
    print("\nThống kê theo lớp:")
    for c in classes:
        print(f"  {c:12s}: {summary.counts[c]} ảnh")
    print(f"  {'TOTAL':12s}: {summary.total} ảnh")
    print(f"\nẢnh lỗi/không đọc được (decode thất bại): {len(summary.corrupt)}")
    for p in summary.corrupt[:10]:
        print(f"  - {p}")
    print(f"Ảnh trùng nội dung (md5): {len(summary.duplicates)} nhóm")
    for h, ps in list(summary.duplicates.items())[:5]:
        print(f"  md5={h[:8]}... → {len(ps)} file: {ps[:3]}")

    # Cảnh báo số lượng khuyến nghị
    low = {c: n for c, n in summary.counts.items() if n < RECOMMENDED_MIN_PER_CLASS}
    if low:
        print(f"\n[CẢNH BÁO] Các lớp có ít hơn {RECOMMENDED_MIN_PER_CLASS} ảnh "
              f"(khuyến nghị tối thiểu mỗi lớp): {low}")

    if summary.total == 0:
        print("\n[CẢNH BÁO] Dataset trống — hãy đặt ảnh vào dataset/raw/<class>/.")
        if args.report is not None:
            rep = write_scan_report(summary, _resolve_report_path(args.report, cfg), classes)
            print(f"Đã ghi báo cáo: {rep}")
        sys.exit(0)

    if args.report is not None:
        rep = write_scan_report(summary, _resolve_report_path(args.report, cfg), classes)
        print(f"Đã ghi báo cáo: {rep}")

    if args.groups_template is not None:
        from src.dataset import generate_groups_template
        out = _resolve_groups_template_path(args.groups_template, cfg)
        generate_groups_template(cfg["paths"]["dataset_raw"], classes, out_path=out)
        print(f"Đã sinh template groups.csv: {out} (điền group_id cho từng quả ớt)")

    if args.prepare:
        try:
            result = prepare_datasets(cfg, force_resplit=args.force_resplit,
                                      copy_to_split_dirs=True)
            print(f"\nChuẩn bị dataset xong: chia tập {result['splits']}")
            print(f"Manifest: {result['manifest']}")
        except DatasetEmptyError as exc:
            print(f"\n[LỖI] {exc}")
            sys.exit(2)


def _resolve_report_path(arg: str, cfg: dict) -> str:
    if arg:
        return arg if os.path.isabs(arg) else os.path.join(os.getcwd(), arg)
    default = cfg["paths"].get("reports_metrics", "reports/metrics")
    return os.path.join(default, "dataset_scan.json")


def _resolve_groups_template_path(arg: str, cfg: dict) -> str:
    if arg:
        return arg if os.path.isabs(arg) else os.path.join(os.getcwd(), arg)
    default = cfg["paths"].get("dataset_raw", "dataset/raw")
    return os.path.join(os.path.dirname(default), "groups.csv")


if __name__ == "__main__":
    main()
