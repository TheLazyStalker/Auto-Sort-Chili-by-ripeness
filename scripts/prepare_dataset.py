"""Chuẩn bị dataset: quét → chia 70/15/15 → lưu manifest → xác minh split.

Dùng:
  python scripts/prepare_dataset.py                  # chia (hoặc dùng manifest sẵn có)
  python scripts/prepare_dataset.py --force-resplit  # chia lại từ đầu
  python scripts/prepare_dataset.py --no-copy        # chỉ tạo manifest, không copy ảnh
  python scripts/prepare_dataset.py --verify-only    # chỉ xác minh manifest hiện có

Xác minh gồm:
  - Không file nào xuất hiện ở >1 tập (chống trùng tuyệt đối).
  - Không Group ID nào xuất hiện ở >1 tập (nếu có groups.csv).
  - Tỷ lệ 70/15/15 theo từng lớp.
  - Ước lượng rò rỉ do chụp burst (ảnh cách nhau <=5 giây nhưng khác tập).

Không xóa/sửa ảnh raw. Không huấn luyện.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    DatasetEmptyError,
    SPLIT_NAMES,
    build_burst_groups,
    find_burst_leaks,
    load_groups_file,
    load_manifest,
    prepare_datasets,
    save_split_manifest,
    scan_raw_dataset,
    stratified_group_split,
    verify_split,
)

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def splits_from_manifest(manifest: dict) -> dict:
    return {
        split: [(item["path"], item["label"]) for item in manifest.get(split, [])]
        for split in SPLIT_NAMES
    }


def report_verification(verification: dict, burst_leaks: list, classes: list) -> bool:
    """In báo cáo xác minh. Trả về True nếu có lỗi nghiêm trọng (trùng lặp/leak group)."""
    ok = True
    print("\n=== Xác minh split ===")
    print(f"Tổng: {verification['total']} ảnh")
    print("Tỷ lệ thực tế: " + ", ".join(
        f"{n}={verification['ratios'][n] * 100:.1f}%" for n in SPLIT_NAMES))

    print("\nSố ảnh mỗi lớp theo tập:")
    print("  %-14s%12s%12s%12s" % ("class", *SPLIT_NAMES))
    for cls in classes:
        row = "  %-14s" % cls + "".join(
            "%12d" % verification["per_split_counts"][n].get(cls, 0) for n in SPLIT_NAMES)
        print(row)

    if verification["overlap_paths"]:
        ok = False
        print(f"\n[LỖI] {len(verification['overlap_paths'])} file xuất hiện ở nhiều tập:")
        for p, ss in list(verification["overlap_paths"].items())[:10]:
            print(f"  - {p} → {ss}")
    else:
        print("\n[OK] Không có file nào xuất hiện ở nhiều tập.")

    if verification["groups_used"]:
        if verification["group_leaks"]:
            ok = False
            print(f"[LỖI] {len(verification['group_leaks'])} Group ID rò rỉ qua nhiều tập:")
            for g, ss in list(verification["group_leaks"].items())[:10]:
                print(f"  - {g} → {ss}")
        else:
            print(f"[OK] {verification['groups_used']} group — không group nào ở nhiều tập.")
    else:
        print("[CẢNH BÁO] Không có Group ID nào được dùng (groups.csv trống hoặc không cấu hình).")
        print("  → Mỗi ảnh được coi là một mẫu độc lập. Không thể loại trừ rò rỉ do")
        print("    nhiều ảnh của CÙNG một quả ớt nằm ở các tập khác nhau.")

    if verification["empty_class_split"]:
        print(f"\n[CẢNH BÁO] {len(verification['empty_class_split'])} tổ hợp lớp/tập rỗng: "
              f"{verification['empty_class_split']}")

    if burst_leaks:
        print(f"\n[CẢNH BÁO] {len(burst_leaks)} cặp ảnh chụp cách nhau <=5s nhưng khác tập "
              "(nguy cơ rò rỉ do burst — CHƯA xác minh cùng quả):")
        for b in burst_leaks[:10]:
            print(f"  - {os.path.basename(b['a'])} [{b['split_a']}] ↔ "
                  f"{os.path.basename(b['b'])} [{b['split_b']}] ({b['gap_s']:.0f}s)")
    else:
        print("\n[OK] Không phát hiện cặp burst (<=5s) nào bị tách sang hai tập.")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Chuẩn bị & xác minh split dataset ớt")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--force-resplit", action="store_true")
    parser.add_argument("--no-copy", action="store_true", help="Không copy ảnh vào train/val/test")
    parser.add_argument("--verify-only", action="store_true", help="Chỉ xác minh manifest hiện có")
    parser.add_argument("--report", default="", help="Đường dẫn JSON báo cáo xác minh")
    parser.add_argument("--max-burst-gap", type=float, default=5.0,
                        help="Ngưỡng giây để coi là cùng chuỗi burst (mặc định 5)")
    parser.add_argument("--burst-groups", type=float, default=None, metavar="GAP_S",
                        help="Sinh split THỬ NGHIỆM theo cụm burst (heuristic, chưa xác minh); "
                             "không ghi đè manifest chính")
    parser.add_argument("--manifest-out", default="",
                        help="Đường dẫn manifest cho chế độ --burst-groups")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["project"]["classes"]
    manifest_path = cfg["paths"]["split_manifest"]
    groups = load_groups_file(cfg["paths"].get("groups_file", ""))

    if args.burst_groups is not None:
        from src.config import get_split_ratios

        summary = scan_raw_dataset(cfg["paths"]["dataset_raw"], classes)
        if summary.total == 0:
            print("[LỖI] Dataset trống.")
            return 2
        heuristic = build_burst_groups(summary.files, max_gap_s=args.burst_groups)
        splits = stratified_group_split(
            summary.files, classes, ratios=get_split_ratios(cfg),
            seed=int(cfg.get("training", {}).get("seed", 42)), groups=heuristic)
        out = args.manifest_out or os.path.join(
            os.path.dirname(manifest_path), "split_manifest_burstgroup.json")
        save_split_manifest(splits, out, classes)
        print(f"[THỬ NGHIỆM] Split theo cụm burst (gap>{args.burst_groups:g}s) — "
              f"group là HEURISTIC, chưa xác minh.")
        print(f"Manifest thử nghiệm: {out} (KHÔNG thay thế manifest chính)")
        verification = verify_split(splits, classes, groups=heuristic)
        burst_leaks = find_burst_leaks(splits, max_gap_s=args.burst_groups)
        ok = report_verification(verification, burst_leaks, classes)
        report_path = args.report or os.path.join(
            cfg["paths"].get("reports_metrics", "reports/metrics"),
            "split_verification_burstgroup.json")
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "manifest": out,
                "heuristic_groups": True,
                "burst_gap_s": args.burst_groups,
                "verification": verification,
                "burst_leaks_count": len(burst_leaks),
            }, f, indent=2, ensure_ascii=False)
        print(f"\nĐã ghi báo cáo: {report_path}")
        return 0 if ok else 1

    if args.verify_only:
        manifest = load_manifest(manifest_path)
        if manifest is None:
            print(f"[LỖI] Không có manifest tại {manifest_path}. Chạy không có --verify-only trước.")
            return 2
        splits = splits_from_manifest(manifest)
    else:
        try:
            result = prepare_datasets(
                cfg, force_resplit=args.force_resplit,
                copy_to_split_dirs=not args.no_copy)
        except DatasetEmptyError as exc:
            print(f"[LỖI] {exc}")
            return 2
        print(f"\nĐã chuẩn bị dataset: {result['splits']}")
        print(f"Manifest: {result['manifest']}")
        manifest = load_manifest(manifest_path)
        if manifest is None:
            print(f"[LỖI] Không đọc được manifest vừa tạo: {manifest_path}")
            return 2
        splits = splits_from_manifest(manifest)

    verification = verify_split(splits, classes, groups=groups)
    burst_leaks = find_burst_leaks(splits, max_gap_s=args.max_burst_gap)
    ok = report_verification(verification, burst_leaks, classes)

    report_path = args.report or os.path.join(
        cfg["paths"].get("reports_metrics", "reports/metrics"), "split_verification.json")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    payload = {
        "manifest": manifest_path,
        "verification": verification,
        "burst_leaks_count": len(burst_leaks),
        "burst_leaks": burst_leaks,
        "max_burst_gap_s": args.max_burst_gap,
        "groups_file": cfg["paths"].get("groups_file", ""),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nĐã ghi báo cáo xác minh: {report_path}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
