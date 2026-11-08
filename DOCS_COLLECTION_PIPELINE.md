# HƯỚNG DẪN THU THẬP & IMPORT ẢNH MỚI (Task 3)

> Áp dụng sau khi nhận ảnh mới (quả vật lý **độc lập** với 473 ảnh cũ).
> Chạy mọi lệnh trong container:
> ```bash
> podman start ai-workspace
> podman exec -it ai-workspace bash -lc 'cd /workspace/chili-ripeness-ai && source /workspace/.venv/bin/activate && bash'
> ```

---

## 1. Trước khi chụp

- Mỗi quả ớt = một `group_id` **duy nhất**, ghi **tại thời điểm chụp** (VD `P_20261108_001`).
- Không tái dùng `group_id` giữa các lớp/phiên.
- Mục tiêu: quả chiếm 25–40% khung; ánh sáng khuếch tán ổn định; nền matte tương phản; khóa WB nếu OV2640 cho phép, nếu không thì cố định lighting/exposure/gain/WB-mode và ghi vào `capture_log.csv`.
- 3–6 ảnh/quả; ≥30–50 quả/lớp (target 50–100+).
- Giữ file gốc + EXIF. Không crop/ghi đè raw.

## 2. Chuẩn bị staging

Đặt ảnh mới vào một thư mục tam (ngoài `dataset/raw`), chia theo lớp:

```
staging/
├── unripe/       # ảnh chưa chín
├── medium_ripe/  # chuyển tiếp
└── ripe/         # chín đỏ
```

**KHÔNG đặt ảnh trực tiếp vào `dataset/raw`** — import qua script để validate + ghi log.

## 3. Import

```bash
python scripts/import_new_images.py \
  --source /path/to/staging \
  --session-id T3_20261108_01 \
  --camera OV2640 \
  --review-status accepted \
  --notes "phiên 1, unripe"
```

- `--dry-run` trước để xem plan: `python scripts/import_new_images.py --source <dir> --session-id <id> --dry-run`.
- Script tự: validate decode; chặn trùng md5 (trong batch và với raw hiện có); chặn filename trùng (không ghi đè); sinh `image_id` duy nhất (`T3_YYYYMMDD_NNNN`); ghi thêm dòng vào `dataset/capture_log.csv`.
- Sau import, **điền `group_id`** cho từng dòng trong `capture_log.csv` (cột `group_id`). Mọi ảnh cùng 1 quả → cùng `group_id`.

## 4. Gán split (50/50 val/test theo group)

```bash
python scripts/split_new_groups.py --seed 42 --dry-run   # xem trước
python scripts/split_new_groups.py --seed 42             # ghi group_splits.csv
```

- Script chỉ gán **group chưa có** trong `group_splits.csv`; không đụng group cũ.
- Mỗi lớp: shuffle theo seed, 50% → validation, 50% → test (theo group).
- Cảnh báo nổi bật nếu một split có <10 group/lớp (evaluation không robust).

## 5. Nhãn / review

```bash
# Đánh dấu ảnh cần rà soát
python scripts/label_review.py --set-status --filename <name> --status pending_review --reason "mờ"
# Giữ nhãn sau khi review
python scripts/label_review.py --set-status --filename <name> --status accepted
# Đổi nhãn (chỉ qua override, không sửa raw/nhãn gốc)
python scripts/label_review.py --override-label --filename <name> --label ripe --approved-by <user>
# Xem trạng thái
python scripts/label_review.py --show --filename <name>
```

Từ vựng trạng thái (bắt buộc): `accepted | pending_review | excluded`. Status lạ → script báo lỗi.

Nếu chỉ có 1 người gán nhãn: ghi `annotator`, đánh dấu ảnh mơ hồ `pending_review`, **không giả vờ** có đo Cohen's κ.

## 6. Sinh derivative + QA + manifest

```bash
# Sinh crop_v1 cho ảnh mới (raw giữ nguyên; ảnh bị loại/pending không tạo derivative)
python scripts/make_crop_derivatives.py
# QA (overlay, size/mode, determinism)
python scripts/qa_crop.py
# Build manifest Task 3 (provisional nếu val/test rỗng; --require-all-splits khi đủ)
python scripts/build_task3_manifest.py
python scripts/build_task3_manifest.py --require-all-splits   # khi có val+test đầy đủ
# Xác minh raw không đổi so với baseline
python scripts/make_crop_derivatives.py --verify-only
```

Ghi chú:
- `make_crop_derivatives.py` không tạo derivative cho ảnh có `review_status ∈ {pending_review, excluded}` (đọc từ `capture_log.csv`).
- `build_task3_manifest.py` áp dụng `dataset/label_overrides.csv` (nếu có) khi build.
- Baseline SHA256 raw được ghi vào `reports/metrics/crop_v1_report.json`; `--verify-only` so sánh lại.

## 7. Những điều KHÔNG được làm

- Không xóa/di chuyển/đổi tên/ghi đè ảnh raw.
- Không dùng ảnh ngoài nguồn do chủ dự án cung cấp.
- Không dùng timestamp/filename/thứ tự để suy group.
- Không gán cùng một group cho 2 split.
- Không tự sửa nhãn cũ; chỉ qua `label_overrides.csv`.
- Không train/evaluate trong Task 3.
