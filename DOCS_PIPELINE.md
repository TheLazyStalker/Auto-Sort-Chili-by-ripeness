# PIPELINE HOÀN CHỈNH — raw → center → dataset → train → export

Một lệnh duy nhất nối toàn bộ quy trình, điều khiển bằng **`pipeline_config.json`**
ở gốc project (sửa file này là mọi bước thay đổi theo).

```
ảnh nguồn ──copy──> staging ──make-object-to-center──> dataset/raw/<class>/
                                          │
                                          ├──> derived/crop_v1/  (crop_v1)
                                          └──> experiments/<dataset>/run_NNN/  (train)
                                                     └──> export/  (TFLite F32/F16)
```

## 1. Config trung tâm: `pipeline_config.json`

| Nhóm | Ý nghĩa |
|---|---|
| `paths` | datasets_root, experiments_root, staging_root, center_tool_dir |
| `dataset` | dataset mặc định + danh sách lớp |
| `centering` | `background_bgr`, `size`, `fruit_frac`, `pad`, `ext`, `suffix`, `enabled` (tham số cho make-object-to-center) |
| `preprocessing` | `pipeline: crop_v1` + `threshold`, `pad_frac`, `border_frac`, `min/max_area_frac`, `dilate_px`, `output_size` |
| `split` | `ratios`, `seed` |
| `training` | epochs/lr/patience/batch/class_weight |
| `model` | base, weights, dropout, fine_tune_layers |

Ví dụ đổi nền centering sang trắng:
```json
"centering": { "background_bgr": [255, 255, 255] }
```
→ lệnh gọi tool đổi thành `--bg 255,255,255`; ảnh mới vào raw có nền trắng.

## 2. Chạy pipeline

```bash
# 1 lớp (toàn bộ ảnh trong thư mục thuộc 1 lớp)
python scripts/pipeline.py --source /path/to/images --class damaged

# nhiều lớp (thư mục con tên = tên lớp)
python scripts/pipeline.py --source /path/to/has_class_subdirs

# nguồn tải trên mạng (zip)
python scripts/pipeline.py --source-url https://host/imgs.zip --class damaged

# kiểm thử nhanh (không train)
python scripts/pipeline.py --source /path/to/images --class damaged --skip-train

# đầy đủ + export TFLite
python scripts/pipeline.py --source /path/to/images --class damaged --export
```

Cờ hữu ích: `--dataset <id>`, `--pipeline-config <json>`, `--run-id`,
`--skip-center`, `--skip-derivatives`, `--skip-train`, `--limit-per-class N`,
`--dry-run`, `--no-copy-source`.

**Nguồn không bao giờ bị di chuyển** — mặc định được COPY vào staging
(`--no-copy-source` để chỉ đọc trực tiếp).

## 3. Kiểm thử độc lập (không đụng dữ liệu thật)

Tạo config tạm trỏ `paths` sang thư mục test, và một dataset tạm:

```bash
mkdir -p /tmp/t/datasets/ds/raw/{ripe,unripe,damaged}
# dataset_config.yaml: classes [ripe, unripe, damaged], paths trỏ về /tmp/t/datasets/ds
python -c "import json;c=json.load(open('pipeline_config.json'));c['paths']={'datasets_root':'/tmp/t/datasets','experiments_root':'/tmp/t/experiments','staging_root':'/tmp/t/staging','center_tool_dir':'$PWD/tools/make-object-to-center'};json.dump(c,open('/tmp/t/pcfg.json','w'))"

cp -r /path/to/some/images /tmp/t/src        # COPY nguồn, không di chuyển
python scripts/pipeline.py --dataset ds --source /tmp/t/src --class damaged \
    --pipeline-config /tmp/t/pcfg.json --export
```

Đã kiểm chứng: nguồn cũ (`/mnt/.../damaged-chili`, 8 ảnh copy), nguồn mới
(vendored examples, qua `--source-url file://…zip`), toàn bộ chuỗi
center → raw → derivative → train → export đều chạy; nguồn gốc nguyên vẹn.

## 4. Các lệnh rời (nếu cần)

```bash
python scripts/make_dataset_derivatives.py --dataset <id>   # chỉ sinh derivative
python scripts/train.py --dataset <id>                      # chỉ train
python scripts/evaluate.py --dataset <id> --split test      # đánh giá
python scripts/export.py --dataset <id> --run <run_NNN>     # chỉ export
```

## 5. Tool centering (vendored)

`tools/make-object-to-center/` — clone từ
`https://github.com/TheLazyStalker/make-object-to-center` (commit `63e2d50`,
xem `PROVENANCE.md`). Không sửa mã nguồn tool.

Dependency: `opencv-python-headless` (đã thêm vào `requirements.txt`).