# chili-ripeness-ai — Phân loại ớt 3 lớp (MobileNetV3Small → TFLite)

Phân loại quả ớt thành **`ripe`** (chín) / **`unripe`** (không chín) / **`damaged`**
(hư hỏng) bằng MobileNetV3Small (ImageNet) + transfer learning, đầu vào RGB
224×224, đầu ra softmax. Xuất TensorFlow Lite **Float32 + Float16** để chạy trên
điện thoại Android qua Termux (CPU), có thể mở rộng giao tiếp ESP32.

Hệ thống quản lý **nhiều bộ dữ liệu độc lập**: mỗi dataset tự mô tả số lớp, tên
lớp, thứ tự nhãn, pipeline tiền xử lý và tỷ lệ chia tập trong
`datasets/<dataset_id>/dataset_config.yaml`. Không hardcode lớp trong mã nguồn.

---

## 1. Bắt đầu nhanh

```bash
podman start ai-workspace
podman exec -it ai-workspace bash -lc 'cd /workspace/chili-ripeness-ai && source /workspace/.venv/bin/activate && bash'

# Mở lớp phủ TUI
python scripts/tui.py
```

## 2. Lớp phủ TUI (khuyến nghị)

`scripts/tui.py` là **lớp phủ mỏng trên terminal**: mỗi mục là một lệnh CLI có sẵn,
mỗi tham số là một công tắc. **Xoá file này không ảnh hưởng gì** — mọi lệnh vẫn
chạy trực tiếp như mục 3.

```
┌──────────────────── CHILI PIPELINE — TUI ────────────────────┐
│ dataset: chili_quality_v2   source: (chưa nhập)   class: auto│
│ [skip_center·]  [skip_train·]  [export✓]                     │
│ ↑/↓ chọn · Enter chạy · t đổi công tắc · d đổi dataset       │
│ c đổi class · s nhập source · q thoát                        │
│                                                              │
│  ▸ Kiểm tra dữ liệu (dry-run)                                │
│    Sinh derivative crop_v1                                   │
│    Train (experiment mới)                                    │
│    Đánh giá split TEST                                       │
│    Đánh giá split VALIDATION                                 │
│    Export TFLite (F32 + F16)                                 │
│    PIPELINE: center → raw → derivative → train               │
│    PIPELINE nhanh trên nguồn mẫu (kiểm thử)                  │
│    Chạy test suite                                           │
└──────────────────────────────────────────────────────────────┘
```

| Phím | Tác dụng |
|---|---|
| `↑` `↓` / `j` `k` | di chuyển giữa các nút |
| `Enter` | chạy nút đang chọn (xuất ra terminal, Enter để quay lại) |
| `t` | bật/tắt công tắc (`skip_center`, `skip_train`, `export`) |
| `d` | đổi dataset |
| `c` | đổi lớp (cho pipeline: `ripe`/`unripe`/`damaged`) |
| `s` | nhập thư mục ảnh nguồn |
| `q` | thoát |

Không có terminal tương tác (CI, script)? Dùng:

```bash
python scripts/tui.py --list                                  # xem mọi nút + lệnh tương ứng
python scripts/tui.py --run verify --dataset chili_quality_v2  # chạy 1 nút
python scripts/tui.py --run pipeline --dataset chili_quality_v2 \
    --source /path/to/images --class damaged --skip-train
```

> Nếu báo "Không khởi động được TUI", chạy trong terminal thật và đặt
> `export TERM=xterm`.

## 3. Lệnh CLI (không cần TUI)

```bash
# Xem dataset có sẵn
python -c "from src.datasets import list_dataset_ids; print(list_dataset_ids())"

# Kiểm tra dữ liệu (dry-run: chỉ in split, không train)
python scripts/train.py --dataset chili_quality_v2 --dry-run

# Sinh derivative crop_v1 (bắt buộc trước khi train)
python scripts/make_dataset_derivatives.py --dataset chili_quality_v2

# Train → experiments/<dataset>/run_NNN/ (không ghi đè run cũ)
python scripts/train.py --dataset chili_quality_v2

# Đánh giá (đọc class mapping từ run)
python scripts/evaluate.py --dataset chili_quality_v2 --split test --save-probs
python scripts/evaluate.py --dataset chili_quality_v2 --split validation

# Export TFLite Float32 + Float16 → <run>/export/
python scripts/export.py --dataset chili_quality_v2
python scripts/export.py --dataset chili_quality_v2 --run run_001 --out /tmp/export

# Dự đoán 1 ảnh bằng model .keras
python -m src.predict --image anh.png \
    --model experiments/chili_quality_v2/run_001/model/best_chili.keras \
    --classes-file experiments/chili_quality_v2/run_001/class_mapping.json

# Test
python -m pytest -q
```

## 4. Pipeline hoàn chỉnh: ảnh raw → center → dataset → train → export

Một lệnh, tham số lấy từ **`pipeline_config.json`** (gốc project):

```bash
python scripts/pipeline.py --source /path/to/images --class damaged --export
python scripts/pipeline.py --source /path/to/has_class_subdirs
python scripts/pipeline.py --source-url https://host/imgs.zip --class damaged
python scripts/pipeline.py --source /path/to/images --class damaged --skip-train  # kiểm thử
```

Luồng: **copy nguồn** (không bao giờ di chuyển) → `make-object-to-center`
(center + nền đồng nhất) → kiểm tra ảnh → `datasets/<id>/raw/<class>/` (không ghi
đè) → derivative `crop_v1` → train (experiment mới) → export.

Chi tiết + cách kiểm thử độc lập: **`DOCS_PIPELINE.md`**.

## 5. Cấu hình

### `pipeline_config.json` (một nguồn cho cả quy trình)

| Nhóm | Nội dung |
|---|---|
| `paths` | `datasets_root`, `experiments_root`, `staging_root`, `center_tool_dir` |
| `dataset` | dataset mặc định + danh sách lớp |
| `centering` | `background_bgr`, `size`, `fruit_frac`, `pad`, `ext`, `suffix` (tham số make-object-to-center) |
| `preprocessing` | `crop_v1`: `threshold`, `pad_frac`, `border_frac`, `min/max_area_frac`, `dilate_px`, `output_size` |
| `split` | `ratios`, `seed` |
| `training` | epochs, lr, patience, batch_size, class_weight |
| `model` | base, weights, dropout, fine_tune_layers |

Sửa JSON → mọi bước đổi theo. Ví dụ đổi nền centering:

```json
"centering": { "background_bgr": [255, 255, 255] }
```

### `datasets/<dataset_id>/dataset_config.yaml`

```yaml
dataset_id: chili_quality_v2
classes: [ripe, unripe, damaged]     # thứ tự = label index
paths: { raw: ..., derived: ..., manifests: ... }
preprocessing: { pipeline: crop_v1, version: v1 }
split: { strategy: group_stratified, ratios: [0.70, 0.15, 0.15], seed: 42 }
```

**Tạo dataset mới:** copy thư mục `datasets/chili_quality_v2/`, sửa `dataset_id`,
`classes`, `paths`. Không cần sửa mã nguồn.

## 6. Cấu trúc thư mục

```text
chili-ripeness-ai/
├── pipeline_config.json          # config trung tâm
├── datasets/
│   ├── legacy_ripeness_v1/dataset_config.yaml   # dataset cũ (trỏ tới dataset/)
│   └── chili_quality_v2/{raw,derived,dataset_config.yaml}
├── dataset/                      # dataset cũ (unripe/medium_ripe/ripe) — giữ nguyên
├── configs/{training,models}/    # config YAML (tùy chọn, override JSON)
├── experiments/<dataset>/run_NNN/{config.yaml, class_mapping.json,
│       dataset_manifest.json, training_history.json, metrics.json,
│       model/best_chili.keras, evaluation/, export/}
├── deployment/task6/             # CLI Termux: predict/benchmark/check_env + model .tflite
├── tools/make-object-to-center/  # tool center ảnh (vendored)
├── src/                          # datasets, experiment, training, model, crop, centering...
├── scripts/                      # tui.py, pipeline.py, train.py, evaluate.py, export.py...
└── tests/                        # 156 unit tests
```

## 7. Dataset hiện có

| dataset_id | classes | ảnh |
|---|---|---|
| `chili_quality_v2` | ripe, unripe, damaged | 596 (272 / 191 / 133) |
| `legacy_ripeness_v1` | unripe, medium_ripe, ripe | 473 |

Ảnh gốc nằm trong `dataset/raw/` và `datasets/*/raw/` (không commit lên git —
gitignore). Model/experiment nằm trong `experiments/`, export trong
`experiments/<id>/run_*/export/`.

## 8. Kết quả tham chiếu (split dev 70/15/15, seed 42)

`chili_quality_v2` · run_001 — MobileNetV3Small, 940.851 tham số:

| Split | Accuracy | Macro F1 |
|---|---|---|
| validation (90) | 0.9889 | 0.9860 |
| test (89) | 0.9888 | 0.9874 |

TFLite export: Float32 3.746 MB, Float16 1.913 MB.

> **Lưu ý:** split chia theo *ảnh*; danh tính quả ớt chưa xác định nên có thể có
> ảnh cùng quả ở cả train và test → số liệu lạc quan. Đây **không phải** benchmark
> độc lập và không phải cam kết độ chính xác thực tế. Muốn đánh giá đúng cần thu
> thêm quả độc lập (`group_id`).

## 9. Triển khai Termux

`deployment/task6/README.md` — CLI `predict.py` (ảnh đơn / cả thư mục / CSV),
`benchmark.py`, `check_env.py`, kèm model Float16/Float32. Preprocessing khớp
pipeline train: RGB `[0,255]`, `crop_v1`, không chia 255 (model tự
`Rescaling(1/128, -1)`).

## 10. Môi trường

Container `ai-workspace` · Python 3.12.3 · TensorFlow 2.17.0 (CPU) · Keras 3.15.1
· numpy 1.26.4 · Pillow 13.0.0 · opencv-python-headless 4.11.0 · pytest.