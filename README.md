# chili-ripeness-ai — Phân loại độ chín ớt 3 lớp (MobileNetV3Small)

Phân loại quả ớt thành `unripe` (chưa chín) / `medium_ripe` (chín vừa) / `ripe` (đã chín) bằng **MobileNetV3Small pretrained ImageNet + Transfer Learning**, đầu vào RGB 224×224, đầu ra Softmax 3 lớp. Mục tiêu triển khai: **chương trình CLI/native chạy trong Termux trên điện thoại Android (điện thoại Android (mẫu đã test))** (dùng LiteRT/TFLite runtime), có khả năng mở rộng giao tiếp ESP32-S3 để điều khiển cơ cấu phân loại.

> **Trạng thái:** Task 1–6 hoàn thành (phần container). Huấn luyện MobileNetV3Small đã chạy trên 463 derivative `crop_v1` (development split 80/20). **Benchmark trên thiết bị thật (điện thoại Android (mẫu đã test)) đang chờ thực hiện qua Termux** — chưa tuyên bố accuracy/hiệu năng thực tế.

Progress: ~80% (container) — chờ đo trên điện thoại + integration ESP32 (Task 7).

## 1. Môi trường đã kiểm chứng (container `ai-workspace`)

| Thành phần | Giá trị |
|---|---|
| Container | `ai-workspace` (Ubuntu 24.04, root) |
| Python | 3.12.3 (`/workspace/.venv`) |
| TensorFlow | 2.17.0 (CPU-only, 8 CPU, 15 GiB RAM, chưa GPU/CUDA) |
| Keras | 3.15.1 |
| NumPy | 1.26.4 (khóa `<2.0.0` vì TF 2.17 yêu cầu) |
| Pillow / PyYAML / scikit-learn / matplotlib / pytest | 13.0.0 / 6.0.3 / 1.5.2 / 3.11.2 / 9.2.0 |
| Volume | Host `<HOST_REPO>/podman-workspace` ↔ Container `/workspace` (bind mount) |
| Dự án | `/workspace/chili-ripeness-ai` (trong volume) |

> **Lưu ý tương thích:** TF 2.17 + Python 3.12 đã kiểm chứng. TensorFlow 2.17 yêu cầu `numpy<2.0.0`, `protobuf<5`. Chỉ dùng CPU.

## 2. Quyết định dữ liệu (đã chốt)

- **Q1=C** — không có mapping ảnh→quả; không suy group từ timestamp/tên file.
- **Q4=B** — 473 ảnh cũ (trừ 10 ảnh bị loại) chỉ dùng **train**; validation/test mới từ quả vật lý độc lập (Task 3).
- **Q5** — 10 ảnh nghi vấn giữ trong `dataset/review_exclusions.csv` (status `excluded`), không đưa vào train.
- **Q6=C** — lưu derivative `crop_v1` riêng (`dataset/derived/crop_v1/`), giữ nguyên raw, cùng quy trình crop khi inference.
- **Q7** — giao thức thu thập Task 3: `group_id` duy nhất/quả, quả chiếm 25–40% khung, ánh sáng khuếch tán, nền matte, 3–6 ảnh/quả, 30–50+ quả/lớp.
- Preprocessing invariant: RGB `[0,255]` → MobileNetV3Small `include_preprocessing=True` (Rescaling 1/128, −1 nhúng trong model) → `[-1,1]`. **Không bao giờ thêm `Rescaling(1/255)` ngoài model; không double-scale.**
- **INT8 đã loại bỏ hoàn toàn** — chỉ Float32 (bản đối chiếu) + Float16 (bản triển khai chính).

## 3. Cấu trúc dự án

```text
chili-ripeness-ai/
├── dataset/
│   ├── raw/{unripe,medium_ripe,ripe}/   # Ảnh gốc (473 ảnh, BẤT BIẾN)
│   ├── derived/crop_v1/                 # Derivative crop 224×224 + metadata.csv
│   ├── review_exclusions.csv            # 10 ảnh bị loại/chờ review
│   ├── capture_log.csv                  # Log chụp Task 3
│   ├── group_splits.csv                 # Gán split theo group (50/50 val/test)
│   ├── label_overrides.csv              # Override nhãn (nếu có)
│   └── split_manifest_task3.json        # Manifest Task 3 (train-only hiện tại)
├── artifacts/task4/                     # Model .keras + training/eval DEV
├── artifacts/task5/                     # TFLite F32/F16 + conversion/eval/bench
├── deployment/task6/                    # Package Termux (predict/benchmark/check_env)
├── models/checkpoints/  saved_model/  tflite/
├── reports/figures/  metrics/
├── src/
│   ├── config.py        # Đọc config.yaml, seed, class mapping, path resolution
│   ├── dataset.py       # Quét ảnh, lỗi/trùng, chia tập, tf.data
│   ├── crop.py          # crop_v1 (mask, largest component, bbox, 224×224)
│   ├── task3.py         # Exclusion/group split/manifest Task 3
│   ├── model.py         # MobileNetV2/V3Small + GAP + Dropout + Softmax, freeze/fine-tune
│   ├── task4.py         # Dev split 80/20, augmentation, hai phase training, metrics
│   ├── train.py         # (legacy) train MobileNetV2 2 giai đoạn
│   ├── evaluate.py      # (legacy) evaluation manifest cũ
│   ├── predict.py       # Dự đoán 1 ảnh
│   ├── export_tflite.py # (legacy) export TFLite cũ
│   └── utils.py         # Class weights, md5, vẽ đường cong
├── scripts/
│   ├── check_dataset.py / audit_dataset.py / prepare_dataset.py
│   ├── import_new_images.py / label_review.py / split_new_groups.py
│   ├── make_crop_derivatives.py / qa_crop.py / build_task3_manifest.py
│   ├── train_task4.py / evaluate_task4.py / measure_latency.py / verify_task4_dataset.py
│   ├── convert_task5.py / evaluate_task5.py / benchmark_task5.py
│   └── smoke_test_model.py
├── tests/                     # Unit tests (118 tests, không cần ảnh thật)
├── config.yaml                # Mọi tham số
├── requirements.txt
├── README.md  DATASET_GUIDE.md  DATASET_AUDIT.md  PROJECT_ROADMAP.md  .gitignore
```

## 4. Cài đặt

```bash
# 1) Vào container (đã chạy sẵn)
podman exec -it ai-workspace bash
cd /workspace/chili-ripeness-ai

# 2) Kích hoạt venv và cài đúng phiên bản đã khóa
source /workspace/.venv/bin/activate
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 5. Pipeline chính (Task 3–5)

```bash
# Xác minh dataset/derivative (SHA256 raw so với baseline)
python scripts/verify_task4_dataset.py --sha256

# Train MobileNetV3Small (dev 80/20 seed 42) → artifacts/task4/
python scripts/train_task4.py

# Evaluation DEV (accuracy, P/R/F1, confusion matrix)
python scripts/evaluate_task4.py --save-probs

# Latency CPU (tham khảo)
python scripts/measure_latency.py --warmup 5 --runs 50

# Convert TFLite (Float32 + Float16, không INT8) → artifacts/task5/
python scripts/convert_task5.py

# Đối chiếu Keras vs TFLite trên dev
python scripts/evaluate_task5.py

# Benchmark TFLite (container)
python scripts/benchmark_task5.py --warmup 5 --runs 50
```

## 6. Triển khai Termux (Task 6) — `deployment/task6/`

Xem `deployment/task6/README.md` để cài đặt từng bước trên điện thoại. Tóm tắt:

```bash
pkg update -y && pkg upgrade -y
pkg install -y python python-pip
pip install 'numpy<2.0.0' Pillow tflite-runtime
# copy deployment/task6/ → ~/task6/ trên máy
cd ~/task6
python check_env.py
python predict.py image.jpg                 # Float16 mặc định
python predict.py --model models/chili_float32.tflite image.jpg
python predict.py --dir images/ --out results.csv
python benchmark.py --model models/chili_float16.tflite --warmup 10 --runs 100
```

- `predict.py` in: filename, class, xác suất 3 lớp, preprocessing_ms, inference_ms, total_ms.
- Preprocessing khớp pipeline container: RGB [0,255] → 224×224 → float32 tensor; crop_v1 tự động nếu ảnh chưa crop.
- Kết quả trên thiết bị thật cần gửi lại để tổng hợp benchmark chính thức.

## 7. Kết quả hiện tại (DEVELOPMENT — chưa phải benchmark độc lập)

| Hạng mục | Giá trị |
|---|---|
| Dataset | 473 raw (unripe 197 / medium 142 / ripe 134); 463 derivative crop_v1 hợp lệ |
| Model | MobileNetV3Small (ImageNet), 940,851 params |
| Dev split | 80/20 image-level, seed 42, leaky (train 371 / val 92) |
| Phase A/B | best val_acc 0.9891 (epoch 9) / 0.9891 (epoch 1, early stop) |
| Dev metrics | accuracy 0.9891, macro F1 0.9877; confusion `[[38,0,0],[0,27,1],[0,0,26]]` |
| TFLite | Float32 3.746 MB (pure), Float16 1.913 MB (primary); agreement 1.0 vs Keras |
| Latency (container) | Keras ~58ms; TFLite F32 ~2.27ms, F16 ~2.20ms (tham khảo) |
| Tests | 118 passed; raw SHA256 473/473 khớp baseline |

> ⚠️ **KHÔNG tuyên bố accuracy thực tế ≥90%** — dev split leaky (chưa biết ảnh cũ thuộc cùng quả hay không); benchmark chính thức cần tập test độc lập + đo trên điện thoại Android (mẫu đã test).

## 8. Các hạn chế / việc tiếp theo

- Thu thập dữ liệu mới (quả độc lập) + điền `group_id` → tạo validation/test độc lập (Task 3 Phase 2–5).
- Đo benchmark trên điện thoại thật (Task 6 §D) — đang chờ người dùng.
- Task 7: tích hợp ESP32/OV2640, throughput toàn hệ thống.
- Xem `PROJECT_ROADMAP.md` cho chi tiết từng task.
