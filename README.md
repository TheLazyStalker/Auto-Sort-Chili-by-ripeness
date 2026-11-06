# chili-ripeness-ai — Phân loại độ chín ớt 3 lớp bằng MobileNetV2

Phân loại quả ớt thành `unripe` (chưa chín) / `medium_ripe` (chín vừa) / `ripe` (đã chín) bằng **MobileNetV2 pretrained ImageNet + Transfer Learning**, đầu vào RGB 224×224, đầu ra Softmax 3 lớp. Mục tiêu triển khai: **chương trình CLI/native chạy trong Termux trên điện thoại Android** (dùng LiteRT/TFLite runtime), có khả năng mở rộng giao tiếp ESP32-S3 để điều khiển cơ cấu phân loại. Phần 1 này là nền tảng huấn luyện trên máy tính.
Progress: Chưa Hoàn Thành - ~25%.

## 1. Môi trường đã kiểm chứng (container `ai-workspace`)

| Thành phần | Giá trị |
|---|---|
| Container | `ai-workspace` (Ubuntu 24.04, root) |
| Python | 3.12.3 (`/workspace/.venv`) |
| TensorFlow | 2.17.0 (CPU-only, 8 CPU, 15 GiB RAM, chưa GPU/CUDA) |
| Keras | 3.15.1 |
| NumPy | 1.26.4 (khóa `<2.0.0` vì TF 2.17 yêu cầu) |
| Pillow / PyYAML / scikit-learn / matplotlib / pytest | 13.0.0 / 6.0.3 / 1.5.2 / 3.11.2 / 9.2.0 |
| Volume | Host `<HOST_REPO>/podman-workspace` ↔ Container `/workspace` (bind mount, giữ lại sau restart) |
| Dự án | `/workspace/chili-ripeness-ai` (trong volume) |

> **Lưu ý tương thích:** TF 2.17 + Python 3.12 đã kiểm chứng. TensorFlow 2.17 yêu cầu `numpy<2.0.0`, `protobuf<5` — đã khóa trong `requirements.txt`. Only for Python 3.12. Sử dụng CPU; pipeline tự co lại theo phần cứng (không giả lập GPU).

## 2. Cấu trúc dự án

```text
chili-ripeness-ai/
├── dataset/
│   ├── raw/{unripe,medium_ripe,ripe}/   # Ảnh gốc đặt ở đây (để trống sẵn)
│   ├── train/  validation/  test/       # Ảnh được copy khi chia tập (script tự tạo)
│   └── split_manifest.json              # Manifest chia tập (tự tạo)
├── models/checkpoints/  saved_model/  tflite/
├── reports/figures/  metrics/
├── src/
│   ├── config.py        # Đọc config.yaml, seed, class mapping
│   ├── dataset.py       # Quét ảnh, kiểm tra lỗi/trùng, chia 70/15/15, tf.data
│   ├── model.py         # MobileNetV2 + GAP + Dropout + Softmax, freeze/fine-tune
│   ├── train.py         # Huấn luyện 2 giai đoạn
│   ├── evaluate.py      # Đánh giá: accuracy, P/R/F1, confusion matrix
│   ├── predict.py       # Dự đoán 1 ảnh
│   ├── export_tflite.py # Xuất SavedModel + .tflite cho Android
│   └── utils.py         # Class weights, md5 trùng ảnh, vẽ đường cong
├── scripts/check_dataset.py     # Kiểm tra/thống kê dataset
├── scripts/smoke_test_model.py  # Khởi tạo model với tensor ngẫu nhiên
├── tests/                        # Unit tests (không cần ảnh thật)
├── config.yaml                   # Mọi tham số huấn luyện
├── requirements.txt
├── README.md  DATASET_GUIDE.md  .gitignore
```

## 3. Cài đặt

```bash
# 1) Vào container (đã chạy sẵn)
podman exec -it ai-workspace bash
cd /workspace/chili-ripeness-ai

# 2) Kích hoạt venv và cài đúng phiên bản đã khóa
source /workspace/.venv/bin/activate
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices())"
```

## 4. Cách dùng

```bash
# Kiểm tra dataset (khi chưa có ảnh sẽ báo rõ ràng và dừng an toàn)
python scripts/check_dataset.py
python scripts/check_dataset.py --prepare          # tạo manifest + chia tập & copy

# Smoke test model (tensor ngẫu nhiên, chưa cần dataset)
python scripts/smoke_test_model.py

# Unit tests
pytest -v

# Huấn luyện (hai giai đoạn: freeze backbone → fine-tune 30 layer cuối)
python -m src.train --force-resplit               # --force-resplit để chia lại khi cần

# Đánh giá trên tập test
python -m src.evaluate --split test

# Dự đoán một ảnh
python -m src.predict dataset/raw/ripe/anh1.jpg

# Xuất TFLite cho Android
python -m src.export_tflite
# → models/tflite/chili_ripeness.tflite
```

## 5. Thiết kế kỹ thuật

- **Model:** Input 224×224×3 → `preprocess_input` → MobileNetV2 (ImageNet, include_top=False) → GlobalAveragePooling2D → Dropout → Dense(3, softmax). Tham số `dropout`, `fine_tune_layers` chỉnh trong `config.yaml`.
- **Hai giai đoạn:** GĐ1 freeze toàn bộ backbone, lr=1e-3 (head); GĐ2 mở `fine_tune_layers` cuối, lr=1e-4. Callback: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau theo `val_accuracy`; class weights dạng balanced.
- **Augmentation nhẹ, không đổi màu mạnh:** flip ngang, xoay ±18°, zoom/translate nhẹ, brightness/contrast ±5%; **không** đổi hue/saturation (màu là đặc trưng chính của độ chín).
- **Dataset:** hỗ trợ JPG/JPEG/PNG, kiểm tra ảnh lỗi/trùng (md5), chia 70/15/15 stratified theo seed=42, hỗ trợ group ID qua `dataset/groups.csv` (`filename,group_id`) để ảnh cùng 1 quả ớt không lọt nhiều tập; manifest `split_manifest.json` giúp **không chia lại** khi không cần; **không xóa ảnh gốc**.

## 6. Trạng thái phần 1

Hoàn thành: môi trường, cấu trúc dự án, toàn bộ mã nguồn, tests + pytest, tài liệu, smoke test model. Đang chờ: **ảnh thật** (đặt vào `dataset/raw/<class>/`) rồi chạy `--prepare` → huấn luyện → đánh giá → xuất TFLite. Xem `DATASET_GUIDE.md` để thu thập & đặt nhãn.
