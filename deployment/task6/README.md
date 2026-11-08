# Deployment Task 6 — Termux (điện thoại Android (mẫu đã test))

Triển khai CLI dự đoán độ chín ớt bằng MobileNetV3Small (TFLite) trên
điện thoại Android (mẫu đã test) qua Termux. Thiết bị mục tiêu: Android 11 / ColorOS 11,
MediaTek SoC ARM64 (MT6771), ARM64, 6GB RAM.

## Nội dung

```
deployment/task6/
├── models/
│   ├── chili_float16.tflite   # ưu tiên (1.91 MB)
│   ├── chili_float32.tflite   # bản đối chiếu (3.75 MB)
│   └── class_mapping.json     # {"unripe":0,"medium_ripe":1,"ripe":2}
├── predict.py                 # CLI predict đơn/thư mục + CSV
├── benchmark.py               # đo latency (warmup, mean/median/p95)
├── check_env.py               # kiểm tra môi trường thiết bị
├── requirements.txt
└── README.md
```

## 1. Cài Termux (nếu chưa có)

1. Cài **Termux** từ F-Droid (khuyến nghị) hoặc GitHub releases.
2. Mở Termux, chạy:

```bash
pkg update -y && pkg upgrade -y
pkg install -y python python-pip
```

## 2. Cài package Python

```bash
pip install 'numpy<2.0.0' Pillow tflite-runtime
```

> Lưu ý: `tflite-runtime` là bản nhẹ, chỉ chạy inference. Nếu `pip install
> tflite-runtime` không có wheel cho ARM64/Android, thử `pip install
> --extra-index-url https://termux-user-repository.github.io/packages/ tflite-runtime`
> hoặc báo cáo lỗi để đánh giá runtime thay thế (C/C++ TFLite). KHÔNG cài
> TensorFlow đầy đủ trên thiết bị.

Kiểm tra môi trường:

```bash
python check_env.py
```

## 3. Copy package lên máy

- Copy thư mục `deployment/task6/` vào Termux home:
  `<TERMUX_HOME>/task6/`
  (dùng `cp -r` sau khi copy qua USB/cloud/storage trong Termux, hoặc `termux-setup-storage`).

## 4. Chạy

```bash
cd ~/task6

# Dự đoán 1 ảnh (mặc định Float16)
python predict.py image.jpg

# Xác định rõ model
python predict.py --model models/chili_float32.tflite image.jpg

# Nhiều ảnh trong thư mục + CSV
python predict.py --dir images/ --out results.csv

# Benchmark (100 runs, warmup 10)
python benchmark.py --model models/chili_float16.tflite --warmup 10 --runs 100
```

## 5. Đầu ra predict

```json
{
  "filename": "image.jpg",
  "class": "ripe",
  "class_id": 2,
  "probabilities": {"unripe": 0.01, "medium_ripe": 0.03, "ripe": 0.96},
  "preprocessing_ms": 12.3,
  "inference_ms": 45.1,
  "total_ms": 57.4
}
```

## 6. Preprocessing (quan trọng)

- Đầu vào: RGB [0,255] (float32 tensor), 224×224.
- Model có `Rescaling(1/128, -1)` nhúng → KHÔNG chia 255, KHÔNG đổi RGB→BGR,
  KHÔNG thêm normalization khác, KHÔNG double-scale.
- Nếu ảnh chưa phải derivative crop_v1 (224×224), `predict.py` tự áp dụng
  `crop_v1` (cùng tham số pipeline gốc: threshold 30, pad 20%). Nếu ảnh đã là
  derivative → bỏ qua crop.

## 7. Security & giới hạn

- KHÔNG root, không đổi hệ thống; không yêu cầu app ngoài Termux.
- Kết quả trên thiết bị KHÔNG đại diện cho benchmark cuối; mục tiêu
  inference ≤1s/ảnh phải đo trực tiếp (Task 6 §D).
- KHÔNG dùng ảnh đó để tuyên bố accuracy thực tế ≥90% (thiếu test độc lập).
