# Deployment Task 6 — Termux (điện thoại + camera OV2640/ESP32)

Triển khai CLI dự đoán độ chín ớt bằng MobileNetV3Small (TFLite) trên Termux.

**Thiết bị đầu cuối:** camera **OV2640** kết nối **ESP32** → trung chuyển ảnh
→ điện thoại Termux (AI chạy trên điện thoại). Camera OV2640 hỗ trợ các chế độ
phổ biến: VGA 640×480, XGA 1024×768, UXGA 1600×1200 (JPEG).

> Quyết định đã chốt: toàn bộ inference/preprocessing chạy trên điện thoại
> (Termux, python-tflite-runtime). ESP32 chỉ chụp/đẩy JPEG (qua BLE/UART/WiFi
> tuỳ tầng) — không chạy AI. Benchmark OV2640 (container, CPU x86 — tham khảo):
> UXGA 1600×1200 ≈ 37ms, XGA ≈ 23ms, VGA ≈ 16ms (preprocessing+inference).

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
- Nếu ảnh đã là derivative 224×224 → **bỏ qua crop** (nhanh nhất, ~1-2 ms).
- Nếu ảnh raw (VD 1024×1024) → `fast_crop.py` tự crop: downscale tính mask/bbox
  + morphology separable + union-find. Benchmark container: **~25 ms** cho 1024²
  (bản cũ ~350-700 ms). Class parity 45/45 so với crop gốc.

> ⚠️ Tối ưu cho điện thoại: **dùng ảnh đã crop 224×224 (hoặc chụp thẳng
> kích thước này)** để bỏ qua crop — tổng thời gian ~3-4 ms. Nếu bắt buộc crop
> raw trên máy, đo lại vì tốc độ phụ thuộc CPU ARM64.

## 7. Security & giới hạn

- KHÔNG root, không đổi hệ thống; không yêu cầu app ngoài Termux.
- Kết quả trên thiết bị KHÔNG đại diện cho benchmark cuối; mục tiêu
  inference ≤1s/ảnh phải đo trực tiếp (Task 6 §D).
- KHÔNG dùng ảnh đó để tuyên bố accuracy thực tế ≥90% (thiếu test độc lập).
