# TERMUX_DEPLOYMENT — Kiến trúc triển khai CLI/native trên Android (Termux)

> **Cập nhật:** 2026-11-06
>
> **Trạng thái:** KIẾN TRÚC MỚI (thay thế Android app/Kotlin). Tài liệu `models/tflite/android_integration.md` (Kotlin/APK/Android Studio/MediaPipe) **đã deprecated** — xem mục 7.

## 1. Thay đổi mục tiêu triển khai

- **BỎ HOÀN TOÀN** hướng Android app (APK/Kotlin/Android Studio/MediaPipe).
- **Thay bằng:** chương trình native dạng CLI/terminal chạy trực tiếp trong **Termux** trên điện thoại Android.
- Ưu tiên: đơn giản, ít phụ thuộc, dễ cài đặt & sử dụng.
- Yêu cầu: nhận ảnh đầu vào → chạy model phân loại 3 lớp (`unripe`/`medium_ripe`/`ripe`) → trả nhãn + độ tin cậy.
- Giữ khả năng mở rộng giao tiếp ESP32-S3 sau này (chưa triển khai ở task này).

## 2. Ràng buộc nền tảng (Termux)

Termux chạy trên **Android kernel + Bionic libc**, KHÔNG phải GNU/Linux glibc thông thường. Do đó:

- **Không mặc định** chương trình Linux thông thường / Python package hiện tại sẽ chạy được trong Termux.
- Các wheel Python prebuilt cho Linux glibc (kể cả `tensorflow`, `tflite-runtime` từ PyPI) **không tương thích** trực tiếp với Bionic.
- Cần runtime được build riêng cho Android/Bionic oraarch64.

## 3. Khảo sát runtime (đã nghiên cứu)

| Runtime | Tương thích Termux/Bionic | Cách cài | Kích thước | Độ phức tạp | Ghi chú |
|---|---|---|---|---|---|
| **`python-tflite-runtime` (gói Termux chính thức)** | ✅ Có (package `python-tflite-runtime` 2.20.0-4 có bản aarch64/arm trong mirror Termux-main) | `pkg install python-tflite-runtime` | ~1.5 MiB | Thấp | Đọc trực tiếp `.tflite`; khớp pipeline export hiện tại; được build cho Android/Bionic |
| **LiteRT/TFLite C/C++** (build với NDK) | ✅ Có nếu tự build | Cần Android NDK / cmake; hoặc dùng AAR Maven | Nhẹ (~2–10 MB) | Cao | Native, hiệu năng tốt; phức tạp build |
| **ncnn** | ✅ Có (framework mobile-first; có thể build aarch64) | Cần build từ source hoặc dùng gói | ~7.5 MiB (archlinux-arm) | Trung bình | Cần convert model → ncnn param/bin; không dùng thẳng `.tflite` |
| **TensorFlow full (Python)** | ❌ Không (prebuilt wheel dành glibc) | — | — | — | Không khả thi trong Termux |
| **PyPI `tflite-runtime`** | ❌ Không (wheel dành Linux glibc) | — | — | — | Không tương thích Bionic |

### Đề xuất runtime (ưu tiên)

**1. Lựa chọn mặc định: `python-tflite-runtime` (gói Termux chính thức).**
- Lý do: (a) là gói chính thức trong Termux-main, build cho Android/Bionic (aarch64), (b) nhẹ (~1.5 MiB), ít phụ thuộc, (c) đọc trực tiếp `.tflite` mà pipeline hiện tại (`export_tflite.py`) đã tạo — **không cần convert model**, (d) giảm đáng kể độ phức tạp so với build C++.
- Python CLI đơn giản: đọc ảnh → preprocess 224×224 (giữ [0,255]) → `tflite_runtime.Interpreter` → softmax → nhãn/confidence.

**2. Lựa chọn thay thế khi cần hiệu năng/native: LiteRT C/C++ (hoặc ncnn).**
- Nếu sau này cần tốc độ tối đa / ít overhead Python, hoặc cần tích hợp sâu, chuyển sang LiteRT C++ (build NDK) hoặc ncnn.
- Lưu ý: ncnn yêu cầu chuyển đổi model (`.tflite` → ncnn params/bin qua `onnx2ncnn` hoặc tool tương tự) — thêm một bước so với `.tflite` trực tiếp.

## 4. Chuẩn tiền xử lý (bất biến)

- Model MobileNetV2 nhận **RGB uint8 `[0,255]`**, 224×224×3, và **tự** `preprocess_input` map `[0,255] → [-1,1]` bên trong.
- **KHÔNG được `Rescaling(1/255)`** (sẽ double-scaling).
- Điều này phải khớp giữa training, `predict.py`, và CLI Termux (xem `models/tflite/model_metadata.json`).

## 5. Kiến trúc triển khai mới

```
[Globalnh/ảnh trên điện thoại]
        │
        ▼
[Termux: CLI program (Python + python-tflite-runtime)]
   - đọc ảnh (Pillow/Termux API)
   - resize 224×224, giữ [0,255]
   - tflite_runtime.Interpreter chạy .tflite
   - softmax 3 lớp → label + confidence (+ uncertain nếu < ngưỡng)
        │
        ▼
[In ra terminal]  (JSON/text: {label, class_id, confidence, uncertain})
        │
        ▼  (khả năng mở rộng sau này, CHƯA triển khai)
[ESP32-S3: nhận kết quả qua UART/BLE → điều khiển cơ cấu]
```

## 6. Kế hoạch chuyển đổi (Task 2+)

| Bước | Nội dung | Trạng thái |
|---|---|---|
| A | Xác minh model `.tflite` export từ `export_tflite.py` (Float32) đọc được bằng `python-tflite-runtime` | Chưa làm (chờ dataset → model thật) |
| B | Viết CLI Termux (Python): lấy ảnh, preprocess, inference, in kết quả | Chưa làm |
| C | Cài đặt trong Termux: `pkg install python-tflite-runtime pillow` + copy model/labels | Chưa làm |
| D | Đo latency/accuracy trên thiết bị thật (CPU trước) | Chưa làm |
| E | Thiết kế giao tiếp ESP32-S3 (protocol), **chưa** implement | Chưa làm |
| F | (Tùy chọn) Đánh giá LiteRT C++ / ncnn nếu cần hiệu năng | Chưa làm |

## 7. Tài liệu cũ — trạng thái deprecated

- **`models/tflite/android_integration.md`**: hướng dẫn Kotlin/Android Studio/MediaPipe/Gradle — **DEPRECATED** (không còn trong phạm vi). Giữ lại để tham khảo, không dùng làm lộ trình.
- **`model_metadata.json` / `labels.txt`**: vẫn còn giá trị (mô tả input/output, thứ tự lớp) — giữ nguyên.
- **`README.md`**: sẽ cập nhật hướng dẫn triển khai mới ở task tiếp theo.

## 8. GPU

- **GPU training** và **GPU inference** là hai vấn đề hoàn toàn khác nhau (xem `PROJECT_STATUS.md` § GPU).
- Trên điện thoại (Termux): **ưu tiên CPU inference**. GPU delegate (Adreno/Mali) cho Termux hiện chưa có gói sẵn sàng và chưa được kiểm chứng — chỉ đề xuất sau khi có phương án thực tế, dễ triển khai và kiểm chứng được.
