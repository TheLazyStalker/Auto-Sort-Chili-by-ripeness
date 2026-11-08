# TASK 6 — DEPLOYMENT & BENCHMARK (điện thoại Android (mẫu đã test) / Termux) — RESULTS

## 1. Trạng thái chung

- Package deployment hoàn chỉnh: `deployment/task6/` (models + scripts + README + requirements + parity_check).
- **Verification trong container đã hoàn thành** (chi tiết §3).
- **Benchmark trên điện thoại thật CHƯA thực hiện** — cần thao tác của người dùng trên điện thoại Android (mẫu đã test) (xem §6). Không suy luận hiệu năng từ container.

## 2. Files trong package

```
deployment/task6/
├── models/chili_float16.tflite (1,913,292 bytes, SHA256 5cf37ac5…)
├── models/chili_float32.tflite (3,746,140 bytes)
├── models/class_mapping.json    ({"unripe":0,"medium_ripe":1,"ripe":2})
├── predict.py     # CLI đơn/--dir/--out CSV; timings; crop_v1 built-in
├── benchmark.py   # warmup + n runs; load_time; mean/median/p95/min/max
├── check_env.py   # Android/ABI/Python/storage/RAM/packages
├── requirements.txt  # numpy<2, Pillow, tflite-runtime
├── README.md      # hướng dẫn từng bước cho người dùng
└── parity_check.json (kết quả §3)
```

## 3. Verification container (parity — đã chạy)

Trên 92 ảnh dev validation, `predict.py` (deployment, Float16) so với container:

| Chỉ số | Giá trị |
|---|---|
| Deployment accuracy (dev) | 0.9891 (91/92) |
| Agreement vs TFLite container (cùng model) | 1.0000 (max |Δprob| = 0.0) |
| Agreement vs Keras gốc | 1.0000 (max |Δprob| = 0.0146 — do Float16) |
| Input tensor | (1,224,224,3) float32, RGB [0,255] |
| Preprocessing | khớp pipeline Task 4/5: Rescaling(1/128,-1) nhúng trong model; không double-scale; crop_v1 cùng tham số |

Kết luận: **pipeline deploy khớp chính xác pipeline train/eval**.

## 4. Kết quả test & integrity

- `pytest -q`: **118 passed / 0 failed** (thêm `tests/test_deployment.py`: load model F32/F16, single predict, dir+CSV, invalid-image handling, crop 1024→224, benchmark).
- Raw SHA256: **473/473 khớp baseline** (không đổi file raw).
- Git: chưa commit (Task 3–6); không reset/clean/stash; chỉ báo cáo.

## 5. Benchmark container (THAM KHẢO, không phải thiết bị)

Float16, 20 runs, warmup 3: mean 2.22ms, median 2.15ms, P95 2.76ms, min 1.93ms, max 2.84ms; load 6.4ms. **Không dùng để kết luận cho điện thoại Android (mẫu đã test).**

## 6. CÁC BƯỚC CẦN NGƯỜI DÙNG CHẠY TRÊN ĐIỆN THOẠI

> Tuần tự, mỗi bước kiểm tra kết quả trước khi sang bước sau.

**Bước 1 — Cài Termux** (F-Droid/GitHub releases). Mở Termux:

```bash
pkg update -y && pkg upgrade -y
pkg install -y python python-pip
```

**Bước 2 — Copy package** lên Termux home:

```bash
# Trong Termux:
termux-setup-storage
mkdir -p ~/task6
# Copy toàn bộ thư mục deployment/task6/* vào ~/task6/
# (dùng cp qua ~/storage/downloads, adb, hoặc cloud — tùy bạn)
```

**Bước 3 — Cài dependencies:**

```bash
cd ~/task6
pip install 'numpy<2.0.0' Pillow tflite-runtime
```

> Nếu `tflite-runtime` không có wheel cho ARM64/Termux: thêm index
> `pip install --extra-index-url https://termux-user-repository.github.io/packages/ tflite-runtime`.
> Nếu vẫn fail → chạy `python check_env.py` và báo cáo lỗi (runtime C/C++ là phương án dự phòng).

**Bước 4 — Kiểm tra môi trường:**

```bash
python check_env.py
```

Mong đợi: `android_version=11`, `abi=arm64-v8a`, `python=3.x`, storage > 100MB.

**Bước 5 — Kiểm tra tính đúng đắn (parity trên máy):**

Copy vài ảnh derivative dev sang máy (`deployment/task6/test_images/` — không kèm trong package do tránh phình; tạo từ `dataset/derived/crop_v1/train/<class>/*.png`), chạy:

```bash
python predict.py --model models/chili_float16.tflite --dir test_images/ --out test_out.csv
```

So sánh output với `artifacts/task5/evaluation_metrics.json` (cùng ảnh): class giống, sai số nhỏ.

**Bước 6 — Benchmark:**

```bash
python benchmark.py --model models/chili_float16.tflite --warmup 10 --runs 100 --out bench_f16.json
python benchmark.py --model models/chili_float32.tflite --warmup 10 --runs 100 --out bench_f32.json
```

**Bước 7 — Chạy liên tục (RAM/ổn định):** xử lý ≥100 ảnh
`python predict.py --dir <dir_100_ảnh> --out stable.csv`; quan sát crash/OOM.

**Bước 8 — Báo cáo kết quả** cho tôi (gửi 3 file JSON + `check_env.py` output) để tôi tổng hợp benchmark chính thức.

## 7. Đánh giá mục tiêu

- Inference ≤1s/ảnh: **chưa xác nhận** cho đến khi có kết quả Bước 6 trên máy; container chỉ là tham khảo.
- Throughput 1 quả/2s: chỉ đánh giá được **inference/image-processing**; throughput toàn hệ thống cần camera OV2640 + ESP32 (Task 7) — không tuyên bố.
- Accuracy ≥90%: dev 0.9891 nhưng không phải benchmark độc lập → không claim.

## 8. Model đề xuất cho Task 7 (sơ bộ)

Float16 (nhỏ hơn 49% so với Float32, parity 1.0 vs Keras). Quyết định cuối dựa trên kết quả benchmark thật (Bước 6).
