# PROJECT_STATUS — Báo cáo khảo sát toàn diện (lập ngày 2026-11-06)

> Báo cáo này được **tự khám phá và xác minh thực tế** — không dựa vào báo cáo cũ.
> Mọi trạng thái dưới đây đều có bằng chứng từ mã nguồn, lệnh chạy thực tế hoặc tài liệu.
> **Không thay đổi mã nguồn; không tạo dữ liệu giả; không huấn luyện/export.**

---

## 0. Tóm tắt điều hành

Dự án **ở giai đoạn nền tảng**: khung toàn bộ pipeline AI (dataset → train → evaluate → predict → export TFLite) **đã viết xong và đã kiểm thử ở mức unit** (28 test pass), nhưng **chưa có bất kỳ dữ liệu ảnh thật nào**, do đó **chưa có model đã huấn luyện**, **chưa có .tflite thật**, và **mọi giai đoạn hạ nguồn (Android, ESP32-S3, phần cứng) đều chưa bắt đầu**.

**Mức độ sẵn sàng của dự án: ~15–20%** (khung AI + môi trường đã sẵn sàng, bị chặn hoàn toàn bởi thiếu dataset ảnh thật).

---

## 1. Bằng chứng khảo sát thực tế (đã xác minh)

| Hạng mục | Kết quả chạy thực tế | Kết luận |
|---|---|---|
| Container | `podman start ai-workspace` OK; Python 3.12.3, TF 2.17.0 (CPU-only), numpy 1.26.4 | Môi trường hoạt động |
| Dataset | `python scripts/check_dataset.py` → 0 ảnh ở cả 3 lớp; `find dataset -type f = 0` | **Dataset trống** |
| Tests | `pytest -q` → **28 passed** (43.74s) | Unit tests pass |
| Smoke test | `python scripts/smoke_test_model.py` → `SMOKE TEST PASSED`, output `(2,3)` softmax; nhưng in `downloaded: False` | Model dựng được; **script kiểm tra sai tên weights** |
| Model đã train | `models/checkpoints/`, `models/saved_model/` rỗng; không có `.keras`/`.tflite`/`.pb` | **Chưa có model huấn luyện** |
| TFLite | `models/tflite/` chỉ có `labels.txt`, `model_metadata.json`, `android_integration.md` (tài liệu) | **Chưa có .tflite thật** |
| Git | `git status` → `fatal: not a git repository` | **Chưa có version control** |
| Android | `glob **/*.{kt,gradle,ino}` → không có file; chỉ có snippet tài liệu | **Chưa có app** |
| ESP32-S3 / phần cứng | Không có `.ino`/firmware/sơ đồ | **Chưa có** |

### Lỗi/thiếu sót đã xác minh trực tiếp

1. **Bug chia tập nhỏ (P0):** probe `stratified_group_split` thực tế:
   - `n=2/lớp` → train=3, val=0, test=3
   - `n=3/lớp` → train=6, val=0, test=3
   - `n=5/lớp` → train=12, val=3, test=0
   - `n=7/lớp` → OK (15/3/3)
   → Với dataset nhỏ (< ~7 ảnh/lớp), val hoặc test **rỗng**; khi đó `build_tf_datasets` sẽ gặp split rỗng (crash `images contains no shape`).
2. **Mâu thuẫn tiền xử lý (P0):** code dùng `preprocess_input` ([0,255]→[-1,1]) — đúng; nhưng `android_integration.md` mục 2 + `model_metadata.json` mẫu vẫn mô tả `Rescaling(1/255)` → [0,1] (sai). Nếu Android chia 255 thì sẽ double-scaling.
3. **Smoke test báo sai weights (P1):** `smoke_test_model.py` kiểm tra file `..._tf_kernels.h5` (không tồn tại) trong khi file thật là `..._1.0_224_no_top.h5` (có trong cache `~/.keras/models/`, 9.4MB). In `downloaded: False` sai.
4. **Fail-safe evaluate/predict chưa đủ (P1):**
   - `evaluate.py` thiếu model → báo đúng (RuntimeError về manifest) nhưng **exit code = 0** (không phải cảnh báo/thoát lỗi rõ).
   - `predict.py` thiếu model → **traceback `ValueError: File not found`** (không có thông điệp hướng dẫn thân thiện).
5. **`evaluate.py` không kiểm tra model trước `load_model`** — nếu model không tồn tại sẽ `load_model` → crash.

---

## 2. Tổng quan kiến trúc

```
dataset/raw/{unripe,medium_ripe,ripe}/  →  scan_raw_dataset (Pillow + md5)
        → stratified_group_split (70/15/15, seed=42, hỗ trợ groups.csv)
        → split_manifest.json → tf.data (augment nhẹ, không hue/sat)
        → MobileNetV2 ImageNet + GAP + Dropout(0.3) + Dense(3, softmax)
        → 2 giai đoạn: freeze → fine-tune 30 layer cuối
        → evaluate (accuracy, P/R/F1, confusion matrix)
        → export_tflite (Float32/Float16/INT8 + validate/compare)
        → Android (chưa có) → ESP32-S3 (chưa có)
```

**Đánh giá tính hợp lý kiến trúc tham khảo:**

- **Số lớp:** code dùng đúng **3 lớp** (`num_classes=3`, output Dense(3, softmax)). Sơ đồ "Dense 4" trong đề bài là **không phù hợp** — dự án chính xác dùng 3.
- **MobileNetV2:** triển khai đúng (`applications.MobileNetV2`, include_top=False, ImageNet). `preprocess_input` nhúng trong model → nhận [0,255] float32, tự map [-1,1].
- **GAP + Dropout + Dense Softmax:** hợp lý, khớp tham khảo.
- **Transfer learning:** freeze_backbone đúng (mở N layer cuối ở giai đoạn 2).
- **Một điểm cần lưu ý (không phải lỗi):** `train.py` gán `class_weight` trong cả 2 giai đoạn fit — hợp lý với data mất cân bằng, nhưng cần kiểm chứng với data thật.

---

## 3. Trạng thái 10 giai đoạn

| # | Giai đoạn | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Nền tảng & cấu trúc | ✅ Khung hoàn thành | src/ scripts/ tests/ config.yaml, tài liệu |
| 2 | Dataset & gán nhãn | ⛔ Bị chặn | **0 ảnh thật**; guide đã soạn |
| 3 | AI build/train/tối ưu | 🔵 Chưa làm | Chờ dataset; bug split nhỏ chưa sửa |
| 4 | Đánh giá mô hình | 🔵 Chưa làm | Cần model đã train |
| 5 | Xuất & kiểm thử TFLite | 🔵 Chưa có model thật | Code + unit test sẵn sàng; không có .tflite thật |
| 6 | Phát triển Android | 🔵 Chưa làm | Chỉ có tài liệu + snippet |
| 7 | Giao tiếp Android–ESP32-S3 | 🔵 Chưa làm | Chỉ mô tả JSON trong tài liệu |
| 8 | Firmware & phần cứng | 🔵 Chưa làm | Không có phần cứng thật |
| 9 | Tích hợp & kiểm thử hệ thống | 🔵 Chưa làm | — |
| 10 | Hoàn thiện & trình diễn | 🔵 Chưa làm | — |

**Lưu ý phân biệt:** "đã viết" ≠ "đã kiểm thử" ≠ "đã hoạt động thực tế".
- Code AI: ✅ đã viết, ✅ đã test unit (28 pass), ❌ **chưa** hoạt động với dữ liệu thật.
- Export TFLite: ✅ code + unit test, ❌ chưa có model thật để export.
- Android/ESP32: ❌ chỉ có tài liệu mô tả, không có code.

---

## 4. Danh sách công việc còn thiếu (ưu tiên)

### GĐ 0 — Sửa blocker đã biết (làm trước, không cần đầu vào ngoài)
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 0.1 | Sửa `stratified_group_split` bỏ `round()`, dùng largest-remainder, đảm bảo val/test ≥1/lớp khi n≥3; warning khi dataset quá nhỏ | P0 | — | Probe n=2,3,5 không còn val/test rỗng; test mới pass |
| 0.2 | Sửa `build_tf_datasets`/`make_ds` guard split rỗng (skip + cảnh báo) thay vì crash | P0 | 0.1 | Không traceback trên manifest có split rỗng |
| 0.3 | Chốt preprocessing [0,255]→[-1,1] làm chuẩn; sửa `android_integration.md` mục 2 + snippet Kotlin (bỏ chia 255) và `model_metadata.json` | P0 | — | `grep` không còn `Rescaling(1/255)` trong docs |
| 0.4 | Sửa `smoke_test_model.py` đường dẫn weights → `..._1.0_224_no_top.h5` | P1 | — | Smoke in `downloaded: True` |
| 0.5 | Fail-safe `evaluate.py`/`predict.py`: kiểm tra model tồn tại trước `load_model`, exit code khác 0 khi thiếu | P1 | — | Không traceback; exit 2 |
| 0.6 | `git init` + commit hiện trạng (hiện chưa phải git repo) | P2 | — | Lịch sử có sẵn |

### GĐ 1 — Dữ liệu
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 1.1 | Chốt giống ớt, điều kiện chụp, tiêu chí nhãn `medium_ripe` vs `ripe` | P0 | Q1/Q2 | Checklist guide được duyệt |
| 1.2 | Thu thập ảnh thật ≥100–200/lớp → `dataset/raw/<class>/` | P0 | 1.1 | `check_dataset.py` báo ≥100/lớp, 0 lỗi |
| 1.3 | Tạo `dataset/groups.csv` (mỗi quả 1 group) + trỏ `paths.groups_file` | P1 | 1.2 | Test group-leak pass |
| 1.4 | Chạy `check_dataset.py --prepare` tạo manifest, kiểm tra trùng md5 | P1 | 1.2–1.3, 0.1 | Manifest có; splits ~70/15/15 |

### GĐ 2 — AI
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 2.1 | Huấn luyện `train.py` 2 giai đoạn trên data thật | P0 | 1.4 | `best_chili.keras`/`chili_final.keras` + `training_summary.json` |
| 2.2 | `evaluate.py --split test`: accuracy, P/R/F1, confusion matrix | P0 | 2.1 | `evaluation_test.json` số liệu thật |
| 2.3 | `predict.py --image` end-to-end (class_id, label, confidence, uncertain) | P1 | 2.1 | JSON đúng schema |
| 2.4 | Tối ưu dropout/fine_tune/lr/augmentation/class_weight | P2 | 2.2 | Báo cáo so sánh trước/sau |
| 2.5 | Hồi quy: giữ 28 pass + test split nhỏ n=5 + guard rỗng | P1 | 0.1–0.2, 2.1 | `pytest` pass |

### GĐ 3 — TFLite + Android
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 3.1 | Export Float32/Float16/INT8 (INT8 cần train thật để calibration) | P1 | 2.1 | 3 file .tflite hiện hữu; `same_label=True` |
| 3.2 | Đồng bộ `labels.txt`/`model_metadata.json`; quyết định MediaPipe vs Task Library | P1 | 0.3, 3.1 | Metadata khớp preprocessing |
| 3.3 | Viết Android app Kotlin (load .tflite + labels, camera/upload, preprocess, hiển thị) | P1 | 3.1–3.2 | App build được; predict khớp `predict.py` |
| 3.4 | Đo latency/accuracy trên thiết bị thật (Float32 vs Float16 vs INT8) | P2 | 3.3 | Bảng đo; kết luận theo số liệu |

### GĐ 4 — Android ↔ ESP32-S3
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 4.1 | Chốt protocol JSON (UART/BLE, baud, retry khi uncertain) | P1 | Q4, 3.3 | Spec 1 trang được duyệt |
| 4.2 | Android serialize + gửi/poll log | P1 | 3.3, 4.1 | Log khớp output predict |

### GĐ 5–6 — Phần cứng + firmware
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 5.1 | Chốt BOM (camera, băng tải, servo, nguồn, khung) | P1 | Q3/Q5 | BOM + sơ đồ wiring duyệt |
| 5.2 | Chốt kiến trúc camera (Android native vs ESP32-CAM) | P1 | 5.1 | Kiến trúc ghi vào roadmap |
| 6.1 | Firmware ESP32-S3 (đọc JSON, điều khiển servo/băng tải, xử lý uncertain) | P1 | 4.1, 5.1 | Nạp thành công lên ESP32 thật |
| 6.2 | Tích hợp end-to-end + đo tỉ lệ đúng/latency | P1 | 3.3, 4.2, 6.1 | Demo thật trên ≥50 quả |

### GĐ 7 — Kiểm thử + tài liệu + demo
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 7.1 | Test toàn hệ thống (pytest + smoke + export compare + Android/ESP thật) | P1 | GĐ 2–6 | Checklist test qua từng tầng |
| 7.2 | Cập nhật README/guide/android_integration với số liệu thật | P2 | 2.2, 3.4, 6.2 | Không còn mô tả [0,1] sai |
| 7.3 | Chuẩn bị trình diễn (video, slide, tái lập pipeline) | P2 | 7.1–7.2 | Người khác chạy lại được |

---

## 5. Thông tin cần người dùng cung cấp (đang chặn)

| Q | Câu hỏi | Chặn giai đoạn |
|---|---|---|
| Q1 | Giống ớt + điều kiện chụp (ánh sáng, nền, khoảng cách)? | 1.1, 1.2 |
| Q2 | Tiêu chí nhãn `medium_ripe` vs `ripe` (giữ >90% diện tích?)? | 1.1 |
| Q3 | Phần cứng thật hay mô phỏng? Có ESP32-S3/camera/băng tải/servo chưa? Ngân sách? | GĐ 5–6 |
| Q4 | Giao tiếp Android–ESP32: UART hay BLE? Cần cloud/logging? | 4.1 |
| Q5 | Camera: Android native hay module riêng (OV5640/ESP32-CAM)? Resolution/FPS? | 5.1–5.2 |
| Q6 | Target accuracy/latency/throughput tối thiểu? | GĐ 2, 6 |
| Q7 | Có được dùng ảnh nguồn công khai không, hay chỉ ảnh tự chụp? | 1.2 |
| Q8 | Môi trường train CPU-only đủ không, hay dự định GPU? | GĐ 2 |

---

## 6. Kết luận và bước tiếp theo

**Bước tiếp theo khuyến nghị (không cần chờ đầu vào ngoài):**
1. Sửa GĐ 0.1–0.5 (bug split nhỏ, crash rỗng, mâu thuẫn preprocessing, smoke weights, fail-safe) + `git init`.
2. Trong lúc chờ ảnh: chốt Q1/Q2 với người dùng → thu thập ảnh thật → `--prepare`.

**Rủi ro chính:**
- **Blocking lớn nhất: không có dataset ảnh thật** → toàn bộ GĐ 2+ bị chặn.
- Bug split nhỏ hiện hữu có thể cho val/test rỗng khi chưa đủ ảnh → cần sửa trước khi data về.
- Mâu thuẫn preprocessing tài liệu có thể gây sai khi triển khai Android nếu không chốt sớm.
- Chưa có git → rủi ro mất lịch sử khi chỉnh sửa tiếp.

*Báo cáo này chỉ ghi nhận hiện trạng; không thay đổi mã nguồn hay dữ liệu.*
