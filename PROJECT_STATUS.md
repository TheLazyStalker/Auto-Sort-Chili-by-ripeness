# PROJECT_STATUS — Báo cáo khảo sát toàn diện (lập ngày 2026-11-06, cập nhật sau TASK 1)

> Báo cáo này được **tự khám phá và xác minh thực tế** — không dựa vào báo cáo cũ.
> Mọi trạng thái dưới đây đều có bằng chứng từ mã nguồn, lệnh chạy thực tế hoặc tài liệu.
> **Không thay đổi mã nguồn; không tạo dữ liệu giả; không huấn luyện/export.**

---

## 0. Tóm tắt điều hành

Dự án **ở giai đoạn nền tảng**: khung toàn bộ pipeline AI (dataset → train → evaluate → predict → export TFLite) **đã viết xong và đã kiểm thử ở mức unit** (36 test pass), nhưng **chưa có bất kỳ dữ liệu ảnh thật nào**, do đó **chưa có model đã huấn luyện**, **chưa có .tflite thật**, và **mọi giai đoạn hạ nguồn (Android, ESP32-S3, phần cứng) đều chưa bắt đầu**.

**Mức độ sẵn sàng của dự án: ~20–25%** (khung AI + môi trường + fix P0/P1/P2 đã hoàn thành; vẫn bị chặn hoàn toàn bởi thiếu dataset ảnh thật).

---

## 1. Bằng chứng khảo sát thực tế (đã xác minh)

| Hạng mục | Kết quả chạy thực tế | Kết luận |
|---|---|---|
| Container | `podman start ai-workspace` OK; Python 3.12.3, TF 2.17.0 (CPU-only), numpy 1.26.4 | Môi trường hoạt động |
| Dataset | `python scripts/check_dataset.py` → 0 ảnh ở cả 3 lớp; `find dataset -type f = 0` | **Dataset trống** |
| Tests | `pytest -q` → **36 passed** (42.7s) | Unit tests pass (28 cũ + 8 mới) |
| Smoke test | `python scripts/smoke_test_model.py` → `SMOKE TEST PASSED`, báo `downloaded: True` (weights no_top tồn tại) | Model dựng được; smoke trỏ đúng file weights |
| Model đã train | `models/checkpoints/`, `models/saved_model/` rỗng; không có `.keras`/`.tflite`/`.pb` | **Chưa có model huấn luyện** |
| TFLite | `models/tflite/` chỉ có `labels.txt`, `model_metadata.json`, `android_integration.md` (tài liệu) | **Chưa có .tflite thật** |
| Git | `git init` xong; nhánh `main`; commit `2402f86` (42 file); `.gitignore` hoàn thiện | **Đã có version control** |
| Android | `glob **/*.{kt,gradle,ino}` → không có file; chỉ có snippet tài liệu | **Chưa có app** |
| ESP32-S3 / phần cứng | Không có `.ino`/firmware/sơ đồ | **Chưa có** |

### Lỗi/thiếu sót đã XỬ LÝ trong TASK 1 (kèm bằng chứng)

| # | Lỗi | Trạng thái | Bằng chứng |
|---|---|---|---|
| P0-1 | `stratified_group_split` dùng `round()` → val/test rỗng với dataset nhỏ | ✅ Đã sửa | Dùng largest-remainder + đảm bảo ≥1 group/split khi đủ 3 group; n=3,5,7,20 → cả 3 tập không rỗng; n=2 → warning rõ. Probe + test `test_split_small_n_gives_nonempty_when_possible` |
| P0-2 | `build_tf_datasets`/`make_ds` crash trên split rỗng (`images contains no shape`) | ✅ Đã sửa | Guard split rỗng → dataset rỗng + cảnh báo; `test_build_tf_datasets_empty_split_no_crash` pass |
| P0-3 | `tf.image.decode_image` mất static shape trong graph mode → crash với cả split không rỗng | ✅ Đã sửa | Thêm `_decode_image_safe` (`expand_animations=False` + `set_shape([None,None,3])`); vi `build_tf_datasets` với ảnh thật |
| P0-4 | Đảm bảo không leak group (cùng quả ớt lọt nhiều tập) | ✅ Đã kiểm chứng | `test_split_no_overlap_even_with_groups` + probe group integrity pass |
| P0-5 | Mâu thuẫn preprocessing: docs nói `Rescaling(1/255)` [0,1] | ✅ Đã sửa | `android_integration.md` mục 2 + snippet Kotlin + `model_metadata.json` đồng bộ `[0,255]→preprocess_input[-1,1]`; `test_android_docs_use_0_255_preprocessing` pass |
| P1-6 | Smoke test kiểm tra sai tên file weights | ✅ Đã sửa | Trỏ đúng `..._1.0_224_no_top.h5`; smoke in `downloaded: True`; `test_smoke_checks_correct_weights_filename` pass |
| P1-7 | `evaluate.py`/`predict.py` thiếu model → exit code sai / traceback | ✅ Đã sửa | Thêm kiểm tra `os.path.exists` + `FileNotFoundError`; CLI bắt lỗi → exit 2, không traceback; `test_evaluate_missing_model_exits_2`, `test_predict_missing_model_exits_2`, `test_evaluate_load_model_checks_existence_first` pass |
| P2-8 | Chưa có Git | ✅ Đã sửa | `git init` → `main`, commit `2402f86`, `.gitignore` hoàn thiện (dataset, cache, model, artifact, Android build, OS) |

### Lỗi/blocker CÒN LẠI

| # | Vấn đề | Trạng thái |
|---|---|---|
| B1 | **Không có dataset ảnh thật** (0 ảnh) → không thể train/evaluate/export thật | Bị chặn (cần ảnh từ người dùng) |
| B2 | `evaluate.py` vẫn cần manifest — khi thiếu manifest báo exit 2 (đã cải thiện) nhưng chưa có model thật để chạy end-to-end | Chờ dataset |
| B3 | INT8 TFLite cần ảnh train thật để calibration — tạm bỏ qua khi chưa có dataset | Chờ dataset |
| B4 | Android app, giao tiếp ESP32-S3, firmware, phần cứng: hoàn toàn chưa làm | Chờ quyết định Q1–Q8 |
| B5 | `predict.py` vẫn có `CLASSES_FALLBACK` (an toàn) — không ảnh hưởng pipeline | Không phải lỗi |

---

## 2. Tổng quan kiến trúc

```
dataset/raw/{unripe,medium_ripe,ripe}/  →  scan_raw_dataset (Pillow + md5)
        → stratified_group_split (70/15/15, largest-remainder, seed=42, groups.csv)
        → split_manifest.json → tf.data (build_tf_datasets chống split rỗng)
        → MobileNetV2 ImageNet + GAP + Dropout(0.3) + Dense(3, softmax)
        → 2 giai đoạn: freeze → fine-tune 30 layer cuối
        → evaluate (accuracy, P/R/F1, confusion matrix)
        → export_tflite (Float32/Float16/INT8 + validate/compare)
        → Android (chưa có) → ESP32-S3 (chưa có)
```

**Đánh giá tính hợp lý kiến trúc tham khảo:**

- ✅ Số lớp đúng **3** (không phải Dense 4).
- ✅ MobileNetV2 ImageNet + GAP + Dropout + Dense(3,softmax) triển khai đúng.
- ✅ `preprocess_input` nhúng trong model → nhận [0,255], tự map [-1,1]; đã đồng bộ docs/Android.
- ✅ Transfer learning 2 giai đoạn hoạt động đúng.

---

## 3. Trạng thái 10 giai đoạn

| # | Giai đoạn | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Nền tảng & cấu trúc | ✅ Hoàn thành (khung) | src/ scripts/ tests/ config.yaml + Git init |
| 2 | Dataset & gán nhãn | ⛔ Bị chặn | **0 ảnh thật**; guide đã soạn |
| 3 | AI build/train/tối ưu | 🔵 Chưa làm | Chờ dataset; đã sửa P0 split/decode |
| 4 | Đánh giá mô hình | 🔵 Chưa làm | Cần model đã train |
| 5 | Xuất & kiểm thử TFLite | 🔵 Chưa có model thật | Code + unit test sẵn sàng; không có .tflite thật |
| 6 | Phát triển Android | 🔵 Chưa làm | Chỉ có tài liệu + snippet (đã sửa preprocessing) |
| 7 | Giao tiếp Android–ESP32-S3 | 🔵 Chưa làm | Chỉ mô tả JSON trong tài liệu |
| 8 | Firmware & phần cứng | 🔵 Chưa làm | Không có phần cứng thật |
| 9 | Tích hợp & kiểm thử hệ thống | 🔵 Chưa làm | — |
| 10 | Hoàn thiện & trình diễn | 🔵 Chưa làm | — |

---

## 4. Danh sách công việc còn thiếu (ưu tiên)

### GĐ 1 — Dữ liệu
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 1.1 | Chốt giống ớt, điều kiện chụp, tiêu chí nhãn `medium_ripe` vs `ripe` | P0 | Q1/Q2 | Checklist guide được duyệt |
| 1.2 | Thu thập ảnh thật ≥100–200/lớp → `dataset/raw/<class>/` | P0 | 1.1 | `check_dataset.py` báo ≥100/lớp, 0 lỗi |
| 1.3 | Tạo `dataset/groups.csv` (mỗi quả 1 group) + trỏ `paths.groups_file` | P1 | 1.2 | Test group-leak pass |
| 1.4 | Chạy `check_dataset.py --prepare` tạo manifest, kiểm tra trùng md5 | P1 | 1.2–1.3 | Manifest có; splits ~70/15/15 |

### GĐ 2 — AI
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 2.1 | Huấn luyện `train.py` 2 giai đoạn trên data thật | P0 | 1.4 | `best_chili.keras`/`chili_final.keras` + `training_summary.json` |
| 2.2 | `evaluate.py --split test`: accuracy, P/R/F1, confusion matrix | P0 | 2.1 | `evaluation_test.json` số liệu thật |
| 2.3 | `predict.py --image` end-to-end (class_id, label, confidence, uncertain) | P1 | 2.1 | JSON đúng schema |
| 2.4 | Tối ưu dropout/fine_tune/lr/augmentation/class_weight | P2 | 2.2 | Báo cáo so sánh trước/sau |

### GĐ 3 — TFLite + Android
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 3.1 | Export Float32/Float16/INT8 (INT8 cần train thật để calibration) | P1 | 2.1 | 3 file .tflite hiện hữu; `same_label=True` |
| 3.2 | Đồng bộ labels/metadata; quyết định MediaPipe vs Task Library | P1 | 3.1 | Metadata khớp preprocessing |
| 3.3 | Viết Android app Kotlin (load .tflite + labels, camera/upload, preprocess) | P1 | 3.1–3.2 | App build được; predict khớp `predict.py` |
| 3.4 | Đo latency/accuracy trên thiết bị thật | P2 | 3.3 | Bảng đo; kết luận theo số liệu |

### GĐ 4–6 — ESP32 + phần cứng
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 4.1 | Chốt protocol JSON (UART/BLE, baud, retry khi uncertain) | P1 | Q4, 3.3 | Spec 1 trang được duyệt |
| 4.2 | Android serialize + gửi/poll log | P1 | 3.3, 4.1 | Log khớp output predict |
| 5.1–5.2 | Chốt BOM + kiến trúc camera | P1 | Q3/Q5 | BOM + wiring duyệt |
| 6.1 | Firmware ESP32-S3 | P1 | 4.1, 5.1 | Nạp thành công lên ESP32 thật |
| 6.2 | Tích hợp end-to-end + đo | P1 | 3.3, 4.2, 6.1 | Demo thật trên ≥50 quả |

### GĐ 7 — Test + tài liệu + demo
| # | Công việc | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 7.1 | Test toàn hệ thống | P1 | GĐ 2–6 | Checklist test qua từng tầng |
| 7.2 | Cập nhật tài liệu với số liệu thật | P2 | 2.2, 3.4, 6.2 | Không còn mô tả sai |
| 7.3 | Chuẩn bị trình diễn | P2 | 7.1–7.2 | Người khác chạy lại được |

---

## 5. Thông tin cần người dùng cung cấp (đang chặn)

| Q | Câu hỏi | Chặn giai đoạn |
|---|---|---|
| Q1 | Giống ớt + điều kiện chụp (ánh sáng, nền, khoảng cách)? | 1.1, 1.2 |
| Q2 | Tiêu chí nhãn `medium_ripe` vs `ripe`? | 1.1 |
| Q3 | Phần cứng thật hay mô phỏng? Ngân sách? | GĐ 5–6 |
| Q4 | Giao tiếp Android–ESP32: UART hay BLE? | 4.1 |
| Q5 | Camera: Android native hay module riêng? | 5.1–5.2 |
| Q6 | Target accuracy/latency/throughput? | GĐ 2, 6 |
| Q7 | Có được dùng ảnh nguồn công khai không? | 1.2 |
| Q8 | Môi trường train CPU-only đủ không? | GĐ 2 |

---

## 6. Kết luận và bước tiếp theo

**Đã hoàn thành trong TASK 1:** mọi fix P0/P1/P2 (split safety, decode_image static shape, preprocessing docs sync, fail-safe CLI, smoke weights path, git init). 36 test pass.

**Bước tiếp theo khuyến nghị:**
1. Chờ Q1/Q2 → thu thập ảnh thật → `--prepare` → train (2.1) → evaluate (2.2) → export (3.1).
2. Trong lúc chờ: chốt Q3–Q8 để chuẩn bị Android/ESP32.

**Rủi ro chính:**
- Blocking lớn nhất: không có dataset ảnh thật → GĐ 2+ bị chặn.
- Khi data về, phải xác minh `check_dataset.py --prepare` và `train.py` end-to-end (chưa thể chạy do thiếu ảnh).

*Báo cáo này ghi nhận hiện trạng sau TASK 1; không đổi kiến trúc, không tạo dữ liệu giả.*
