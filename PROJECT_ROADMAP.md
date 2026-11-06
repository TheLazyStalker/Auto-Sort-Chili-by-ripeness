# PROJECT_ROADMAP — chili-ripeness-ai

> Khảo sát thực tế trong container `ai-workspace` tại `/workspace/chili-ripeness-ai`
> (không chỉ dựa vào README). Quy ước trạng thái:
> **Đã hoàn thành** = đã viết mã + kiểm chứng thực tế; **Đang làm** = đã có mã/draft nhưng chưa kiểm chứng end-to-end hoặc đã biết có lỗi;
> **Chưa làm** = chưa có code/artefact; **Bị chặn** = không thể tiến triển nếu thiếu đầu vào bên ngoài (ảnh thật, phần cứng, quyết định người dùng).
> Độ ưu tiên: **P0** (blocker toàn hệ thống) > **P1** (blocker giai đoạn) > **P2** (nên có) > **P3** (polish).

## 0. Hiện trạng kiểm chứng (bằng chứng)

| Hạng mục | Bằng chứng thực tế | Trạng thái |
|---|---|---|
| Cấu trúc dự án | `src/` (10 file .py), `scripts/` (2 file), `tests/` (7 file, gồm test_failsafe.py), `config.yaml`, `requirements.txt`, `README.md`, `DATASET_GUIDE.md`, `models/tflite/` (chỉ .md/.txt/.json) | Đã hoàn thành (khung) |
| Dataset ảnh | `dataset/raw/{unripe,medium_ripe,ripe}/` **rỗng**, `dataset/train|validation|test/` rỗng, không có `split_manifest.json`; `find dataset -type f = 0` | Bị chặn (thiếu ảnh thật) |
| Model đã huấn luyện | `models/checkpoints/`, `models/saved_model/` rỗng; `reports/` rỗng; chưa có `.keras`/`.tflite` nào | Chưa làm (do thiếu dataset) |
| Tests | `pytest -q`: **36 passed** (khảo sát ngày 2026-11-06, 42.7s) | Đã hoàn thành (unit-level) |
| Smoke test | `python scripts/smoke_test_model.py` → `SMOKE TEST PASSED`, báo `downloaded: True` (đã sửa đường dẫn weights no_top) | Đã hoàn thành (random tensor) |
| Fail-safe khi thiếu đầu vào | `check_dataset.py` báo Dataset trống exit 0; `train` dừng an toàn SystemExit 2; `evaluate`/`predict` thiếu model → exit 2, thông báo rõ, không traceback (đã sửa) | Đã hoàn thành |
| Bug chia tập nhỏ (P0) | Đã sửa: largest-remainder + đảm bảo ≥1 group/split khi đủ 3 group. Probe: `n=3,5,7,20` → cả 3 tập không rỗng; `n=2` → warning rõ. `build_tf_datasets` chống split rỗng (không crash) | Đã hoàn thành |
| Mâu thuẫn tiền xử lý Android (P0) | Đã đồng bộ: `android_integration.md` mục 2 + snippet Kotlin + `model_metadata.json` đều dùng `[0,255]→preprocess_input[-1,1]`, không còn `Rescaling(1/255)` như chuẩn | Đã hoàn thành |
| Android app | Không có `*.kt`/`*.gradle`, không có `app/` — chỉ có tài liệu + snippet tham khảo trong `models/tflite/android_integration.md` | Chưa làm |
| ESP32-S3 / phần cứng | Không có `*.ino`/firmware, không có sơ đồ/diagram; chỉ có mô tả giao tiếp JSON trong mục 6 của `android_integration.md` | Chưa làm |
| Git/version control | `git init` → nhánh `main`, commit `2402f86` (42 file), `.gitignore` hoàn thiện (dataset, cache, model, artifact, Android build) | Đã hoàn thành |
| Môi trường | Python 3.12.3, TF 2.17.0 CPU, Keras 3.15.1, NumPy 1.26.4 (`<2.0.0`), pytest 9.2.0; weights ImageNet `mobilenet_v2_..._1.0_224_no_top.h5` (9.4 MB) đã cache; smoke test giờ kiểm tra đúng file | Đã hoàn thành |

## 1. Mục tiêu và kiến trúc tổng thể

**Mục tiêu:** Hệ thống phân loại quả ớt 3 mức chín (`unripe` → 0, `medium_ripe` → 1, `ripe` → 2) chạy thời gian thực trên Android, gửi kết quả cho ESP32-S3 để điều khiển cơ cấu phân loại (băng tải / servo / đèn / motor). Ranh giới `medium_ripe`/`ripe` là nhiệm vụ khó nhất vì phụ thuộc tỉ lệ màu chín đồng đều.

**Kiến trúc (đề xuất, suy ra từ code hiện có — chưa được triển khai phần cứng):**

```
[Camera/Capture] → [Android App: resize 224x224, preprocess, TFLite inference]
                 → [Android gửi JSON {label,class_id,confidence,uncertain} qua UART/BLE]
                 → [ESP32-S3: firmware điều khiển servo / băng tải / đèn]
                 → [Cơ cấu phân loại cơ khí]
```

**AI pipeline (đã có khung code):** `dataset/raw/<class>/` → `scan_raw_dataset` (Pillow + md5 trùng/lỗi) → `stratified_group_split` 70/15/15 (seed 42, hỗ trợ `groups.csv`) → `split_manifest.json` → `tf.data` (augment nhẹ, không đổi hue/saturation) → MobileNetV2 ImageNet + GAP + Dropout(0.3) + Dense(3, softmax), 2 giai đoạn (freeze → fine-tune 30 layer) → checkpoint `best_chili.keras` / `chili_final.keras` → `evaluate.py` (accuracy, P/R/F1, confusion matrix) → `export_tflite.py` (Float32/Float16/INT8 + validate/compare) → `.tflite` + `labels.txt` + `model_metadata.json` → Android.

**Nguyên tắc không phá vỡ:** không tạo ảnh giả, không xoá ảnh gốc, không bịa metric; chia tập theo `group_id` để tránh rò rỉ; màu là tín hiệu chính nên không augment hue/saturation mạnh.

## 2. Danh sách công việc theo giai đoạn

### GĐ 0 — Sửa lỗi blocker đã biết (quyền ưu tiên cao nhất trước khi thu thập dữ liệu)

| # | Công việc | Trạng thái | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|---|
| 0.1 | Sửa `stratified_group_split`: đảm bảo val/test tối thiểu ≥1 ảnh/lớp khi `n ≥ 3` (dùng largest-remainder thay vì `round()`), và báo warning rõ khi dataset quá nhỏ để chia 3 tập | Đã hoàn thành | P0 | — | Probe `n=2,3,5,7,20` không còn split rỗng khi có thể chia; `n=2` → warning rõ; test mới pass; `pytest` 36 pass |
| 0.2 | Sửa `build_tf_datasets`/`make_ds`: guard split rỗng (skip + cảnh báo) thay vì crash `images contains no shape`; fix `decode_image` mất static shape trong graph mode (`_decode_image_safe`) | Đã hoàn thành | P0 | 0.1 | `build_tf_datasets` với manifest có split rỗng không traceback; load ảnh thật hoạt động |
| 0.3 | Chốt tiền xử lý duy nhất: đọc `model.py` (`preprocess_input` [0,255]→[-1,1]) là chuẩn, sửa `android_integration.md` mục 2 + snippet Kotlin (không chia 255) và đồng bộ `model_metadata.json` mẫu; ghi quyết định vào README | Đã hoàn thành | P0 | — | `grep` không còn `Rescaling(1/255)` như chuẩn trong docs; snippet Android khớp `predict.py` |
| 0.4 | Sửa `smoke_test_model.py`: sửa đường dẫn weights `..._tf_kernels.h5` → `..._1.0_224_no_top.h5` (đúng file thực tế trong cache) và assert file tồn tại | Đã hoàn thành | P1 | — | Smoke in `downloaded: True`; không còn báo False dù weights có sẵn |
| 0.5 | Cải thiện fail-safe `evaluate.py`/`predict.py`: kiểm tra model file tồn tại trước `load_model`; CLI exit code 2 khi thiếu model, không traceback | Đã hoàn thành | P1 | — | Cả hai module in hướng dẫn rõ + exit code 2 khi thiếu model, không traceback |
| 0.6 | Khởi tạo Git (`git init` → main, commit hiện trạng), hoàn thiện `.gitignore` (dataset, cache, model, artifact, Android build, OS) | Đã hoàn thành | P2 | — | `git log` có commit; `git status` sạch; `.gitignore` loại trừ đúng |

### GĐ 1 — Dữ liệu (thu thập, gán nhãn, xử lý)

| # | Công việc | Trạng thái | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|---|
| 1.1 | Chốt giống ớt, môi trường chụp, tiêu chí nhãn `medium_ripe` vs `ripe` (>90% diện tích màu chín = `ripe`); ghi vào `DATASET_GUIDE.md` | Bị chặn (cần người dùng xác nhận giống + điều kiện chụp) | P0 | §5 Q1–Q2 | Guide có checklist đã duyệt; 2 người kiểm tra nhãn độc lập trên mẫu |
| 1.2 | Thu thập ảnh thật ≥ 100–200/lớp (mục tiêu khởi điểm; tăng dần theo đường cong học), đặt vào `dataset/raw/<class>/`, chụp gần, nền nhất quán, JPG ≥1080p | Bị chặn (thiếu ảnh) | P0 | 1.1 | `check_dataset.py` báo ≥100/lớp, 0 lỗi decode; không dùng ảnh Internet không rõ bản quyền nếu có yêu cầu |
| 1.3 | Tạo `dataset/groups.csv` (`filename,group_id`): mỗi quả ớt một group, mọi góc chụp cùng group; trỏ `paths.groups_file` trong `config.yaml` | Chưa làm | P1 | 1.2 | Mỗi ảnh có group; test group-leak pass với dữ liệu thật |
| 1.4 | Chạy `check_dataset.py --prepare` tạo `split_manifest.json`, kiểm tra trùng md5, thôi chia lại khi không cần (`--force-resplit` chỉ khi cần) | Chưa làm | P1 | 1.2–1.3, 0.1 | Manifest tồn tại, splits tỉ lệ ~70/15/15, ảnh gốc không mất, mô tả kết quả trong `reports/` |

### GĐ 2 — AI: xây dựng, huấn luyện, đánh giá, tối ưu

| # | Công việc | Trạng thái | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|---|
| 2.1 | Kiểm chứng kiến trúc MobileNetV2 + freeze/fine-tune với dataset thật (chạy `train.py` 2 giai đoạn); lưu `best_chili.keras`/`chili_final.keras` | Chưa làm (chờ dataset) | P0 | 1.4 | Training xong không crash; `training_summary.json` + `training_curves.png` tồn tại |
| 2.2 | Đánh giá `evaluate.py --split test`: accuracy, P/R/F1 theo lớp, confusion matrix (PNG+JSON); phân tích nhầm lẫn `medium_ripe`↔`ripe` | Chưa làm | P0 | 2.1 | `reports/metrics/evaluation_test.json` + confusion matrix có số liệu thật (không bịa) |
| 2.3 | Kiểm tra `predict.py --image` end-to-end: class_id, label, confidence, uncertain (ngưỡng 0.6 trong `config.yaml`) | Chưa làm | P1 | 2.1 | Predict ra JSON đúng schema trên ảnh test thật; `uncertain` hoạt động khi confidence thấp |
| 2.4 | Tối ưu: tune dropout/fine_tune_layers/lr, augmentation (vẫn cấm hue mạnh), class_weight, mở rộng dataset khi F1 thấp; ghi mọi thay đổi vào `training_summary.json` | Chưa làm | P2 | 2.2 | Có báo cáo so sánh trước/sau bằng số liệu test (không khẳng định số nếu chưa đo) |
| 2.5 | Khiểm thử hồi quy: giữ `pytest` 28 pass + thêm test split nhỏ (`n=5`) và guard rỗng; đo thời gian inference CPU nếu cần | Đang làm (28 pass) / bổ sung Chưa làm | P1 | 0.1–0.2, 2.1 | `pytest -q` pass toàn bộ, không còn `_split probe` val/test=0 |

### GĐ 3 — TFLite + Android

| # | Công việc | Trạng thái | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|---|
| 3.1 | Xuất TFLite Float32/Float16/INT8 bằng `export_tflite.py`; đọc `export_report.json` (size, validate shape, compare Keras↔TFLite) | Đang làm (code + unit test đã pass trên model chưa train) / Chưa làm (chưa có model thật) | P1 | 2.1 | 3 file `.tflite` tồn tại; `same_label=True`; INT8 chỉ xuất khi có train thật (đã chặn đúng) |
| 3.2 | Đồng bộ `labels.txt`/`model_metadata.json` do export sinh ra; xác nhận MediaPipe cần metadata writer (tài liệu đã cảnh báo đúng — không tuyên bố tương thích nếu chưa nhúng) | Đang làm (tài liệu) / Chưa làm (metadata writer) | P1 | 0.3, 3.1 | Metadata khớp tiền xử lý đã chốt ở 0.3; có hướng dẫn nhúng metadata nếu đi MediaPipe |
| 3.3 | Viết Android app Kotlin mới (dự án Gradle riêng): load `.tflite` + labels vào `assets/`, camera/upload ảnh, resize 224×224 + preprocess khớp 0.3, hiển thị label/confidence/uncertain | Chưa làm | P1 | 3.1–3.2 | App build được, predict trên máy thật khớp `predict.py` (cùng ảnh, cùng label) |
| 3.4 | Đo latency/accuracy trên thiết bị thật (Float32 vs Float16 vs INT8), chọn cấu hình mặc định theo số liệu | Chưa làm | P2 | 3.3 | Có bảng đo latency/accuracy theo thiết bị; không kết luận nếu chưa đo |

### GĐ 4 — Android ↔ ESP32-S3

| # | Công việc | Trạng thái | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|---|
| 4.1 | Chốt protocol: JSON `{label,class_id,confidence}` hoặc `{"status":"uncertain"}` qua UART hay BLE, baud/interval, cơ chế retry khi `uncertain` (chụp lại / chờ người dùng) | Chưa làm (mới có mô tả) | P1 | §5 Q4, 3.3 | Spec 1 trang được duyệt; Android và ESP32 cùng implement theo spec |
| 4.2 | Android: serialize + gửi/poll (UART/BLE) theo protocol, log `class_id/label/confidence/probabilities` | Chưa làm | P1 | 3.3, 4.1 | Log khớp output `predict.py`; `uncertain` không phát động cơ cấu |
| 4.3 | Chưa có sản phẩm firmware ESP32-S3 nếu không có phần cứng thật (xem GĐ 5) | Bị chặn | — | 4.1 | — |

### GĐ 5 — Phần cứng: camera, băng tải, cảm biến, servo

| # | Công việc | Trạng thái | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|---|
| 5.1 | Chốt BOM phần cứng: camera (OV5640/ESP32-CAM/global shutter?), băng tải, cảm biến có/không quả, servo cơ cấu phân loại 3 ngăn, nguồn/điện áp, khung đỡ | Bị chặn (cần người dùng xác nhận: có phần cứng thật không, ngân sách, nhất là camera đơn vs Android camera trực tiếp) | P1 | §5 Q3–Q5 | BOM + sơ đồ wiring được duyệt trước khi code firmware |
| 5.2 | Nếu camera là Android native (không cần ESP32-CAM): chốt kiến trúc “AI trên Android → ESP32 chỉ điều khiển cơ cấu” và bỏ qua ESP32-CAM | Chưa làm | P1 | 5.1 | Kiến trúc đã chốt, ghi vào roadmap (không assumptions) |

### GĐ 6 — Firmware ESP32-S3 + tích hợp toàn hệ thống

| # | Công việc | Trạng thái | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|---|
| 6.1 | Viết firmware ESP32-S3 (Arduino/ESP-IDF): đọc JSON, điều khiển servo/băng tải/đèn, xử lý `uncertain` (không di chuyển hoặc về vị trí chờ), watchdog, log UART | Chưa làm | P1 | 4.1, 5.1 | Nạp thành công lên ESP32 thật, servo di chuyển đúng theo label |
| 6.2 | Tích hợp end-to-end: ảnh → Android → ESP32 → cơ cấu phân loại; đo tỉ lệ phân loại đúng trên quả thật (so với nhãn người) và latency toàn hệ thống | Chưa làm | P1 | 3.3, 4.2, 6.1 | Video/log demo end-to-end; bảng kết quả thật trên ≥50 quả (không bịa) |

### GĐ 7 — Kiểm thử, tài liệu, trình diễn

| # | Công việc | Trạng thái | Ưu tiên | Phụ thuộc | Tiêu chí hoàn thành |
|---|---|---|---|---|---|
| 7.1 | Test toàn hệ thống: `pytest`, smoke, export compare, kiểm tra Android trên thiết bị, phân loại trên băng tải | Đang làm (unit 28 pass) / phần còn lại Chưa làm | P1 | GĐ 2–6 | Checklist test pass theo từng tầng, không còn chỉ unit với tensor ngẫu nhiên |
| 7.2 | Cập nhật `README.md`/`DATASET_GUIDE.md`/`android_integration.md`: số liệu thật, kinh nghiệm, giới hạn (giống ớt, ánh sáng), link weights/model | Đang làm (draft) / Chưa làm (số liệu thật) | P2 | 2.2, 3.4, 6.2 | Tài liệu phản ánh thực tế đã đo, không còn mô tả [0,1] sai |
| 7.3 | Chuẩn bị trình diễn: video, slide, danh mục file model/dataset, cách tái lập (`pip install -r requirements.txt`, train, export) | Chưa làm | P2 | 7.1–7.2 | Người khác chạy được pipeline từ README trong môi trường sạch |

## 3. Vấn đề chưa rõ — cần người dùng xác nhận (§5)

- **Q1 Giống ớt + điều kiện chụp:** chốt 1–2 giống, ánh sáng, nền, khoảng cách không? (Quyết định ranh giới màu chín.)
- **Q2 Tiêu chí nhãn `medium_ripe` vs `ripe`:** giữ ngưỡng >90% diện tích đồng đều như `DATASET_GUIDE.md`, hay theo chỉ thị khác (độ cứng, đường kính, chỉ thiên vs sừng)?
- **Q3 Phần cứng là thật hay mô phỏng:** đã có ESP32-S3 / camera / băng tải / servo chưa? Ngân sách? Hay chỉ demo Android + mô hình cơ khí?
- **Q4 Giao tiếp:** UART hay BLE giữa Android và ESP32? Có cần hỗ trợ cloud/logging?
- **Q5 Camera:** dùng camera Android chụp trực tiếp, hay camera module riêng (OV5640, ESP32-CAM)? Độ phân giải / FPS yêu cầu?
- **Q6 Yêu cầu độ chính xác/latency:** target accuracy tối thiểu, thời gian phân loại tối đa cho 1 quả, throughput băng tải (quả/giây)?
- **Q7 Quyền ảnh:** có được dùng ảnh từ nguồn công khai (bản quyền/street view) hay chỉ ảnh tự chụp?
- **Q8 Môi trường train:** CPU-only hiện tại đủ (train chậm nhưng khả thi) hay có dự định thêm GPU?

## 4. Thứ tự triển khai đề xuất + bước tiếp theo

1. **Tuần 1 (đã hoàn thành):** GĐ 0 (0.1–0.6) → `pytest` 36 pass + probe split nhỏ → chốt preprocessing → git init + commit hiện trạng (`2402f86`).
2. **Tuần 2:** Chờ Q1/Q2 → thu thập ảnh (1.2) + `groups.csv` (1.3) → `check_dataset.py --prepare` (1.4).
3. **Tuần 3:** Huấn luyện (2.1) → đánh giá (2.2) → predict end-to-end (2.3) → export TFLite (3.1).
4. **Tuần 4:** Android app (3.3) → đo trên thiết bị (3.4) → chốt protocol (4.1) → chờ Q3/Q5 để đi GĐ 5–6.
5. **Tuần 5+:** Firmware + tích hợp (6.1–6.2) → test/tài liệu/demo (GĐ 7).

**Việc quan trọng nhất tiếp theo:** chờ người dùng xác nhận Q1/Q2 rồi thu thập ảnh thật (1.2) → `check_dataset.py --prepare` (1.4) → huấn luyện (2.1) → đánh giá (2.2) → export TFLite (3.1). GĐ 0 (fix P0/P1/P2 + git) đã hoàn thành. Mọi việc còn lại đều bị chặn bởi dataset thật và các quyết định ở §5.
