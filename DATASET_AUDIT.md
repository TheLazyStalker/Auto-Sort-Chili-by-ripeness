# DATASET_AUDIT — Kiểm tra dữ liệu ảnh thật (ớt sim)

> Task 2 — Dataset audit & preparation. Mọi thao tác chạy trong container `ai-workspace`
> tại `/workspace/chili-ripeness-ai`. Không tải ảnh công khai, không tạo ảnh giả,
> không xóa/sửa/đổi nhãn ảnh raw, không huấn luyện model.
>
> Số liệu dưới đây sinh từ `reports/metrics/dataset_audit.json` bằng
> `scripts/audit_dataset.py`; split từ `scripts/prepare_dataset.py`.
> Contact sheet nội bộ: `reports/figures/contact_sheets/` (không upload dịch vụ ngoài).

---

## 0. Tóm tắt điều hành

| Hạng mục | Kết quả |
|---|---|
| Tổng số ảnh | **473** |
| `unripe` | **197** (41,6%) |
| `medium_ripe` | **142** (30,0%) |
| `ripe` | **134** (28,3%) |
| Định dạng | 100% JPG, RGB, **1024×1024** |
| Dung lượng | 26.684–50.384 bytes (TB 35.604) |
| Ảnh lỗi / không đọc được | **0** |
| Ảnh quá nhỏ (≤32 px) | **0** |
| Trùng nội dung tuyệt đối (md5) | **0** |
| Gần trùng (perceptual hash) | **Không kết luận được** — xem §5 |
| Mất cân bằng lớp (max/min) | **1,47 : 1** |
| Nền | **1 biến thể duy nhất** (≈ RGB 220/235/235) cho cả 473 ảnh |
| Diện tích quả trong khung (median) | **3,3–3,7%** diện tích ảnh |
| Số ảnh không có EXIF | **473 / 473** |
| Split chính (70/15/15) | 332 / 71 / 70 → **70,2 / 15,0 / 14,8%** |
| Trùng file giữa các tập | **0** |
| Group ID xác minh | **0** (chưa xác minh — người dùng chọn tách rời) |
| Cặp burst (≤5 s) bị tách sang 2 tập | **183 cặp** ⚠️ |

**Phán quyết ngắn gọn:**

- Dataset **sạch về mặt kỹ thuật** (không lỗi, không trùng tuyệt đối, đúng 3 lớp, nhãn nhìn chung nhất quán).
- Dataset **chưa đủ đa dạng và chưa đủ "quả độc lập"** để đánh giá mô hình một cách đáng tin:
  ảnh được chụp theo chuỗi burst liên tiếp, nền/ánh sáng/khoảng cách gần như đồng nhất,
  quả chỉ chiếm ~3,5% khung hình.
- **Chưa có cơ sở để cam kết Accuracy ≥ 90%.** Có thể chạy pipeline để kiểm tra kỹ thuật,
  nhưng metric trên split hiện tại sẽ bị lạc quan do rò rỉ burst (xem §11).

---

## 1. Phạm vi & phương pháp

Đã kiểm tra **toàn bộ** 473 ảnh trong `dataset/raw/{unripe,medium_ripe,ripe}/`.

Mỗi ảnh được đo:

- SHA256 (manifest), md5 (phát hiện trùng tuyệt đối), dung lượng, độ phân giải, mode màu.
- Kiểm tra decode bằng Pillow (`verify()` + mở lại).
- **dHash 64-bit** toàn khung và **dHash trên vùng quả đã crop** (tự cài bằng Pillow + NumPy;
  môi trường không có `imagehash`/`cv2`).
- Tách nền bằng màu 4 góc (khoảng cách RGB > 45) → tỷ lệ diện tích quả, bounding box, độ nét
  (phương sai Laplacian trên vùng quả).
- Thống kê HSV **chỉ trên vùng quả** (loại nền): hue vòng tròn, saturation, value,
  tỷ lệ pixel xanh/vàng/cam/đỏ.
- Timestamp từ tên file `IMG_YYYYMMDD_HHMMSS` → khoảng cách giữa các lần chụp, cụm burst.
- Chẩn đoán tách lớp bằng **1-NN leave-one-out** trên 7 đặc trưng màu (chỉ để đánh giá
  mức nhất quán nhãn; **không phải** accuracy của model dự án).

Kiểm tra bổ sung: EXIF, ảnh mờ nhất, ảnh có quả nhỏ nhất, ảnh biên về màu.

> Lưu ý về tên thư mục: thực tế dữ liệu nằm ở `dataset/raw/medium_ripe/` (dấu gạch dưới).
> Yêu cầu `medium-ripe` → nhãn chuẩn `medium_ripe` đã được xử lý bằng alias trong
> `src/dataset.py` (`CLASS_DIR_ALIASES`), **không đổi tên thư mục raw**.

---

## 2. Số lượng, định dạng, độ phân giải

| Lớp | Số ảnh | Tỷ lệ |
|---|---:|---:|
| `unripe` | 197 | 41,6% |
| `medium_ripe` | 142 | 30,0% |
| `ripe` | 134 | 28,3% |
| **Tổng** | **473** | 100% |

- 100% `.jpg`, mode `RGB`.
- 100% **1024×1024** — không có ảnh dị kích thước.
- Không có `.gitkeep`/file lạ bị tính nhầm (`.gitkeep` 0 byte bị loại).
- **Không có EXIF** ở bất kỳ ảnh nào. Ảnh vuông tuyệt đối + không EXIF + dung lượng thấp
  (27–50 KB) cho thấy ảnh **đã được crop/resize/nén lại** trước khi đưa vào dataset.
  Hệ quả: không thể kiểm chứng thiết bị chụp, thông số phơi sáng, hướng ảnh gốc.

---

## 3. Ảnh lỗi / quá nhỏ / bất thường

- **0 ảnh lỗi decode.**
- **0 ảnh quá nhỏ** (ngưỡng ≤32 px mỗi chiều).
- **0 ảnh dị kích thước.**
- Không phát hiện ảnh đơn sắc hoàn toàn hay ảnh rỗng nội dung.

**Ảnh mờ nhất** (vẫn đọc được, nhưng độ nét thấp hơn phần còn lại):

| Ảnh | Lớp | Độ nét (Laplacian var) |
|---|---|---:|
| `IMG_20261107_142529.jpg` | ripe | 19,0 |
| `IMG_20261107_142534.jpg` | ripe | 19,5 |
| `IMG_20261107_142526.jpg` | ripe | 28,5 |
| `IMG_20261107_142519.jpg` | ripe | 29,3 |
| `IMG_20261107_151210.jpg` | unripe | 43,3 |
| `IMG_20261107_142559.jpg` | ripe | 54,7 |

Độ nét trung vị toàn dataset: 341–412 (theo lớp). 6 ảnh trên nằm ở nhóm thấp nhất
(4 ảnh ripe đầu tiên của phiên chụp). **Không xóa** — chỉ ghi nhận để người dùng cân nhắc
chụp lại nếu muốn.

**Ảnh có quả nhỏ nhất** (diện tích quả thấp):

| Ảnh | Lớp | Diện tích quả |
|---|---|---:|
| `IMG_20261107_150950.jpg` | unripe | 1,58% |
| `IMG_20261107_143128.jpg` | ripe | 1,95% |
| `IMG_20261107_150946.jpg` | unripe | 1,97% |
| `IMG_20261107_151314.jpg` | unripe | 2,17% |
| `IMG_20261107_143124.jpg` | ripe | 2,21% |

---

## 4. Trùng nội dung tuyệt đối (md5 / SHA256)

- **0 nhóm trùng md5, 0 file trùng.**
- Mỗi ảnh có SHA256 riêng trong manifest.

→ Không có ảnh copy nguyên byte.

---

## 5. Gần trùng bằng perceptual hash — KHÔNG kết luận được (trung thực)

Đây là phần **phải nói rõ hạn chế**, không được báo một con số gây hiểu nhầm:

1. **dHash toàn khung** (64-bit, ngưỡng ≤6) báo **9.315 cặp**. Con số này **không có nghĩa
   "gần trùng"**: nền xám đồng nhất chiếm >95% khung, nên dHash chủ yếu mã hóa nền.
   Phân bố khoảng cách Hamming giữa các ảnh *cùng lớp* có trung vị ~9/64 — tức là **mọi cặp
   đều "gần"** theo thang này.
2. **dHash trên vùng quả đã crop** (bỏ nền) phân biệt tốt hơn: trung vị ~24–25/64.
   Với ngưỡng ≤4 còn **346 cặp**. Tuy nhiên ớt sim mảnh, dài, hình dạng tương tự nhau,
   nên 8×8 dHash vẫn không đủ phân biệt.
3. **Kiểm tra trực quan 8 cặp gần nhất** (kể cả cặp Hamming = 0): tất cả là **các quả ớt
   khác nhau** (khác hình dạng, khác độ dài cuống, khác góc xoay), **không phải ảnh trùng**.

**Kết luận:** không có bằng chứng ảnh trùng/gần trùng thực sự. Các con số 9.315 / 346 là
**cảnh báo của công cụ**, không phải kết luận dữ liệu bẩn. Muốn xác minh gần trùng một cách
đáng tin cần embedding (ví dụ MobileNet features) hoặc so khớp pixel sau khi căn chỉnh —
**ngoài phạm vi Task 2** và không cần thiết khi md5 = 0.

---

## 6. Mất cân bằng lớp

| Lớp | Số ảnh | Tỷ lệ |
|---|---:|---:|
| `unripe` | 197 | 41,6% |
| `medium_ripe` | 142 | 30,0% |
| `ripe` | 134 | 28,3% |

- Tỷ lệ max/min = **1,47 : 1** → mất cân bằng **nhẹ**, chấp nhận được.
- `config.yaml` đã đặt `training.class_weight: balanced`, phù hợp.
- Không cần oversample nhân tạo (tránh tạo dữ liệu giả).

---

## 7. Điều kiện chụp (ánh sáng, nền, góc, khoảng cách, độ nét, đa dạng)

### 7.1 Nền — đồng nhất tuyệt đối

- **Chỉ 1 biến thể màu nền** cho cả 473 ảnh: ≈ RGB (220, 235, 235).
- Ưu điểm: nhất quán, dễ tách nền.
- Rủi ro: model có thể học "nền + màu quả" thay vì hình thái quả. Khi triển khai trên băng tải
  (nền băng tải, ánh sáng khác), **domain shift sẽ lớn** → accuracy thực tế có thể giảm mạnh.

### 7.2 Khoảng cách — quả quá nhỏ trong khung

| Lớp | Diện tích quả p10 | p50 | p90 | BBox p50 |
|---|---:|---:|---:|---:|
| unripe | 2,61% | **3,73%** | 5,45% | 10,5% |
| medium_ripe | 2,54% | **3,55%** | 5,04% | 12,6% |
| ripe | 2,63% | **3,28%** | 5,00% | 13,1% |

- Quả chỉ chiếm ~3,5% diện tích ảnh; bounding box ~10–13%.
- Sau khi resize 224×224, vùng quả còn ~40×40 px → **quá nhỏ** để học hình thái/đốm màu tốt.
- Khuyến nghị: chụp gần hơn để quả chiếm **≥25–40% diện tích khung** (tăng khoảng 3× cạnh).

### 7.3 Độ nét

- Trung vị 341–412 (Laplacian var trên vùng quả); p10 của `ripe` chỉ 83.
- 6 ảnh mờ nhất ở §3 (4 ảnh ripe mở đầu phiên chụp).
- Nhìn chung **đủ nét**, trừ nhóm nhỏ cần lưu ý.

### 7.4 Ánh sáng / phản xạ

- Ánh sáng nhìn chung đều, nền sáng.
- Một số ảnh có **điểm chói (specular highlight)** trên vỏ làm pixel chuyển trắng/vàng nhạt,
  có thể gây nhầm với màu chín. Cần kiểm soát khi chụp lại.

### 7.5 Góc chụp & khoảng cách

- Quả được đặt nhiều hướng khác nhau (cuống trái/phải/lên/xuống) — **đa dạng góc tốt**.
- Khoảng cách gần như cố định (diện tích quả dao động hẹp p10–p90 ~2,5–5,5%).

### 7.6 Chuỗi chụp (burst) — vấn đề lớn nhất

| Lớp | Số ảnh | Thời gian | Gap ≤5 s | Gap >60 s | Gap lớn nhất |
|---|---:|---|---:|---:|---:|
| unripe | 197 | 15:08:35 → 15:14:50 (6 ph 15 s) | **196/196** | 0 | 5 s |
| medium_ripe | 142 | 15:00:13 → 15:05:44 (5 ph 31 s) | 136/141 | 0 | 14 s |
| ripe | 134 | 14:25:19 → 14:59:01 (33 ph 42 s) | 104/133 | 2 | 1278 s |

**Số cụm chụp ước lượng** (ngưỡng gap):

| Lớp | gap > 2 s | gap > 5 s | gap > 10 s | gap > 30 s |
|---|---:|---:|---:|---:|
| unripe | 31 | **1** | **1** | 1 |
| medium_ripe | 39 | 6 | 3 | 1 |
| ripe | 93 | 30 | 11 | 4 |

→ `unripe` gần như **một phiên chụp liên tục duy nhất** (197 ảnh, mọi gap ≤5 s).
Đây là dấu hiệu mạnh rằng **số quả ớt vật lý độc lập rất ít** so với 473 ảnh — có thể chỉ
một vài quả được xoay/đổi vị trí liên tục. **Chưa xác minh** đây có phải cùng quả hay không,
nhưng đây là rủi ro nghiêm trọng nhất của dataset.

---

## 8. Đa dạng dữ liệu & "kích thước mẫu hiệu dụng"

- 473 ảnh nhưng **không phải 473 quả độc lập**. Theo cụm burst (gap >10 s):
  `unripe` 1 cụm, `medium_ripe` 3 cụm, `ripe` 11 cụm → **tổng ~15 cụm**.
- Nếu mỗi cụm ≈ một quả (giả định chưa xác minh), kích thước mẫu hiệu dụng chỉ ~15,
  **nhỏ hơn hàng chục lần** so với 473.
- Toàn bộ ảnh cùng nền, cùng điều kiện ánh sáng, cùng khoảng cách, cùng thiết bị/ quy trình
  xử lý → **đa dạng môi trường gần như bằng 0**.

Hệ quả: model rất dễ đạt accuracy cao trên split ngẫu nhiên nhưng **không tổng quát hóa**
sang ảnh chụp trên băng tải/OV2640.

---

## 9. Q1 — Đặc điểm hình thái & màu sắc từng lớp (từ dữ liệu thực tế)

Số liệu HSV **trên vùng quả** (trung bình lớp):

| Lớp | Hue trung vị | Saturation | Value | Xanh | Vàng | Cam | Đỏ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `unripe` | **83,3°** | 0,744 | 0,406 | **0,668** | 0,262 | 0,030 | **0,000** |
| `medium_ripe` | **52,7°** | 0,823 | 0,484 | 0,173 | 0,053 | **0,492** | 0,075 |
| `ripe` | **32,3°** | 0,782 | 0,453 | 0,147 | 0,029 | 0,022 | **0,558** |

Phân vị theo lớp (p10–p90):

| Lớp | Hue | Tỷ lệ xanh | Tỷ lệ cam | Tỷ lệ đỏ |
|---|---|---|---|---|
| `unripe` | 75,0–91,7° | 0,52–0,81 | 0,00–0,03 | 0,00–0,00 |
| `medium_ripe` | 42,4–64,6° | 0,08–0,24 | 0,35–0,61 | 0,00–0,19 |
| `ripe` | 19,4–44,4° | 0,05–0,25 | 0,00–0,06 | 0,46–0,67 |

**Quan sát (chỉ mô tả, không khẳng định sinh học):**

- `unripe`: vỏ **xanh lá chiếm ưu thế rõ rệt**, hầu như không có pixel đỏ. Một phần đáng kể
  pixel nằm ở dải vàng-lục (yellow_frac trung bình 0,26) — có thể do giống ớt có ánh vàng
  hoặc do phản xạ ánh sáng; **chưa đủ bằng chứng để kết luận**.
- `medium_ripe`: màu **cam chiếm ưu thế**, vẫn còn một ít xanh (thường ở cuống/đầu quả),
  đỏ mới xuất hiện ít → đúng nghĩa "đang chuyển".
- `ripe`: **đỏ chiếm ưu thế** (trung bình 56% pixel vùng quả), xanh còn lại chủ yếu ở cuống.
- Hình thái: cả 3 lớp đều là quả nhỏ, thuôn dài, cuống dài; **không thấy khác biệt hình thái
  rõ ràng giữa các lớp trong ảnh** — màu là tín hiệu phân biệt chính, khớp với giả định dự án.

> Cảnh báo: hue chịu ảnh hưởng của **cân bằng trắng và ánh sáng**. Các giá trị trên chỉ đúng
> trong điều kiện chụp hiện tại; không nên dùng làm ngưỡng tuyệt đối cho môi trường khác.

---

## 10. Q2 — Tiêu chí phân biệt 3 lớp (đề xuất, KHÔNG áp đặt ngưỡng)

Theo yêu cầu, **không** đặt trước ngưỡng màu/% diện tích, **không** mặc định ">90% màu chín
= ripe". Dữ liệu hiện tại cho thấy 3 lớp tách theo **thứ tự chuyển màu**, không theo một
ngưỡng diện tích cố định:

1. **`unripe`** — vỏ **xanh lá là màu chủ đạo**, gần như không có mảng đỏ/cam rõ rệt.
   (Dữ liệu: xanh 0,52–0,81; đỏ ≈ 0.)
2. **`medium_ripe`** — **trạng thái chuyển tiếp**: vỏ có cả mảng xanh và mảng cam/đỏ,
   không màu nào áp đảo hoàn toàn; thường còn xanh ở cuống/đầu quả.
   (Dữ liệu: cam 0,35–0,61; xanh 0,08–0,24; đỏ 0–0,19.)
3. **`ripe`** — **đỏ là màu chủ đạo** trên thân quả; xanh còn lại chủ yếu ở cuống.
   (Dữ liệu: đỏ 0,46–0,67; xanh 0,05–0,25.)

**Quy tắc quyết định đề xuất (dạng thứ tự, không phải công thức cứng):**

- Nếu vỏ **xanh chiếm ưu thế** và không có mảng chín rõ → `unripe`.
- Nếu vỏ **vừa xanh vừa cam/đỏ**, chưa màu nào áp đảo → `medium_ripe`.
- Nếu vỏ **đỏ chiếm ưu thế** trên thân quả → `ripe`.
- Ảnh nằm đúng ranh giới (ví dụ vàng-lục, cam-đỏ) → **đánh dấu "cần duyệt"**, không tự gán.

**Đề xuất để chốt ngưỡng đúng (khi có dữ liệu mới):**

- Chụp **nhiều quả khác nhau** ở từng giai đoạn, mỗi quả một `group_id`.
- Gán nhãn độc lập bởi **2 người**, tính đồng thuận; tranh chấp → loại hoặc xử lý riêng.
- Từ tập gán nhãn đó, mới ước lượng ngưỡng (ví dụ theo phân vị) và **kiểm chứng trên quả
  chưa từng thấy**. Không lấy ngưỡng từ dataset hiện tại vì phụ thuộc burst.

---

## 11. Nhất quán nhãn & ảnh nghi vấn (CHỈ LIỆT KÊ — KHÔNG SỬA)

Chẩn đoán 1-NN leave-one-out trên 7 đặc trưng màu: **98,94%** (468/473 ảnh có "hàng xóm
gần nhất" cùng lớp). Đây là **chẩn đoán đặc trưng màu**, không phải accuracy của model.

Ma trận (hàng = nhãn thật, cột = lớp gần nhất):

| Thật \ Gần nhất | unripe | medium_ripe | ripe |
|---|---:|---:|---:|
| unripe | 196 | 1 | 0 |
| medium_ripe | 1 | 140 | 1 |
| ripe | 0 | 2 | 132 |

**5 ảnh lệch (cần người dùng duyệt, KHÔNG tự đổi nhãn):**

| Ảnh | Nhãn hiện tại | Gần lớp | Lý do |
|---|---|---|---|
| `IMG_20261107_150901.jpg` | unripe | medium_ripe | Vỏ xanh-vàng, đầu quả ngả vàng rõ; nằm giữa unripe và medium. |
| `IMG_20261107_150329.jpg` | medium_ripe | ripe | Quả nâu-đỏ khá sẫm, ít mảng xanh; có thể đã chín hơn nhãn. |
| `IMG_20261107_150511.jpg` | medium_ripe | unripe | Quả vàng-cam nhưng còn nhiều xanh ở cuống/thân; ranh giới mờ. |
| `IMG_20261107_145600.jpg` | ripe | medium_ripe | Đỏ nhưng còn ánh cam; độ chín có thể chưa đồng đều. |
| `IMG_20261107_145847.jpg` | ripe | medium_ripe | Cam-đỏ, chưa đỏ sẫm; ranh giới medium/ripe. |

**Ảnh `unripe` ngả vàng đáng chú ý** (chưa chắc sai nhãn, nhưng nên xem lại):
`IMG_20261107_151125.jpg`, `IMG_20261107_151349.jpg`, `IMG_20261107_151342.jpg`,
`IMG_20261107_150950.jpg`, `IMG_20261107_150946.jpg`.

→ **Không di chuyển, không đổi tên, không sửa nhãn.** Đây là danh sách để người dùng quyết định.

---

## 12. Group ID

- `dataset/groups.csv` đã được sinh cho **toàn bộ 473 ảnh**, schema:

  ```csv
  filename,group_id,verified
  IMG_20261107_150835.jpg,,no
  ```

- `group_id` **để trống** và `verified=no` cho mọi ảnh → **chưa xác minh** ảnh nào cùng một quả.
- Theo lựa chọn của người dùng: **không gán group (tách rời)** — split hiện tại coi mỗi ảnh
  là một mẫu độc lập. Điều này **không** đồng nghĩa khẳng định mỗi ảnh là một quả khác nhau;
  nó chỉ là cách chia tạm thời khi chưa có thông tin group.
- Manifest (`reports/metrics/dataset_audit.csv`) có cột `group_id` + `group_verified`
  (tất cả `no`).
- Khi người dùng điền `group_id` thật vào `groups.csv`, `paths.groups_file` đã trỏ sẵn
  (`dataset/groups.csv`) → lần chia lại sẽ tự động chia theo nhóm.

---

## 13. Chia tập & kiểm tra rò rỉ

### 13.1 Split chính (theo lựa chọn "tách rời", seed 42)

| Tập | Số ảnh | Tỷ lệ |
|---|---:|---:|
| train | 332 | 70,2% |
| validation | 71 | 15,0% |
| test | 70 | 14,8% |

Theo lớp:

| Lớp | train | validation | test |
|---|---:|---:|---:|
| unripe | 138 | 30 | 29 |
| medium_ripe | 100 | 21 | 21 |
| ripe | 94 | 20 | 20 |

- **Không file nào ở >1 tập** (kiểm tra bằng đường dẫn tuyệt đối).
- **Không Group ID nào ở >1 tập** — nhưng vì không có group nào được dùng, kiểm tra này
  không có hiệu lực bảo vệ.
- **183 cặp ảnh chụp cách nhau ≤5 s bị tách sang hai tập khác nhau** → **rò rỉ burst
  gần như chắc chắn** (nếu các cặp đó là cùng một quả, điều chưa được xác minh nhưng rất
  có khả năng).

### 13.2 Split thử nghiệm theo cụm burst (heuristic, KHÔNG thay manifest chính)

Chạy `python scripts/prepare_dataset.py --burst-groups 10` → ghi
`dataset/split_manifest_burstgroup.json`:

| Lớp | train | validation | test |
|---|---:|---:|---:|
| unripe | **197** | **0** | **0** |
| medium_ripe | 63 | 68 | 11 |
| ripe | 89 | 28 | 17 |

- Tổng: 349 / 96 / 28 (73,8 / 20,3 / 5,9%) — **lệch** so với 70/15/15.
- `unripe` chỉ có **1 group** → **không thể** chia thành 3 tập không rỗng.
- Đây là **bằng chứng định lượng** rằng dataset hiện tại không đủ số quả độc lập để đánh giá
  công bằng, dù có gán group theo burst.

---

## 14. Mức độ sẵn sàng huấn luyện

| Tiêu chí | Trạng thái |
|---|---|
| Đủ 3 lớp, ảnh đọc được, không trùng tuyệt đối | ✅ Đạt |
| Cân bằng lớp | ✅ Đạt (1,47:1) |
| Có manifest + split 70/15/15 | ✅ Đạt (kỹ thuật) |
| Không rò rỉ giữa các tập | ❌ **Không đạt** (183 cặp burst; không có group) |
| Đủ số quả độc lập | ❌ **Không đạt** (~15 cụm burst; unripe 1 cụm) |
| Đa dạng nền/ánh sáng/khoảng cách | ❌ **Không đạt** (1 nền, 1 điều kiện) |
| Kích thước quả trong khung phù hợp | ❌ **Không đạt** (~3,5% diện tích) |
| Đủ cơ sở cam kết Accuracy ≥ 90% | ❌ **Chưa** |

**Kết luận:** có thể chạy `python -m src.train` để **kiểm tra kỹ thuật pipeline**, nhưng:
- Metric trên split hiện tại **không đáng tin** (rò rỉ burst).
- **Không được báo cáo ≥90%** dựa trên split này.
- Muốn có metric có ý nghĩa: bổ sung nhiều quả độc lập + `group_id`, rồi chia lại.

---

## 15. Đề xuất khắc phục & quy chuẩn chụp (cho lần thu thập tiếp theo)

### 15.1 Ưu tiên P0 — tăng số quả độc lập

- Mỗi **quả ớt riêng biệt** = 1 `group_id`; ghi ngay khi chụp (không suy đoán từ timestamp).
- Mục tiêu tối thiểu: **≥50 quả độc lập/lớp** (tốt hơn: 100+), tổng **≥300–500 quả**.
- Với mỗi quả: 3–6 ảnh ở các góc khác nhau, **cùng một giai đoạn chín** trong một lần chụp.
- Không chụp nhiều chục ảnh cho một quả rồi coi là nhiều mẫu.

### 15.2 Ưu tiên P0 — khung hình & khoảng cách

- Cho quả chiếm **≥25–40% diện tích khung** (hiện ~3,5%): lại gần khoảng 3×.
- Giữ khoảng cách cố định giữa các lần chụp; tránh zoom số.
- Một quả/khung hình; không để quả khác lọt vào.

### 15.3 Ưu tiên P1 — kiểm soát ánh sáng & nền

- Ánh sáng khuếch tán, không flash trực tiếp; tránh điểm chói trên vỏ.
- Cân bằng trắng cố định (khóa WB) để hue không trôi giữa các phiên.
- Nền trung tính, mờ (matte), không cùng màu với vỏ; **đa dạng nền có kiểm soát**
  (ghi lại loại nền) để model không học nền.
- Chụp thêm vài phiên ở điều kiện ánh sáng khác (trong nhà/ngoài trời) để tăng đa dạng,
  nhưng **ghi nhãn điều kiện** để phân tích sau.

### 15.4 Ưu tiên P1 — quy trình gán nhãn

- 2 người gán độc lập; thống kê bất đồng; ảnh tranh chấp → "cần duyệt", không ép nhãn.
- Dùng tiêu chí thứ tự ở §10, **không** dùng ngưỡng >90% mặc định.
- Ghi lại tiêu chí đã chốt vào `DATASET_GUIDE.md` sau khi người dùng duyệt.

### 15.5 Ưu tiên P2 — kỹ thuật

- Giữ ảnh gốc **có EXIF** (không crop/resize trước khi lưu raw) để truy vết thiết bị/thông số.
- Độ phân giải ≥1080p; JPG chất lượng cao; không nén lại nhiều lần.
- Tránh ảnh mờ: kiểm tra nhanh sau khi chụp; chụp lại các ảnh độ nét thấp.

---

## 16. Hạn chế của audit này

- **Không xác minh được** ảnh nào cùng một quả ớt (không có group thật, không có EXIF,
  không có ảnh chụp kèm nhận dạng). Mọi suy luận burst chỉ là **giả thuyết**.
- **Không xác minh được** nhãn đúng/sai tuyệt đối; chỉ phát hiện ảnh **lệch đặc trưng màu**
  và ảnh biên. Việc duyệt nhãn cuối cùng thuộc về người dùng.
- Perceptual hash **không đáng tin** cho ớt mảnh + nền đồng nhất; đã kiểm tra trực quan
  một mẫu nhỏ (8 cặp gần nhất), **không** duyệt thủ công toàn bộ 473 ảnh.
- Không đánh giá được **chất lượng cảm quan** (độ tươi, tổn thương, nấm mốc) vì không có
  tiêu chí/nhãn tương ứng.
- Không huấn luyện, không suy ra accuracy thật. Mọi con số "98,94%" chỉ là chẩn đoán
  đặc trưng màu.

---

## 17. Cách tái lập

```bash
# trong container ai-workspace
podman exec -it ai-workspace bash
cd /workspace/chili-ripeness-ai
source /workspace/.venv/bin/activate

# 1) Audit + manifest + groups.csv + contact sheet
python scripts/audit_dataset.py --contact --audit-md reports/metrics/dataset_audit.md

# 2) Split chính 70/15/15 + xác minh rò rỉ
python scripts/prepare_dataset.py --force-resplit

# 3) (Tùy chọn) Split thử nghiệm theo cụm burst — KHÔNG thay manifest chính
python scripts/prepare_dataset.py --burst-groups 10

# 4) Test
pytest -q
```

**Artifact sinh ra (không commit — đã nằm trong `.gitignore`):**

- `reports/metrics/dataset_audit.json` / `.csv` — manifest đầy đủ + thống kê.
- `reports/metrics/dataset_audit.md` — báo cáo máy sinh.
- `reports/metrics/split_verification.json` — xác minh split chính.
- `reports/metrics/split_verification_burstgroup.json` — xác minh split thử nghiệm.
- `reports/figures/contact_sheets/contact_{unripe,medium_ripe,ripe}.png` — contact sheet nội bộ.
- `dataset/split_manifest.json` — manifest split chính.
- `dataset/groups.csv` — group_id trống (chưa xác minh).

---

## 18. Phụ lục — Manifest

Cột tối thiểu trong `reports/metrics/dataset_audit.csv`:

`path, label, sha256, size_bytes, width, height, group_id, group_verified, filename`

- `label`: nhãn chuẩn (`unripe` / `medium_ripe` / `ripe`).
- `group_id`: trống (chưa xác minh).
- `group_verified`: `no` cho toàn bộ.

Ví dụ:

```csv
path,label,sha256,size_bytes,width,height,group_id,group_verified,filename
/workspace/chili-ripeness-ai/dataset/raw/unripe/IMG_20261107_150835.jpg,unripe,0a6420…fab5,31900,1024,1024,,no,IMG_20261107_150835.jpg
```
