# IMAGE_HANDLING_REVIEW — Task 2.5 (Dataset review & image handling decisions)

> **Trạng thái:** phân tích & đề xuất — **chưa thay đổi dataset**.
> Mọi thao tác đọc/kiểm tra chạy trong container `ai-workspace` tại
> `/workspace/chili-ripeness-ai`. Không train/evaluate, không thêm/tải/tạo ảnh,
> không sửa/xóa/đổi tên/đổi nhãn ảnh raw, không đổi `groups.csv`, manifest hay split.
>
> **Ngày:** 2026-11-07
> **Người thực hiện:** AI agent (OpenCode), theo yêu cầu Task 2.5.
> **Ảnh raw đã kiểm tra lại SHA256:** 473/473 không đổi.
>
> **CẬP NHẬT 2026-11-07 — Q1–Q7 đã chốt.** Kế hoạch triển khai mới:
> `reports/task3_data_plan.md`. Các mục §0.1, §2.2, §2.4, §3.6, §4.4, §5.5, §6.5,
> §7, §8, §11 đã được cập nhật theo quyết định; phần phân tích số liệu gốc được giữ
> nguyên. Khi có mâu thuẫn, **bảng §0.1 và `task3_data_plan.md` là nguồn chính**.
> Chưa thay đổi raw dataset; chưa bắt đầu Task 3.

---

## 0. Kết luận nhanh (TL;DR)

1. **473 ảnh không phải 473 quả ớt, và số quả là UNKNOWN.** Người dùng ước lượng ~3–6
   ảnh/quả và ~30 quả/lớp, nhưng **Q3 xác nhận đây là ước lượng, không phải số đếm**.
   Số học chỉ cho thấy "≥30 quả/lớp" *có thể* đúng nếu trung bình 4–4,5 ảnh/quả; nếu
   đúng 5 ảnh/quả thì `medium_ripe`/`ripe` chỉ còn ~27–28 quả/lớp. Không dùng con số
   này như fact.
2. **Không có mapping ảnh → quả** trong dữ liệu hiện có: tên file chỉ là timestamp,
   thư mục chỉ là lớp, không EXIF, `groups.csv` trống, không có ghi chú.
3. **Ảnh liên tiếp theo thời gian không phải cùng một quả.** Kiểm tra trực quan 25 ảnh
   đầu mỗi lớp cho thấy các ảnh cạnh nhau khác nhau rõ về kích thước/hình dạng/góc xoay.
   Vì vậy **không thể dùng timestamp/burst để gán group**.
4. **"~15 cụm burst" của Task 2 không phải số quả độc lập.** Đó chỉ là số cụm theo
   khoảng cách thời gian; `unripe` là một phiên chụp liên tục nhưng có thể chứa ~30–40 quả.
   Kết luận cũ "unripe có thể chỉ một vài quả" là **suy diễn quá mức** và cần bỏ.
5. **183 cặp ≤5 giây bị tách tập không phải bằng chứng 183 quả bị leak.** Đó là các cặp
   ảnh *kề nhau về thời gian*; phần lớn có thể là các quả khác nhau. Tuy nhiên **rủi ro
   leak thật vẫn cao** vì mỗi quả có nhiều ảnh bị chia ngẫu nhiên: nếu ~5 ảnh/quả thì
   xác suất một quả có ít nhất 1 ảnh trong validation/test là ~83%.
6. **Split hiện tại không đáng tin để báo cáo accuracy cuối cùng**, nhưng vẫn dùng được
   để kiểm tra kỹ thuật pipeline. Không nên tự gán group/resplit khi chưa có quyết định.
7. **10 ảnh nghi vấn:** 5 ảnh lệch đặc trưng màu (có thể có ảnh đúng nhãn, ví dụ `145600`
   bị lệch do cuống xanh) + 5 ảnh `unripe` từng bị gắn cờ vàng (thực tế phần lớn vẫn
   xanh; chỉ `151125` ngả vàng/cam rõ). **Không đổi nhãn.**
8. **Đã chốt (2026-11-07):** không có mapping ảnh→quả (Q1=C). Dùng **473 ảnh, trừ 10
   ảnh bị loại**, cho **train**; thu **validation/test mới từ quả vật lý độc lập** ở
   Task 3. Crop derivative **`crop_v1`** lưu riêng, giữ raw, dùng **cùng quy trình crop
   khi inference** (Q6=C). 10 ảnh nghi vấn vào `dataset/review_exclusions.csv`, không
   xóa/đổi nhãn (Q5). Task 3 thu **30–50+ quả/lớp** (mục tiêu 50–100+), có `group_id`
   ngay khi chụp, quả chiếm 25–40% khung (Q7).

### 0.1 Quyết định đã chốt (2026-11-07)

| Mã | Quyết định | Hệ quả cho kế hoạch |
|---|---|---|
| **Q1** | **C — không có mapping ảnh→quả** | Không suy đoán group từ tên file/timestamp/burst/visual similarity. 473 ảnh chỉ dùng train. |
| **Q2** | **A — thường chụp liên tiếp từng quả, nhưng không nhớ thứ tự chính xác** | Ghi nhận nhưng **không dùng** để tái tạo group: số ảnh/quả ước lượng 3–6, thứ tự không nhớ, và 25 ảnh đầu cho thấy ảnh cạnh nhau khác quả. Không dùng heuristic "block 5 ảnh = 1 quả". |
| **Q3** | Số quả và số ảnh/quả là **ước lượng**, không chính xác | Không báo cáo dataset hiện có là "≥30 quả/lớp" như fact. Mục tiêu 30–50+ quả/lớp thuộc Task 3. |
| **Q4** | **B — dùng 473 ảnh cho train; thu quả mới độc lập cho validation/test** | Split 332/71/70 hiện tại chỉ là **smoke-test**, không phải benchmark cuối. |
| **Q5** | **Loại 10 ảnh flagged khỏi training; giữ review/exclusion list; không xóa/sửa raw** | `dataset/review_exclusions.csv`; 10 ảnh không vào train/val/test cho đến khi review. |
| **Q6** | **C — lưu crop derivative riêng, giữ raw, dùng cùng quy trình crop khi inference** | `crop_v1` trong `dataset/derived/crop_v1/`; raw không đổi; inference dùng đúng hàm crop. |
| **Q7** | **YES với các điều kiện** | `group_id` khi chụp; 25–40% khung; ánh sáng khuếch tán + nền matte; khóa WB nếu được, nếu không thì cố định và ghi lại; 3–6 ảnh/quả; 30–50+ quả/lớp (mục tiêu 50–100+); giữ EXIF/raw; không crop/ghi đè raw. |

Chi tiết triển khai: `reports/task3_data_plan.md`.

---

## 1. Tóm tắt dữ kiện

### 1.1 Đã đọc

- `DATASET_AUDIT.md`
- `PROJECT_STATUS.md`
- `DATASET_GUIDE.md`
- `dataset/groups.csv`
- `dataset/split_manifest.json`
- `dataset/split_manifest_burstgroup.json`
- `reports/metrics/split_verification.json`
- `reports/metrics/split_verification_burstgroup.json`
- `reports/metrics/dataset_audit.json` / `.csv` / `.md`
- `src/dataset.py`
- `scripts/check_dataset.py`
- `scripts/prepare_dataset.py`
- Contact sheet hiện có: `reports/figures/contact_sheets/contact_{unripe,medium_ripe,ripe}.png`

### 1.2 Đã kiểm tra / xác minh lại

| Kiểm tra | Kết quả |
|---|---|
| Số ảnh thực tế 3 lớp | `unripe` 197, `medium_ripe` 142, `ripe` 134; tổng **473** |
| `groups.csv` | **473 dòng**, `group_id` trống **473/473**, `verified=no` **473/473** |
| Split hiện tại | train 332 / validation 71 / test 70 = **70,2 / 15,0 / 14,8%** |
| Trùng file giữa các tập | **0** |
| Group ID dùng trong split | **0** |
| Cặp ảnh kề nhau ≤5 giây bị tách tập | **183** (đếm theo cặp kề timestamp) |
| Split thử nghiệm theo cụm burst | `unripe` 197/0/0 → **không chia được 3 tập** |
| Ảnh raw bị thay đổi | **0/473** (SHA256 khớp manifest) |
| Test | **45 passed** (Task 2; Task 2.5 không sửa code) |

### 1.3 Số liệu ảnh (từ audit)

| Hạng mục | Giá trị |
|---|---|
| Định dạng | 100% JPG, RGB |
| Độ phân giải | 100% **1024×1024** |
| Dung lượng | 26.684–50.384 bytes (trung bình 35.604) |
| Ảnh lỗi/không decode | **0** |
| Ảnh quá nhỏ (≤32 px) | **0** |
| Trùng md5 tuyệt đối | **0 nhóm / 0 file** |
| Nền | **1 biến thể** ≈ RGB (220, 235, 235) cho cả 473 ảnh |
| Diện tích quả (median) | **3,28–3,73%** khung hình |
| Bounding box quả (median) | **10,5–13,1%** khung hình |
| Độ nét (Laplacian var, median) | 341–412; p10 của `ripe` chỉ **83** |
| EXIF | **0/473 ảnh có EXIF** |
| Mất cân bằng lớp (max/min) | **1,47 : 1** |

### 1.4 Phạm vi review trực quan

Task 2.5 **không duyệt thủ công từng ảnh trong toàn bộ 473 ảnh**. Phạm vi đã xem:

- 3 contact sheet theo lớp từ Task 2 (`contact_{unripe,medium_ripe,ripe}.png`) — có sẵn
  để đối chiếu.
- 3 contact sheet thumbnail theo **thứ tự timestamp cho toàn bộ** 197/142/134 ảnh
  (`chrono_{unripe,medium_ripe,ripe}.png`) — đã xem để nắm bố cục tổng thể.
- **Kiểm tra chi tiết 25 ảnh đầu tiên** theo thứ tự timestamp của mỗi lớp (3 sheet tạm,
  không lưu vào repo).
- **10 ảnh nghi vấn** (sheet mới lưu trong repo, xem §10).
- Toàn bộ số liệu màu/hình học từ `dataset_audit.json`.

Kết luận trực quan quan trọng nhất: **các ảnh cạnh nhau trong cùng lớp là các quả khác
nhau**, không phải 5 ảnh liên tiếp của cùng một quả. Kết luận này được kiểm tra chi tiết
ở 25 ảnh đầu mỗi lớp; contact sheet toàn bộ cũng không cho thấy khối 5 ảnh liên tiếp nào
rõ ràng là cùng một quả, nhưng **chưa kiểm tra thủ công từng ảnh trong 473 ảnh**.

### 1.5 Những điểm của Task 2 cần hiệu chỉnh trong báo cáo này

Các điểm sau **không phải lỗi dữ liệu**, mà là cách diễn giải cần sửa:

| Task 2 đã ghi | Hiệu chỉnh ở Task 2.5 |
|---|---|
| "~15 cụm burst → có thể chỉ ~15 quả độc lập" | Cụm burst chỉ đo khoảng cách thời gian. Vì các quả được chụp xen kẽ, cụm burst **không phải** số quả. Không dùng con số này để kết luận số quả. |
| "unripe gần như 1 phiên chụp → có thể chỉ một vài quả" | `unripe` là 1 phiên liên tục, nhưng có thể chứa ~30–40 quả. Số ảnh (197) và thông tin người dùng (~5 ảnh/quả, ≥30 quả) **không mâu thuẫn** với một phiên chụp. |
| "183 cặp burst bị tách tập → rò rỉ gần như chắc chắn" | 183 là số cặp **kề timestamp**, không phải số cặp cùng quả. Phần lớn cặp kề nhau có thể là quả khác nhau. Đây là **cảnh báo**, không phải bằng chứng leak. Rủi ro leak thật đến từ nhiều ảnh/quả bị chia ngẫu nhiên (xem §3.3). |
| "1-NN màu 98,94%" | Chỉ là **chẩn đoán đặc trưng màu**, không phải accuracy của model. Không dùng để cam kết chất lượng. |

---

## 2. Mapping ảnh–quả: đã biết / chưa biết

### 2.1 Những gì biết chắc **[FACT]**

- Mỗi ảnh có tên `IMG_YYYYMMDD_HHMMSS.jpg`; timestamp **2026-11-07**.
- Thư mục `dataset/raw/<class>/` chỉ cho biết **nhãn lớp**, không cho biết quả.
- Không có EXIF, không có ghi chú, không có file log, không có `group_id`.
- `groups.csv` có 473 dòng nhưng **tất cả group_id đều trống** và `verified=no`.
- Không có ảnh trùng md5; mỗi ảnh là một file riêng.
- Các phiên chụp theo lớp (timestamp file):
  - `ripe`: 14:25:19 → 14:59:01; có 2 khoảng nghỉ lớn: **111 giây** (14:28:57→14:30:48)
    và **1.278 giây ≈ 21,3 phút** (14:31:40→14:52:58) → **3 khối chụp**.
  - `medium_ripe`: 15:00:13 → 15:05:44; gap lớn nhất 14 giây → **1 khối**.
  - `unripe`: 15:08:35 → 15:14:50; gap lớn nhất 5 giây → **1 khối liên tục**.
- Trong mỗi lớp, khoảng cách giữa các ảnh chủ yếu **1–5 giây**; `unripe` không có gap >5 giây.

### 2.2 Thông tin người dùng cung cấp **[USER]**

- Mỗi mẫu ớt được chụp **khoảng 5 ảnh**.
- Có **ít nhất 30 quả độc lập mỗi lớp**.
- Ảnh chỉ tự chụp, không dùng ảnh công khai.
- **Q3 (2026-11-07):** đây là **ước lượng**, không phải số đếm chính xác. Không được
  báo cáo "dataset hiện có ≥30 quả/lớp" như fact; chỉ là khả năng.
- **Q2 (2026-11-07):** người dùng nói **thường chụp liên tiếp từng quả**, nhưng không
  nhớ thứ tự chính xác. Xem §2.4 để biết vì sao vẫn không thể tái tạo group.

### 2.3 Đối chiếu số học **[INFERENCE]**

Nếu mỗi quả có `k` ảnh, số quả ước lượng = số ảnh / `k`:

| Lớp | Số ảnh | Nếu k=5 | Nếu k=4 | Nếu k=4,5 | k tối đa để có ≥30 quả |
|---|---:|---:|---:|---:|---:|
| `unripe` | 197 | 39,4 | 49,2 | 43,8 | 6,57 |
| `medium_ripe` | 142 | 28,4 | 35,5 | 31,6 | **4,73** |
| `ripe` | 134 | 26,8 | 33,5 | 29,8 | **4,47** |

**Đọc bảng:**

- Nếu đúng **5 ảnh/quả**, `medium_ripe` ~28 quả và `ripe` ~27 quả → **hơi dưới 30**.
- Nếu trung bình **4–4,5 ảnh/quả**, cả 3 lớp đều **≥30 quả** → thông tin người dùng
  hoàn toàn hợp lý.
- **Q3 đã trả lời (2026-11-07):** số quả và số ảnh/quả là **ước lượng**, không chính
  xác. Vì vậy bảng trên chỉ là **minh hoạ số học**, không phải số đo.

**Kết luận:** thông tin "≥30 quả/lớp" **không bị dữ liệu phủ định**, nhưng cũng
**không được xác minh**. Không dùng con số ước lượng này làm fact trong báo cáo hay
cam kết chất lượng. Task 3 phải đếm `group_id` thực tế.

### 2.4 Bằng chứng trực quan: ảnh liên tiếp không phải cùng quả **[INFERENCE từ quan sát]**

Đã xem 25 ảnh đầu tiên theo timestamp của từng lớp (sheet tạm, không lưu repo). Ở
`unripe` #01–#05, năm ảnh liên tiếp khác nhau rõ về **kích thước quả, hướng xoay, độ dài
cuống và độ thon**; không có cặp nào trông như hai góc chụp của cùng một quả. Ví dụ #02
nằm ngang với cuống ngắn, trong khi #03–#05 chéo/đứng với cuống dài hơn và thân quả dày
hơn.

Đây **không phải** 5 góc xoay của cùng một quả; kích thước và hình dạng khác nhau rõ.
`medium_ripe` và `ripe` cũng tương tự: các ảnh cạnh nhau khác quả.

Hệ quả:

- **Không thể chia 5 ảnh liên tiếp thành một group.**
- **Không thể dùng timestamp/burst để gán group.**
- **Xung đột cần ghi nhận:** Q2=A nói người dùng *thường* chụp liên tiếp từng quả,
  nhưng (a) không nhớ thứ tự chính xác, (b) số ảnh/quả là ước lượng 3–6, (c) 25 ảnh
  đầu cho thấy ảnh cạnh nhau khác quả rõ. Có thể người dùng xoay/đổi khoảng cách khi
  chụp cùng quả, hoặc trí nhớ không khớp. Cả hai trường hợp đều dẫn đến **không thể
  tái tạo group chính xác từ timestamp**. Vì 473 ảnh chỉ dùng train, điều này không
  ảnh hưởng đánh giá cuối, nhưng **cấm mọi heuristic "block 5 ảnh = 1 quả"**.

Đã thử thêm hai chẩn đoán định lượng (chỉ để tham khảo, không kết luận):

- Khoảng cách đặc trưng màu+size+shape theo offset 1–80: offset **1–3 luôn nhỏ nhất**
  (ảnh kề nhau giống nhau hơn), nhưng **không có chu kỳ đều ~30–40** như giả thuyết
  "luân phiên đều"; các local minimum rời rạc (unripe 25, medium 18, ripe 21) không nhất
  quán giữa các lớp.
- Top-1 nearest neighbor theo đặc trưng màu nằm ở offset 1–4 cho **21,3% / 35,2% / 26,9%**
  ảnh (unripe / medium / ripe); không đủ để suy ra group.

**Kết luận:** không có cách tự động đáng tin để khôi phục mapping từ dữ liệu hiện có.
Cần người dùng cung cấp log/trí nhớ, hoặc chấp nhận không có group.

### 2.5 Vì sao "~15 cụm burst" không phải số quả

Task 2 tính cụm bằng cách gộp các ảnh có khoảng cách >10 giây. Bảng dưới đây ghi
**số cụm** (không phải số gap) nếu tách tại từng ngưỡng; số cụm = 1 + số gap vượt ngưỡng:

| Lớp | Số cụm nếu tách tại gap >2s | >5s | >10s | >30s |
|---|---:|---:|---:|---:|
| `unripe` | 31 | 1 | **1** | 1 |
| `medium_ripe` | 39 | 6 | **3** | 1 |
| `ripe` | 93 | 30 | **11** | 4 |

Ví dụ: `unripe` có 30 gap >2 giây → 31 cụm; nhưng **0 gap >5 giây** → tách tại >5s
chỉ còn 1 cụm. Vì vậy "số cụm" thay đổi mạnh theo ngưỡng và **không phải** số quả.

Nếu người dùng chụp xen kẽ nhiều quả với nhịp ~2 giây/ảnh, thì **mọi quả trong một
phiên sẽ bị gộp thành 1 cụm** — đúng như `unripe`. Vì vậy:

- "1 cụm" = "một phiên chụp liên tục", **không phải** "một quả".
- "15 cụm" = "15 khoảng thời gian tách biệt", **không phải** "15 quả".
- Split theo cụm burst (gap>10s) không phải split theo quả; nó chỉ tách theo thời gian.

### 2.6 Kết luận mapping

| Câu hỏi | Trả lời |
|---|---|
| Có mapping ảnh→quả trong filename/folder/EXIF/groups.csv? | **Không** |
| Có thể suy ra từ timestamp/burst? | **Không** — ảnh liên tiếp khác quả |
| Có thể suy ra từ contact sheet? | Chỉ khi người dùng **trực tiếp nhận ra** quả; không tự động |
| Số quả mỗi lớp có khớp ≥30? | **Chưa xác minh** — Q3: chỉ là ước lượng; không báo cáo như fact |
| 473 ảnh = 473 quả? | **Không** |
| 15 cụm burst = 15 quả? | **Không** |

**Trạng thái:** `image → pepper` là **UNKNOWN**. Không được tự gán group.

---

## 3. Phân tích leakage và split

### 3.1 Split hiện tại — facts

| Tập | Số ảnh | Tỷ lệ |
|---|---:|---:|
| train | 332 | 70,2% |
| validation | 71 | 15,0% |
| test | 70 | 14,8% |

Theo lớp:

| Lớp | train | validation | test |
|---|---:|---:|---:|
| `unripe` | 138 | 30 | 29 |
| `medium_ripe` | 100 | 21 | 21 |
| `ripe` | 94 | 20 | 20 |

- **0 file** xuất hiện ở >1 tập.
- **0 group** được dùng → kiểm tra group-leak **không có hiệu lực**.
- **183 cặp ảnh kề nhau ≤5 giây** bị tách sang hai tập khác nhau.
- Split thử nghiệm `--burst-groups 10`: `unripe` 197/0/0; `medium_ripe` 63/68/11;
  `ripe` 89/28/17; tổng 349/96/28 (73,8/20,3/5,9%) — lệch và có lớp/tập rỗng.

### 3.2 Hiệu chỉnh cách hiểu "183 burst leak"

**Fact:** 183 cặp ảnh *kề nhau về timestamp* (≤5 giây) nằm ở hai tập khác nhau.

**Không phải fact:** 183 cặp đó là cùng một quả.

Bằng chứng trực quan ở §2.4 cho thấy các ảnh kề nhau thường là **quả khác nhau**. Vì vậy:

- 183 là **cận trên theo kiểu "temporal proximity"**, không phải số leak thật.
- Nó vẫn hữu ích như cảnh báo: nếu có cặp nào thực sự cùng quả, chúng đã bị tách.
- Nhưng **không được báo cáo "183 quả bị leak"** hoặc dùng con số này làm metric leakage.

### 3.3 Rủi ro leakage thật: nhiều ảnh/quả bị chia ngẫu nhiên **[INFERENCE]**

Đây là rủi ro chính, không phụ thuộc timestamp:

- Mỗi quả có ~`k` ảnh (người dùng nói ~5).
- Split hiện tại chia **từng ảnh** ngẫu nhiên theo lớp (không có group).
- Xác suất **tất cả `k` ảnh của một quả đều rơi vào train** = `0,7^k`.
- Xác suất **có ít nhất 1 ảnh trong validation hoặc test** = `1 − 0,7^k`.

| k (ảnh/quả) | P(ít nhất 1 ảnh ở val/test) | Số quả bị leak nếu 30 quả/lớp | P(ít nhất 1 ảnh ở test) |
|---:|---:|---:|---:|
| 3 | 65,7% | ~20 | 38,6% |
| 4 | 76,0% | ~23 | 47,8% |
| **5** | **83,2%** | **~25** | **55,6%** |
| 6 | 88,2% | ~26 | 62,3% |

**Đọc bảng:** nếu mỗi quả có ~5 ảnh và split ngẫu nhiên, khoảng **83% số quả** sẽ có
ít nhất một ảnh trong validation/test. Với 30 quả/lớp, ~25 quả/lớp bị leak. Đây là
**ước lượng xác suất**, không phải số đo chính xác; nhưng đủ để kết luận split hiện tại
**không dùng được cho đánh giá cuối**.

Lưu ý: nếu các ảnh của cùng một quả rất giống nhau (cùng góc, cùng ánh sáng), leak còn
nguy hiểm hơn vì model có thể "nhớ" quả. Nếu các ảnh khác góc/ánh sáng, leak vẫn làm
metric lạc quan nhưng ít hơn. Chưa đo được mức độ giống nhau vì thiếu mapping.

### 3.4 Split theo cụm burst — không dùng được

Split `--burst-groups 10` cho `unripe` 197/0/0. Task 2 diễn giải là "chỉ có 1 group".
Nhưng theo §2.5, "1 cụm" chỉ là "1 phiên chụp". Vì vậy:

- Split này **không phải** split theo quả.
- Kết quả `unripe` 197/0/0 **không chứng minh** chỉ có 1 quả; nó chỉ chứng minh
  không có khoảng nghỉ >10 giây trong phiên `unripe`.
- **Không dùng split này** cho train/evaluate.

### 3.5 Các phương án split

| Mã | Phương án | Ưu điểm | Nhược điểm | Khi nào dùng |
|---|---|---|---|---|
| **S0** | Giữ split ngẫu nhiên hiện tại (không group) | Đã có; 70/15/15; đủ lớp; 0 trùng file | Leak cùng quả rất cao (~83% quả nếu k=5); metric lạc quan; không đại diện | **Chỉ smoke-test pipeline**, không báo accuracy cuối |
| **S1** | Người dùng cung cấp `group_id` thật → stratified group split 70/15/15 | Đánh giá đúng; dùng được 473 ảnh; kiểm tra group-leak có hiệu lực | Cần mapping 473 ảnh → ~90–120 quả; tốn thời gian; nếu mapping sai thì split sai | Khi có log/trí nhớ/nhận diện được quả |
| **S2** | Bán tự động: embedding/similarity đề xuất group → người dùng duyệt | Giảm công thủ công; có thể cứu dữ liệu cũ | Chưa có model/embedding đáng tin; ớt nhỏ, xoay nhiều; dễ gộp/tách sai; cần thêm code & thời gian | Chỉ khi không có mapping và muốn thử cứu dữ liệu |
| **S3** | Dùng 473 ảnh hiện tại cho **train/pretraining**, thu tập **validation/test mới** từ quả vật lý khác ở Task 3 | Không cần mapping cũ; đánh giá sạch trên quả mới; tận dụng được 473 ảnh | Cần thu thập mới trước khi evaluate; test mới phải là quả **khác** với 473 ảnh cũ | **Khuyến nghị nếu không có mapping** |
| **S4** | Chia theo phiên/thời gian (temporal/session split) | Không cần mapping; kiểm tra domain shift theo phiên | `unripe`/`medium` chỉ 1 phiên; không chia 3 tập; quả xen kẽ nên vẫn leak | Không khuyến nghị |
| **S5** | Chia theo cụm burst | Không cần mapping; 0 burst leak theo định nghĩa | Cụm ≠ quả; `unripe` 197/0/0; split lệch | **Loại bỏ** |
| **S6** | Chia ảnh + "buffer" thời gian (hold-out một khối thời gian) | Giảm gần-trùng nếu ảnh cùng quả nằm sát nhau | Ảnh xen kẽ nên cùng quả vẫn xuất hiện khắp nơi; phân bố lớp có thể lệch | Dự phòng, không dùng cho kết luận cuối |

### 3.6 Khuyến nghị split

**Đã chốt: S3 (Q4=B).**

- Dùng **473 ảnh, trừ 10 ảnh trong exclusion list**, cho **train**.
- **Không** dùng split 332/71/70 hiện tại làm benchmark cuối; chỉ smoke-test pipeline.
- Task 3 thu **quả mới, vật lý khác** cho **validation và test**; chia theo group,
  không quả nào nằm ở >1 tập.
- Test set phải là quả **chưa từng xuất hiện** trong 473 ảnh cũ và trong train mới.
- Nếu cần dev trước khi có dữ liệu mới: chỉ dùng một subset nhỏ của 473, gắn nhãn
  `dev_leaky`, không báo accuracy.
- Khi đó metric test mới có ý nghĩa generalization.

**Nếu sau này có mapping (không phải hiện tại): S1.**

- Điền `groups.csv`, chạy `python scripts/prepare_dataset.py --force-resplit`.
- Kiểm tra `group_leaks = 0`, `overlap_paths = 0`, cả 3 lớp có mặt ở 3 tập.
- Có thể chia theo group cho cả 473 ảnh; val/test sẽ là các quả chưa thấy trong train.

**Trả lời câu hỏi "có nên giữ ảnh burst để train, chỉ dùng quả độc lập cho val/test?"**

- **Có**, nếu biết group: train được dùng nhiều ảnh/quả (3–5 ảnh, có augmentation);
  validation/test chỉ dùng **quả chưa thấy trong train**. Đây là cách chuẩn.
- Nếu **không biết group**: không thể đảm bảo. Khi đó dùng S3 (test mới toàn bộ).
- Không nên đưa một ảnh lẻ của quả vào val/test nếu các ảnh khác của quả đó nằm trong
  train — dù chỉ 1 ảnh cũng đủ leak.

---

## 4. Bảng 10 ảnh nghi vấn

### 4.1 Cách đọc bảng

- **Không đổi nhãn.** Bảng chỉ để người dùng duyệt.
- `hue` là hue trung vị trên vùng quả (đã tách nền).
- Các dải màu **chồng lấn**:
  - xanh: 60–180°
  - vàng: 40–70°
  - cam: 20–60°
  - đỏ: <20° hoặc ≥330°
  - Vì chồng lấn, tổng các `frac` có thể >1; chỉ dùng để mô tả, **không phải tỷ lệ
    diện tích độc lập**.
- Vùng foreground hiện tại **có thể bao gồm cuống xanh**, nên ảnh `ripe` có thể có
  `greenish_frac` cao do cuống, không phải do vỏ chưa chín.
- `split` là tập hiện tại; `burst pair` là cặp kề timestamp ≤5s bị tách tập (nếu có).
  Cặp này **không xác nhận cùng quả**.

### 4.2 Bảng

| # | Filename | Nhãn hiện tại | Đặc điểm quan sát (số liệu) | Nhãn thay thế có thể | Lý do | Mức chắc chắn | Cần user xác nhận? | Split cũ (smoke-test) / burst pair — **nay đều bị loại** |
|---:|---|---|---|---|---|---|---|---|
| 1 | `IMG_20261107_150901.jpg` | `unripe` | hue 68,4°; xanh 0,25; vàng 0,60; cam 0,48; đỏ 0,00; fg 2,36% | `medium_ripe` | Vỏ xanh-vàng, đầu quả ngả vàng/cam rõ; nằm giữa unripe và medium | Thấp–trung bình | **Có** | train; không có cặp ≤5s trong danh sách |
| 2 | `IMG_20261107_150329.jpg` | `medium_ripe` | hue 44,8°; đỏ 0,43; cam 0,15; xanh 0,20; fg 4,30% | `ripe` | Quả nâu-đỏ sẫm, ít mảng xanh; có thể chín hơn nhãn medium | Trung bình | **Có** | train; ↔ `150326` test (3s) — có thể khác quả |
| 3 | `IMG_20261107_150511.jpg` | `medium_ripe` | hue 69,7°; xanh 0,29; vàng 0,46; cam 0,51; đỏ 0,00; fg 3,85% | `unripe` | Thân quả còn xanh nhiều, đầu quả vàng-cam; ranh giới mờ | Thấp–trung bình | **Có** | train; không có cặp ≤5s trong danh sách |
| 4 | `IMG_20261107_145600.jpg` | `ripe` | hue 51,0°; đỏ 0,47; xanh 0,19; cam 0,01; fg 3,23% | `medium_ripe` (theo màu) | Thân quả đỏ nhưng cuống xanh dài; đặc trưng màu bị cuống làm lệch. Nhìn ảnh: thân đỏ, nhãn `ripe` **có khả năng đúng** | **Cao rằng nhãn đúng** | Chỉ cần xác nhận quy tắc loại cuống | train; ↔ `145558` test (2s) — có thể khác quả |
| 5 | `IMG_20261107_145847.jpg` | `ripe` | hue 47,2°; đỏ 0,28; cam 0,33; xanh 0,27; fg 2,78% | `medium_ripe` | Cam-đỏ, chưa đỏ sẫm; ranh giới medium/ripe | Thấp–trung bình | **Có** | train; không có cặp ≤5s trong danh sách |
| 6 | `IMG_20261107_151125.jpg` | `unripe` | hue 68,1°; xanh 0,10; vàng 0,50; cam 0,58; đỏ 0,00; fg 2,48% | `medium_ripe` | Vỏ vàng-cam rõ, xanh rất ít; khác biệt với phần lớn `unripe` xanh | Thấp–trung bình | **Có** | train; ↔ `151124` test (1s) — có thể khác quả |
| 7 | `IMG_20261107_150946.jpg` | `unripe` | hue 97,5°; xanh 0,61; vàng 0,01; cam 0,00; đỏ 0,00; fg 1,97% | Giữ `unripe` | Vẫn xanh rõ; quả nhỏ (fg 1,97%). Từng bị gắn cờ "vàng" nhưng số liệu không ủng hộ | **Cao rằng nhãn đúng** | Chỉ cần xác nhận chính sách ảnh quả nhỏ | validation; ↔ `150947` train (1s) — có thể khác quả |
| 8 | `IMG_20261107_151349.jpg` | `unripe` | hue 74,7°; xanh 0,85; vàng 0,61; cam 0,00; đỏ 0,00; fg 3,01% | Giữ `unripe` (hoặc `medium` nếu ưu tiên ánh vàng) | Xanh-vàng nhưng xanh vẫn chiếm ưu thế; quả lớn, bbox rộng | Trung bình | **Có** (nếu thấy vàng rõ) | train; ↔ `151347` val (2s), `151352` test (3s) — có thể khác quả |
| 9 | `IMG_20261107_151342.jpg` | `unripe` | hue 76,8°; xanh 0,68; vàng 0,50; cam 0,00; đỏ 0,00; fg 2,96% | Giữ `unripe` (hoặc `medium` nếu ưu tiên ánh vàng) | Xanh-vàng, xanh chiếm ưu thế; tương tự #8 | Trung bình | **Có** (nếu thấy vàng rõ) | train; không có cặp ≤5s trong danh sách |
| 10 | `IMG_20261107_150950.jpg` | `unripe` | hue 88,5°; xanh 0,71; vàng 0,01; cam 0,00; đỏ 0,00; fg **1,58%** | Giữ `unripe` | Xanh rõ; quả nhỏ nhất dataset. Từng bị gắn cờ "vàng" nhưng số liệu không ủng hộ | **Cao rằng nhãn đúng** | Chỉ cần xác nhận chính sách ảnh quả nhỏ | train; không có cặp ≤5s trong danh sách |

> **Lưu ý (Q5):** cột "Split cũ" chỉ là lịch sử của split smoke-test 332/71/70.
> **Cả 10 ảnh hiện bị loại khỏi train/validation/test** và nằm trong
> `dataset/review_exclusions.csv` cho đến khi người dùng review. Không ảnh nào bị
> xóa, đổi tên, hay đổi nhãn.

### 4.3 Nhận xét

- **5 ảnh lệch màu** (`150901`, `150329`, `150511`, `145600`, `145847`) **không đồng nghĩa
  5 nhãn sai**. Ít nhất `145600` rất có thể đúng nhãn `ripe`; lệch là do cuống xanh.
- **5 ảnh `unripe` từng bị gắn cờ vàng**: thực tế `150946` và `150950` vẫn **xanh rõ**;
  `151349`/`151342` là xanh-vàng nhưng xanh chiếm ưu thế; chỉ `151125` là **vàng-cam rõ**.
  Danh sách cũ đã **over-flag**.
- Các ca khó nhất là **ranh giới chuyển màu**:
  - xanh-vàng (`150901`, `151125`, `151349`, `151342`)
  - cam-đỏ (`150329`, `145847`)
  - thân xanh + đầu vàng (`150511`)
- Không có ảnh nào trong 10 ảnh là **chắc chắn sai nhãn** dựa trên số liệu hiện có.
  Cần người dùng nhìn ảnh gốc và quyết định.

### 4.4 Xử lý đã chốt (Q5, 2026-11-07)

**Loại cả 10 ảnh khỏi training; không xóa/sửa/đổi nhãn; lưu vào
`dataset/review_exclusions.csv` để review sau.**

| Filename | Nhãn hiện tại | Lý do loại tạm thời |
|---|---|---|
| `150901` | `unripe` | màu gần `medium_ripe`; đầu quả vàng/cam |
| `150329` | `medium_ripe` | màu gần `ripe`; thân nâu-đỏ sẫm |
| `150511` | `medium_ripe` | màu gần `unripe`; thân xanh, đầu vàng |
| `145600` | `ripe` | màu gần `medium_ripe`; cuống xanh dài (nhãn có thể đúng) |
| `145847` | `ripe` | màu gần `medium_ripe`; cam-đỏ ranh giới |
| `151125` | `unripe` | dải vàng/cam rõ; ranh giới unripe/medium |
| `150946` | `unripe` | quả nhỏ; bị cờ "vàng" cũ |
| `151349` | `unripe` | xanh-vàng; bbox lớn |
| `151342` | `unripe` | xanh-vàng; bbox lớn |
| `150950` | `unripe` | quả nhỏ nhất; bị cờ "vàng" cũ |

- Đây **không phải** kết luận 10 ảnh sai nhãn; chỉ là quyết định thận trọng.
- `dataset/review_exclusions.csv` dự kiến có cột:
  `filename,current_label,reason,status,reviewed,reviewed_label,notes`.
- Manifest builder phải **bỏ 10 ảnh này** khỏi train/validation/test; audit/contact
  sheet vẫn hiển thị để review.
- Nếu sau này review và giữ nhãn: đổi `status=accepted`.
- Nếu đổi nhãn: ghi `reviewed_label` và dùng `dataset/label_overrides.csv`; **không
  di chuyển raw file** nếu người dùng chưa cho phép.
- Không tạo derivative cho 10 ảnh này cho đến khi review xong.

---

## 5. Đánh giá crop và chất lượng ảnh

### 5.1 Hình học & độ phân giải hiệu dụng

| Lớp | Diện tích quả p10 | p50 | p90 | BBox p50 | Độ nét p10 | p50 |
|---|---:|---:|---:|---:|---:|---:|
| `unripe` | 2,61% | **3,73%** | 5,45% | 10,5% | 202,8 | 412,2 |
| `medium_ripe` | 2,54% | **3,55%** | 5,04% | 12,6% | 156,5 | 340,9 |
| `ripe` | 2,63% | **3,28%** | 5,00% | 13,1% | **83,1** | 396,8 |

- Ảnh gốc 1024×1024. Bbox quả (median) chỉ chiếm 10,5–13,1% diện tích khung → cạnh bbox
  ≈ 330–370 px ở 1024, tương đương **≈ 72–80 px ở 224**. Quả lại thon dài và chỉ lấp đầy
  ~30% diện tích bbox, nên bề dày quả khi resize thẳng còn **ước lượng ~25–35 px**, chiều
  dài ~50–60 px → **quá nhỏ** để học hình thái/đốm màu.
- Nếu crop đúng bbox (khoảng 330–370 px mỗi cạnh ở 1024) rồi resize về 224, quả chiếm
  ~30% diện tích crop và **giữ được nhiều chi tiết hơn** so với resize toàn khung. Đây là
  lợi ích kỹ thuật thật của crop.
- Tuy nhiên crop **không sửa được ảnh mờ**; ảnh mờ vẫn mờ sau crop.

### 5.2 Độ nét / ánh sáng / nền

- Độ nét trung vị 341–412; p10 của `ripe` chỉ 83 → một số ảnh `ripe` mờ hơn.
- 6 ảnh mờ nhất (chỉ ghi nhận, **không xóa**):
  - `IMG_20261107_142529.jpg` (ripe, 19,0)
  - `IMG_20261107_142534.jpg` (ripe, 19,5)
  - `IMG_20261107_142526.jpg` (ripe, 28,5)
  - `IMG_20261107_142519.jpg` (ripe, 29,3)
  - `IMG_20261107_151210.jpg` (unripe, 43,3)
  - `IMG_20261107_142559.jpg` (ripe, 54,7)
- Ánh sáng nhìn chung đều, nền sáng; có **điểm chói (specular highlight)** trên vỏ ở một
  số ảnh, có thể làm pixel chuyển trắng/vàng nhạt → dễ nhầm với màu chín.
- **Nền chỉ 1 biến thể** (≈220,235,235) cho cả 473 ảnh → model có thể học nền; domain
  shift sang băng tải/OV2640 sẽ lớn.
- Khoảng cách chụp gần như cố định (diện tích quả p10–p90 hẹp); góc xoay đa dạng (tốt).

### 5.3 Thông tin màu có đủ phân biệt 3 lớp không?

- Trong điều kiện chụp hiện tại: **có**, theo số liệu HSV:
  - `unripe`: hue ~83°, xanh 0,67, đỏ 0,00
  - `medium_ripe`: hue ~53°, cam 0,49, xanh 0,17
  - `ripe`: hue ~32°, đỏ 0,56, xanh 0,15
- Nhưng:
  - Hue phụ thuộc **cân bằng trắng/ánh sáng**; khi sang OV2640/băng tải, hue có thể trôi.
  - Vùng foreground có thể gồm **cuống xanh**, làm lệch đặc trưng (ví dụ `145600`).
  - **1-NN 98,94%** chỉ là chẩn đoán đặc trưng màu, **không phải accuracy model**; nó
    không chứng minh model sẽ tổng quát hóa.
- Kết luận: màu là tín hiệu chính và **đủ mạnh trong điều kiện hiện tại**, nhưng cần
  kiểm soát WB/ánh sáng khi thu thập và kiểm thử trên ảnh mới.

### 5.4 Các phương án crop

| Mã | Phương án | Ưu điểm | Nhược điểm | Leakage / derivative |
|---|---|---|---|---|
| **C1** | **Không crop** — resize toàn khung 1024→224 | Đơn giản; khớp pipeline hiện tại; không derivative; không cần detector | Quả quá nhỏ; model dễ học nền; ít chi tiết | Không tạo derivative; leakage phụ thuộc group, không phụ thuộc crop |
| **C2** | Crop bbox + padding, **lưu derivative riêng** (raw giữ nguyên) | Quả to hơn ~3×; giữ chi tiết; có thể chuẩn hóa | Cần detector/ngưỡng nền; có thể cắt cuống/đầu quả; tạo thêm file; phải map group/split cho derivative; inference phải crop giống hệt | Derivative **phải cùng group** với ảnh gốc; nếu không, có thể tạo leak mới |
| **C3** | **Crop/zoom on-the-fly** trong `tf.data` (không lưu file) | Không tạo derivative; tăng đa dạng scale; raw là nguồn duy nhất | Cần bbox/metadata hoặc random crop; có thể cắt mất quả; inference phải mô phỏng cùng phân bố | Không tạo file; vẫn phải chia group trước khi split |
| **C4** | **Chụp gần hơn** ở Task 3 (quả 25–40% khung), dùng full-frame | Tốt nhất cho triển khai; không cần crop; tăng chi tiết thật; đa dạng nền/ánh sáng | Cần thu thập mới; cần chuẩn hóa khoảng cách/ánh sáng | Không derivative; group do người dùng gán |

### 5.5 Crop & framing đã chốt (Q6=C)

**Đã chốt: crop derivative riêng (`crop_v1`), giữ raw, dùng cùng quy trình khi
inference.** Chi tiết đầy đủ ở `reports/task3_data_plan.md` §4.

1. **Raw giữ nguyên**; derivative ở `dataset/derived/crop_v1/` (PNG 224×224×3, RGB
   `[0,255]`).
2. **Quy trình `crop_v1`:**
   - Nền: median viền ảnh (473 ảnh cũ) hoặc background reference frame (deployment).
   - Mask: khoảng cách màu RGB/Lab > ngưỡng (calibrate trên mẫu; gợi ý 25–40 đơn vị
     RGB); morphology; largest component; dilation 2–3 px để giữ cuống.
   - Bbox + **padding 20%**; square; clamp/pad nền; resize 224×224; lưu PNG.
   - Metadata: source, SHA256, bbox, pad, output, group_id, split, label, crop_status.
3. **Inference:** dùng đúng hàm `crop_v1`; chụp background reference lúc khởi động;
   nếu crop fail → `crop_failed`, **không dự đoán**.
4. **Leakage:** derivative kế thừa `group_id`/split; không trộn group giữa các tập.
5. **QA:** overlay mask/bbox trên ≥30 ảnh mẫu; mask area 0.5–60%; output đúng 224×224;
   không cắt quả; so sánh màu với raw.
6. **Không tạo derivative cho 10 ảnh bị loại** (Q5).
7. **Train trên derivative**; không trộn raw full-frame trong `crop_v1`.
8. **Không crop trước khi chia group**: chia group trước, hoặc để derivative kế thừa
   group rồi mới split.

**Trả lời trực tiếp:**

- *Có nên crop quả từ ảnh hiện tại không?* — **Có** (Q6=C). Crop giúp quả to hơn ~3×
  so với resize toàn khung.
- *Có nên lưu derivative riêng và giữ raw?* — **Có**, bắt buộc. Raw là nguồn duy nhất;
  derivative có thể xóa/tái tạo.
- *Crop có làm mất thông tin/leakage không?* — Có thể mất một phần cuống/đầu quả nếu
  padding quá nhỏ; **không tạo leakage nếu group được giữ đúng**, nhưng **không sửa
  được leakage** nếu cùng quả đã nằm nhiều tập.
- *Cần chụp bổ sung ảnh toàn cảnh hay gần quả?* — **Ưu tiên ảnh gần quả** (25–40%
  khung) cho train/deploy; có thể chụp thêm một ít **toàn cảnh** để kiểm tra crop/
  detector, nhưng nguồn chính là ảnh gần.

### 5.6 Mục tiêu OV2640 (đề xuất — chưa kiểm chứng)

Đây là **mục tiêu thiết kế**, không phải thông số đã đo trên phần cứng:

- Quả chiếm **25–40% diện tích khung** tại vị trí chụp.
- Camera cố định, khoảng cách cố định; một quả/khung; không zoom số.
- Ánh sáng khuếch tán, không flash trực tiếp; **khóa WB**; nền matte, khác màu vỏ.
- Input model 224×224×3; resize từ ảnh gốc (OV2640 2 MP hoặc cấu hình phù hợp).
- Nếu dùng full-frame resize: 25–40% diện tích quả cho effective resolution tốt hơn
  nhiều so với 3,5% hiện tại.
- Cần kiểm thử thật trên OV2640 + Termux để xác nhận độ nét, màu, latency; **chưa có
  số liệu phần cứng** ở Task 2.5.

---

## 6. Các phương án xử lý & ưu/nhược điểm

### 6.1 Nhóm A — Group ID & burst

| Mã | Phương án | Ưu | Nhược | Đề xuất |
|---|---|---|---|---|
| A1 | Người dùng cung cấp mapping/log → điền `groups.csv` | Chính xác nhất; dùng được 473 ảnh; split đúng | Cần công sức; nếu log không đầy đủ thì khó | **Ưu tiên nếu có** |
| A2 | Người dùng nhận diện quả qua contact sheet → điền group | Không cần log; trực quan | 473 ảnh, quả xoay nhiều, dễ nhầm; rất tốn thời gian | Chỉ khi A1 không có |
| A3 | Bán tự động: similarity/embedding đề xuất group → user duyệt | Giảm công; có thể cứu dữ liệu | Chưa có embedding đáng tin; dễ sai; cần code/thời gian | Thử nghiệm, không mặc định |
| A4 | Không gán group; dùng 473 ảnh cho train; test mới ở Task 3 | Không cần mapping; đánh giá sạch nếu test mới | Cần thu thập mới; không dùng được 473 ảnh cho test | **Đã chốt (Q1=C, Q4=B)** |
| A5 | Chia theo cụm burst | Không cần mapping | Cụm ≠ quả; `unripe` 197/0/0 | **Loại bỏ** |
| A6 | Gán group theo block 5 ảnh liên tiếp | Nhanh | Sai vì ảnh liên tiếp khác quả (đã kiểm tra trực quan) | **Loại bỏ** |

### 6.2 Nhóm B — Nhãn

| Mã | Phương án | Ưu | Nhược | Đề xuất |
|---|---|---|---|---|
| B1 | Giữ nguyên nhãn 463 ảnh còn lại | Không rủi ro thay đổi; đơn giản | 10 ảnh flagged bị loại khỏi train | **Đã chốt một phần** |
| B2 | Người dùng duyệt 10 ảnh → đổi nhãn các ca rõ ràng | Giảm noise | Cần thời gian; có thể tranh cãi | Để sau (Q5) |
| B3 | Loại 10 ảnh flagged khỏi train; giữ `review_exclusions.csv`; không đổi nhãn | An toàn; minh bạch | Cần metadata; giảm 10 ảnh | **Đã chốt (Q5)** |
| B4 | 2 người gán độc lập toàn bộ dataset | Chất lượng cao; đo đồng thuận | Tốn công; chưa có người thứ 2 | Cho Task 3 |
| B5 | Dùng ngưỡng màu cứng để tự động sửa | Nhanh | Sai lệch; vi phạm yêu cầu; bỏ qua ngữ cảnh | **Không làm** |

### 6.3 Nhóm C — Crop/chất lượng

Đã trình bày ở §5.4. Tóm tắt:

- **C2** (crop derivative `crop_v1`) — **đã chốt (Q6=C)** cho 473 ảnh hiện tại và
  dữ liệu mới; raw giữ nguyên; inference dùng cùng quy trình.
- **C4** (chụp gần 25–40% khung) — **đã chốt (Q7)** cho Task 3.
- **C1** (không crop) — không dùng cho train cuối; chỉ có thể dùng cho smoke-test.
- **C3** (on-the-fly crop/zoom) — không chọn; `crop_v1` derivative rõ ràng hơn.

### 6.4 Nhóm D — Split

Đã trình bày ở §3.5. Tóm tắt:

- **S0** — chỉ smoke-test.
- **S1** — chỉ nếu sau này có mapping (hiện tại Q1=C, không dùng).
- **S3** — **đã chốt (Q4=B)**.
- **S4/S5/S6** — không khuyến nghị.

### 6.5 Bảng quyết định tóm tắt (đã chốt)

| Quyết định | Đã chốt | Ghi chú triển khai |
|---|---|---|
| Gán group ID | **Không tự gán cho 473 ảnh** | Q1=C; 473 ảnh chỉ train |
| Split | **S3**: 463 ảnh train; val/test mới từ quả độc lập | Q4=B; split cũ chỉ smoke-test |
| Nhãn 10 ảnh | **Loại khỏi train; giữ exclusion list** | Q5; không xóa/đổi nhãn |
| Crop | **C2 — `crop_v1` derivative, giữ raw** | Q6=C; inference dùng cùng hàm |
| Chụp Task 3 | **Group ID, 25–40% khung, WB/ánh sáng, 3–6 ảnh/quả, 30–50+ quả/lớp** | Q7 |
| Accuracy ≥90% | **Chưa cam kết** | Chỉ sau test sạch + train/evaluate thật |

---

## 7. Quyết định đã chốt & tham số còn mở

### 7.1 Đã chốt (2026-11-07)

| Mã | Trả lời | Ghi chú |
|---|---|---|
| **Q1** | **C — không có mapping ảnh→quả** | Không tự gán group; 473 ảnh chỉ train |
| **Q2** | **A — thường chụp liên tiếp từng quả, không nhớ thứ tự** | Không dùng timestamp/block để gán group; xem §2.4 |
| **Q3** | **Số quả/ảnh là ước lượng** | Không báo cáo ≥30 quả/lớp như fact |
| **Q4** | **B — 473 ảnh train; quả mới cho val/test** | S3; split cũ chỉ smoke-test |
| **Q5** | **Loại 10 ảnh flagged khỏi train; giữ exclusion list** | Không xóa/đổi nhãn; review sau |
| **Q6** | **C — crop derivative riêng, giữ raw, cùng crop khi inference** | `crop_v1` |
| **Q7** | **YES với điều kiện** | group_id, 25–40%, WB/ánh sáng, 3–6 ảnh/quả, 30–50+ quả/lớp, giữ EXIF/raw |

### 7.2 Tham số còn mở (đề xuất mặc định)

| Tham số | Đề xuất | Cần xác nhận? |
|---|---|---|
| Crop padding | 20% mỗi cạnh | Có |
| Crop output | PNG 224×224×3, RGB `[0,255]` | Có |
| Crop threshold | Calibrate; gợi ý 25–40 RGB distance | Có |
| Background estimate | Border median (ảnh cũ); background reference (deployment) | Có |
| Val/test allocation | 50/50 theo group, stratified theo lớp | Có |
| New train peppers | Không bắt buộc; mặc định chỉ 473 + val/test mới | Có |
| Exclusion list | `dataset/review_exclusions.csv` | Có |
| Capture log | `dataset/capture_log.csv` | Có |
| Review 10 ảnh | Sau; không chặn Phase 1 | Không |
| Thời điểm tạo crop | Sau khi plan được confirm, trước pilot | Có |

Chi tiết đầy đủ: `reports/task3_data_plan.md` §11.

---

## 8. Kế hoạch Task 3 đã cập nhật

Chi tiết đầy đủ ở `reports/task3_data_plan.md`. Tóm tắt:

### 8.1 Nguyên tắc

- Raw bất biến; derivative riêng, tái tạo được.
- Group ID khi chụp; không quả nào qua nhiều tập.
- Không ngưỡng màu cứng khi gán nhãn.
- Preprocessing: RGB `[0,255]` → `preprocess_input`; không `Rescaling(1/255)`.
- Không cam kết ≥90% trước test sạch.

### 8.2 Các phase

| Phase | Nội dung | Đầu ra |
|---|---|---|
| 0 | Confirm plan | Quyết định Q1–Q7 + tham số §7.2 |
| 1 | Metadata/tooling; `crop_v1` cho 463 ảnh | exclusion CSV, crop module, derivatives, QA |
| 2 | Pilot 3–5 quả/lớp | capture log, group IDs, crop QA, label calibration |
| 3 | Thu đủ 30–50+ quả/lớp (mục tiêu 50–100+) | raw mới + EXIF + group IDs |
| 4 | Gán nhãn 2 người | label_status, agreement |
| 5 | Audit, derivative, manifest; freeze test | manifest hash, verification |
| 6 | Train/evaluate | image-level + pepper-level metrics, CI |
| 7 | Export/Termux/OV2640 | TFLite + on-device test |

### 8.3 Tiêu chí chấp nhận

| Hạng mục | Tiêu chí |
|---|---|
| Raw | 473/473 SHA256 không đổi; EXIF giữ |
| Exclusion | 10 ảnh không có trong train/val/test |
| Old data | 463 ảnh chỉ train |
| New data | ≥30–50 quả/lớp (mục tiêu 50–100+); báo cáo số thực |
| Images/pepper | 3–6 |
| Group ID | 100% ảnh mới có `group_id` |
| Split | 0 group leak, 0 file overlap, đủ 3 lớp |
| Test | Quả vật lý khác; manifest frozen |
| Crop | `crop_v1` dùng cho train/val/test/predict/export |
| Labels | 2 người; agreement; unresolved → `needs_review` |
| Metrics | Image-level + pepper-level; confusion matrix; 95% CI theo pepper |
| Honesty | Không báo ≥90% trước test; 1-NN không phải model accuracy |

### 8.4 Rủi ro

- Crop segmentation trên nền/ánh sáng mới; test pilot trước.
- Số quả thu thập ít hơn mục tiêu; báo cáo CI rộng, không overclaim.
- Q2 conflict; không dùng timestamp block.
- Domain shift OV2640/Termux; test on-device.
- Công sức gán nhãn 2 người.
- Class imbalance; báo per-class.

---

## 9. Những gì KHÔNG làm trong Task 2.5

- Không train/evaluate.
- Không thêm/tải/tạo ảnh; không dùng ảnh công khai.
- Không sửa/xóa/đổi tên/đổi nhãn ảnh raw.
- Không crop/resize/augment/tạo derivative cho dataset (chỉ tạo **contact sheet review**
  từ ảnh hiện có, không lưu ảnh crop riêng lẻ).
- Không đổi `groups.csv`, `split_manifest.json`, split hay code.
- Không bắt đầu Task 3.
- Không triển khai Android/ESP32/UART/BLE/CUDA.
- Không commit/push thay đổi dataset.

**Đã tạo (review aid, không phải dataset):**

- `reports/image_handling_review.md` (báo cáo này)
- `reports/figures/contact_sheets/chrono_unripe.png`
- `reports/figures/contact_sheets/chrono_medium_ripe.png`
- `reports/figures/contact_sheets/chrono_ripe.png`
- `reports/figures/contact_sheets/review_suspects_10.png`
- `reports/metrics/chrono_review_index.csv` (index, filename, timestamp, split, màu,
  diện tích, độ nét — để bạn tham chiếu khi duyệt contact sheet)

**Không thay đổi:**

- `dataset/raw/**` — SHA256 473/473 khớp.
- `dataset/groups.csv` — vẫn 473 dòng, group_id trống, `verified=no`.
- `dataset/split_manifest.json` — vẫn 332/71/70.
- `dataset/split_manifest_burstgroup.json` — vẫn là artifact thử nghiệm cũ.
- `src/`, `scripts/`, `tests/` — không sửa.

---

## 10. Artifact & cách tái lập

### 10.1 Contact sheet review

- `reports/figures/contact_sheets/chrono_unripe.png` — 197 ảnh, thứ tự timestamp,
  nhãn `#index HHMMSS`.
- `reports/figures/contact_sheets/chrono_medium_ripe.png` — 142 ảnh.
- `reports/figures/contact_sheets/chrono_ripe.png` — 134 ảnh.
- `reports/figures/contact_sheets/review_suspects_10.png` — 10 ảnh nghi vấn, nhãn hiện
  tại + lý do gắn cờ.
- `reports/metrics/chrono_review_index.csv` — bảng index để tra cứu.

**Cách dùng:** mở contact sheet, dùng `#index` để chỉ ra các ảnh **không liền kề** mà
bạn nhận ra là cùng một quả, ví dụ "#001, #037, #073 là cùng quả". Không dùng các ảnh
liền kề làm một nhóm — §2.4 đã chỉ ra ảnh cạnh nhau thường là quả khác. Nếu bạn có log
gốc, càng tốt.

### 10.2 Lệnh tái lập (trong container)

```bash
podman exec -it ai-workspace bash
cd /workspace/chili-ripeness-ai
source /workspace/.venv/bin/activate

# Kiểm tra trạng thái (không sửa gì)
python scripts/check_dataset.py
python scripts/prepare_dataset.py --verify-only

# (Task 2.5) đọc số liệu; không tạo derivative dataset
python - <<'PY'
import json
a = json.load(open("reports/metrics/dataset_audit.json"))
print(a["summary"]["counts"])
print(a["summary"]["class_color"])
PY
```

### 10.3 Ghi chú về contact sheet

Contact sheet là **công cụ review**, không phải dataset derivative. Ảnh raw không bị
sửa; không có ảnh crop/resize riêng lẻ nào được lưu. Nếu người dùng muốn, có thể xóa
các file contact sheet sau khi duyệt (chúng nằm trong `reports/figures/`, đã gitignore).

---

## 11. Kết luận cuối

- Dataset hiện tại **sạch về kỹ thuật** nhưng **thiếu mapping ảnh→quả** và **thiếu đa
  dạng môi trường**.
- **Q1=C/Q3:** không có mapping và số quả chỉ là ước lượng → 473 ảnh được mô tả là
  **473 ảnh với số quả chưa biết**, không phải "473 quả" hay "≥30 quả/lớp".
- **Q2=A** được ghi nhận, nhưng không thể tái tạo group từ timestamp/block 5 ảnh; xem
  §2.4.
- **Q4=B/S3:** 463 ảnh (trừ 10 ảnh bị loại) chỉ dùng **train**; validation/test mới từ
  quả vật lý độc lập. Split 332/71/70 chỉ là smoke-test.
- **Q5:** 10 ảnh flagged vào `dataset/review_exclusions.csv`; không xóa/đổi nhãn.
- **Q6=C:** crop derivative `crop_v1`, giữ raw, dùng cùng quy trình khi inference.
- **Q7:** Task 3 thu 30–50+ quả/lớp (mục tiêu 50–100+), có `group_id`, 25–40% khung,
  ánh sáng/WB ổn định, 3–6 ảnh/quả, giữ EXIF/raw.
- **Không báo accuracy cuối** trên split hiện tại; chỉ sau test sạch.
- **Trạng thái:** kế hoạch đã cập nhật ở `reports/task3_data_plan.md`; **chưa thay đổi
  raw dataset, chưa bắt đầu Task 3**. Chờ người dùng xác nhận kế hoạch và các tham số
  còn mở (§7.2) trước khi triển khai.

---

*Hết báo cáo Task 2.5. Mọi thay đổi dataset chỉ được thực hiện sau khi người dùng xác nhận.*
