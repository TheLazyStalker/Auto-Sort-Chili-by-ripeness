# DATASET_GUIDE — Hướng dẫn thu thập & đặt nhãn dataset ớt

## 1. Ba lớp

| Lớp | Dấu hiệu | Ví dụ màu (tùy giống) |
|---|---|---|
| `unripe` | Chưa chín, vỏ xanh đậm, cứng, bóng mờ | Xanh lá đậm → xanh hơi trắng ở đầu cuống |
| `medium_ripe` | Chín vừa, chuyển màu dở dang, còn vệt xanh xen kẽ chín | Xanh xen đỏ/đỏ cam, mỗi quả khác nhau |
| `ripe` | Đã chín, màu chín đồng đều, vỏ bóng, cuống hơi khô | Đỏ / đỏ sậm / vàng cam tùy giống |

> Ghi chú: màu chín phụ thuộc giống ớt (ớt sừng, ớt chỉ thiên, ớt ngọt…). Hãy chọn **một hoặc ít giống**, chụp trong cùng điều kiện ánh sáng để model học được ranh giới màu, thay vì trộn nhiều giống khác nhau ngay từ đầu. Chưa đặt ngưỡng màu cố định khi chưa xác định giống ớt.

## 2. Cấu trúc thư mục dataset

```
dataset/
├── raw/
│   ├── unripe/          # ảnh chưa chín
│   ├── medium_ripe/     # ảnh chín vừa
│   └── ripe/            # ảnh đã chín
├── train/               # (tự sinh bởi --prepare, không sửa tay)
├── validation/          # (tự sinh bởi --prepare)
├── test/                # (tự sinh bởi --prepare)
├── groups.csv           # [tùy chọn] filename,group_id — phân nhóm theo quả
├── groups.example.csv    # template định dạng groups.csv (tham khảo)
└── split_manifest.json   # (tự sinh bởi --prepare) siêu dữ liệu chia tập
```

Không có ảnh thật trong `dataset/raw/<class>/` khi `--prepare` được kiểm tra an toàn: script báo rõ "Dataset trống" và dừng, không tạo kết quả giả.

## 3. Chọn giống & môi trường chụp

- Ưu tiên 1 giống chiếm đa số; nếu nhiều giống, ghi lại giống của mỗi ảnh trong `groups.csv` (hoặc metadata riêng).
- **Khung hình:** một quả/khung; cho quả chiếm **≥25–40% diện tích ảnh** (dataset hiện tại chỉ ~3,5% — quá xa).
- **Khoảng cách:** cố định giữa các lần chụp; không dùng zoom số.
- **Nền:** đơn giản, matte, trung tính, không cùng màu với vỏ; ghi lại loại nền. Tránh để nền đồng nhất
  tuyệt đối cho toàn bộ dataset (model dễ học nền thay vì quả).
- **Ánh sáng:** khuếch tán, không flash trực tiếp, tránh điểm chói trên vỏ; **khóa cân bằng trắng (WB)**
  để màu không trôi giữa các phiên.
- **Góc chụp:** điện thoại ngang hàng quả ớt; xoay quả để có nhiều góc nhưng **giữ cùng giai đoạn chín**
  trong một lần chụp.
- **Định dạng:** JPG chất lượng cao, độ phân giải ≥1080p; **giữ EXIF gốc** (không crop/resize trước khi lưu raw).
- **Số lượng khuyến nghị:** tối thiểu **≥50 quả độc lập/lớp** (tốt hơn 100+), mỗi quả 3–6 ảnh.
  Tổng mục tiêu **≥300–500 quả độc lập** — quan trọng hơn tổng số ảnh.
  Nhiều ảnh của cùng một quả **không** thay thế được nhiều quả khác nhau.

## 4. Thu thập theo nhóm (quan trọng nhất)

- **Gán group ID cho mỗi quả ớt:** mọi ảnh của cùng 1 quả (chụp góc khác, ánh sáng khác, khung hình khác) phải chung `group_id`. File `dataset/groups.csv` có dạng:

  ```csv
  filename,group_id
  0001_unripe_1.jpg,pepper_0001
  0002_unripe_2.jpg,pepper_0002
  ```

  Xem `dataset/groups.example.csv` để biết template đầy đủ.

- Nếu đặt `groups_file` trong `config.yaml`: `paths.groups_file: dataset/groups.csv`. Dataset loader sẽ chia theo **nhóm** để ảnh cùng 1 quả **không bao giờ** lọt sang nhiều tập — nếu không, model "nhớ mặt" quả ớt và metric trên tập test sẽ ảo (rò rỉ dữ liệu).
- `groups.csv` có thể có thêm cột `verified` (`yes`/`no`). Nếu `group_id` để trống và `verified=no`, nghĩa là **chưa xác minh** ảnh nào cùng một quả; khi đó split coi mỗi ảnh là mẫu độc lập (có cảnh báo rò rỉ burst).
- **Không suy đoán group từ timestamp** rồi coi là sự thật. Chỉ dùng cụm burst như **phương án thử nghiệm** (`--burst-groups`), ghi rõ là heuristic chưa xác minh.

## 5. Đặt nhãn nhất quán

- Mỗi quả ớt chỉ thuộc **một nhãn** duy nhất; nếu không chắc chắn, bỏ qua quả đó hoặc hỏi người thứ hai xác minh.
- **Không dùng một ngưỡng diện tích màu cố định** (ví dụ ">90% màu chín = ripe"). Dữ liệu ảnh thật
  cho thấy 3 lớp tách theo **thứ tự chuyển màu**, không theo một tỷ lệ % cố định:
  - `unripe`: vỏ **xanh lá chiếm ưu thế**, gần như không có mảng đỏ/cam rõ rệt.
  - `medium_ripe`: **chuyển tiếp** — vỏ có cả mảng xanh và cam/đỏ, chưa màu nào áp đảo; thường còn xanh ở cuống/đầu quả.
  - `ripe`: **đỏ chiếm ưu thế** trên thân quả; xanh còn lại chủ yếu ở cuống.
- Ảnh nằm đúng ranh giới (vàng-lục, cam-đỏ) → đánh dấu **"cần duyệt"**, không tự gán nhãn.
- Số liệu phân bố màu thực tế, danh sách ảnh nghi vấn và tiêu chí đề xuất chi tiết: xem `DATASET_AUDIT.md` (§9–§11).
- Ngưỡng chính thức chỉ nên chốt sau khi có **nhiều quả độc lập**, gán nhãn bởi 2 người, rồi kiểm chứng
  trên quả chưa từng thấy — không lấy ngưỡng từ dataset hiện tại (phụ thuộc burst).
- Kiểm tra lại ngẫu nhiên 10% ảnh sau khi chia tập; ảnh trùng nội dung sẽ được báo trong `reports`/script.

## 6. Quy trình đưa vào dự án

1. Đặt ảnh vào `dataset/raw/unripe/`, `medium_ripe/` (hoặc `medium-ripe/`), `ripe/` (file `.jpg/.jpeg/.png`).
2. (Bắt buộc khi có nhiều ảnh/quả) Tạo `dataset/groups.csv` và điền `group_id` cho từng quả; `paths.groups_file` đã trỏ sẵn file này.
3. Chạy `python scripts/check_dataset.py` → kiểm tra nhanh số lượng, ảnh lỗi, trùng lặp.
4. Chạy `python scripts/audit_dataset.py --contact --audit-md reports/metrics/dataset_audit.md` → audit chuyên sâu + manifest + contact sheet.
5. Chạy `python scripts/prepare_dataset.py` → tạo `dataset/split_manifest.json`, copy vào `train/validation/test` (70/15/15, stratified, seed 42) và **in báo cáo xác minh rò rỉ**. Ảnh gốc **không bị xóa**.
6. Đọc kỹ cảnh báo: nếu có cặp burst bị tách tập hoặc lớp/tập rỗng → **bổ sung dữ liệu/gán group** trước khi tin metric.
7. Huấn luyện: `python -m src.train`; đánh giá: `python -m src.evaluate --split test`.

> Không tự chia lại nếu không cần: script sẽ dùng manifest sẵn có. Chỉ thêm `--force-resplit` khi muốn chia lại từ đầu (ví dụ sau khi bổ sung ảnh mới vào `raw/` hoặc sau khi điền `groups.csv`).

## 7. Các script hỗ trợ

| Lệnh | Chức năng |
|---|---|
| `python scripts/check_dataset.py` | Kiểm tra nhanh dataset (đếm, lỗi, trùng md5, cảnh báo ít ảnh) |
| `python scripts/audit_dataset.py --contact --audit-md reports/metrics/dataset_audit.md` | Audit chuyên sâu: SHA256, dHash, màu HSV trên vùng quả, độ nét, timestamp; ghi manifest CSV/JSON + contact sheet |
| `python scripts/prepare_dataset.py` | Tạo manifest + chia 70/15/15 + **xác minh rò rỉ** (file trùng, group, burst) |
| `python scripts/prepare_dataset.py --force-resplit` | Chia lại từ đầu, bỏ qua manifest cũ |
| `python scripts/prepare_dataset.py --verify-only` | Chỉ xác minh manifest hiện có |
| `python scripts/prepare_dataset.py --burst-groups 10` | Split **thử nghiệm** theo cụm burst (heuristic, chưa xác minh) — ghi manifest riêng, không thay manifest chính |
| `python scripts/check_dataset.py --prepare` | (tương thích) Tạo manifest + chia tập + copy vào train/val/test |
| `python scripts/check_dataset.py --report` | Ghi báo cáo scan ra `reports/metrics/dataset_scan.json` |
| `python scripts/check_dataset.py --groups-template` | Sinh template `groups.csv` (group_id trống) |

> Ghi chú thư mục: `dataset/raw/medium-ripe/` (nếu có) được **ánh xạ tự động** sang lớp chuẩn
> `medium_ripe`; **không đổi tên thư mục raw**. Xem `CLASS_DIR_ALIASES` trong `src/dataset.py`.
