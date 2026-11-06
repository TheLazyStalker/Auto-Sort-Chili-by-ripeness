# DATASET_GUIDE — Hướng dẫn thu thập & đặt nhãn dataset ớt

## 1. Ba lớp

| Lớp | Dấu hiệu | Ví dụ màu (tùy giống) |
|---|---|---|
| `unripe` | Chưa chín, vỏ xanh đậm, cứng, bóng mờ | Xanh lá đậm → xanh hơi trắng ở đầu cuống |
| `medium_ripe` | Chín vừa, chuyển màu dở dang, còn vệt xanh xen kẽ chín | Xanh xen đỏ/đỏ cam, mỗi quả khác nhau |
| `ripe` | Đã chín, màu chín đồng đều, vỏ bóng, cuống hơi khô | Đỏ / đỏ sậm / vàng cam tùy giống |

> Ghi chú: màu chín phụ thuộc giống ớt (ớt sừng, ớt chỉ thiên, ớt ngọt…). Hãy chọn **một hoặc ít giống**, chụp trong cùng điều kiện ánh sáng để model học được ranh giới màu, thay vì trộn nhiều giống khác nhau ngay từ đầu.

## 2. Chọn giống & môi trường chụp

- Ưu tiên 1 giống chiếm đa số; nếu nhiều giống, ghi lại giống của mỗi ảnh trong `groups.csv` (hoặc metadata riêng).
- Chụp gần, quả ớt chiếm phần lớn khung hình; nền đơn giản và nhất quán (tránh nền tô màu tương tự vỏ để không rối model).
- Ánh sáng tự nhiên đồng đều (ban ngày, tránh nắng gắt), tránh flash, tránh họng chiếu bóng làm đổi màu.
- Điện thoại ngang hàng quả ớt, khoảng cách ổn định; ưu tiên độ phân giải ≥ 1080p, format JPG.

## 3. Thu thập theo nhóm (quan trọng nhất)

- **Gán group ID cho mỗi quả ớt:** mọi ảnh của cùng 1 quả (chụp góc khác, góc khác, khung hình khác) phải chung `group_id`. File `dataset/groups.csv` có dạng:

  ```csv
  filename,group_id
  0001_unripe_1.jpg,pepper_0001
  0002_unripe_1.jpg,pepper_0002
  ```

- Nếu đặt `groups_file` trong `config.yaml`: `paths.groups_file: dataset/groups.csv`. Dataset loader sẽ chia theo **nhóm** để ảnh cùng 1 quả **không bao giờ** lọt sang nhiều tập — nếu không, model “nhớ mặt” quả ớt và metric trên tập test sẽ ảo (rò rỉ dữ liệu).

## 4. Đặt nhãn nhất quán

- Mỗi quả ớt chỉ thuộc **một nhãn** duy nhất; nếu không chắc chắn, bỏ qua quả đó hoặc hỏi người thứ hai xác minh.
- Phân biệt rõ `medium_ripe` và `ripe`: khi vỏ đã đồng đều màu chín (>90% diện tích) thì là `ripe`; còn xen vệt xanh rõ rệt là `medium_ripe`.
- Kiểm tra lại ngẫu nhiên 10% ảnh sau khi chia tập; ảnh trùng nội dung sẽ được báo trong `reports`/script.

## 5. Quy trình đưa vào dự án

1. Đặt ảnh vào `dataset/raw/unripe/`, `medium_ripe/`, `ripe/` (file `.jpg/.jpeg/.png`).
2. (Tùy chọn) Tạo `dataset/groups.csv` và trỏ `paths.groups_file` trong `config.yaml`.
3. Chạy `python scripts/check_dataset.py` → kiểm tra số lượng, ảnh lỗi, trùng lặp.
4. Chạy `python scripts/check_dataset.py --prepare` → tạo `dataset/split_manifest.json` và copy vào `train/validation/test` (70/15/15, stratified, seed 42). Ảnh gốc **không bị xóa**.
5. Huấn luyện: `python -m src.train`; đánh giá: `python -m src.evaluate --split test`.

> Không tự chia lại nếu không cần: script sẽ dùng manifest sẵn có. Chỉ thêm `--force-resplit` (hoặc `--prepare` sau khi xóa manifest) khi muốn chia lại từ đầu.
