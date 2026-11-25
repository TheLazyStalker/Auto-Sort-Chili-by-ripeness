# Contributing

Thanks for your interest in improving **make-object-to-center**! This project is
open source under the MIT license — anyone is free to fork, modify and
redistribute it.

## Ways to contribute

- Report bugs and edge cases (please attach a sample image if possible).
- Improve segmentation robustness (shadows, very bright backgrounds, pale objects).
- Add features (multi-object support, PNG output, config files, CLI presets).
- Improve documentation and translations.

## Workflow

1. **Fork** the repository and clone your fork.
2. Create a branch:
   ```bash
   git checkout -b feature/my-improvement
   ```
3. Make your changes.
4. Test on a few images with the visualizer:
   ```bash
   python3 crop_chili.py --input ./examples/red-chili/input --output ./out --visualize 12
   ```
5. Commit with a clear message and open a **Pull Request** describing:
   - what you changed,
   - why,
   - which images improved (before/after if possible).

## Where to tune

Most adjustments live in `crop_chili.py`:

- `segment_pepper()` — background/object separation (HSV saturation thresholds,
  "darker than background" logic, morphology).
- `make_crop()` — framing (`--fruit-frac`, `--pad`), centering and compositing.
- `process()` / `main()` — batch I/O.

## Code style

- Python 3.9+, standard library + OpenCV + NumPy only.
- Keep functions small and documented.
- Do not break the existing CLI flags; add new ones as optional.
- Verify that every output file is actually written (check `cv2.imwrite` result).

## License

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).

---

# Đóng góp (Tiếng Việt)

Cảm ơn bạn muốn cải thiện **make-object-to-center**! Dự án **mã nguồn mở (MIT)** —
ai cũng có thể fork, sửa và phân phối lại.

## Các cách đóng góp

- Báo lỗi / trường hợp biên (kèm ảnh mẫu nếu được).
- Cải thiện độ bền khi tách nền (bóng đổ, nền quá sáng, vật thể nhạt màu).
- Thêm tính năng (hỗ trợ nhiều vật thể, xuất PNG, file cấu hình, preset CLI).
- Cải thiện tài liệu và bản dịch.

## Quy trình

1. **Fork** repo và clone bản fork.
2. Tạo nhánh:
   ```bash
   git checkout -b feature/cai-tien-cua-ban
   ```
3. Sửa code.
4. Chạy thử vài ảnh với `--visualize`:
   ```bash
   python3 crop_chili.py --input ./examples/red-chili/input --output ./out --visualize 12
   ```
5. Commit với thông điệp rõ ràng và mở **Pull Request** mô tả:
   - bạn đã thay đổi gì,
   - vì sao,
   - ảnh nào được cải thiện (kèm before/after nếu có).

## Nên tinh chỉnh ở đâu

Phần lớn nằm trong `crop_chili.py`:

- `segment_pepper()` — tách vật thể/nền (ngưỡng HSV saturation, logic "tối hơn nền", morphology).
- `make_crop()` — đóng khung (`--fruit-frac`, `--pad`), canh tâm và ghép nền.
- `process()` / `main()` — xử lý batch và I/O.

## Phong cách code

- Python 3.9+, chỉ dùng thư viện chuẩn + OpenCV + NumPy.
- Hàm nhỏ, có chú thích.
- Không phá vỡ các cờ CLI cũ; cờ mới thêm dạng tùy chọn.
- Kiểm tra file output thật sự được ghi (dùng giá trị trả về của `cv2.imwrite`).

## Giấy phép

Khi đóng góp, bạn đồng ý phần đóng góp của mình theo [giấy phép MIT](LICENSE).
