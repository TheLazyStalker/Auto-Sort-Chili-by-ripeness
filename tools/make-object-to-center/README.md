# make-object-to-center

**English** · [Tiếng Việt](#tiếng-việt)

Batch-crop product photos (chili peppers) so the object is **perfectly centered** and placed on a **uniform background**. The tool automatically removes the original background (shadows, vignetting, under-lit corners) and re-composites the object on a flat color.

![before / after](examples/red-chili/preview.jpg)

> Works well for colorful objects on a light/gray background. Robust to soft shadows, gradients and uneven lighting.

**Open source (MIT).** Anyone is free to fork, modify, use and redistribute this project. See [Contributing](#contributing) and [LICENSE](LICENSE).

---

## Features

- **Robust background removal** using HSV saturation + "darker than background" logic — handles vignetting, gradients, and under-lit areas.
- **Centers on the fruit body** (center of mass), not the stem.
- **Stem always stays inside the frame** — the crop expands toward the stem automatically.
- **Uniform background** (configurable color; default white).
- **Square output**, default `1024×1024`.
- **Fast batch processing** (~0.4 s per 4K image, ~200 images/minute).
- Safe fallback if segmentation fails, and it verifies that every file is actually written.
- Preserves original filenames.

---

## Requirements

- Python 3.9+
- OpenCV + NumPy

```bash
python3 -m pip install -r requirements.txt
```

`requirements.txt`:

```
opencv-python-headless>=4.8
numpy>=1.24
```

> On some systems you may need `--user --break-system-packages`:
> `python3 -m pip install --user --break-system-packages -r requirements.txt`

---

## Quick start

```bash
# Crop a folder of images
python3 crop_chili.py --input ./examples/red-chili/input --output ./out

# Preview first 12 images as side-by-side comparisons (original | mask | result)
python3 crop_chili.py --input ./examples/red-chili/input --output ./out --visualize 12
```

Results are written to `./out`, keeping the original file names.

---

## Options

| Option | Default | Description |
|---|---|---|
| `--input PATH` | *(required)* | Folder with the original images |
| `--output PATH` | *(required)* | Folder for the cropped results |
| `--size N` | `1024` | Side length of the square output (px) |
| `--fruit-frac F` | `0.5` | Max fraction of the frame the fruit body may occupy (`0.5` = 50%) |
| `--pad P` | `0.12` | Safety margin around the whole object + stem |
| `--bg B,G,R` | `255,255,255` | Output background color in **BGR** order |
| `--visualize N` | `0` | Write comparison images for the first `N` files |
| `--ext .jpg` | `.jpg` | Input file extension to scan |
| `--suffix STR` | `_cropped` | Suffix added to output names (use `--suffix ""` to keep names) |

### Examples

```bash
# Light gray background  (RGB 240,240,240  ->  BGR 240,240,240)
python3 crop_chili.py -i in -o out --bg 240,240,240

# Light cyan background  (RGB 220,235,235  ->  BGR 235,235,220)
python3 crop_chili.py -i in -o out --bg 235,235,220

# Larger object in frame
python3 crop_chili.py -i in -o out --fruit-frac 0.6

# Keep original filenames
python3 crop_chili.py -i in -o out --suffix ""
```

> **Color order:** `--bg` expects **B,G,R** (OpenCV order). For RGB `(220, 235, 235)` pass `--bg 235,235,220`.

---

## How it works

1. **Detect the fruit body** — the largest high-saturation region (HSV `S`), which is stable even when the object is under-lit. A small morphological opening removes the thin stem from the body mask.
2. **Compute the center** — the center of mass of the fruit body becomes the center of the output frame.
3. **Detect the stem** — pixels that are both *colored* and *darker than the background*, connected to the fruit (shadows and background cast are excluded).
4. **Size the frame** —
   `side = max( fruit_dim / fruit_frac , 2 · R · (1 + pad) )`
   where `R` is the farthest distance from the fruit center to any object pixel. This guarantees the stem stays inside.
5. **Composite** — the object is pasted onto a flat background color with a feathered alpha edge.

Because the stem must always fit, an object with a very long stem will occupy **less than** `fruit-frac` of the frame (the value is a maximum, not a fixed size).

---

## Example dataset

`examples/red-chili/` contains 12 representative red-chili photos:

```
examples/red-chili/
├── input/     # 12 original 4K photos
├── output/    # 12 cropped 1024×1024 results
└── preview.jpg
```

The sample outputs use a light cyan background `RGB(220,235,235)`.

---

## Tips & troubleshooting

- **Object touches the edge / too tight** → increase `--pad` (e.g. `0.15`) or decrease `--fruit-frac`.
- **Object too small** → increase `--fruit-frac` (e.g. `0.6`); note the stem may get closer to the edge.
- **Halo / gray patches included** → the object likely has a colored background cast; try a different background or tune thresholds in `segment_pepper()`.
- **Only one object per image** — the tool keeps the largest object. Images with multiple objects are not supported out of the box.
- **Exact background color** — JPEG is lossy, so the background may differ by ±1. Use PNG output or JPEG quality 100 if you need exact pixels.

---

## Limitations & known issues

This project is a **proof of concept** and has **only been validated on chili peppers** so far.

- **Tested on chili peppers only.** Other fruits/objects may require tuning the thresholds in `segment_pepper()`.
- **Shadows are not guaranteed to be preserved.** The tool targets the object; soft shadows are usually removed (and may occasionally be partially included).
- **Severe failures: ~2–3%** (≈ 5/203 images) — mostly fruits photographed on a **very bright/light background**, where object/background separation is weak.
- **Minor issues: ~7–8%** — small artifacts such as a thin bright line along the stem, a slightly clipped stem tip, or a small halo.
- **One object per image** — multiple objects are not supported out of the box.

These numbers come from a real batch (~200 images per color). Contributions that improve robustness are very welcome.

---

## Contributing

This is an open-source project — **anyone is welcome to fork, modify and improve it** under the MIT license.

1. Fork the repo and create a branch: `git checkout -b feature/my-improvement`
2. Make your changes. Most tuning happens in `segment_pepper()` and `make_crop()` in `crop_chili.py`.
3. Test on a few images with `--visualize`.
4. Open a Pull Request describing what you changed and which images improved.

Ideas for contributions: better shadow handling, multi-object support, alternative color spaces, PNG output, a config file / CLI presets.

---

## Repo structure

```
make-object-to-center/
├── crop_chili.py
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── examples/
    └── red-chili/
        ├── input/
        ├── output/
        └── preview.jpg
```

---

## English quick start

```bash
pip install -r requirements.txt
python3 crop_chili.py --input ./examples/red-chili/input --output ./out
```

Centers the object on its body (not the stem), keeps the stem inside the frame, and pastes it on a uniform background (`--bg B,G,R`, default white). Output is `1024×1024` by default.

---

## License

MIT — see [LICENSE](LICENSE).

---

# Tiếng Việt

> Bản dịch tiếng Việt của hướng dẫn trên. Dự án **mã nguồn mở (MIT)** — ai cũng có thể fork, sửa và đóng góp.

## Giới thiệu

Công cụ crop hàng loạt ảnh sản phẩm (ớt) sao cho vật thể **nằm đúng tâm** và được đặt trên **nền đồng nhất**. Tool tự tách nền cũ (bóng đổ, gradient, góc thiếu sáng) rồi ghép vật thể lên một màu nền phẳng.

![trước / sau](examples/red-chili/preview.jpg)

## Tính năng

- Tách nền bền vững bằng HSV Saturation + logic "tối hơn nền" — xử lý được gradient, thiếu sáng, bóng đổ mềm.
- **Canh tâm theo thân quả** (trọng tâm), không phải cuống.
- **Cuống luôn nằm trong khung** — khung tự nới về phía cuống.
- **Nền đồng nhất** (đổi màu tùy ý; mặc định trắng).
- Ảnh vuông, mặc định `1024×1024`.
- Chạy batch nhanh (~0.4 s/ảnh 4K, ~200 ảnh/phút).
- Có fallback an toàn và kiểm tra ghi file thật sự.
- Giữ nguyên tên file gốc.

## Cài đặt

```bash
python3 -m pip install -r requirements.txt
```

Trên một số hệ thống cần thêm `--user --break-system-packages`:

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

## Dùng nhanh

```bash
python3 crop_chili.py --input ./examples/red-chili/input --output ./out

# Xem thử 12 ảnh đầu (gốc | mask | kết quả)
python3 crop_chili.py --input ./examples/red-chili/input --output ./out --visualize 12
```

## Tùy chọn

| Tùy chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--input PATH` | *(bắt buộc)* | Thư mục ảnh gốc |
| `--output PATH` | *(bắt buộc)* | Thư mục xuất kết quả |
| `--size N` | `1024` | Cạnh ảnh vuông output |
| `--fruit-frac F` | `0.5` | Thân quả chiếm tối đa bao nhiêu khung (`0.5` = 50%) |
| `--pad P` | `0.12` | Biên an toàn quanh toàn bộ ớt + cuống |
| `--bg B,G,R` | `255,255,255` | Màu nền output, thứ tự **BGR** |
| `--visualize N` | `0` | Xuất ảnh so sánh cho N file đầu |
| `--ext .jpg` | `.jpg` | Đuôi file cần quét |
| `--suffix STR` | `_cropped` | Hậu tố tên file (dùng `--suffix ""` để giữ nguyên tên) |

## Ví dụ

```bash
# Nền xám nhạt (RGB 240,240,240)
python3 crop_chili.py -i in -o out --bg 240,240,240

# Nền xanh ngọc nhạt (RGB 220,235,235 -> BGR 235,235,220)
python3 crop_chili.py -i in -o out --bg 235,235,220

# Ớt to hơn trong khung
python3 crop_chili.py -i in -o out --fruit-frac 0.6

# Giữ nguyên tên file
python3 crop_chili.py -i in -o out --suffix ""
```

## Nguyên lý hoạt động

1. **Tách thân quả** — vùng bão hòa màu cao nhất (HSV `S`), ổn định cả khi thiếu sáng; phép mở morphology bỏ cuống mảnh.
2. **Lấy trọng tâm thân quả** làm tâm khung.
3. **Tách cuống** — pixel vừa có màu vừa tối hơn nền, nối liền thân quả (loại bóng xám/quầng nền).
4. **Tính cạnh khung** — `side = max(fruit_dim / fruit_frac, 2·R·(1+pad))`, với `R` là khoảng cách xa nhất từ tâm quả tới mọi điểm ớt+cuống ⇒ cuống luôn nằm trong.
5. **Ghép** vật thể lên nền phẳng một màu, viền mềm (feather).

> Vì cuống luôn phải nằm trong khung nên quả có cuống rất dài sẽ chiếm **dưới** `fruit-frac` (giá trị này là mức tối đa, không cố định).

## Hạn chế đã biết

Đây là bản thử nghiệm, **mới chỉ kiểm chứng trên quả ớt**:

- **Chỉ mới thử nghiệm với quả ớt.** Trái cây/vật thể khác có thể cần chỉnh ngưỡng trong `segment_pepper()`.
- **Bóng đổ không đảm bảo được giữ lại.** Tool nhắm vào vật thể; bóng mềm thường bị loại (đôi khi có thể bị tính nhầm vào vật thể).
- **Lỗi nặng ~2–3%** (≈ 5/203 ảnh) — chủ yếu là trái cây có **màu nền quá sáng**, tách vật thể/nền kém.
- **Lỗi nhẹ ~7–8%** — vài lỗi nhỏ như vệt sáng mảnh dọc cuống, đầu cuống bị cắt nhẹ, hoặc quầng nhỏ.
- **Mỗi ảnh một vật thể** — chưa hỗ trợ nhiều vật thể trong cùng một ảnh.

Các con số trên đến từ một batch thực tế (~200 ảnh/màu). Rất hoan nghênh đóng góp để cải thiện.

## Đóng góp / Mã nguồn mở

Dự án **mã nguồn mở theo giấy phép MIT** — ai cũng có thể fork, sửa, dùng cho mục đích riêng và phân phối lại.

1. Fork repo và tạo nhánh: `git checkout -b feature/cai-tien-cua-ban`
2. Sửa code (phần lớn việc tinh chỉnh nằm ở `segment_pepper()` và `make_crop()` trong `crop_chili.py`).
3. Chạy thử vài ảnh với `--visualize`.
4. Mở Pull Request mô tả thay đổi và ảnh nào được cải thiện.

Ý tưởng đóng góp: xử lý bóng đổ tốt hơn, hỗ trợ nhiều vật thể, thêm đầu ra PNG, file cấu hình/CLI preset.

## Giấy phép

MIT — xem [LICENSE](LICENSE).
