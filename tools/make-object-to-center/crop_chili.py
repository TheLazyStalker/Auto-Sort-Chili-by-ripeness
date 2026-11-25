#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop hàng loạt ảnh ớt (nền sáng) về đúng tâm, nền đồng nhất.

- Trọng tâm khung = TRỌNG TÂM THÂN QUẢ (không phải cuống).
- Thân quả chiếm ~50% khung (tùy chỉnh --fruit-frac).
- Khung đủ rộng để CUỐNG LUÔN NẰM TRONG ảnh.
- Thay nền cũ bằng MỘT MÀU ĐỒNG NHẤT (khử cả vùng thiếu sáng/gradient).

Cách dùng:
    python3 crop_chili.py --input <thư_mục_ảnh> --output <thư_mục_ra>

Tùy chọn chính:
    --size 1024        Cạnh ảnh vuông output (px)
    --fruit-frac 0.5   Tỉ lệ thân quả chiếm trong khung (0.5 = 50%)
    --pad 0.12         Biên an toàn quanh toàn bộ ớt+cuống (phân số)
    --bg 255,255,255   Màu nền output (B,G,R) — mặc định trắng
    --visualize N      Xuất ảnh so sánh (gốc | mask | kết quả) cho N tấm đầu
    --ext .jpg         Đuôi file cần quét
    --suffix _cropped  Hậu tố tên file output
"""
import argparse
import math
import os
import sys
import numpy as np
import cv2


# ----------------------------------------------------------------------------
# Tách nền
# ----------------------------------------------------------------------------
def _bg_chroma_p99(ch):
    """Ước lượng mức chroma nền từ dải viền ảnh (percentile 99)."""
    H, W = ch.shape[:2]
    band = max(2, int(min(H, W) * 0.02))
    border = np.concatenate([
        ch[:band, :].ravel(), ch[-band:, :].ravel(),
        ch[:, :band].ravel(), ch[:, -band:].ravel(),
    ])
    return float(np.percentile(border, 99))


def segment_pepper(bgr, work=1200):
    """Tách ớt: thân quả theo HSV Saturation, cuống theo 'có màu & tối hơn nền'.

    Trả về (fg_full, fcx, fcy, fruit_dim) hoặc None nếu thất bại:
      fg_full   : mask nhị phân (0/255) của TOÀN BỘ ớt + cuống, kích thước gốc.
      fcx, fcy  : trọng tâm THÂN QUẢ (toạ độ ảnh gốc).
      fruit_dim : cạnh lớn của bounding box thân quả (px ảnh gốc).
    """
    h, w = bgr.shape[:2]
    scale = work / max(h, w)
    if scale < 1.0:
        small = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = bgr

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Ước lượng nền từ dải viền
    Hh, Ww = S.shape[:2]
    band = max(2, int(min(Hh, Ww) * 0.02))
    def border_stats(chan):
        v = np.concatenate([
            chan[:band, :].ravel(), chan[-band:, :].ravel(),
            chan[:, :band].ravel(), chan[:, -band:].ravel(),
        ])
        return float(np.median(v)), float(np.percentile(v, 99))
    bgS_med, bgS_p99 = border_stats(S)
    bgL_med, _ = border_stats(L)

    thr_high = max(95.0, bgS_p99 + 8.0)   # thân quả: bão hòa cao
    thr_low = max(55.0, bgS_med + 20.0)   # cuống: bão hòa vừa
    dark = L < (bgL_med - 12.0)           # tối hơn nền rõ rệt

    strong = (S > thr_high).astype(np.uint8)
    weak = (((S > thr_low) & dark) | (S > thr_high)).astype(np.uint8)

    # --- Thân quả: thành phần lớn nhất của "strong", mở để bỏ cuống mảnh
    n, lab_s, stats_s, _ = cv2.connectedComponentsWithStats(strong, 8)
    if n <= 1:
        return None
    areas = stats_s[1:, cv2.CC_STAT_AREA]
    fi = int(np.argmax(areas)) + 1
    fruit = (lab_s == fi).astype(np.uint8)

    fx, fy, fw, fh, _ = stats_s[fi]
    ksize = max(3, int(min(fw, fh) * 0.15))
    if ksize % 2 == 0:
        ksize += 1
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    fruit = cv2.morphologyEx(fruit, cv2.MORPH_OPEN, k_open)
    if cv2.countNonZero(fruit) == 0:
        fruit = (lab_s == fi).astype(np.uint8)

    # --- Toàn bộ ớt+cuống: thành phần "weak" chứa thân quả
    nw, lab_w, _, _ = cv2.connectedComponentsWithStats(weak, 8)
    labs = lab_w[fruit > 0]
    if labs.size > 0:
        counts = np.bincount(labs)
        counts[0] = 0
        best = int(np.argmax(counts))
        fg = (lab_w == best).astype(np.uint8)
    else:
        fg = fruit.copy()

    # lấp lỗ nhỏ (lóa sáng trên vỏ)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3)
    # lấp mọi lỗ bên trong bằng contour ngoài
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    fg = filled

    # --- Scale mask lên kích thước gốc
    if scale < 1.0:
        fg_full = cv2.resize(fg * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        fruit_full = cv2.resize(fruit * 255, (w, h),
                                interpolation=cv2.INTER_NEAREST)
    else:
        fg_full = fg * 255
        fruit_full = fruit * 255

    # --- Trọng tâm thân quả (toạ độ gốc) từ moment
    M = cv2.moments(fruit_full, True)
    if M["m00"] > 0:
        fcx = M["m10"] / M["m00"]
        fcy = M["m01"] / M["m00"]
    else:
        fcx = (fx + fw / 2.0) / scale
        fcy = (fy + fh / 2.0) / scale

    fruit_dim = max(fw, fh) / scale
    return fg_full, fcx, fcy, fruit_dim


# ----------------------------------------------------------------------------
# Cắt + ghép nền đồng nhất
# ----------------------------------------------------------------------------
def _crop_pad(img, x0, y0, side, border):
    H, W = img.shape[:2]
    pt = max(0, -y0)
    pb = max(0, (y0 + side) - H)
    pl = max(0, -x0)
    pr = max(0, (x0 + side) - W)
    if pt or pb or pl or pr:
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, border)
        x0 += pl
        y0 += pt
    return img[y0:y0 + side, x0:x0 + side]


def make_crop(bgr, fg_full, fcx, fcy, fruit_dim, args):
    H, W = bgr.shape[:2]
    bg = np.array(args.bg, np.float32)

    # Bán kính xa nhất từ trọng tâm quả tới mọi điểm ớt+cuống
    ys, xs = np.where(fg_full > 0)
    if xs.size == 0:
        return None
    d2 = (xs - fcx) ** 2 + (ys - fcy) ** 2
    R = math.sqrt(float(d2.max()))

    side_fruit = fruit_dim / max(0.05, args.fruit_frac)   # quả chiếm ~fruit_frac
    side_cover = 2.0 * R * (1.0 + args.pad)               # đủ chứa cả cuống
    side = int(math.ceil(max(side_fruit, side_cover, 16)))

    x0 = int(round(fcx - side / 2.0))
    y0 = int(round(fcy - side / 2.0))

    crop = _crop_pad(bgr, x0, y0, side, cv2.BORDER_REPLICATE)
    alpha = _crop_pad(fg_full, x0, y0, side, cv2.BORDER_CONSTANT)
    alpha = cv2.GaussianBlur(alpha, (0, 0),
                             sigmaX=max(1.0, side * 0.0015))
    a3 = (alpha.astype(np.float32) / 255.0)[..., None]

    out = bg * (1.0 - a3) + crop.astype(np.float32) * a3
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.resize(out, (args.size, args.size),
                      interpolation=cv2.INTER_AREA)


def fallback_crop(bgr, args):
    H, W = bgr.shape[:2]
    side = int(min(H, W) * 0.9)
    cy, cx = H // 2, W // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    crop = bgr[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (args.size, args.size),
                      interpolation=cv2.INTER_AREA)


def make_visualize(orig, fg, result):
    h = 220
    def fit(im):
        s = h / im.shape[0]
        return cv2.resize(im, (max(1, int(im.shape[1] * s)), h),
                          interpolation=cv2.INTER_AREA)
    a = fit(orig)
    b = fit(cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR))
    c = fit(result)
    gap = 8
    canvas = np.full((h, a.shape[1] + b.shape[1] + c.shape[1] + gap * 2, 3),
                     180, np.uint8)
    canvas[:, :a.shape[1]] = a
    canvas[:, a.shape[1] + gap:a.shape[1] + gap + b.shape[1]] = b
    canvas[:, a.shape[1] + b.shape[1] + gap * 2:] = c
    return canvas


# ----------------------------------------------------------------------------
def parse_bg(s):
    parts = [int(x) for x in s.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--bg phải là 'B,G,R'")
    return tuple(parts)


def process(path, args):
    img = cv2.imread(path)
    if img is None:
        return None, "không đọc được file"
    seg = segment_pepper(img)
    if seg is None:
        return fallback_crop(img, args), (None, "fallback: không tách được nền")
    fg_full, fcx, fcy, fruit_dim = seg
    result = make_crop(img, fg_full, fcx, fcy, fruit_dim, args)
    if result is None:
        return fallback_crop(img, args), (None, "fallback: mask rỗng")
    vis = make_visualize(img, fg_full, result) if args.visualize else None
    return result, (vis, None)


def main():
    ap = argparse.ArgumentParser(description="Crop hàng loạt ảnh ớt")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--fruit-frac", type=float, default=0.5,
                    help="Tỉ lệ thân quả trong khung (0.5 = 50%%)")
    ap.add_argument("--pad", type=float, default=0.12,
                    help="Biên an toàn quanh toàn bộ ớt+cuống")
    ap.add_argument("--bg", type=parse_bg, default=(255, 255, 255),
                    help="Màu nền output 'B,G,R' (mặc định 255,255,255)")
    ap.add_argument("--visualize", type=int, default=0)
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--suffix", default="_cropped")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        sys.exit(f"Không thấy thư mục input: {args.input}")
    os.makedirs(args.output, exist_ok=True)

    exts = (args.ext.lower(), args.ext.upper())
    files = sorted(f for f in os.listdir(args.input)
                   if f.lower().endswith(exts) and not f.startswith('.'))
    if not files:
        sys.exit(f"Không có file {args.ext} trong {args.input}")

    vis_dir = os.path.join(args.output, "visualize") if args.visualize else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    print(f"Tìm thấy {len(files)} ảnh. Đang xử lý...")
    ok = warn = 0
    failed = []
    for i, fname in enumerate(files, 1):
        result, meta = process(os.path.join(args.input, fname), args)
        if result is None:
            failed.append(fname)
            print(f"[{i}/{len(files)}] LỖI {fname}: {meta}")
            continue
        base, _ = os.path.splitext(fname)
        dst = os.path.join(args.output, base + args.suffix + ".jpg")
        wrote = cv2.imwrite(dst, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not wrote:
            failed.append(fname)
            print(f"[{i}/{len(files)}] LỖI GHI {fname}")
            continue
        ok += 1
        vis, warning = meta
        if warning:
            warn += 1
            print(f"[{i}/{len(files)}] cảnh báo {fname}: {warning}")
        if vis is not None:
            cv2.imwrite(os.path.join(vis_dir, base + "_vis.jpg"), vis)
        if i % 50 == 0 or i == len(files):
            print(f"  ... {i}/{len(files)}")

    print("\n=== TỔNG KẾT ===")
    print(f"Thành công: {ok}/{len(files)}")
    print(f"Fallback: {warn}")
    if failed:
        print(f"Lỗi ({len(failed)}): {', '.join(failed[:20])}")
    print(f"Kết quả: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
