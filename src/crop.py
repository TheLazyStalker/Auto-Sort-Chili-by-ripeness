"""Crop v1: tách quả ớt khỏi nền và xuất ảnh vuông 224×224.

Quy ước quan trọng:
- Ảnh đầu vào là RGB uint8 ``[0, 255]``; KHÔNG rescale, KHÔNG đổi sang ``[0, 1]``.
- Raw luôn giữ nguyên; module này chỉ tạo ảnh derivative trong bộ nhớ.
- Cùng một hàm ``crop_v1`` phải được dùng khi tạo derivative và khi inference.

Thuật toán (``crop_v1``):
1. Ước lượng nền: median dải viền, hoặc ``background_ref`` do người gọi truyền.
2. Mask foreground: khoảng cách Euclid RGB > ``threshold``.
3. Morphology mở/đóng, giữ component lớn nhất, dilation để giữ cuống.
4. Bbox + padding ``pad_frac`` × max(w, h), vuông hoá, clamp/pad bằng màu nền.
5. Resize bilinear về ``output_size`` (mặc định 224).

Trạng thái:
- ``ok``: crop dùng được.
- ``needs_review``: mask quá nhỏ/quá lớn; vẫn xuất ảnh để kiểm tra nhưng KHÔNG
  đưa vào training cho đến khi duyệt.
- ``failed``: không có foreground; inference phải trả ``crop_failed`` và không dự đoán.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

CROP_VERSION = "crop_v1"

DEFAULT_THRESHOLD = 30.0
DEFAULT_PAD_FRAC = 0.20
DEFAULT_OUTPUT_SIZE = 224
DEFAULT_BORDER_FRAC = 0.05
DEFAULT_MIN_AREA_FRAC = 0.005
DEFAULT_MAX_AREA_FRAC = 0.60
DEFAULT_DILATE_PX = 2


@dataclass
class CropResult:
    """Kết quả crop_v1.

    ``image`` là RGB uint8 ``(output_size, output_size, 3)`` hoặc ``None`` nếu failed.
    ``mask`` chỉ dùng cho QA/debug, không serialize vào metadata.
    """

    image: np.ndarray | None
    bbox: tuple[int, int, int, int] | None
    crop_box: tuple[int, int, int, int] | None
    background_rgb: tuple[int, int, int]
    threshold: float
    status: str
    reason: str
    mask_area_frac: float
    pad_frac: float
    output_size: int
    source_size: tuple[int, int]
    mask: np.ndarray | None = field(default=None, repr=False)


def _as_rgb_array(image) -> np.ndarray:
    """Chuyển PIL Image / numpy array về RGB uint8 ``(H, W, 3)`` liên tục."""
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"Ảnh phải có 3 hoặc 4 kênh, nhận được shape {arr.shape}")
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def load_image_rgb(path: str) -> np.ndarray:
    """Đọc ảnh từ đĩa → RGB uint8 ``[0, 255]`` (không đổi giá trị màu)."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def estimate_background(image, border_frac: float = DEFAULT_BORDER_FRAC) -> np.ndarray:
    """Ước lượng màu nền bằng median dải viền ảnh.

    Trả về ``np.ndarray`` float32 shape ``(3,)``. Ảnh cũ có nền đồng nhất
    ≈ RGB(220, 235, 235); deployment nên truyền ``background_ref`` thay vì
    dựa vào viền nếu nền không đồng nhất.
    """
    arr = _as_rgb_array(image)
    h, w = arr.shape[:2]
    band = max(1, int(round(min(h, w) * float(border_frac))))
    parts = [
        arr[:band, :, :].reshape(-1, 3),
        arr[-band:, :, :].reshape(-1, 3),
        arr[:, :band, :].reshape(-1, 3),
        arr[:, -band:, :].reshape(-1, 3),
    ]
    pixels = np.concatenate(parts, axis=0)
    return np.median(pixels, axis=0).astype(np.float32)


def foreground_mask(image, background_rgb, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Mask boolean: pixel có khoảng cách Euclid RGB tới nền > ``threshold``."""
    arr = _as_rgb_array(image)
    bg = np.asarray(background_rgb, dtype=np.float32).reshape(1, 1, 3)
    diff = arr.astype(np.float32) - bg
    distance = np.sqrt(np.sum(diff * diff, axis=-1))
    return distance > float(threshold)


def _apply_filter(mask: np.ndarray, filt) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    return np.asarray(img.filter(filt)) > 127


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Các đoạn ``True`` liên tiếp trong một hàng → [(start, end), ...]."""
    if not row.any():
        return []
    padded = np.concatenate(([False], row, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Giữ component liên thông lớn nhất (run-length + union-find, không cần scipy)."""
    h, _w = mask.shape
    rows = [_row_runs(mask[y]) for y in range(h)]

    parent: list[int] = []
    length: list[int] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
            length[ra] += length[rb]
        else:
            parent[ra] = rb
            length[rb] += length[ra]

    run_ids: list[list[int]] = []
    for runs in rows:
        ids = []
        for start, end in runs:
            rid = len(parent)
            parent.append(rid)
            length.append(end - start)
            ids.append(rid)
        run_ids.append(ids)

    if not parent:
        return np.zeros_like(mask, dtype=bool)

    for y in range(1, h):
        prev, cur = rows[y - 1], rows[y]
        i = j = 0
        while i < len(prev) and j < len(cur):
            s1, e1 = prev[i]
            s2, e2 = cur[j]
            if s1 < e2 and s2 < e1:
                union(run_ids[y - 1][i], run_ids[y][j])
            if e1 <= e2:
                i += 1
            else:
                j += 1

    best_root = max(range(len(parent)), key=lambda i: length[find(i)])
    best_root = find(best_root)

    out = np.zeros_like(mask, dtype=bool)
    for y, runs in enumerate(rows):
        for (start, end), rid in zip(runs, run_ids[y]):
            if find(rid) == best_root:
                out[y, start:end] = True
    return out


def clean_mask(
    mask: np.ndarray,
    dilate_px: int = DEFAULT_DILATE_PX,
    open_size: int = 3,
    close_size: int = 5,
) -> np.ndarray:
    """Mở/đóng morphology → giữ component lớn nhất → dilation.

    Dilation giúp giữ cuống mảnh và tránh cắt sát mép quả. Không dùng scipy để
    giữ đường inference trên Termux nhẹ.
    """
    out = mask.astype(bool)
    if open_size >= 3:
        out = _apply_filter(out, ImageFilter.MinFilter(open_size))
        out = _apply_filter(out, ImageFilter.MaxFilter(open_size))
    if close_size >= 3:
        out = _apply_filter(out, ImageFilter.MaxFilter(close_size))
        out = _apply_filter(out, ImageFilter.MinFilter(close_size))
    out = _largest_component(out)
    if dilate_px > 0:
        size = 2 * int(dilate_px) + 1
        out = _apply_filter(out, ImageFilter.MaxFilter(size))
    return out


def crop_v1(
    image,
    background_ref=None,
    threshold: float = DEFAULT_THRESHOLD,
    pad_frac: float = DEFAULT_PAD_FRAC,
    output_size: int = DEFAULT_OUTPUT_SIZE,
    border_frac: float = DEFAULT_BORDER_FRAC,
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
    max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
    dilate_px: int = DEFAULT_DILATE_PX,
) -> CropResult:
    """Crop một ảnh RGB theo ``crop_v1``.

    ``background_ref``: màu nền (tuple/list/array 3 phần tử) hoặc ``None`` để
    ước lượng từ viền ảnh. Deployment nên truyền frame nền tham chiếu.
    """
    arr = _as_rgb_array(image)
    h, w = arr.shape[:2]
    source_size = (int(w), int(h))

    if background_ref is None:
        bg = estimate_background(arr, border_frac=border_frac)
    else:
        bg = np.asarray(background_ref, dtype=np.float32).reshape(-1)[:3]

    mask = foreground_mask(arr, bg, threshold=threshold)
    mask = clean_mask(mask, dilate_px=dilate_px)
    area_frac = float(mask.mean())

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return CropResult(
            image=None, bbox=None, crop_box=None,
            background_rgb=tuple(int(round(float(v))) for v in bg),
            threshold=float(threshold), status="failed", reason="empty_mask",
            mask_area_frac=0.0, pad_frac=float(pad_frac),
            output_size=int(output_size), source_size=source_size, mask=mask,
        )

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0

    pad = float(pad_frac) * max(bw, bh)
    side = max(1, int(round(max(bw, bh) + 2.0 * pad)))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    sx0 = int(round(cx - side / 2.0))
    sy0 = int(round(cy - side / 2.0))
    sx1 = sx0 + side
    sy1 = sy0 + side

    bg_u8 = np.clip(np.rint(bg), 0, 255).astype(np.uint8)
    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[...] = bg_u8

    ix0, iy0 = max(0, sx0), max(0, sy0)
    ix1, iy1 = min(w, sx1), min(h, sy1)
    if ix1 > ix0 and iy1 > iy0:
        canvas[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = arr[iy0:iy1, ix0:ix1]

    out_img = Image.fromarray(canvas).resize(
        (int(output_size), int(output_size)), resample=Image.Resampling.BILINEAR)
    out = np.asarray(out_img, dtype=np.uint8)

    status, reason = "ok", ""
    if area_frac < float(min_area_frac):
        status, reason = "needs_review", f"mask_too_small:{area_frac:.5f}"
    elif area_frac > float(max_area_frac):
        status, reason = "needs_review", f"mask_too_large:{area_frac:.5f}"

    return CropResult(
        image=out, bbox=(x0, y0, x1, y1), crop_box=(sx0, sy0, sx1, sy1),
        background_rgb=tuple(int(round(float(v))) for v in bg),
        threshold=float(threshold), status=status, reason=reason,
        mask_area_frac=area_frac, pad_frac=float(pad_frac),
        output_size=int(output_size), source_size=source_size, mask=mask,
    )
