"""Fast crop_v1 for deployment — thuần NumPy, không PIL rankfilter, không DFS Python.

Tối ưu so với bản cũ (~350-700 ms cho 1024×1024):
1. Tính mask/bbox trên ảnh DOWNSCALE (work_size=256) → rẻ hơn ~16×; bbox sau đó
   map về toạ độ ảnh gốc và crop ở full-res (chất lượng crop không đổi đáng kể).
2. Morphology separable 1D (min/max theo hàng rồi theo cột) thay vì 2D window.
3. Largest component: run-length + union-find (không DFS từng pixel).

Cùng tham số logic: threshold 30, border 5%, pad 20%, dilate 2px, output 224 bilinear.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

INPUT_SIZE = 224
THRESHOLD = 30.0
PAD_FRAC = 0.20
BORDER_FRAC = 0.05
DILATE_PX = 2
WORK_SIZE = 256  # kích thước ảnh làm việc để tính mask/bbox


def estimate_background(arr, border_frac=BORDER_FRAC):
    h, w = arr.shape[:2]
    band = max(1, int(round(min(h, w) * border_frac)))
    parts = [arr[:band].reshape(-1, 3), arr[-band:].reshape(-1, 3),
             arr[:, :band].reshape(-1, 3), arr[:, -band:].reshape(-1, 3)]
    return np.median(np.concatenate(parts, axis=0), axis=0).astype(np.float32)


def _minmax_1d(m, k, is_max):
    """Min/max filter 1D dọc trục 1 (separable). m: 2D bool → bool."""
    pad = k // 2
    if is_max:
        p = np.pad(m.astype(np.uint8), ((0, 0), (pad, pad)), constant_values=0)
        v = np.lib.stride_tricks.sliding_window_view(p, k, axis=1)
        r = v.max(axis=-1)
    else:
        p = np.pad(m.astype(np.uint8), ((0, 0), (pad, pad)), constant_values=1)
        v = np.lib.stride_tricks.sliding_window_view(p, k, axis=1)
        r = v.min(axis=-1)
    return r > 0


def _morph(mask, k, is_max):
    """Separable min/max 2D: theo hàng rồi theo cột."""
    r = _minmax_1d(mask, k, is_max)
    r = _minmax_1d(r.T, k, is_max).T
    return r


def _erode(mask, k):
    return _morph(mask, k, is_max=False)


def _dilate(mask, k):
    return _morph(mask, k, is_max=True)


def _row_runs(row):
    if not row.any():
        return []
    padded = np.concatenate(([False], row, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _largest_component(mask):
    """Run-length + union-find (không cần scipy, không DFS từng pixel)."""
    h = mask.shape[0]
    rows = [_row_runs(mask[y]) for y in range(h)]
    parent, length = [], []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
            length[ra] += length[rb]
        else:
            parent[ra] = rb
            length[rb] += length[ra]

    run_ids = []
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

    best_root = find(max(range(len(parent)), key=lambda i: length[find(i)]))
    out = np.zeros_like(mask, dtype=bool)
    for y, runs in enumerate(rows):
        for (start, end), rid in zip(runs, run_ids[y]):
            if find(rid) == best_root:
                out[y, start:end] = True
    return out


def clean_mask(mask, dilate_px=DILATE_PX):
    """Mở 3×3 → đóng 5×5 → largest component → dilation (2·dilate+1)."""
    out = mask.astype(bool)
    out = _dilate(_erode(out, 3), 3)      # open
    out = _erode(_dilate(out, 5), 5)      # close
    out = _largest_component(out)
    if dilate_px > 0:
        out = _dilate(out, 2 * dilate_px + 1)
    return out


def foreground_bbox(arr, work_size=WORK_SIZE):
    """Tính bbox foreground (toạ độ ảnh gốc) bằng cách xử lý trên ảnh downscale.

    Trả về (x0, y0, x1, y1, area_frac_work) hoặc None nếu mask rỗng.
    """
    h, w = arr.shape[:2]
    if max(h, w) > work_size:
        scale = work_size / max(h, w)
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = np.asarray(Image.fromarray(arr).resize((sw, sh), Image.Resampling.BILINEAR),
                           dtype=np.uint8)
    else:
        small = arr
        sw, sh = w, h
    bg = estimate_background(small)
    diff = small.astype(np.float32) - bg.reshape(1, 1, 3)
    mask = np.sqrt(np.sum(diff * diff, axis=-1)) > THRESHOLD
    mask = clean_mask(mask)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    fx = w / sw
    fy = h / sh
    x0 = int(np.floor(xs.min() * fx))
    x1 = int(np.ceil((xs.max() + 1) * fx))
    y0 = int(np.floor(ys.min() * fy))
    y1 = int(np.ceil((ys.max() + 1) * fy))
    return x0, y0, x1, y1, float(mask.mean())


def crop_v1(arr, work_size=WORK_SIZE):
    """Crop ảnh RGB → 224×224 uint8 (cùng logic crop_v1, nhanh hơn nhiều)."""
    h, w = arr.shape[:2]
    bbox = foreground_bbox(arr, work_size=work_size)
    if bbox is None:
        raise ValueError("empty_mask: không tìm thấy foreground (ảnh nền trơn?)")
    x0, y0, x1, y1, _area = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad = PAD_FRAC * max(bw, bh)
    side = max(1, int(round(max(bw, bh) + 2.0 * pad)))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sx0, sy0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    bg = estimate_background(arr)
    bg_u8 = np.clip(np.rint(bg), 0, 255).astype(np.uint8)
    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[...] = bg_u8
    ix0, iy0 = max(0, sx0), max(0, sy0)
    ix1, iy1 = min(w, sx0 + side), min(h, sy0 + side)
    if ix1 > ix0 and iy1 > iy0:
        canvas[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = arr[iy0:iy1, ix0:ix1]
    img = Image.fromarray(canvas).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)
