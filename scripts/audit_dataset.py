"""Script audit dữ liệu ảnh thật (Task 2).

Chức năng:
  - Quét toàn bộ dataset/raw/<class>/ (JPG/JPEG/PNG).
  - Tính SHA256, kích thước file, độ phân giải, mode ảnh.
  - Kiểm tra ảnh lỗi/không đọc được/quá nhỏ.
  - Phát hiện ảnh trùng nội dung (md5) và gần trùng (perceptual dHash).
  - Thống kê màu theo không gian HSV theo từng lớp (trung bình, percentile).
  - Phân tích timestamp (chuỗi chụp burst, khoảng nghỉ lớn).
  - Ghi manifest đầy đủ (đường dẫn, nhãn chuẩn, SHA256, size, resolution, group).
  - Ghi báo cáo JSON + in tóm tắt; tạo contact sheet nội bộ (không upload).

Dùng (chạy trong container ai-workspace, venv đã kích hoạt):
  python scripts/audit_dataset.py [--config config.yaml] [--manifest dataset/manifest.json]
  python scripts/audit_dataset.py --contact  # tạo contact sheet theo lớp

Không tải ảnh công khai; không tạo/sửa/xóa ảnh raw; không train.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.dataset import SUPPORTED_EXTENSIONS  # noqa: E402

MIN_RESOLUTION_MIN = 32          # ảnh nhỏ hơn cả hai chiều → cảnh báo quá nhỏ
# dHash 64-bit. Dataset này có nền xám đồng nhất chiếm phần lớn khung hình
# → dHash của các ảnh cùng lớp rất gần nhau (median ~9/64).
# Ngưỡng chỉ nên bắt các cặp RẤT gần (nguy cơ trùng lặp thực sự) chứ không
# phải mọi cặp giống nhau do nền. Dùng ngưỡng chặt 6 (khoảng 10% cặp).
PERCEPTUAL_HAMMING_THRESHOLD = 6


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_dhash(img, hash_size: int = 8) -> int:
    """dHash 64-bit: so sánh độ sáng pixel kề nhau theo hàng.

    Trả về int 64 bit. Không cần imagehash/cv2 — dùng Pillow + numpy.
    """
    import numpy as np
    from PIL import Image

    gray = img.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    # arr[i, j] < arr[i, j+1] → bit 1
    diff = arr[:, 1:] >= arr[:, :-1]
    bits = diff.ravel()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def load_image_safe(path: str):
    """Trả về (Image, error) — không raise ngoại lệ."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.verify()
        im = Image.open(path)
        return im, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def parse_timestamp(fname: str):
    m = re.match(r"IMG_(\d{8})_(\d{6})", fname)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return None
    return None


def rgb_to_hsv(rgb):
    import numpy as np

    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 1e-6
    h[mask & (mx == r)] = 60 * (((g - b) / diff)[mask & (mx == r)] % 6)
    h[mask & (mx == g)] = 60 * (((b - r) / diff + 2)[mask & (mx == g)])
    h[mask & (mx == b)] = 60 * (((r - g) / diff + 4)[mask & (mx == b)])
    h = (h + 360.0) % 360.0
    s = np.where(mx > 0, diff / np.maximum(mx, 1e-6), 0.0)
    v = mx
    return h, s, v


def analyze_image(img, sample: int = 20_000):
    """Thống kê màu (chỉ vùng quả) + tỷ lệ diện tích quả + độ nét.

    Nền được ước lượng từ 4 góc; pixel khác nền >45 (RGB Euclid) là vùng quả.
    Không dùng ngưỡng màu để phân lớp — chỉ mô tả dữ liệu.
    """
    import numpy as np

    im = img.convert("RGB")
    arr = np.asarray(im, dtype=np.float32)
    h, w, _ = arr.shape
    k = 40
    corners = [arr[ys:ys + k, xs:xs + k].reshape(-1, 3)
               for ys, xs in [(0, 0), (0, w - k), (h - k, 0), (h - k, w - k)]]
    bg = np.median(np.concatenate(corners), axis=0)
    dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
    mask = dist > 45.0
    fg_frac = float(mask.mean())
    ys, xs = np.where(mask)
    if len(ys) > 0:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        bbox_frac = ((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)) / (w * h)
    else:
        bbox, bbox_frac = None, 0.0

    # độ nét: phương sai Laplacian trên vùng quả
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    lap = (4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
           - gray[1:-1, :-2] - gray[1:-1, 2:])
    m = mask[1:-1, 1:-1]
    sharpness = float(lap[m].var()) if m.sum() > 0 else float(lap.var())

    # màu HSV chỉ trên vùng quả (nếu quá ít pixel thì dùng toàn ảnh)
    px = arr[mask] if mask.sum() >= 50 else arr.reshape(-1, 3)
    if px.shape[0] > sample:
        idx = np.linspace(0, px.shape[0] - 1, sample).astype(int)
        px = px[idx]
    px = px / 255.0
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mx = np.max(px, axis=1)
    mn = np.min(px, axis=1)
    diff = mx - mn
    hue = np.zeros_like(mx)
    msk = diff > 1e-6
    i = msk & (mx == r)
    hue[i] = 60 * (((g - b) / np.maximum(diff, 1e-6))[i] % 6)
    i = msk & (mx == g)
    hue[i] = 60 * (((b - r) / np.maximum(diff, 1e-6) + 2)[i])
    i = msk & (mx == b)
    hue[i] = 60 * (((r - g) / np.maximum(diff, 1e-6) + 4)[i])
    hue = (hue + 360.0) % 360.0
    sat = np.where(mx > 0, diff / np.maximum(mx, 1e-6), 0.0)
    val = mx
    ang = np.deg2rad(hue)
    hue_circ = float(np.rad2deg(np.arctan2(np.sin(ang).mean(),
                                           np.cos(ang).mean())) % 360.0)
    strong = (sat > 0.25) & (val > 0.15)
    return {
        "bg_rgb": [round(float(x), 1) for x in bg],
        "fg_frac": round(fg_frac, 4),
        "bbox_frac": round(bbox_frac, 4),
        "bbox": bbox,
        "sharpness": round(sharpness, 1),
        "hue_median": round(hue_circ, 1),
        "hue_mean": round(float(np.mean(hue)), 1),
        "sat_mean": round(float(np.mean(sat[strong])) if strong.any() else 0.0, 4),
        "sat_median": round(float(np.median(sat[strong])) if strong.any() else 0.0, 4),
        "val_mean": round(float(np.mean(val[strong])) if strong.any() else 0.0, 4),
        "val_median": round(float(np.median(val[strong])) if strong.any() else 0.0, 4),
        "greenish_frac": round(float(np.mean((hue >= 60) & (hue <= 180) & strong)), 4),
        "orange_frac": round(float(np.mean((hue >= 20) & (hue < 60) & strong)), 4),
        "red_frac": round(float(np.mean(((hue < 20) | (hue >= 330)) & strong)), 4),
        "yellow_frac": round(float(np.mean((hue >= 40) & (hue < 70) & strong)), 4),
        "samples": int(px.shape[0]),
    }


def scan_audit(raw_root: str, classes: list) -> dict:
    from PIL import Image

    records = []          # mỗi ảnh
    corrupt = []
    tiny = []
    errors = []
    md5_map = defaultdict(list)
    class_records = defaultdict(list)

    for cls in classes:
        class_dir = os.path.join(raw_root, cls)
        if not os.path.isdir(class_dir):
            continue
        fnames = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith(SUPPORTED_EXTENSIONS) and os.path.isfile(os.path.join(class_dir, f))
        )
        for fname in fnames:
            path = os.path.join(class_dir, fname)
            img, err = load_image_safe(path)
            rec = {
                "path": path,
                "filename": fname,
                "class": cls,
                "label": cls,
                "size_bytes": os.path.getsize(path),
                "width": None,
                "height": None,
                "mode": None,
                "sha256": None,
                "md5": None,
                "dhash": None,
                "timestamp": None,
            }
            if err:
                rec["error"] = err
                errors.append(rec)
                corrupt.append(path)
                records.append(rec)
                continue
            rec["width"], rec["height"] = img.size
            rec["mode"] = img.mode
            rec["md5"] = md5_file(path)
            rec["sha256"] = sha256_file(path)
            md5_map[rec["md5"]].append(path)
            if rec["width"] <= MIN_RESOLUTION_MIN or rec["height"] <= MIN_RESOLUTION_MIN:
                tiny.append(path)
            rec["timestamp"] = parse_timestamp(fname).isoformat() if parse_timestamp(fname) else None
            rec["color"] = analyze_image(img)
            rec["dhash"] = perceptual_dhash(img)
            records.append(rec)
            class_records[cls].append(rec)
            img.close()

    # trùng nội dung (md5)
    duplicates = {h: ps for h, ps in md5_map.items() if len(ps) > 1}

    # gần trùng perceptual hash (trong cùng lớp để giảm nhiễu; chặn so sánh ảnh khác lớp)
    near_duplicates = []
    for cls in classes:
        recs = class_records.get(cls, [])
        n = len(recs)
        for i in range(n):
            if recs[i]["dhash"] is None:
                continue
            for j in range(i + 1, n):
                if recs[j]["dhash"] is None:
                    continue
                d = hamming(recs[i]["dhash"], recs[j]["dhash"])
                if d <= PERCEPTUAL_HAMMING_THRESHOLD:
                    near_duplicates.append({
                        "class": cls,
                        "a": recs[i]["path"],
                        "b": recs[j]["path"],
                        "hamming": d,
                    })

    return {
        "classes": classes,
        "total": sum(len(class_records[c]) for c in classes),
        "counts": {c: len(class_records[c]) for c in classes},
        "corrupt": corrupt,
        "tiny": tiny,
        "errors": errors,
        "duplicates": duplicates,
        "near_duplicates": near_duplicates,
        "records": records,
        "class_records": {c: [r["path"] for r in class_records[c]] for c in classes},
    }


def color_separability(audit: dict) -> dict:
    """Chẩn đoán thô: 1-NN leave-one-out trên đặc trưng màu (đã z-score).

    KHÔNG phải model phân loại và KHÔNG dùng để báo cáo accuracy của dự án.
    Chỉ để trả lời: "các lớp có tách được bằng màu đơn giản không?".
    """
    import numpy as np

    feats = ["hue_median", "sat_mean", "val_mean", "greenish_frac",
             "orange_frac", "red_frac", "yellow_frac"]
    recs = [r for r in audit["records"] if r.get("color")]
    if len(recs) < 4:
        return {}
    X = np.array([[r["color"][f] for f in feats] for r in recs], dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    labels = [r["class"] for r in recs]
    classes = list(dict.fromkeys(labels))
    pred = []
    for i in range(len(Z)):
        dist = np.linalg.norm(Z - Z[i], axis=1)
        dist[i] = 1e9
        pred.append(labels[int(np.argmin(dist))])
    acc = sum(p == l for p, l in zip(pred, labels)) / len(labels)
    mismatches = [
        {"filename": recs[i]["filename"], "class": labels[i], "nearest_class": pred[i]}
        for i in range(len(labels)) if pred[i] != labels[i]
    ]
    confusion = {
        cls: {c: sum(1 for i in range(len(labels))
                     if labels[i] == cls and pred[i] == c) for c in classes}
        for cls in classes
    }
    return {
        "features": feats,
        "leave_one_out_1nn_accuracy": round(acc, 4),
        "confusion": confusion,
        "mismatches": mismatches,
        "note": "Chẩn đoán đặc trưng màu, không phải accuracy của model dự án.",
    }


def summarize_audit(audit: dict) -> dict:
    out = {
        "total": audit["total"],
        "counts": audit["counts"],
        "corrupt_count": len(audit["corrupt"]),
        "tiny_count": len(audit["tiny"]),
        "duplicate_groups": len(audit["duplicates"]),
        "duplicate_files": sum(len(v) for v in audit["duplicates"].values()),
        "near_duplicate_pairs": len(audit["near_duplicates"]),
    }
    # kích thước
    sizes = [r["size_bytes"] for r in audit["records"] if r["size_bytes"]]
    if sizes:
        out["file_size_bytes"] = {"min": min(sizes), "max": max(sizes),
                                  "mean": round(sum(sizes) / len(sizes), 1)}
    res = [(r["width"], r["height"]) for r in audit["records"] if r["width"]]
    if res:
        out["resolutions"] = {f"{w}x{h}": n for (w, h), n in Counter(res).items()}
    # cân bằng lớp
    counts = audit["counts"]
    maxc = max(counts.values()) if counts else 0
    minc = min(counts.values()) if counts else 0
    out["imbalance_ratio_max_over_min"] = round(maxc / minc, 3) if minc else None
    # màu trung bình theo lớp
    out["class_color"] = {}
    out["class_geometry"] = {}
    for cls in audit["classes"]:
        recs = [r for r in audit["records"] if r["class"] == cls and r.get("color")]
        if recs:
            agg = {}
            for key in ("hue_median", "hue_mean", "sat_mean", "sat_median",
                        "val_mean", "val_median", "greenish_frac",
                        "orange_frac", "red_frac", "yellow_frac"):
                agg[key] = round(sum(r["color"][key] for r in recs) / len(recs), 4)
            out["class_color"][cls] = agg

            fg = sorted(r["color"]["fg_frac"] for r in recs)
            bb = sorted(r["color"]["bbox_frac"] for r in recs)
            sh = sorted(r["color"]["sharpness"] for r in recs)

            def pct(v, q):
                return round(v[min(len(v) - 1, int(q * (len(v) - 1)))], 4)

            out["class_geometry"][cls] = {
                "fg_frac_p10": pct(fg, 0.10),
                "fg_frac_p50": pct(fg, 0.50),
                "fg_frac_p90": pct(fg, 0.90),
                "bbox_frac_p50": pct(bb, 0.50),
                "sharpness_p10": pct(sh, 0.10),
                "sharpness_p50": pct(sh, 0.50),
            }
    # nền: đếm màu nền khác nhau (làm tròn 5)
    bg_keys = set()
    for r in audit["records"]:
        c = r.get("color")
        if c:
            bg_keys.add(tuple(round(v / 5) * 5 for v in c["bg_rgb"]))
    out["background_variants"] = len(bg_keys)
    out["background_common"] = sorted(
        ([list(k), sum(1 for r in audit["records"]
                       if r.get("color") and tuple(round(v / 5) * 5 for v in r["color"]["bg_rgb"]) == k)]
         for k in bg_keys), key=lambda x: -x[1])[:3]

    # ảnh mờ nhất / quả nhỏ nhất (chỉ liệt kê)
    scored = [(r["color"]["sharpness"], r["path"]) for r in audit["records"] if r.get("color")]
    out["blurriest"] = [{"sharpness": s, "path": p} for s, p in sorted(scored)[:10]]
    small = [(r["color"]["fg_frac"], r["path"]) for r in audit["records"] if r.get("color")]
    out["smallest_object"] = [{"fg_frac": f, "path": p} for f, p in sorted(small)[:10]]

    out["color_separability"] = color_separability(audit)
    # timestamp clustering
    out["timestamp"] = {}
    for cls in audit["classes"]:
        times = []
        for r in audit["records"]:
            if r["class"] == cls and r["timestamp"]:
                times.append(datetime.fromisoformat(r["timestamp"]))
        times.sort()
        gaps = [(times[i] - times[i - 1]).total_seconds() for i in range(1, len(times))]
        big = [g for g in gaps if g > 60]
        out["timestamp"][cls] = {
            "count": len(times),
            "first": times[0].isoformat() if times else None,
            "last": times[-1].isoformat() if times else None,
            "gap_leq5s": sum(1 for g in gaps if g <= 5),
            "gap_gt60s": len(big),
            "max_gap_s": round(max(gaps), 1) if gaps else None,
        }
    return out


def write_manifest(audit: dict, manifest_path: str, groups: dict | None = None) -> str:
    """Ghi manifest đầy đủ (CSV + JSON). groups: {filename: group_id} hoặc None."""
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    groups = groups or {}
    json_path = manifest_path if manifest_path.endswith(".json") else manifest_path + ".json"
    csv_path = os.path.splitext(json_path)[0] + ".csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "sha256", "size_bytes", "width", "height",
                         "group_id", "group_verified", "filename"])
        for r in audit["records"]:
            gid = groups.get(r["filename"], "")
            writer.writerow([
                r["path"], r["label"], r["sha256"], r["size_bytes"],
                r["width"], r["height"], gid,
                "yes" if gid else "no", r["filename"],
            ])
    return csv_path


def generate_groups_csv(audit: dict, out_path: str) -> str:
    """Sinh dataset/groups.csv: mọi ảnh, group_id để trống, cột verified=no.

    group_id trống + verified=no nghĩa là "chưa xác minh ảnh nào cùng một quả".
    Theo quyết định của người dùng, split hiện tại coi mỗi ảnh là một mẫu độc lập;
    khi nào điền group_id thật, đặt paths.groups_file để chia theo nhóm.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "group_id", "verified"])
        for cls in audit["classes"]:
            for r in audit["records"]:
                if r["class"] == cls and r["filename"]:
                    writer.writerow([r["filename"], "", "no"])
    return out_path


def create_contact_sheet(audit: dict, out_dir: str, cols: int = 6, thumb: int = 200) -> str:
    """Tạo contact sheet theo lớp (nội bộ, không upload)."""
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    made = []
    for cls in audit["classes"]:
        recs = [r for r in audit["records"] if r["class"] == cls and not r.get("error")]
        if not recs:
            continue
        rows = (len(recs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(recs):
            try:
                im = Image.open(r["path"]).convert("RGB").resize((thumb, thumb))
            except Exception:  # noqa: BLE001
                continue
            x = (i % cols) * thumb
            y = (i // cols) * thumb
            sheet.paste(im, (x, y))
            draw.text((x + 4, y + 4), r["filename"], fill=(255, 0, 0))
        out = os.path.join(out_dir, f"contact_{cls}.png")
        sheet.save(out)
        made.append(out)
    return ", ".join(made) if made else ""


def render_audit_markdown(audit: dict, summary: dict, groups_path: str, manifest_path: str) -> str:
    md = []
    md.append("# DATASET_AUDIT — Kiểm tra dữ liệu ảnh thật (ớt sim)\n")
    md.append(f"- Ngày chạy: {datetime.now().isoformat(timespec='seconds')}")
    md.append(f"- Tổng số ảnh: **{summary['total']}**")
    md.append(f"- Phân bố: `{summary['counts']}`")
    md.append(f"- Ảnh lỗi/không đọc được: **{summary['corrupt_count']}**")
    md.append(f"- Ảnh quá nhỏ (≤{MIN_RESOLUTION_MIN}px): **{summary['tiny_count']}**")
    md.append(f"- Trùng nội dung (md5): **{summary['duplicate_groups']} nhóm / {summary['duplicate_files']} file**")
    md.append(f"- Gần trùng (perceptual dHash ≤{PERCEPTUAL_HAMMING_THRESHOLD}): **{summary['near_duplicate_pairs']} cặp**")
    md.append(f"- Chênh lệch lớp (max/min): **{summary['imbalance_ratio_max_over_min']}**")
    md.append(f"- Manifest: `{manifest_path}`")
    if groups_path:
        md.append(f"- groups.csv: `{groups_path}` (group_id trống = chưa xác minh, mỗi ảnh 1 nhóm riêng)")
    md.append("")

    md.append("## 1. Kích thước & độ phân giải\n")
    if "file_size_bytes" in summary:
        s = summary["file_size_bytes"]
        md.append(f"- Kích thước file: {s['min']}–{s['max']} bytes (trung bình {s['mean']})")
    if "resolutions" in summary:
        md.append(f"- Độ phân giải: {summary['resolutions']}")
    md.append("")

    md.append("## 2. Cân bằng lớp\n")
    for c, n in summary["counts"].items():
        frac = n / summary["total"] * 100 if summary["total"] else 0
        md.append(f"- `{c}`: {n} ảnh ({frac:.1f}%)")
    md.append("")

    md.append("## 3. Màu sắc theo lớp (HSV trên vùng quả, trung bình lớp)\n")
    md.append("| Lớp | Hue median | Sat mean | Val mean | Xanh | Vàng | Cam | Đỏ |")
    md.append("|---|---|---|---|---|---|---|---|")
    for cls, col in summary.get("class_color", {}).items():
        md.append(f"| {cls} | {col['hue_median']} | {col['sat_mean']} | {col['val_mean']} "
                  f"| {col['greenish_frac']} | {col.get('yellow_frac', '')} "
                  f"| {col['orange_frac']} | {col['red_frac']} |")
    md.append("")

    md.append("## 3b. Hình học: quả chiếm bao nhiêu khung + độ nét\n")
    md.append("| Lớp | Diện tích quả p10 | p50 | p90 | BBox p50 | Độ nét p10 | p50 |")
    md.append("|---|---|---|---|---|---|---|")
    for cls, g in summary.get("class_geometry", {}).items():
        md.append(f"| {cls} | {g['fg_frac_p10']} | {g['fg_frac_p50']} | {g['fg_frac_p90']} "
                  f"| {g['bbox_frac_p50']} | {g['sharpness_p10']} | {g['sharpness_p50']} |")
    md.append("")
    md.append(f"- Số biến thể màu nền (làm tròn 5): **{summary.get('background_variants')}** "
              f"— phổ biến: {summary.get('background_common')}")
    md.append("")

    sep = summary.get("color_separability") or {}
    if sep:
        md.append("## 3c. Chẩn đoán tách lớp bằng màu (1-NN leave-one-out)\n")
        md.append(f"- Accuracy 1-NN trên đặc trưng màu: **{sep['leave_one_out_1nn_accuracy']}**")
        md.append("- Đây là chẩn đoán đặc trưng, **không phải accuracy của model dự án**.")
        md.append("")
        md.append("| Thật \\ Gần nhất | " + " | ".join(sep["confusion"]) + " |")
        md.append("|---|" + "---|" * len(sep["confusion"]))
        for cls, row in sep["confusion"].items():
            md.append(f"| {cls} | " + " | ".join(str(v) for v in row.values()) + " |")
        md.append("")
        if sep["mismatches"]:
            md.append(f"Ảnh có đặc trưng màu gần lớp khác ({len(sep['mismatches'])}):")
            for m in sep["mismatches"]:
                md.append(f"- `{m['filename']}` nhãn `{m['class']}` ↔ gần `{m['nearest_class']}`")
        md.append("")

    md.append("## 4. Điều kiện chụp (timestamp)\n")
    md.append("| Lớp | Số ảnh | Đầu | Cuối | Gap ≤5s | Gap >60s | Gap lớn nhất |")
    md.append("|---|---|---|---|---|---|---|")
    for cls, t in summary.get("timestamp", {}).items():
        md.append(f"| {cls} | {t['count']} | {t['first']} | {t['last']} "
                  f"| {t['gap_leq5s']} | {t['gap_gt60s']} | {t['max_gap_s']}s |")
    md.append("")

    md.append("## 5. Ảnh lỗi / quá nhỏ\n")
    if audit["corrupt"]:
        for p in audit["corrupt"]:
            md.append(f"- ❌ `{p}`")
    else:
        md.append("- Không có ảnh lỗi.")
    if audit["tiny"]:
        md.append("\nẢnh quá nhỏ:")
        for p in audit["tiny"]:
            md.append(f"- ⚠️ `{p}`")
    md.append("")

    md.append("## 6. Trùng nội dung (md5)\n")
    if audit["duplicates"]:
        for h, ps in audit["duplicates"].items():
            md.append(f"- md5={h[:8]}… → {len(ps)} file: `{ps[:3]}`")
    else:
        md.append("- Không có ảnh trùng nội dung.")
    md.append("")

    md.append(f"## 7. Gần trùng (perceptual dHash ≤ {PERCEPTUAL_HAMMING_THRESHOLD})\n")
    if audit["near_duplicates"]:
        for nd in audit["near_duplicates"][:50]:
            md.append(f"- ({nd['hamming']}) `{os.path.basename(nd['a'])}` ↔ `{os.path.basename(nd['b'])}` (lớp {nd['class']})")
        if len(audit["near_duplicates"]) > 50:
            md.append(f"- … và {len(audit['near_duplicates']) - 50} cặp nữa.")
    else:
        md.append("- Không có cặp gần trùng.")
    md.append("")

    md.append("## 8. Nhận xét & đề xuất\n")
    md.append("- (Ghi sau khi xem contact sheet và phân tích dữ liệu thực tế.)")
    return "\n".join(md)


def main(argv=None) -> None:
    from PIL import Image

    parser = argparse.ArgumentParser(description="Audit dataset ảnh thật ớt")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--groups", default="")
    parser.add_argument("--contact", action="store_true", help="Tạo contact sheet theo lớp")
    parser.add_argument("--audit-md", default="")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["project"]["classes"]
    raw = cfg["paths"]["dataset_raw"]

    print(f"Audit dataset: {raw}")
    audit = scan_audit(raw, classes)
    summary = summarize_audit(audit)
    audit["summary"] = summary

    print(f"\nSố ảnh: {summary['total']} → {summary['counts']}")
    print(f"Lỗi: {summary['corrupt_count']}, quá nhỏ: {summary['tiny_count']}")
    print(f"Trùng nội dung: {summary['duplicate_groups']} nhóm / {summary['duplicate_files']} file")
    print(f"Gần trùng: {summary['near_duplicate_pairs']} cặp")
    if "file_size_bytes" in summary:
        print(f"Kích thước file: {summary['file_size_bytes']}")
    if "resolutions" in summary:
        print(f"Độ phân giải: {summary['resolutions']}")
    print(f"Chênh lệch lớp (max/min): {summary['imbalance_ratio_max_over_min']}")

    manifest_path = args.manifest or os.path.join(cfg["paths"].get("reports_metrics", "reports/metrics"),
                                                  "dataset_audit.json")
    groups_path = None
    if args.groups:
        groups_path = args.groups
    else:
        # mặc định: dataset/groups.csv
        groups_path = os.path.join(os.path.dirname(cfg["paths"]["dataset_raw"]), "groups.csv")
    write_manifest(audit, manifest_path, groups=None)
    generate_groups_csv(audit, groups_path)
    print(f"Đã ghi manifest: {manifest_path}")
    print(f"Đã ghi groups.csv: {groups_path} (group_id trống = chưa xác minh)")

    if args.contact:
        out = os.path.join(cfg["paths"].get("reports_figures", "reports/figures"), "contact_sheets")
        made = create_contact_sheet(audit, out)
        print(f"Contact sheet đã tạo: {made}")

    if args.audit_md:
        md = render_audit_markdown(audit, summary, groups_path, manifest_path)
        with open(args.audit_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Đã ghi báo cáo markdown: {args.audit_md}")


if __name__ == "__main__":
    main()
