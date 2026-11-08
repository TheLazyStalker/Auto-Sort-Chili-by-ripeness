# TASK 4 — TRAINING & DEVELOPMENT EVALUATION (MobileNetV3Small)

> Trạng thái: **DEVELOPMENT** — mọi số liệu dưới đây là trên dev split 80/20
> image-level **leaky** (identity quả không rõ), KHÔNG phải final benchmark.
> Không claim accuracy generalization thực tế.

## 1. Môi trường

- Container `ai-workspace` (Podman), project `/workspace/chili-ripeness-ai`.
- Venv `/workspace/.venv`; Python 3.12.3; TensorFlow 2.17.0 CPU-only.
- Không cài dependency mới; không đổi host/GPU/CUDA config.

## 2. Preprocessing MobileNetV3Small (đã verify trong container)

- Keras 3 / TF 2.17: `MobileNetV3Small(include_preprocessing=True)` nhúng layer
  `Rescaling(scale=0.0078431, offset=-1.0)` ngay sau Input → `[0,255]` → `[-1,1]`.
- `keras.applications.mobilenet_v3.preprocess_input()` là **placeholder no-op**
  (backward compatibility) — KHÔNG gọi, KHÔNG thêm `Rescaling(1/255)` ngoài model.
- Invariant: mọi consumer truyền RGB `[0,255]` (uint8 hoặc float32 chưa chia 255);
  model tự xử lý bên trong. Không áp dụng preprocessing 2 lần.

## 3. Dữ liệu

- Nguồn: `dataset/split_manifest_task3.json` (train-only 463 derivative crop_v1;
  10 ảnh bị loại không có trong manifest — đã verify).
- Dev split: `src/task4.py::create_dev_split` — **image-level 80/20**, stratified
  theo lớp, seed 42. Không group-aware → **leaky có chủ đích**, chỉ cho development.
- Augmentation (chỉ train): RandomFlip(h/v), RandomRotation(0.1), RandomZoom(0.1),
  RandomTranslation(0.05), RandomBrightness(0.05) — **không hue/saturation**.
- Class weights: balanced (nếu `training.class_weight != null`).

## 4. Huấn luyện (2 phase)

| Phase | Backbone | Epochs (max) | LR | Callbacks |
|---|---|---|---|---|
| A | frozen | 30 | 1e-3 | EarlyStopping(val_acc, patience 7), ModelCheckpoint(best_phase_a), ReduceLROnPlateau |
| B | fine-tune 20 layer cuối | 20 | 1e-4 | như trên + ModelCheckpoint(best_chili) |

- Seeds: Python/NumPy/TF qua `set_seeds(cfg)` (seed 42).
- Best checkpoint giữ (save_best_only); không lưu epoch cuối.

## 5. Lệnh chạy

```bash
# Train (artifacts/task4/)
python scripts/train_task4.py                    # mặc định seed 42
python scripts/train_task4.py --epochs-a 30 --epochs-b 20

# Evaluation DEV
python scripts/evaluate_task4.py --save-probs

# Latency CPU (preliminary)
python scripts/measure_latency.py --warmup 5 --runs 50
```

## 6. Artifacts (`artifacts/task4/`)

`best_chili.keras`, `best_phase_a.keras`, `training_config.json`,
`class_mapping.json`, `dev_split_manifest.json`, `training_history.json`,
`training_history_summary.json`, `training_curves.png`, `evaluation_dev.json`,
`confusion_matrix.png|json`, `latency_cpu.json`, `dev_prediction_probs.csv`.

## 7. Giới hạn

- Dev split leaky → accuracy chỉ phản ánh quá khứ trên ảnh đã thấy; chỉ dùng để
  monitoring/model selection.
- Latency CPU trong container là sơ bộ — chưa phải điện thoại Android (mẫu đã test) / OV2640.
- TFLite/export/Termux thuộc Task 5.

## 8. Kết quả (2026-11-08, DEVELOPMENT)

| Hạng mục | Giá trị |
|---|---|
| Params | 940,851 |
| best_chili.keras | 7.17 MB |
| Phase A | best val_acc 0.9891 (epoch 9/17 — early stop) |
| Phase B | best val_acc 0.9891 (epoch 1/8 — early stop) |
| DEV accuracy | 0.9891 (92 ảnh) |
| Macro F1 | 0.9877 |
| Confusion | unripe [38,0,0]; medium [0,27,1]; ripe [0,0,26] |
| Latency CPU | mean 71ms / median 66ms / P95 113ms |

Mọi con số là DEVELOPMENT trên split leaky — không claim generalization thực tế.
