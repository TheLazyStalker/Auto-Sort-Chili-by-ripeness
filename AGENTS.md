# AGENTS.md — chili-ripeness-ai

## Use the container, not host Python
- The host repo `<HOST_REPO>/podman-workspace/chili-ripeness-ai` is bind-mounted as `/workspace/chili-ripeness-ai` in the podman container `ai-workspace`.
- Run project commands inside the container:
  ```bash
  podman start ai-workspace
  podman exec -it ai-workspace bash
  cd /workspace/chili-ripeness-ai
  source /workspace/.venv/bin/activate
  ```
- Host `python3` is not the project runtime (Python 3.14, no TensorFlow). The container uses Python 3.12.3 + TensorFlow 2.17.0 CPU-only. Keep `numpy<2.0.0` and `protobuf<5`; CUDA warnings are expected.
- The repo is not currently a git repository; do not assume branch/PR workflows or that `.gitignore` protects files if git is initialized later.

## Verify
- Tests: `pytest -q` (28 unit tests; no real images required). Single test: `pytest tests/test_model.py -q`.
- Dataset check: `python scripts/check_dataset.py`; prepare splits: `python scripts/check_dataset.py --prepare`.
- Smoke test: `python scripts/smoke_test_model.py`. This only proves model construction with random tensors; it does not validate data, training, Android, or deployment.
- Pipeline: `python -m src.train` → `python -m src.evaluate --split test` → `python -m src.export_tflite`.
- Missing data/model paths are designed to stop safely. Do not fabricate images, labels, metrics, calibration data, or placeholder model results to make the pipeline pass.

## Preprocessing invariant
- The MobileNetV2 model internally applies `preprocess_input` (`[0,255]` → `[-1,1]`). Dataset, predict, evaluate, export, and Android integration must pass RGB `[0,255]` and must not add `Rescaling(1/255)` (would double-scale).
- Existing `models/tflite/model_metadata.json` and `models/tflite/android_integration.md` currently describe stale `[0,1]`/`Rescaling` preprocessing. Trust `src/model.py`, `src/predict.py`, and `src/export_tflite.py`; fix docs if touching them.

## Dataset rules
- Real images belong in `dataset/raw/{unripe,medium_ripe,ripe}/`. Never use synthetic/Internet placeholder images as project data without explicit user approval.
- `--prepare` copies images into split folders and never deletes originals. `dataset/split_manifest.json` prevents accidental re-splitting; use `--force-resplit` only when explicitly needed.
- Use `dataset/groups.csv` (`filename,group_id`) and set `paths.groups_file` in `config.yaml` when multiple shots of the same pepper exist; otherwise the split can leak the same pepper across train/validation/test.
- Keep augmentation hue/saturation-free; color is the primary ripeness signal (`config.yaml` sets both factors to `0.0`).
- See `DATASET_GUIDE.md` for labeling rules and `PROJECT_ROADMAP.md` for backlog/known issues; verify roadmap status against the code before relying on it.

## Known traps
- `stratified_group_split` uses `round()` and tiny datasets can produce empty validation/test splits; `build_tf_datasets` then fails on empty splits. Treat this as a bug to fix, not expected behavior.
- `PROJECT_ROADMAP.md` contains known P0/P1 issues and may have stale status labels. Inspect the code and rerun tests before marking anything fixed.
- `paths.*` in `config.yaml` are relative to the project root; `src/config.py` resolves them to absolute paths at load time.
- Generated datasets, models, reports, and split manifests are intentionally not meant to be committed. Do not generate them without real inputs.
