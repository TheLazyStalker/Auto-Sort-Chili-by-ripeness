# TASK3_DATA_PLAN — Confirmed data-handling plan

> **Status:** proposed update based on the user's decisions Q1–Q7 (2026-11-07).
> **Awaiting confirmation.**
>
> **No raw image, label, group file, manifest, split, or dataset has been changed.**
> **No new images have been collected. No training/evaluation has been started.**
>
> This plan supersedes the recommendations in `reports/image_handling_review.md`
> §4.4, §5.5, §6.5, §7 and §8 where they conflict. The analysis and evidence in
> that report remain valid.

---

## 0. Confirmed decisions

| ID | Decision | Consequence |
|---|---|---|
| **Q1** | **C — no image→pepper mapping.** | Do not infer groups from filenames, timestamps, bursts, or visual similarity. The existing 473 images are training-only. |
| **Q2** | **A — usually consecutive shots per pepper, but exact sequence is not remembered.** | Recorded, but **not usable** to reconstruct groups: shot counts varied (≈3–6), sequence is unknown, and the 25-image visual check shows adjacent images often differ. No "block of 5 = one pepper" heuristic. |
| **Q3** | Pepper counts and images/pepper are **estimates**, not exact. | Do not claim the existing dataset has exactly/at least 30 independent peppers per class. The 30–50+ target applies to Task 3 collection. |
| **Q4** | **B — use the existing 473 images for training; collect new, independent peppers for validation and test.** | Current 332/71/70 split is **smoke-test only**, never a final benchmark. Final validation/test must come from new peppers. |
| **Q5** | **Exclude the 10 flagged images from training for now; keep a separate review/exclusion list; do not delete or modify raw images.** | `dataset/review_exclusions.csv`; all 10 excluded from train/val/test until reviewed. Raw files stay in place. |
| **Q6** | **C — save cropped derivatives separately, preserve all raw images, use the same crop procedure at inference.** | Versioned `crop_v1`; derivatives in `dataset/derived/crop_v1/`; raw untouched; inference uses the identical crop function. |
| **Q7** | **YES, with the stated conditions.** | Capture protocol: unique `group_id`, 25–40% frame, diffuse light + matte background, lock WB if reliable (otherwise document/stabilize), 3–6 images/pepper, 30–50+ peppers/class initially (target 50–100+), preserve originals + EXIF, never crop/overwrite raw. |

---

## 1. Invariants

1. **Raw data is immutable.** No delete, move, rename, relabel, crop, resize, or
   overwrite of `dataset/raw/**`. Record a SHA256 baseline and re-verify after every
   processing step.
2. **Derivatives are separate and reproducible.** `dataset/derived/crop_v1/` is
   generated from raw + a frozen crop specification. Deleting and regenerating
   derivatives must be safe.
3. **Group identity is metadata, not a folder guess.** New peppers get a unique
   `group_id` at capture. No pepper may appear in more than one split.
4. **Preprocessing invariant:** images are RGB `[0,255]`; the model internally applies
   MobileNetV2 `preprocess_input` → `[-1,1]`. **Never add `Rescaling(1/255)`** in the
   dataset, crop, predict, evaluate, export, or Termux/Android path.
5. **Labels are human decisions.** No hard HSV/RGB thresholds may auto-assign or
   auto-correct labels. Color statistics are audit aids only.
6. **The 1-NN color diagnostic (98.94%) is not model accuracy.** It must never be
   reported as a model result.
7. **No final accuracy claim** until training/evaluation on a clean, group-disjoint
   test set of new peppers.

---

## 2. Existing 473 images

### 2.1 Facts

| Item | Value |
|---|---|
| Raw images | 473 (unripe 197 / medium_ripe 142 / ripe 134) |
| Format | JPG, RGB, 1024×1024 |
| EXIF | 0/473 |
| md5 duplicates | 0 |
| Current split | 332 / 71 / 70 (70.2 / 15.0 / 14.8%) |
| Burst pairs ≤5 s across splits | 183 (temporal warning, not 183 leaked peppers) |
| Pepper area (median) | 3.28–3.73% of frame |
| Pepper bbox (median) | 10.5–13.1% of frame |

### 2.2 How the 473 images will be used

- **All 463 included images → training only.**
- **10 flagged images → excluded** (see §3).
- **No validation/test image** is taken from the 473.
- The existing `dataset/split_manifest.json` (332/71/70) is kept only for
  pipeline smoke tests and must be labelled **`smoke_only / leaky`** in any report.
- If a small dev subset is needed before new data exists, it may be carved from the
  473 **only** for overfitting monitoring, labelled **`dev_leaky`**, and never used
  for a final metric.

### 2.3 Q2 conflict — documented, not resolved by guessing

The user says they usually shot consecutive photos of one pepper before moving on.
That cannot be used to reconstruct groups because:

- the exact sequence is not remembered;
- the number of shots per pepper is an estimate (≈3–6);
- the 25-image visual check shows adjacent images differ clearly in size, shape,
  orientation, and stem length.

Possible explanations: the same pepper was rotated/moved between shots, or the
memory does not match the actual sequence. Either way, **timestamp blocks are not a
valid group assignment**. Since all 473 are training-only, this does not affect the
final evaluation; it only forbids using the old split as a benchmark.

### 2.4 Q3 — counts are estimates

The arithmetic in `image_handling_review.md` §2.3 (≈39/28/27 peppers if exactly
5 images/pepper) is an **illustration**, not a verified count. The dataset must be
described as **473 images of unknown pepper multiplicity**, not "≥30 peppers/class".
Task 3 must report the actual number of unique `group_id`s collected.

---

## 3. Review / exclusion list

### 3.1 File

`dataset/review_exclusions.csv`

Proposed columns:

```csv
filename,current_label,reason,status,reviewed,reviewed_label,notes
```

- `status`: `excluded` initially (vocabulary: `accepted | pending_review | excluded`).
- `reviewed`: `no` initially.
- `reviewed_label`: empty until the user reviews.
- Do **not** move, rename, delete, or relabel the raw file.

### 3.2 The 10 excluded images

| Filename | Current label | Reason for temporary exclusion |
|---|---|---|
| `IMG_20261107_150901.jpg` | unripe | color-nearest medium_ripe; yellow/orange tip |
| `IMG_20261107_150329.jpg` | medium_ripe | color-nearest ripe; dark red-brown body |
| `IMG_20261107_150511.jpg` | medium_ripe | color-nearest unripe; green body with yellow tip |
| `IMG_20261107_145600.jpg` | ripe | color-nearest medium_ripe; long green stem (label may be correct) |
| `IMG_20261107_145847.jpg` | ripe | color-nearest medium_ripe; orange-red boundary |
| `IMG_20261107_151125.jpg` | unripe | yellow/orange bands; strongest unripe/medium boundary |
| `IMG_20261107_150946.jpg` | unripe | small object; flagged by the old "yellow" heuristic |
| `IMG_20261107_151349.jpg` | unripe | yellow-green; large bbox |
| `IMG_20261107_151342.jpg` | unripe | yellow-green; large bbox |
| `IMG_20261107_150950.jpg` | unripe | smallest object; flagged by the old "yellow" heuristic |

Exclusion is **not** a statement that the label is wrong. It is a conservative
"do not train on ambiguous examples until reviewed" decision.

### 3.3 How exclusion is enforced

- The Task 3 manifest builder reads `review_exclusions.csv` and **omits** these files
  from train/validation/test.
- The audit/contact sheets still include them so the user can review them.
- Reports show both counts: `raw_total=473`, `trainable=463`, `excluded=10`.
- If the user later marks an image as correct, set `status=accepted` and remove
  it from exclusion on the next manifest build.
- If the user later changes a label, record `reviewed_label` and apply a
  `dataset/label_overrides.csv` (`filename,new_label,reason,approved_by,date`) in the
  manifest builder. **Do not move the raw file** unless the user explicitly approves
  a physical reorganisation.

---

## 4. `crop_v1` — derivative crop policy (Q6=C)

### 4.1 Why crop

At 1024×1024 the pepper occupies only ~3.3–3.7% of the frame. Resizing the full frame
to 224×224 leaves the pepper only ~25–35 px thick (estimate from bbox and foreground
area). A bbox crop preserves much more detail. The raw image is always kept.

### 4.2 Directory layout

```text
dataset/derived/crop_v1/
├── train/
│   ├── unripe/
│   ├── medium_ripe/
│   └── ripe/
├── validation/          # populated only from new, independent peppers
├── test/                # populated only from new, independent peppers
├── metadata.csv
└── manifest.json
```

- Derivatives are **PNG** (lossless) at 224×224×3, RGB `[0,255]`.
- `dataset/derived/**` is generated and gitignored.
- Raw stays in `dataset/raw/**`, unchanged.

### 4.3 Crop algorithm (`crop_v1`)

1. Read raw image as RGB.
2. Estimate the background:
   - Existing 473 images: median color of a border band (the background is uniform
     ≈ RGB 220,235,235).
   - Deployment/OV2640: capture a **background reference frame** under the same
     lighting at startup; use its median color. If no reference is available, fall
     back to the border-band median and log a warning.
3. Build a foreground mask by color distance from the background in RGB or CIELAB.
   Calibrate the threshold on a sample; an initial candidate range is 25–40 RGB
   units. Do not hard-code a threshold without checking the sample.
4. Clean the mask: small morphological open/close, remove components below a minimum
   area, keep the largest component; dilate 2–3 px so a thin stem is included.
5. Compute the bounding box `(x0, y0, x1, y1)`.
6. Expand by **20% padding** of `max(width, height)` on each side (enough for
   rotation/zoom/translation augmentation and to avoid cutting the tip/stem).
7. Make the crop square, centred on the bbox; clamp to image bounds; if the square
   extends beyond the image, pad with the estimated background colour.
8. Resize to **224×224** with a deterministic resampling method (e.g., bilinear).
9. Save as PNG.
10. Record metadata (see §4.4).

If the mask area is `<0.5%` or `>60%` of the frame, or the bbox is implausible, set
`crop_status=needs_review` and do **not** silently use a bad crop. For training,
exclude `needs_review` derivatives until inspected. For inference, return
`crop_failed` and do not predict.

### 4.4 Metadata

`dataset/derived/crop_v1/metadata.csv` columns:

```csv
source_path,source_sha256,crop_version,bbox_x0,bbox_y0,bbox_x1,bbox_y1,
pad_frac,output_path,output_sha256,width,height,label,group_id,split,
crop_status,background_rgb,threshold,notes
```

- `crop_version = crop_v1`.
- `group_id` and `split` are inherited from the source image / capture log.
- `label` is the final label after overrides, not necessarily the folder name.

### 4.5 Inference parity

- Implement one function, e.g. `src/crop.py::crop_v1(image, background_ref=None)`.
- The derivative generator and `src/predict.py` **must call the same function**.
- `src/evaluate.py` must evaluate on the derivative test set, or apply `crop_v1` to
  raw test images before resize; do not compare a cropped-trained model against
  full-frame test images.
- `src/export_tflite.py` INT8 representative dataset must use `crop_v1` derivatives,
  not raw full-frame images.
- At deployment, capture a background reference frame at startup. If `crop_v1`
  fails, return `crop_failed` / `uncertain`; do not feed a full-frame image to a
  crop-trained model.
- Do not add `Rescaling(1/255)` anywhere.

### 4.6 Leakage rules

- A derivative **inherits** the source `group_id` and split.
- Never put a derivative of a training pepper into validation/test.
- Since the 473 are training-only, no old-image derivative may enter val/test.
- New data: assign groups to splits **before** generating derivatives, or generate
  derivatives and then split by inherited group; never split by derivative filename.

### 4.7 QA before training

- Overlay the mask/bbox on at least 30 sampled images (all classes, small/large
  objects, blurry images) and inspect visually.
- Verify output size, RGB mode, and that colour statistics are not shifted.
- Verify raw SHA256 is unchanged after derivative generation.
- Count `crop_status=ok / needs_review / failed`; require a high success rate before
  full training. If failures cluster by class/size, fix the crop before proceeding.
- Do not generate derivatives for the 10 excluded images.

### 4.8 Versioning

- Any change to threshold, padding, background estimation, or resize method creates
  `crop_v2`, etc.
- The crop version must be recorded in the model metadata, evaluation report, and
  Termux deployment docs. Training and inference must use the same version.

---

## 5. Task 3 capture protocol (Q7)

### 5.1 Capture log

`dataset/capture_log.csv` (created after plan confirmation):

```csv
filename,group_id,class,session_id,captured_at,lighting,background,wb_setting,
exposure,gain,distance_cm,orientation,provisional_label,notes
```

At minimum: `filename, group_id, class, session_id, captured_at, lighting,
background, wb_setting, notes`.

- `group_id` is unique per physical pepper and is filled **at capture time**.
- Do not reuse a `group_id` across classes or sessions.
- If camera filenames collide across sessions, copy into `dataset/raw/<class>/` with
  a session prefix; keep the original file.

### 5.2 Framing and optics

- One pepper per frame.
- Pepper occupies **25–40% of the frame area**.
- Fixed camera distance and fixed zoom; no digital zoom.
- Capture multiple orientations per pepper; include stem and tip.
- Avoid cutting the pepper at frame edges.

### 5.3 Lighting, background, white balance

- Diffuse, consistent lighting; no direct flash.
- Matte background that contrasts with the pepper (not the current pale background
  only; use 2–3 controlled backgrounds if possible).
- Use a diffuser to reduce specular highlights.
- Lock white balance if the OV2640 supports reliable manual control. If not, fix the
  lighting, exposure, gain, and WB mode, and document them in the capture log.
- Keep lighting/background consistent within a session; record any change.

### 5.4 Counts

- **3–6 images per pepper.**
- **At least 30–50 independent peppers per class initially; target 50–100+.**
- Balance classes and ripeness stages; do not collect all unripe first.
- More peppers are better than more images of the same pepper. Do not compensate for
  too few peppers by taking 20 shots each.

### 5.5 Raw preservation

- Preserve original files and EXIF.
- Never crop, resize, or overwrite raw images.
- Derivatives are generated later by `crop_v1`.
- Do not use Internet or synthetic images.

### 5.6 Test-pepper isolation

- Test peppers must be **physically different** from every pepper used in training or
  validation.
- Decide group→split assignment before or at capture (e.g., separate sessions for
  validation and test).
- Keep a written list of test `group_id`s; freeze it before training.

---

## 6. New-data split

### 6.1 Group split, not image split

- Split by `group_id`, never by image.
- No group may appear in more than one of train/validation/test.
- The 473 old images are all train; new peppers are validation/test.
- Optional: collect extra new train peppers. If so, include them in the group split.

### 6.2 Proposed allocation

- If new collection is used only for validation/test: **50/50 by pepper group**,
  stratified by class.
- If new train peppers are also collected: **70/15/15 by group**, stratified by class.
- Target **≥10–15 peppers/class in validation** and **≥10–15 peppers/class in test**
  (with 30–50 peppers/class collected, 50/50 gives 15–25 each).
- If collection falls short, prioritise the test set; report the actual counts and do
  not claim the target was met.

### 6.3 Test freeze

- Before training, write the test manifest and record its SHA256.
- Do not tune on test. Use validation for early stopping/model selection.
- Evaluate test once for the final report; if you must re-evaluate after a code
  change, document it and keep the test set unchanged.

---

## 7. Labeling

- Two annotators label independently.
- Pilot 30 images/class, then compare and agree on criteria:
  - colour transition order;
  - how to treat the green stem;
  - boundary cases (green-yellow, orange-red);
  - small/partially occluded peppers.
- Use anchor/reference images for each class; **no hard colour thresholds**.
- Record `label_status` (`confirmed` / `needs_review`) and annotator IDs.
- Report inter-annotator agreement (e.g., Cohen's kappa or % agreement).
- Unresolved disagreements → `needs_review`; exclude from the test set if still
  unresolved.
- Do not fabricate labels or copy labels from filenames.

---

## 8. Manifest and training

### 8.1 Task 3 manifest

`dataset/split_manifest_task3.json`:

- `train`: 463 old `crop_v1` derivatives (excluding the 10) + optional new train
  peppers.
- `validation`: new validation peppers' `crop_v1` derivatives.
- `test`: new test peppers' `crop_v1` derivatives.
- Paths point to derivatives; labels are class IDs.
- Keep a separate `manifest_metadata.json` with crop version, creation time,
  exclusion-list hash, group-count summary, and test-manifest SHA256.

### 8.2 Verification before training

- 0 file overlap across splits.
- 0 group leak across splits.
- All 3 classes present in train/validation/test.
- No excluded filename appears in any split.
- Unique `group_id` counts per class/split are reported.
- Validation/test peppers are physically distinct from train peppers.
- Crop version is `crop_v1` for all images.

### 8.3 Training changes (implementation later)

- Extend `src/train.py` to accept `--manifest dataset/split_manifest_task3.json` and
  **skip** `prepare_datasets()` auto-resplitting when a preset manifest is supplied.
- Verify the manifest hash before training; refuse to train if verification fails.
- Train on `crop_v1` derivatives only; do not mix raw full-frame and cropped images
  in `crop_v1` training.
- Keep class weights, two-stage transfer learning, and no hue/saturation
  augmentation.
- Check the augmentation rotation range: Keras `RandomRotation(0.1)` is approximately
  ±36°, while `config.yaml` comments say ±18°. The 20% crop padding is chosen to
  tolerate the larger range; confirm before training.

### 8.4 Evaluation

- Report **image-level** and **pepper-level** metrics:
  - image-level accuracy, macro F1, per-class precision/recall, confusion matrix;
  - pepper-level aggregation (mean probability or majority vote per `group_id`);
  - 95% confidence interval by **cluster bootstrap over peppers**, not images.
- Report the number of test peppers and images per class.
- Report uncertain rate at the configured confidence threshold.
- Do not report the 1-NN colour diagnostic as model accuracy.
- No "≥90%" claim until the clean test evaluation is complete.

---

## 9. Implementation phases

### Phase 0 — Plan confirmation (current)

- User confirms this plan and the open parameters in §11.
- No dataset changes, no collection.

### Phase 1 — Metadata and tooling (no new images)

Deliverables:

- `dataset/review_exclusions.csv` with the 10 rows.
- `dataset/capture_log.csv` template.
- `src/crop.py` with `crop_v1`.
- `scripts/make_crop_derivatives.py` (reads raw + exclusions, writes derivatives +
  metadata).
- `scripts/build_task3_manifest.py` (combines old train + new val/test, verifies
  groups/overlaps).
- `scripts/qa_crop.py` (mask/bbox overlays, success counts).
- Config keys for `dataset_derived`, `split_manifest_task3`, `exclusions_file`,
  `capture_log`.
- Tests for crop determinism, exclusion filtering, and manifest verification.

Actions:

- Generate `crop_v1` derivatives for the **463 included** old images.
- Run QA on a sample; verify raw SHA256 unchanged.
- Create a provisional train-only manifest; do **not** train yet.

Acceptance:

- 463 derivatives, 0 from excluded images.
- Raw SHA256 473/473 unchanged.
- Crop success rate acceptable; failures listed.

### Phase 2 — Pilot collection

- 3–5 peppers/class, 3–6 shots each, full Q7 protocol.
- Fill `capture_log.csv` and `groups.csv` for the pilot.
- Generate `crop_v1` derivatives; run QA.
- Label with 2 people; calibrate criteria.
- Acceptance: crop works on pilot; group IDs complete; labels agree; no raw changes.

### Phase 3 — Full collection

- 30–50+ peppers/class (target 50–100+).
- Assign groups to validation/test (and optional train-new).
- Keep test peppers isolated.
- Acceptance: counts, group IDs, framing, lighting, EXIF, no raw crop/overwrite.

### Phase 4 — Labeling

- 2 annotators; pilot calibration; full labeling; adjudication.
- Acceptance: agreement reported; unresolved cases marked; no hard thresholds.

### Phase 5 — Audit, derivatives, manifest

- Audit new raw: counts, corrupt, duplicates, EXIF, colour/geometry.
- Generate `crop_v1` derivatives for included images.
- Build `split_manifest_task3.json`; verify §8.2.
- Freeze test manifest + hash.
- Acceptance: 0 overlap, 0 group leak, all classes, no exclusions, crop QA passed.

### Phase 6 — Train/evaluate

- Train on train manifest; validate on new validation peppers.
- Evaluate once on frozen test; report image- and pepper-level metrics + CI.
- Acceptance: reproducible run; metrics reported honestly; no test tuning.

### Phase 7 — Export/deploy

- Only after a clean test: TFLite export, Termux CLI, OV2640 integration.
- Inference uses `crop_v1`; verify on-device crop/colour/latency.
- Update deployment docs to the `[0,255]` → `preprocess_input` invariant and crop
  version.

---

## 10. Acceptance criteria

| Area | Criterion |
|---|---|
| Raw integrity | 473/473 SHA256 unchanged; EXIF preserved; no raw crop/overwrite |
| Exclusions | 10 flagged images absent from train/val/test; raw still present |
| Old data | 463 included images used for training only |
| New peppers | ≥30–50/class initially, target 50–100+; actual counts reported |
| Images/pepper | 3–6 |
| Group IDs | 100% of new images have a unique pepper `group_id` |
| Split | 0 group leak, 0 file overlap; all classes in all splits |
| Test isolation | Test peppers physically distinct from train/val; manifest frozen |
| Crop | `crop_v1` used for train, val, test, predict, export; QA success recorded |
| Labels | 2 annotators; agreement reported; unresolved → needs_review |
| Metrics | Image-level + pepper-level; confusion matrix; 95% CI by pepper |
| Honesty | No ≥90% claim before clean test; 1-NN not reported as model accuracy |

---

## 11. Open parameters for confirmation

These are proposed defaults; confirm or change them before Phase 1.

| Parameter | Proposed default |
|---|---|
| Crop padding | 20% of `max(bbox_w, bbox_h)` on each side |
| Crop output | PNG, 224×224×3, RGB `[0,255]` |
| Crop threshold | Calibrated on a sample; initial candidate 25–40 RGB distance |
| Background estimate | Border median (old images); background reference frame (deployment) |
| Validation/test allocation | 50/50 by pepper group, stratified by class |
| New train peppers | Optional; default is 473 old + new val/test only |
| Exclusion list | `dataset/review_exclusions.csv` with the columns in §3.1 |
| Capture log | `dataset/capture_log.csv` with the columns in §5.1 |
| Review of 10 images | Later; no blocker for Phase 1 |
| Crop generation timing | After plan confirmation, before pilot collection |

---

## 12. Risks and unknowns

- **Crop generalisation:** background/lighting on the conveyor may differ from the
  current pale background. The background-reference approach must be tested on the
  pilot before full collection.
- **Small collection:** if fewer than 30 peppers/class are collected, the test set
  will be small; report wide confidence intervals and do not overclaim.
- **Q2 conflict:** the user's "consecutive" memory does not match the visual check;
  do not use timestamp blocks as groups.
- **Domain shift:** current images are close-up, uniform background, 1024×1024;
  OV2640/Termux images will differ. The pilot and on-device test are mandatory.
- **Annotation labour:** two independent annotators for 3 classes × 30–100 peppers
  is significant; plan time accordingly.
- **Class imbalance:** keep class weights and report per-class metrics; do not hide
  a weak class behind overall accuracy.
- **Augmentation range:** verify the actual Keras rotation range before training.

---

## 13. What will not change before confirmation

- `dataset/raw/**` — no changes.
- `dataset/groups.csv` — no changes.
- `dataset/split_manifest.json` — no changes.
- Raw labels — no changes.
- No new images, no synthetic/Internet images.
- No training, evaluation, export, or Task 3 collection.

---

*End of plan. Implementation starts only after the user confirms this plan and the
open parameters in §11.*
