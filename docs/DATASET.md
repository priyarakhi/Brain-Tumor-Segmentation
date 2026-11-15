# Dataset and annotation provenance

## What the supplied archive proves

- The original Google Colab notebook unpacks a file named `datas.zip` into `/content/mydata`, then reads
  `datas/annotations_all.json` and images from `datas/all/`.
- The annotations are stored in a VIA-style mapping with image `filename`, `regions`, and polygon arrays.
- Saved notebook output reports generating masks for `y0.jpg` through `y700.jpg` (**701 messages**).
  This is **not an independent audit of the raw archive**, which was not supplied.
- Images were resized to **256 × 256**, and the original training used an approximately 80/20
  image-level train/validation split (`random_state=42`).
- The exact source dataset name, patient identifiers, permissions, scanning modality, and license
  **cannot be confirmed** from the uploaded materials. In particular, the old README's claim
  that the data are BraTS or MRI/CT has not been verified. The screenshot depicts a brain scan,
  but does not verify provenance.

The old README listed this [Google Drive link](https://drive.google.com/file/d/1kVXx67TqTMc4IsK_E2jqSFnvMWBqZSp3/view?usp=drive_link).
Its contents and license could not be verified; treat it as an **unverified historical reference**,
not as an endorsed download or proof of dataset origin.

## New mask preparation

`python scripts/prepare_masks.py --images data/raw/datas/all --annotations data/raw/datas/annotations_all.json`

The script writes **PNG** files into `data/processed/masks/` with exactly 0 and 255 pixel values,
plus a `manifest.json`. It rejects invalid polygons, missing images, unsafe filenames,
and ambiguous basenames rather than silently skipping examples.

## Evaluation and leakage

Use `--group-regex` when filenames encode patient IDs. This creates group-disjoint train,
validation, and held-out test splits. Example (only if your filenames really have this structure):

`--group-regex '^(patient_[0-9]+)_'`

Without patient IDs, the code can only split **by image**. Multiple slices from one patient can
appear across splits, exaggerating evaluation results. Do **not** describe image-level results as
patient-independent, generalizable, or clinically validated.

Masks are resized with nearest-neighbor interpolation and checked for binary values.
Inputs and masks are aligned by basename. For newly trained models, color preprocessing is
RGB/255 for U-Net and architecture-specific preprocessing for the transfer models.
The separately supplied **legacy model** expects BGR/255 as used in the original notebook.
