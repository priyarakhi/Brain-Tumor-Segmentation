# Historical U-Net checkpoint: model card

**Artifact:** `uNet_tumor_segmentation_model.h5` (supplied in the original archive)

- Format: legacy Keras HDF5 (`.h5`); saved with Keras 3.8.0
- Serialized layer count: 11
- Serialized model input: `[None, 256, 256, 3]` (256 × 256 RGB-shaped tensor; original training **BGR** order)
- File size: 4,954,480 bytes
- SHA-256: `5971016f2d6f5a8a2fe58dd6dba1ec59235cdea72f9e76dd425629fe203bc51a`

## Provenance and purpose

The checkpoint was supplied by the author along with a Colab notebook and a historical
Gradio screenshot. The notebook constructs a shallow U-Net-style CNN, trains using BGR
images normalized to [0, 1], and saves it under this filename. There is no raw training
dataset or independent labeled test set in the uploaded archive.

The checkpoint is **not** identical to the revised `build_model('unet')` network; it remains
an unchanged historical artifact. It can be used for an educational prediction demo with
`scripts/predict.py` or `scripts/app.py`. By default, the legacy threshold is 0.30.

## Verification status

The HDF5 structure, serialized metadata, expected input shape, and checksum were inspected.
No new sensitivity, specificity, Dice, IoU, or clinical-generalization claim is made.

## Limitations

Research-only. Not for diagnosis or clinical decision making. Input data provenance,
participant consent, license, demographic coverage, external validity, and robust performance
remain unverified. Confirm redistribution rights **before** publishing this checkpoint.
