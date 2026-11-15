# Historical results — extracted from the supplied notebook

**These are historical logged image-level *validation pixel accuracies*, not new test results.**
They are not Dice, IoU, patient-independent sensitivity, or clinical evidence. The original
source dataset is not present, and the experiments have not been rerun.

| Model as trained in original notebook | Epochs recorded | Highest logged validation pixel accuracy | Epoch with highest val accuracy | Lowest recorded val loss |
|---|---:|---:|---:|---:|
| Shallow U-Net-style CNN | 17 | **95.33%** | 16 | 0.1220 (epoch 12) |
| MobileNetV2 DeepLabV3+ | 9 | **97.04%** | 4 | 0.0599 (epoch 4) |
| ResNet50 U-Net | 22 | **96.00%** | 17 | 0.1230 (epoch 17) |

These entries were read from the output logs of the three `.fit()` cells in the supplied
`Brain_Tumor_Segmentation_using_Deep_Learning.ipynb` (cells 15, 22, and 29 with
zero-based indexing). They are *per-epoch maxima*, not a common selected-checkpoint
comparison or independent test accuracy. Different model training choices and
potential class imbalance further limit direct comparison.

**Severe reproducibility issue:** In the original ResNet50 U-Net section, the prediction
and save operations use `model`, which still references DeepLabV3+, **not** the
trained `resnet_unet_model`. Therefore, the saved file originally named
`resnet50_unet_tumor_segmentation.h5` and plotted predictions cannot be assumed to
represent the ResNet50 U-Net. The uploaded archive itself contains only the genuine
`uNet_tumor_segmentation_model.h5` artifact, which is preserved here.

## What would be needed for an evidence-backed comparison

1. Permission to use the actual source scans and verified ground-truth masks.
2. Patient identifiers and a patient-disjoint split, if available.
3. Same held-out test images across the three independently trained models.
4. Per-image **Dice, IoU, positive-only Dice/IoU**, and empty-reference counts.
5. Saved model checkpoints and evaluation output with exact code/configuration versions.

The revised pipeline is designed to generate these artifacts. **Until then, no new
comparison or superiority claim is supported.**
