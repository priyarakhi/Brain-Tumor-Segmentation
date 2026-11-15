# Original experimental workflow vs. corrected implementation

## Original experiment (as found in the supplied notebook)

1. Unzip `datas.zip` in Colab, read polygon annotations and generate binary mask files.
2. Resize images and masks to 256 × 256, normalize both by dividing by 255.
3. Make a single 80/20 **image-level** train/validation split.
4. Train (a) shallow U-Net-style CNN, (b) MobileNetV2-based DeepLabV3+,
   and (c) ResNet50 encoder–decoder with a frozen pretrained backbone.
5. Log binary cross-entropy and per-pixel accuracy, visualize a few masks.
6. Save an HDF5 U-Net model and show a Gradio demonstration.

## Specific problems corrected

- Original masks were written using their JPEG names, creating lossy categorical targets.
  **New:** rasterized masks are saved as lossless PNG (0/255), and resized nearest-neighbor.
- In the original ResNet50 section, `plot_model_history(history)`, `model.predict(...)`
  and `model.save(...)` refer to the last DeepLabV3+ model instead of
  `history_resnet_unet` / `resnet_unet_model`. **New:** explicit model selection
  is passed through a single training/evaluation pipeline; each architecture has its own checkpoint.
- The original experiments measured mainly pixel accuracy, which can remain high even
  when the model predicts background. **New:** held-out Dice and IoU are reported,
  including tumor-positive-only summaries and the number of empty reference masks.
- The original split contained no separate test partition or patient grouping.
  **New:** 70/15/15 splits and optional grouped splitting if reliable patient IDs exist.
- The original DeepLab code used fragile fixed upsampling factors and printed feature-map
  sizes that did not match the selected layers. **New:** resize each feature branch to the
  actual low-level resolution and back to input resolution.
- The original Gradio code started a public `share=True` tunnel and displayed a palette
  across the entire field. **New:** local-only demo, threshold slider, and foreground-only overlay.
- The original Colab notebook embedded installation commands, hard-coded `/content/` paths,
  repetitive training code, and legacy `.h5` saves. **New:** package and CLI scripts,
  portable paths, `.keras` checkpoints, metadata, exact split lists, and tests.

## Scope of changes

The **new U-Net architecture** has three encoder stages rather than the shallow original
network. The training setup and image preprocessing for pretrained models are also revised.
The new experiments are consequently **not exact reproductions** of the historical notebook.
No new score is asserted until training is performed on the actual dataset.

The included `models/uNet_tumor_segmentation_model.h5` is the unchanged historical
checkpoint. The `scripts/app.py` and `scripts/predict.py` commands use historical BGR/255
preprocessing when this checkpoint is selected.

## Inference limitations

Outputs are research-only binary regions, not diagnostic findings. Neither uncertainty
calibration, patient-level clinical validation, demographic performance, nor external
hospital data validation was provided. Use synthetic images for basic unit tests only;
such tests **do not establish segmentation accuracy**.
