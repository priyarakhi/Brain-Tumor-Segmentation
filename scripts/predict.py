"""Run a model on one image and write binary mask + overlay to files."""
import argparse
import json
from pathlib import Path

import cv2
import tensorflow as tf

from brain_tumor_segmentation.inference import run_prediction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('models/uNet_tumor_segmentation_model.h5'))
    p.add_argument('--metadata', type=Path, help='metadata.json for newly trained models; omit for legacy model')
    p.add_argument('--output', type=Path, default=Path('outputs'))
    p.add_argument('--threshold', type=float, default=None)
    args = p.parse_args()
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Image cannot be opened: {args.image}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if args.metadata:
        meta = json.loads(args.metadata.read_text(encoding='utf8'))
        arch, size, legacy = meta['architecture'], int(meta['image_size']), False
        threshold = float(meta.get('threshold', 0.5)) if args.threshold is None else args.threshold
    else:
        if args.model.name != 'uNet_tumor_segmentation_model.h5':
            p.error('--metadata is required for newly trained model checkpoints')
        arch, size, legacy = 'unet', 256, True
        threshold = 0.3 if args.threshold is None else args.threshold  # Original demo threshold
    model = tf.keras.models.load_model(str(args.model), compile=False)
    original, mask, overlay = run_prediction(model, rgb, arch, size, threshold, legacy)
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / 'predicted_mask.png'), mask)
    cv2.imwrite(str(args.output / 'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f'Saved results to {args.output} (threshold {threshold})')


if __name__ == '__main__':
    main()
