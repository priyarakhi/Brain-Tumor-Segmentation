"""Evaluate a trained checkpoint using its saved split assignment."""
import argparse
import json
from pathlib import Path
import numpy as np

from brain_tumor_segmentation.data import list_pairs, load_arrays
from brain_tumor_segmentation.metrics import binary_metrics
from brain_tumor_segmentation.models import preprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='Training output directory')
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    args = p.parse_args()
    import tensorflow as tf
    metadata = json.loads((args.run / 'metadata.json').read_text(encoding='utf8'))
    assignments = json.loads((args.run / 'splits.json').read_text(encoding='utf8'))
    requested = assignments[args.split]
    lookup = {img.name: (img, mask) for img, mask in list_pairs(args.images, args.masks)}
    if len(requested) != len(set(requested)):
        raise ValueError('Repeated split filenames')
    missing = set(requested) - set(lookup)
    if missing:
        raise FileNotFoundError(f"Cannot find split images: {sorted(missing)[:3]}")
    images, masks = load_arrays([lookup[name] for name in requested], int(metadata['image_size']))
    model = tf.keras.models.load_model(args.run / 'best.keras', compile=False)
    predictions = model.predict(preprocess(images, metadata['architecture']), verbose=0)
    print(json.dumps(binary_metrics(masks, predictions, float(metadata['threshold'])), indent=2))


if __name__ == '__main__':
    main()
