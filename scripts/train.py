"""Train a chosen architecture and report evaluation on held-out images."""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from brain_tumor_segmentation.data import list_pairs, load_arrays, split_indices
from brain_tumor_segmentation.metrics import binary_metrics
from brain_tumor_segmentation.models import build_model, preprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--model', choices=['unet', 'deeplabv3plus', 'resnet50_unet'], default='unet')
    p.add_argument('--out', type=Path, default=Path('runs/unet'))
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--group-regex', type=str, default=None,
                   help='Capture patient ID in group 1; REQUIRED for patient-level evaluation')
    p.add_argument('--no-imagenet', action='store_true', help='Initialize transfer backbones randomly')
    args = p.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        p.error('epochs and batch-size must be positive')
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    pairs = list_pairs(args.images, args.masks)
    names = [im.name for im, _ in pairs]
    splits = split_indices(names, args.seed, args.group_regex)
    images, masks = load_arrays(pairs, size=args.image_size)
    prepared = preprocess(images, args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    # Split assignment is saved before training and can be audited later.
    assignments = {part: [names[i] for i in values] for part, values in splits.items()}
    (args.out / 'splits.json').write_text(json.dumps(assignments, indent=2), encoding='utf8')
    metadata = {'architecture': args.model, 'image_size': args.image_size,
                'seed': args.seed, 'preprocessing': args.model, 'color': 'RGB',
                'imagenet_weights': not args.no_imagenet and args.model != 'unet',
                'group_regex': args.group_regex,
                'split_unit': 'patient/group' if args.group_regex else 'image (NOT patient independent)',
                'threshold': 0.5, 'source': 'new training run; not original notebook'}
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')

    model = build_model(args.model, args.image_size, imagenet_weights=not args.no_imagenet)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='pixel_accuracy')])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(args.out / 'best.keras'), save_best_only=True,
                                           monitor='val_loss', mode='min'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(args.out / 'training_history.csv')),
    ]
    history = model.fit(prepared[splits['train']], masks[splits['train']],
                        validation_data=(prepared[splits['val']], masks[splits['val']]),
                        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=2)
    # Restore exactly the checkpoint used for reporting.
    model = tf.keras.models.load_model(args.out / 'best.keras', compile=False)
    metrics = {}
    for part in ('val', 'test'):
        preds = model.predict(prepared[splits[part]], batch_size=args.batch_size, verbose=0)
        metrics[part] = binary_metrics(masks[splits[part]], preds, threshold=0.5)
    (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    print(json.dumps(metrics, indent=2))
    if not args.group_regex:
        print('WARNING: Image-level splits may mix subjects; do not claim patient-independent performance.')
    print('Saved:', args.out / 'best.keras')


if __name__ == '__main__':
    main()
