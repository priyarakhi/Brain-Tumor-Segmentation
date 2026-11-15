"""Generate lossless binary PNG masks from VIA polygon annotations."""
import argparse
import json
from pathlib import Path

from brain_tumor_segmentation.data import create_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True, help='Directory with MRI slice image files')
    parser.add_argument('--annotations', type=Path, required=True, help='VIA annotations_all.json')
    parser.add_argument('--output', type=Path, default=Path('data/processed/masks'))
    args = parser.parse_args()
    result = create_masks(args.images, args.annotations, args.output)
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(f"Created {result['num_images']} PNG masks; {result['num_empty_masks']} have no annotated tumor")


if __name__ == '__main__':
    main()
