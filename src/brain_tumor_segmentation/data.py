"""Dataset preparation with explicit binary-mask integrity checks.

This project used a VGG Image Annotator (VIA) JSON export containing polygon
annotations keyed by image identifier. The images themselves are NOT bundled.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _safe_name(value: str) -> str:
    """Reject filenames which could escape the image/mask directories."""
    path = Path(value)
    if not value or path.name != value or value in {".", ".."} or "\\" in value:
        raise ValueError(f"Unsafe image filename in annotations: {value!r}")
    return value


def iter_via_annotations(path: str | Path) -> Iterable[tuple[str, list[dict]]]:
    """Read list- or dict-based regions from a VIA JSON annotation export."""
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Expected a top-level JSON object keyed by image ID")
    seen: set[str] = set()
    for image_info in payload.values():
        if not isinstance(image_info, dict) or "filename" not in image_info:
            raise ValueError("Expected each image entry to include 'filename'")
        name = _safe_name(str(image_info["filename"]))
        if name in seen:
            raise ValueError(f"Repeated image filename in JSON: {name}")
        seen.add(name)
        regions = image_info.get("regions", [])
        if isinstance(regions, dict):
            regions = list(regions.values())
        if not isinstance(regions, list):
            raise ValueError(f"Regions for {name} must be a list or object")
        yield name, regions


def polygon_mask(height: int, width: int, regions: list[dict]) -> np.ndarray:
    """Rasterize union of annotated polygons as a lossless {0,255} mask."""
    if height <= 0 or width <= 0:
        raise ValueError("Image width/height must be positive")
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        shape = region.get("shape_attributes", {})
        if shape.get("name") != "polygon":
            raise ValueError(f"Unsupported region shape: {shape.get('name')!r}; polygon required")
        xs, ys = shape.get("all_points_x"), shape.get("all_points_y")
        if xs is None or ys is None or len(xs) < 3 or len(xs) != len(ys):
            raise ValueError("Each polygon requires at least three paired coordinates")
        coords = np.column_stack((xs, ys)).astype(float)
        if not np.all(np.isfinite(coords)):
            raise ValueError("Non-finite polygon coordinate")
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.rint(coords).astype(np.int32)], color=255)
    return mask


def create_masks(images_dir: str | Path, annotations_file: str | Path, output_dir: str | Path) -> dict:
    """Generate PNG masks; missing input images fail rather than disappearing."""
    images_dir, output_dir = Path(images_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    used_stems: set[str] = set()
    for name, regions in iter_via_annotations(annotations_file):
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {name}")
        if Path(name).stem in used_stems:
            raise ValueError(f"Multiple image filenames share a stem: {name}")
        used_stems.add(Path(name).stem)
        image = cv2.imread(str(images_dir / name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read annotated image: {images_dir / name}")
        height, width = image.shape[:2]
        mask = polygon_mask(height, width, regions)
        target = output_dir / (Path(name).stem + ".png")
        if not cv2.imwrite(str(target), mask):
            raise OSError(f"Failed to write mask: {target}")
        manifest.append({"image": name, "mask": target.name, "height": height,
                         "width": width, "foreground_pixels": int(np.count_nonzero(mask))})
    if not manifest:
        raise ValueError("No annotations were found")
    return {"num_images": len(manifest), "num_empty_masks": sum(v["foreground_pixels"] == 0 for v in manifest),
            "masks": manifest}


def list_pairs(images_dir: str | Path, masks_dir: str | Path) -> list[tuple[Path, Path]]:
    """Match images with lossless PNG masks by basename, rejecting missing pairs."""
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError("Both images and masks directories must exist")
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images in {images_dir}")
    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        raise ValueError("Image basenames are not unique")
    pairs = [(p, masks_dir / (p.stem + ".png")) for p in images]
    missing = [str(mask) for _, mask in pairs if not mask.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing PNG masks for {len(missing)} images, e.g. {missing[0]}")
    return pairs


def load_arrays(pairs: list[tuple[Path, Path]], size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB uint8 images and binary float32 masks; nearest resize for masks."""
    if size <= 0 or size % 32 != 0:
        raise ValueError("Image size must be a positive multiple of 32")
    images, masks = [], []
    for image_path, mask_path in pairs:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or gray is None:
            raise ValueError(f"Unreadable image or mask: {image_path}, {mask_path}")
        if bgr.shape[:2] != gray.shape:
            raise ValueError(f"Unaligned raw image/mask: {image_path.name}")
        raw_values = set(np.unique(gray).tolist())
        if not raw_values.issubset({0, 255}):
            raise ValueError(f"Raw PNG mask {mask_path.name} must be binary (0/255)")
        rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        resized_mask = cv2.resize(gray, (size, size), interpolation=cv2.INTER_NEAREST)
        unique = set(np.unique(resized_mask).tolist())
        if not unique.issubset({0, 255}):
            raise ValueError(f"Mask {mask_path.name} must contain only 0 and 255; got {sorted(unique)[:10]}")
        images.append(rgb)
        masks.append((resized_mask > 0).astype(np.float32)[..., None])
    return np.stack(images), np.stack(masks)


def split_indices(names: list[str], seed: int = 42, group_regex: str | None = None,
                  test_fraction: float = 0.15, val_fraction: float = 0.15) -> dict[str, list[int]]:
    """Split into disjoint train/validation/test indices; optional subject-level grouping.

    A group_regex should capture one patient/subject ID in the first capture group.
    Without it, leakage between scans from one subject cannot be excluded.
    """
    if len(names) < 7:
        raise ValueError("At least seven images required for 3-way split")
    if test_fraction <= 0 or val_fraction <= 0 or test_fraction + val_fraction >= 1:
        raise ValueError("Invalid validation/test fractions")
    ids = np.arange(len(names))
    if group_regex:
        pattern = re.compile(group_regex)
        matches = [pattern.search(Path(name).stem) for name in names]
        if any(m is None or m.lastindex is None for m in matches):
            raise ValueError("group_regex must match every filename and contain a capture group")
        groups = np.array([m.group(1) for m in matches])
        if len(set(groups)) < 3:
            raise ValueError("Grouped splitting requires at least 3 distinct patient IDs")
        first = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        pool, test = next(first.split(ids, groups=groups))
        second = GroupShuffleSplit(n_splits=1, test_size=val_fraction / (1-test_fraction), random_state=seed)
        ti, vi = next(second.split(ids[pool], groups=groups[pool]))
        train, val = pool[ti], pool[vi]
    else:
        pool, test = train_test_split(ids, test_size=test_fraction, random_state=seed)
        train, val = train_test_split(pool, test_size=val_fraction/(1-test_fraction), random_state=seed)
    result = {"train": sorted(train.tolist()), "val": sorted(val.tolist()), "test": sorted(test.tolist())}
    if any(not v for v in result.values()):
        raise ValueError("Split produced an empty partition; adjust fractions/groups")
    assert len(set(result['train']+result['val']+result['test'])) == len(names)
    return result
