"""Synthetic tests: data invariants and arithmetic, not trained-model accuracy."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from brain_tumor_segmentation.data import (
    create_masks, iter_via_annotations, list_pairs, load_arrays, polygon_mask, split_indices
)
from brain_tumor_segmentation.inference import prepare_image, postprocess, make_overlay, run_prediction
from brain_tumor_segmentation.metrics import binary_metrics


def test_polygon_mask_is_binary_and_clips():
    regions = [{'shape_attributes': {'name': 'polygon',
                                     'all_points_x': [-4, 6, 6], 'all_points_y': [0, 0, 6]}}]
    mask = polygon_mask(8, 8, regions)
    assert mask.shape == (8, 8)
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert np.count_nonzero(mask) > 0


def test_invalid_polygon_fails():
    with pytest.raises(ValueError, match='three'):
        polygon_mask(10, 10, [{'shape_attributes': {'name': 'polygon',
                                                    'all_points_x': [1, 2], 'all_points_y': [1, 2]}}])


def test_annotation_to_lossless_masks_and_load(tmp_path):
    imgs = tmp_path / 'images'
    imgs.mkdir()
    gray = np.full((18, 24, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(imgs/'y0.jpg'), gray)
    annotations = {'key0': {'filename': 'y0.jpg', 'regions': [{'shape_attributes': {
        'name': 'polygon', 'all_points_x': [1, 13, 8], 'all_points_y': [1, 5, 15]}}]}}
    ann = tmp_path / 'annotations.json'
    ann.write_text(json.dumps(annotations), encoding='utf8')
    out = tmp_path / 'masks'
    result = create_masks(imgs, ann, out)
    assert result['num_images'] == 1 and result['num_empty_masks'] == 0
    raw = cv2.imread(str(out/'y0.png'), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(raw).tolist()) == {0, 255}
    pairs = list_pairs(imgs, out)
    images, masks = load_arrays(pairs, size=64)
    assert images.shape == (1, 64, 64, 3) and images.dtype == np.uint8
    assert masks.shape == (1, 64, 64, 1) and set(np.unique(masks).tolist()) == {0.0, 1.0}


def test_dangerous_filename_rejected(tmp_path):
    ann = tmp_path/'ann.json'
    ann.write_text(json.dumps({'x': {'filename': '../private.jpg', 'regions': []}}))
    with pytest.raises(ValueError, match='Unsafe'):
        list(iter_via_annotations(ann))


def test_missing_mask_is_error(tmp_path):
    im, ma = tmp_path/'images', tmp_path/'masks'
    im.mkdir(); ma.mkdir()
    cv2.imwrite(str(im/'a.png'), np.zeros((8,8,3), np.uint8))
    with pytest.raises(FileNotFoundError, match='Missing PNG'):
        list_pairs(im, ma)


def test_image_split_is_complete_and_reproducible():
    names = [f'x{i}.jpg' for i in range(40)]
    a = split_indices(names, seed=42)
    b = split_indices(names, seed=42)
    assert a == b
    assert sorted(a['train'] + a['val'] + a['test']) == list(range(40))
    assert all(a.values())


def test_patient_group_split_does_not_leak():
    names = [f'patient_{i}_slice_{j}.jpg' for i in range(15) for j in range(2)]
    splits = split_indices(names, group_regex=r'^(patient_\d+)_')
    ids = {part: {names[i].split('_slice')[0] for i in values} for part, values in splits.items()}
    assert not ids['train'] & ids['val']
    assert not ids['train'] & ids['test']
    assert not ids['val'] & ids['test']


def test_dice_iou_perfect_overlap():
    truth = np.array([[[0, 1],[1,0]]], dtype=np.float32)[...,None]
    result = binary_metrics(truth, truth)
    assert result['mean_dice'] == 1.0
    assert result['positive_only_iou'] == 1.0
    assert result['mean_pixel_accuracy'] == 1.0


def test_dice_iou_partial_overlap():
    actual = np.array([[[1, 1],[0,0]]], np.float32)
    predicted = np.array([[[1,0],[1,0]]], np.float32)
    result = binary_metrics(actual, predicted)
    assert result['mean_dice'] == pytest.approx(0.5)
    assert result['mean_iou'] == pytest.approx(1 / 3)


def test_empty_mask_metrics_are_explicit():
    truth = np.zeros((2, 3, 3, 1), np.float32)
    pred = truth.copy()
    metrics = binary_metrics(truth, pred)
    assert metrics['empty_reference_images'] == 2
    assert metrics['positive_only_dice'] is None
    assert metrics['mean_dice'] == 1.0


def test_foreground_overlay_only_colors_mask():
    original = np.ones((2, 2, 3), dtype=np.uint8) * 120
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 1] = 255
    merged = make_overlay(original, mask)
    assert (merged[1, 0] == original[1, 0]).all()
    assert (merged[0, 1] != original[0, 1]).any()


def test_probability_is_thresholded_only_after_resizing():
    prob = np.array([[[0.0], [1.0]], [[0.0], [1.0]]], dtype=np.float32)
    mask = postprocess(prob, width=10, height=10, threshold=0.5)
    assert mask.shape == (10, 10)
    assert set(np.unique(mask).tolist()) == {0, 255}


def test_legacy_preprocessing_matches_original_bgr_div_255():
    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    ready = prepare_image(rgb, 'unet', size=1, legacy=True)
    np.testing.assert_allclose(ready[0,0,0], [30/255, 20/255, 10/255])


class FakeModel:
    def predict(self, x, verbose=0):
        return np.full((1, x.shape[1], x.shape[2], 1), 0.9, dtype=np.float32)


def test_prediction_uses_mask_and_preserves_original_size():
    image = np.full((40, 48, 3), 127, dtype=np.uint8)
    orig, mask, overlay = run_prediction(FakeModel(), image, 'unet', 64, legacy=True)
    assert orig.shape == overlay.shape == (40,48,3)
    assert mask.shape == (40,48) and mask.min() == 255


def test_invalid_probability_rejected():
    truth = np.zeros((1,2,2), dtype=np.float32)
    probs = np.full_like(truth, np.nan)
    with pytest.raises(ValueError, match='finite'):
        binary_metrics(truth, probs)


def test_nonbinary_png_mask_rejected(tmp_path):
    images, masks = tmp_path/'images', tmp_path/'masks'
    images.mkdir(); masks.mkdir()
    cv2.imwrite(str(images/'one.jpg'), np.full((32,32,3), 50, dtype=np.uint8))
    cv2.imwrite(str(masks/'one.png'), np.full((32,32), 128, dtype=np.uint8))
    with pytest.raises(ValueError, match='binary'):
        load_arrays(list_pairs(images, masks), 64)
