"""Post-threshold binary tumor-segmentation metrics; framework-independent."""
from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute per-image metrics and dataset means.

    Dice/IoU are 1.0 if BOTH the reference and prediction are empty;
    'positive_only' scores report images with annotated tumor only.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one")
    actual, prob = np.asarray(y_true), np.asarray(y_prob)
    if actual.shape != prob.shape or actual.ndim not in (3, 4):
        raise ValueError("Expected equal-shaped (N,H,W) or (N,H,W,1) arrays")
    if actual.ndim == 4:
        if actual.shape[-1] != 1:
            raise ValueError("Binary segmentation expects one channel")
        actual, prob = actual[..., 0], prob[..., 0]
    if not np.all(np.isfinite(prob)) or not np.all((prob >= 0) & (prob <= 1)):
        raise ValueError("Predictions must be finite probabilities in [0,1]")
    if not np.all(np.isin(actual, [0, 1])):
        raise ValueError("Ground-truth masks must contain only 0/1")
    actual = actual.astype(bool)
    predicted = prob >= threshold
    axes = (1, 2)
    tp = np.sum(actual & predicted, axis=axes)
    fp = np.sum(~actual & predicted, axis=axes)
    fn = np.sum(actual & ~predicted, axis=axes)
    tn = np.sum(~actual & ~predicted, axis=axes)
    dice = np.divide(2*tp, 2*tp+fp+fn, out=np.ones_like(tp,dtype=float), where=(2*tp+fp+fn)!=0)
    iou = np.divide(tp, tp+fp+fn, out=np.ones_like(tp,dtype=float), where=(tp+fp+fn)!=0)
    pixel_accuracy = (tp + tn) / (tp + fp + fn + tn)
    pos = np.sum(actual, axis=axes) > 0
    return {
        "image_count": int(len(actual)), "positive_images": int(pos.sum()), "empty_reference_images": int((~pos).sum()),
        "threshold": float(threshold), "mean_dice": float(dice.mean()), "mean_iou": float(iou.mean()),
        "positive_only_dice": float(dice[pos].mean()) if pos.any() else None,
        "positive_only_iou": float(iou[pos].mean()) if pos.any() else None,
        "mean_pixel_accuracy": float(pixel_accuracy.mean()),
    }
