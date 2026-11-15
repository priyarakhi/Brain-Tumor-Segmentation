"""Safe, transparent preprocessing and visualization for inference."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def prepare_image(rgb: np.ndarray, model_name: str, size: int = 256, legacy: bool = False) -> np.ndarray:
    """Input: RGB image. Original model used BGR / 255; new models use RGB."""
    if rgb is None:
        raise ValueError("An input image is required")
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected an RGB array of shape (H, W, 3)")
    if size <= 0:
        raise ValueError("size must be positive")
    resized = cv2.resize(rgb.astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA)
    if legacy:
        return (resized[..., ::-1].astype(np.float32) / 255.0)[None, ...]
    from .models import preprocess
    return preprocess(resized[None, ...], model_name)


def postprocess(probability: np.ndarray, width: int, height: int, threshold: float = 0.5) -> np.ndarray:
    """Resize probabilities to original size BEFORE thresholding; mask values 0/255."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    prob = np.squeeze(np.asarray(probability))
    if prob.ndim != 2 or not np.all(np.isfinite(prob)):
        raise ValueError("Model must output one finite, two-dimensional probability mask")
    prob = cv2.resize(prob.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return (prob >= threshold).astype(np.uint8) * 255


def make_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend RED overlay over only the predicted positive region."""
    if rgb.shape[:2] != mask.shape or not (0 <= alpha <= 1):
        raise ValueError("Image/mask dimensions disagree, or invalid alpha")
    overlay = rgb.copy().astype(np.float32)
    foreground = mask > 0
    overlay[foreground] = (1-alpha)*overlay[foreground] + alpha*np.array([255, 60, 55])
    return np.clip(overlay, 0, 255).astype(np.uint8)


def run_prediction(model, rgb: np.ndarray, model_name: str, image_size: int = 256,
                   threshold: float = 0.5, legacy: bool = False):
    """Returns original RGB, mask uint8, and RGB overlay."""
    batch = prepare_image(rgb, model_name, image_size, legacy=legacy)
    pred = model.predict(batch, verbose=0)
    if pred.shape != (1, image_size, image_size, 1):
        raise ValueError(f"Unexpected model output shape: {pred.shape}")
    mask = postprocess(pred[0], rgb.shape[1], rgb.shape[0], threshold=threshold)
    return rgb, mask, make_overlay(rgb, mask)
