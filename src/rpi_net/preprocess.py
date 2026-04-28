from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def remove_hair_artifacts(rgb: np.ndarray) -> np.ndarray:
    """Apply black-hat + inpaint hair removal similar to paper preprocessing."""
    if cv2 is None:
        return rgb

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    cleaned = cv2.inpaint(rgb, hair_mask, 3, cv2.INPAINT_TELEA)
    return cleaned


def pil_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)
