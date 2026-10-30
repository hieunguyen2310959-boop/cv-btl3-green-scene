"""RGB preprocessing; only downscale, never distort aspect ratio."""
import cv2
import numpy as np


def preprocess_views(images, config=None):
    config = config or {}
    max_side = config.get('max_side', 960)
    if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 16:
        raise ValueError('max_side must be an integer >= 16')
    if not images:
        raise ValueError('At least one image is required')
    result = []
    for image in images:
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1):
            raise ValueError('Expected non-empty RGB uint8 image')
        scale = min(1.0, max_side / max(image.shape[:2]))
        if scale < 1:
            h, w = image.shape[:2]
            image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            image = image.copy()
        result.append(image)
    return result
