"""Deterministic RGB vegetation masks using HSV or excess green."""
import numbers
import cv2
import numpy as np


def _validate_image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape[:2]):
        raise ValueError('image must be a nonempty HxWx3 RGB uint8 array')


def _validate_mask(mask, shape, name):
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or
            mask.ndim != 2 or mask.shape != shape):
        raise ValueError(f'{name} must be a matching HxW boolean array')


def segment(panorama, valid_mask, config=None):
    """Return a boolean mask, valid-support pixel ratio and resolved parameters.

    HSV uses OpenCV's 0..179 hue scale and inclusive thresholds. ExG uses
    r=R/(R+G+B), g=G/(R+G+B), b=B/(R+G+B), selecting 2*g-r-b >= exg_min;
    black pixels have undefined chromaticity and are never selected by ExG.
    Optional cleanup opens then closes with an elliptical radius-r kernel,
    clips to valid support, then drops 8-connected components smaller than
    min_component_area. Zero disables the corresponding cleanup operation.
    Inputs are never mutated. Invalid arrays/configuration raise ValueError.
    """
    _validate_image(panorama)
    _validate_mask(valid_mask, panorama.shape[:2], 'valid_mask')
    if not np.any(valid_mask):
        raise ValueError('valid_mask must contain at least one valid pixel')
    parameters = dict(method='hsv', h_min=25, h_max=95, s_min=35, v_min=25,
                      min_component_area=0, morphology_radius=0)
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError('config must be a dictionary or None')
        if set(config) - (set(parameters) | {'exg_min'}):
            raise ValueError('unknown segmentation configuration key')
        parameters.update(config)
    if not isinstance(parameters['method'], str) or parameters['method'] not in ('hsv', 'exg'):
        raise ValueError('method must be hsv or exg')
    for name, lower, upper in (('h_min', 0, 179), ('h_max', 0, 179),
                                ('s_min', 0, 255), ('v_min', 0, 255),
                                ('exg_min', -1, 2)):
        value = parameters.get(name, .05)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not lower <= value <= upper or not np.isfinite(value)):
            raise ValueError(f'{name} must be finite and in [{lower}, {upper}]')
    if parameters['h_min'] > parameters['h_max']:
        raise ValueError('h_min must not exceed h_max')
    for name in ('min_component_area', 'morphology_radius'):
        value = parameters[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if parameters['method'] == 'exg':
        parameters.setdefault('exg_min', .05)
        rgb = panorama.astype(np.float64)
        total = rgb.sum(axis=2)
        chroma = np.divide(rgb, total[..., None], out=np.zeros_like(rgb),
                           where=total[..., None] != 0)
        exg = 2 * chroma[..., 1] - chroma[..., 0] - chroma[..., 2]
        mask = (exg >= parameters['exg_min']) & (total > 0)
    else:
        hsv = cv2.cvtColor(panorama, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        mask = ((h >= parameters['h_min']) & (h <= parameters['h_max']) &
                (s >= parameters['s_min']) & (v >= parameters['v_min']))
    mask &= valid_mask
    radius = parameters['morphology_radius']
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        mask &= valid_mask
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        mask &= valid_mask
    min_area = parameters['min_component_area']
    if min_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        mask = keep[labels]
    ratio = float(np.count_nonzero(mask) / np.count_nonzero(valid_mask))
    return dict(vegetation_mask=mask, vegetation_pixel_ratio=ratio, parameters=parameters)


def overlay(image, mask, valid_mask=None, alpha=.4):
    """Blend selected valid pixels toward RGB (0, 255, 0), rounding to uint8.

    An all-false support mask is allowed and leaves the image unchanged.
    Alpha must be finite in [0, 1]; arrays follow segment's dtype/shape rules.
    """
    _validate_image(image)
    _validate_mask(mask, image.shape[:2], 'mask')
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, numbers.Real)
            or not 0 <= alpha <= 1 or not np.isfinite(alpha)):
        raise ValueError('alpha must be finite and in [0, 1]')
    if valid_mask is not None:
        _validate_mask(valid_mask, image.shape[:2], 'valid_mask')
        mask = mask & valid_mask
    result = image.copy()
    result[mask] = np.rint((1 - alpha) * image[mask].astype(np.float64) +
                           alpha * np.array([0, 255, 0])).astype(np.uint8)
    return result
