"""Feature-based RGB panorama geometry. Written independently for this project."""
import cv2
import numpy as np


def _validate_image(image, minimum=1):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < minimum or
            max(image.shape[:2]) >= 32767):
        raise ValueError(f'image must be RGB uint8, with dimensions >= {minimum} and < 32767')


def _checked_homography(H, shape):
    """Reject singular maps and poles anywhere on the source pixel footprint."""
    if np.iscomplexobj(H):
        raise ValueError('homography must be real')
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or not np.any(H):
        raise ValueError('homography must be finite, nonzero and 3x3')
    H = H / np.max(np.abs(H))  # Homographies are invariant to scale.
    singular_values = np.linalg.svd(H, compute_uv=False)
    if singular_values[-1] <= singular_values[0] * 1e-12:
        raise ValueError('singular or ill-conditioned homography')
    height, width = shape[:2]
    footprint = np.array([[-.5, -.5, 1.], [width - .5, -.5, 1.],
                          [width - .5, height - .5, 1.], [-.5, height - .5, 1.]])
    denominators = footprint @ H[2]
    tolerance = 1e-10 * np.max(np.abs(denominators))
    if not (np.all(denominators > tolerance) or np.all(denominators < -tolerance)):
        raise ValueError('homography has a projective pole on the source footprint')
    # The source origin lies inside the checked footprint, so H[2, 2] is nonzero.
    # Conventional unit bottom-right scale also avoids OpenCV's tiny-denominator
    # cutoff when projecting otherwise valid, homogeneous-equivalent matrices.
    return H / H[2, 2]


def stitch(views: list, config: dict | None = None) -> dict:
    """Stitch RGB uint8 views with ORB/SIFT+RANSAC, or return explicit failure.

    Defaults: method='orb', max_features=3000, ratio_test=.75,
    ransac_threshold=3., min_inliers=8, max_canvas_pixels=12000000,
    edge_threshold=7 and fast_threshold=7 (the latter two affect ORB only).
    Transforms map source pixel coordinates to canvas pixel coordinates.
    A single view succeeds as identity; multiple views require real registration.
    Undefined RMSE on failed pairs is None, never a fabricated zero.
    """
    try:
        return _stitch(views, config)
    except (cv2.error, MemoryError, np.linalg.LinAlgError) as exc:
        return {'status': 'failed', 'reason': f'geometry backend failure: {exc}',
                'panorama': None, 'valid_mask': None, 'transforms_to_canvas': [],
                'pair_metrics': []}


def _stitch(views, config=None):
    """Register adjacent RGB views, chaining source-to-first transforms.

    Pair metrics describe current-source -> previous-view RANSAC fits.
    Failed registrations never return partial panoramas or identity substitutes.
    """
    cfg = {'method': 'orb', 'max_features': 3000, 'ratio_test': .75,
           'ransac_threshold': 3., 'min_inliers': 8,
           'max_canvas_pixels': 12000000, 'edge_threshold': 7, 'fast_threshold': 7}
    result = {'status': 'failed', 'reason': '', 'panorama': None,
              'valid_mask': None, 'transforms_to_canvas': [], 'pair_metrics': []}
    try:
        if config is not None and not isinstance(config, dict):
            raise ValueError('config must be a dict or None')
        cfg.update(config or {})
        if cfg['method'] not in ('orb', 'sift'):
            raise ValueError('method must be orb or sift')
        for key, minimum in (('max_features', 1), ('min_inliers', 4),
                             ('max_canvas_pixels', 1), ('edge_threshold', 0), ('fast_threshold', 0)):
            value = cfg[key]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not minimum <= value <= 2147483647:
                raise ValueError(f'{key} must be an integer >= {minimum} and <= 2147483647')
            cfg[key] = int(value)
        for key in ('ratio_test', 'ransac_threshold'):
            value = cfg[key]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
            cfg[key] = float(value)
        if cfg['ratio_test'] >= 1:
            raise ValueError('ratio_test must be strictly between 0 and 1')
        if not isinstance(views, list) or not views:
            raise ValueError('views must be a nonempty list of RGB uint8 arrays')
        for image in views:
            _validate_image(image, minimum=16)
    except (ValueError, TypeError) as exc:
        result['reason'] = f'invalid input: {exc}'
        return result
    detector = (cv2.SIFT_create(nfeatures=cfg['max_features']) if cfg['method'] == 'sift'
                else cv2.ORB_create(nfeatures=cfg['max_features'],
                                    edgeThreshold=cfg['edge_threshold'], fastThreshold=cfg['fast_threshold']))
    features = [detector.detectAndCompute(cv2.cvtColor(v, cv2.COLOR_RGB2GRAY), None)
                for v in views]
    matcher = cv2.BFMatcher(cv2.NORM_L2 if cfg['method'] == 'sift' else cv2.NORM_HAMMING)
    transforms = [np.eye(3)]
    for index in range(1, len(views)):
        previous_kp, previous_desc = features[index - 1]
        current_kp, current_desc = features[index]
        metric = {'source_index': index, 'target_index': index - 1,
                  'match_count': 0, 'inlier_count': 0,
                  'inlier_ratio': 0., 'inlier_rmse': None}
        result['pair_metrics'].append(metric)
        if current_desc is None or previous_desc is None or len(previous_desc) < 2:
            result['reason'] = 'insufficient features'
            return result
        neighbors = matcher.knnMatch(current_desc, previous_desc, k=2)
        matches = [pair[0] for pair in neighbors if len(pair) == 2 and
                   pair[0].distance < cfg['ratio_test'] * pair[1].distance]
        metric['match_count'] = len(matches)
        if len(matches) < 4:
            result['reason'] = 'insufficient ratio-test matches'
            return result
        source = np.float64([current_kp[m.queryIdx].pt for m in matches])
        target = np.float64([previous_kp[m.trainIdx].pt for m in matches])
        H, inliers = cv2.findHomography(source, target, cv2.RANSAC, cfg['ransac_threshold'])
        if H is None or inliers is None:
            result['reason'] = 'homography estimation failed'
            return result
        selected = inliers.ravel().astype(bool)
        count = int(selected.sum())
        metric['inlier_count'] = count
        metric['inlier_ratio'] = count / len(matches)
        if count < cfg['min_inliers']:
            result['reason'] = 'insufficient RANSAC inliers'
            return result
        try:
            H = _checked_homography(H, views[index].shape)
            chained = _checked_homography(transforms[-1] @ H, views[index].shape)
        except (ValueError, np.linalg.LinAlgError) as exc:
            result['reason'] = f'unsafe homography: {exc}'
            return result
        projected_h = np.column_stack((source[selected], np.ones(count))) @ H.T
        projected = projected_h[:, :2] / projected_h[:, 2:]
        rmse = float(np.sqrt(np.mean(np.sum((projected - target[selected]) ** 2, axis=1))))
        if not np.isfinite(rmse):
            result['reason'] = 'nonfinite inlier reprojection error'
            return result
        metric['inlier_rmse'] = rmse
        transforms.append(chained)
    corners = []
    for image, H in zip(views, transforms):
        height, width = image.shape[:2]
        box = np.float64([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        corners.append(cv2.perspectiveTransform(box[None], H)[0])
    corners = np.concatenate(corners)
    lower = np.floor(corners.min(axis=0))
    upper = np.ceil(corners.max(axis=0))
    width, height = (int(value) for value in upper - lower + 1)
    if width < 1 or height < 1 or width * height > cfg['max_canvas_pixels'] or max(width, height) >= 32767:
        result['reason'] = 'canvas bounds exceed pixel or OpenCV dimension limit'
        return result
    translation = np.array([[1., 0., -lower[0]], [0., 1., -lower[1]], [0., 0., 1.]])
    canvas_transforms = [translation @ H for H in transforms]
    total = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    for image, H in zip(views, canvas_transforms):
        support = cv2.warpPerspective(np.ones(image.shape[:2], dtype=np.float32), H, (width, height))
        color = cv2.warpPerspective(image.astype(np.float32), H, (width, height))
        total += color
        weight += support
    panorama = np.clip(np.rint(total / np.maximum(weight[..., None], 1e-12)), 0, 255).astype(np.uint8)
    result.update(status='success', reason='ok', panorama=panorama,
                  valid_mask=weight > 0, transforms_to_canvas=canvas_transforms)
    return result


def warp_image(image, H, output_shape):
    """Warp source to canvas; output_shape is (height, width), not OpenCV order."""
    _validate_image(image)
    H = _checked_homography(H, image.shape)
    if (not isinstance(output_shape, (tuple, list)) or len(output_shape) != 2 or
            any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or
                not 1 <= v < 32767 for v in output_shape)):
        raise ValueError('output_shape must contain two positive integer dimensions < 32767')
    height, width = map(int, output_shape)
    rgb = cv2.warpPerspective(image.astype(np.float32), H, (width, height))
    support = cv2.warpPerspective(np.ones(image.shape[:2], dtype=np.float32),
                                  np.asarray(H, dtype=np.float64), (width, height))
    valid = support > 0
    rgb = np.clip(np.rint(rgb.astype(np.float64) / np.maximum(support[..., None], 1e-12)), 0, 255).astype(np.uint8)
    return rgb, valid
