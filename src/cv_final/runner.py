"""CLI orchestration. Failures are explicit and never produce a fake panorama."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
from PIL import Image
from .preprocess import preprocess_views
from .segmentation import segment, overlay
from .evaluate import evaluate_segmentation

ARTIFACTS = ('panorama.png', 'valid_mask.png', 'vegetation_mask.png', 'overlay.png',
             'edges.png', 'lines.png', 'inputs.png', 'ground_truth.png',
             'status.json', 'config.json', 'geometry_metrics.json', 'evaluation.json')


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8', newline='\n')


def controlled_views(image):
    """Overlapping crops from ONE frame: controlled registration, not camera views."""
    w = image.shape[1]
    width = int(round(.65 * w))
    origins = [0, int(round(.175 * w)), w - width]
    return [image[:, x:x + width].copy() for x in origins], origins


def run_scene(entry, config, output_dir, base_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unknown = {p.name for p in output_dir.iterdir()} - set(ARTIFACTS)
    if unknown:
        raise ValueError('Output directory contains unowned files; use a dedicated empty directory')
    for name in ARTIFACTS:
        p = output_dir / name
        if p.is_file():
            p.unlink()
    base = Path(base_dir or '.')
    t0 = time.perf_counter()
    result = {'scene_id': entry.get('scene_id', 'custom'), 'status': 'failed', 'reason': None}
    save_json(output_dir / 'config.json', config)
    try:
        paths = entry.get('view_paths') or [entry['image_path']]
        images, hashes = [], []
        for p in paths:
            path = base / p
            hashes.append({'path': str(p), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            with Image.open(path) as im:
                images.append(np.asarray(im.convert('RGB')))
        images = preprocess_views(images, config.get('preprocess'))
        if entry.get('controlled_crops') and len(images) != 1:
            raise ValueError('Controlled crops require exactly one real source frame')
        views = controlled_views(images[0])[0] if entry.get('controlled_crops') else images
        t1 = time.perf_counter()
        if len(views) == 1:
            geometry = {'status': 'success', 'reason': None, 'panorama': views[0],
                        'valid_mask': np.ones(views[0].shape[:2], bool),
                        'transforms_to_canvas': [np.eye(3)], 'pair_metrics': []}
        else:
            from .geometry import stitch
            geometry = stitch(views, config.get('geometry'))
        t2 = time.perf_counter()
        gm = {'status': geometry['status'], 'reason': geometry['reason'],
              'pair_metrics': geometry['pair_metrics'],
              'transforms_to_canvas': [x.tolist() for x in geometry['transforms_to_canvas']],
              'input_sha256': hashes,
              'experiment_type': 'controlled_crops_from_one_real_frame' if entry.get('controlled_crops')
                                 else ('independent_views' if len(views) > 1 else 'single_frame_edges_lines')}
        save_json(output_dir / 'geometry_metrics.json', gm)
        if geometry['status'] != 'success':
            result['reason'] = geometry['reason']
            save_json(output_dir / 'status.json', result)
            return result
        pano, valid = geometry['panorama'], geometry['valid_mask']
        seg = segment(pano, valid, config.get('segmentation'))
        t3 = time.perf_counter()
        gray = cv2.cvtColor(pano, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        edges[~valid] = 0
        line_image = pano.copy()
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=25, maxLineGap=8)
        if lines is not None:
            for line in np.asarray(lines).reshape(-1, 4)[:200]:
                x1, y1, x2, y2 = map(int, line)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 70, 30), 1)
        corners = cv2.goodFeaturesToTrack(gray, 150, .02, 8, mask=valid.astype(np.uint8) * 255)
        if corners is not None:
            for corner in corners:
                cv2.circle(line_image, tuple(np.round(corner[0]).astype(int)), 2, (30, 160, 255), -1)
        arrays = {'panorama': pano, 'valid_mask': valid.astype(np.uint8) * 255,
                  'vegetation_mask': seg['vegetation_mask'].astype(np.uint8) * 255,
                  'overlay': overlay(pano, seg['vegetation_mask'], valid),
                  'edges': edges, 'lines': line_image}
        thumbs = [cv2.resize(x, (240, max(1, round(x.shape[0] * 240 / x.shape[1])))) for x in views]
        height = max(x.shape[0] for x in thumbs)
        contact = np.full((height, 240 * len(thumbs), 3), 245, np.uint8)
        for i, thumb in enumerate(thumbs):
            contact[:thumb.shape[0], i*240:(i+1)*240] = thumb
        arrays['inputs'] = contact
        metrics = None
        if entry.get('annotation_path'):
            if len(images) != 1:
                raise ValueError('Annotation must refer to a single source or controlled-crop source')
            with Image.open(base / entry['annotation_path']) as im:
                labels = np.asarray(im)
            if labels.ndim != 2:
                raise ValueError('Expected integer class-id annotation, not RGB palette rendering')
            labels = cv2.resize(labels, (images[0].shape[1], images[0].shape[0]), interpolation=cv2.INTER_NEAREST)
            if entry.get('controlled_crops'):
                h, w = valid.shape
                labels = cv2.warpPerspective(labels, geometry['transforms_to_canvas'][0], (w, h),
                                             flags=cv2.INTER_NEAREST, borderValue=255)
            gt = np.isin(labels, entry.get('positive_class_ids', [5]))
            evaluation_valid = valid & ~np.isin(labels, entry.get('ignore_class_ids', [11]) + [255])
            metrics = evaluate_segmentation(seg['vegetation_mask'], gt, evaluation_valid)
            metrics['scope'] = 'valid non-ignored pixels; Tree class only'
            arrays['ground_truth'] = overlay(pano, gt, evaluation_valid)
            save_json(output_dir / 'evaluation.json', metrics)
        for name, array in arrays.items():
            Image.fromarray(array).save(output_dir / (name + '.png'))
        result.update(status='success', reason=None, vegetation_pixel_ratio=seg['vegetation_pixel_ratio'],
                      parameters=seg['parameters'], geometry=gm, evaluation=metrics,
                      valid_pixels=int(valid.sum()), line_count=0 if lines is None else len(lines),
                      corner_count=0 if corners is None else len(corners),
                      timing_seconds={'preprocess': t1-t0, 'geometry': t2-t1, 'segmentation': t3-t2,
                                      'total_including_io': time.perf_counter()-t0})
    except (ValueError, KeyError, OSError, cv2.error) as exc:
        # Remove partial success artifacts if failure happened after geometry.
        for name in ARTIFACTS:
            if name.endswith('.png') or name == 'evaluation.json':
                (output_dir / name).unlink(missing_ok=True)
        result.update(status='failed', reason=str(exc))
    save_json(output_dir / 'status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifest.json')
    parser.add_argument('--scene', help='Scene id in manifest')
    parser.add_argument('--views', nargs='+', help='Your images left-to-right, overrides manifest')
    parser.add_argument('--controlled-crops', action='store_true', help='Test crops from one image, NOT camera views')
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--output', default='outputs/custom')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.views:
        entry = {'scene_id': 'custom', 'view_paths': args.views}
    else:
        manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        entries = [e for e in manifest['scenes'] if e['scene_id'] == args.scene]
        if not entries:
            parser.error('Scene not found; pass --scene or --views')
        entry = dict(entries[0])
    entry['controlled_crops'] = args.controlled_crops
    result = run_scene(entry, config, args.output)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0 if result['status'] == 'success' else 2


if __name__ == '__main__':
    raise SystemExit(main())
