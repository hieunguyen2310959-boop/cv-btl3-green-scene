"""Fit thresholds on dev ONLY, freeze config, evaluate every held-out scene.
Real-image crops are labeled as controlled geometry experiments, not camera captures.
"""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from cv_final.segmentation import segment, overlay
from cv_final.evaluate import evaluate_segmentation
from cv_final.runner import run_scene, save_json, controlled_views

ROOT = Path(__file__).resolve().parents[1]


def load(entry):
    image = np.asarray(Image.open(ROOT / entry['image_path']).convert('RGB'))
    labels = np.asarray(Image.open(ROOT / entry['annotation_path']))
    return image, labels == 5, labels != 11


def aggregate(rows):
    if not rows:
        return {'n': 0, 'macro_iou': None, 'macro_f1': None}
    sums = {k: sum(r[k] for r in rows) for k in ['tp', 'fp', 'fn', 'tn']}
    den = sums['tp'] + sums['fp'] + sums['fn']
    return {'n': len(rows), 'macro_iou': float(np.mean([r['iou'] for r in rows])),
            'macro_f1': float(np.mean([r['f1'] for r in rows])),
            'macro_precision': float(np.mean([r['precision'] for r in rows])),
            'macro_recall': float(np.mean([r['recall'] for r in rows])),
            'micro_iou': sums['tp'] / den if den else 1.0,
            'mean_absolute_ratio_error': float(np.mean([abs(r['fp'] - r['fn']) / r['valid_pixels'] for r in rows])),
            **sums}


def panel(paths, titles, destination):
    thumb_w = 400
    canvas = Image.new('RGB', (thumb_w * len(paths), 340), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (path, title) in enumerate(zip(paths, titles)):
        im = Image.open(path).convert('RGB')
        im.thumbnail((thumb_w, 300))
        canvas.paste(im, (i * thumb_w, 34))
        draw.text((i * thumb_w + 8, 10), title, fill='black')
    canvas.save(destination)


def main():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(2310959)
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    entries = manifest['scenes']
    assert len({x['scene_id'] for x in entries}) == len(entries)
    dev = [e for e in entries if e['split'] == 'dev']
    test = [e for e in entries if e['split'] == 'test']
    assert not ({e['sequence'] for e in dev} & {e['sequence'] for e in test})
    # Verify assets against frozen provenance, not just their presence.
    for entry in entries:
        for kind in ['image', 'annotation']:
            assert hashlib.sha256((ROOT / entry[kind + '_path']).read_bytes()).hexdigest() == entry[kind + '_sha256']
    default = json.loads((ROOT / 'configs/default.json').read_text(encoding='utf-8'))
    development = [(e, *load(e)) for e in dev]
    candidates = []
    for h_min in [20, 25, 30]:
        for h_max in [85, 95]:
            for s_min in [20, 35]:
                for v_min in [20, 40]:
                    candidates.append({'method': 'hsv', 'h_min': h_min, 'h_max': h_max,
                                       's_min': s_min, 'v_min': v_min})
    candidates.extend({'method': 'exg', 'exg_min': value} for value in [.02, .05, .1, .15, .2])
    trials = []
    for parameters in candidates:
        metrics = []
        for _, image, gt, valid in development:
            pred = segment(image, np.ones(valid.shape, bool), parameters)['vegetation_mask']
            metrics.append(evaluate_segmentation(pred, gt, valid))
        trials.append({'parameters': parameters, **aggregate(metrics)})
    best = {method: max([t for t in trials if t['parameters']['method'] == method],
                        key=lambda x: x['macro_iou']) for method in ['hsv', 'exg']}
    config = deepcopy(default)
    config['alternatives'] = {m: best[m]['parameters'] for m in best}
    winner = max(best, key=lambda m: best[m]['macro_iou'])
    config['segmentation'] = config['alternatives'][winner]
    config['selection'] = {'criterion': 'dev macro IoU', 'dev_frames': len(dev),
                           'test_not_used_for_selection': True, 'winner': winner}
    # Freeze before evaluating the first test image.
    save_json(ROOT / 'configs/selected.json', config)
    config_hash = hashlib.sha256((ROOT / 'configs/selected.json').read_bytes()).hexdigest()
    results_dir = ROOT / 'reports/results'
    figures_dir = ROOT / 'reports/figures'
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_dir / 'development_trials.json', trials)
    rows, geometry_rows, controls = [], [], []
    for entry in test:
        for method in ['hsv', 'exg']:
            cfg = deepcopy(config)
            cfg['segmentation'] = cfg['alternatives'][method]
            out = ROOT / 'outputs/experiments/single' / method / entry['scene_id']
            status = run_scene(entry, cfg, out, ROOT)
            if status['status'] != 'success':
                raise RuntimeError(f"Unexpected single-scene failure: {status}")
            row = {'scene_id': entry['scene_id'], 'sequence': entry['sequence'], 'method': method,
                   'segmentation_seconds': status['timing_seconds']['segmentation'], **status['evaluation']}
            rows.append(row)
        for method in ['orb', 'sift']:
            cfg = deepcopy(config)
            cfg['geometry']['method'] = method
            crop_entry = dict(entry, controlled_crops=True)
            out = ROOT / 'outputs/experiments/crops' / method / entry['scene_id']
            status = run_scene(crop_entry, cfg, out, ROOT)
            row = {'scene_id': entry['scene_id'], 'method': method, 'status': status['status'],
                   'reason': status['reason'], 'corner_error_pixels': None, 'evaluation': status.get('evaluation')}
            if status['status'] == 'success':
                gm = status['geometry']
                transforms = [np.array(x, float) for x in gm['transforms_to_canvas']]
                image = load(entry)[0]
                views, origins = controlled_views(image)
                errors = []
                for view, x, transform in zip(views, origins, transforms):
                    h, w = view.shape[:2]
                    corners = np.array([[0., 0.], [w-1., 0.], [w-1., h-1.], [0., h-1.]])
                    rel = np.linalg.inv(transforms[0]) @ transform
                    predicted = cv2.perspectiveTransform(corners[None], rel)[0]
                    truth = corners + [x, 0]
                    errors.extend(np.linalg.norm(predicted - truth, axis=1).tolist())
                row.update(corner_error_pixels=float(np.mean(errors)), pair_metrics=gm['pair_metrics'],
                           geometry_seconds=status['timing_seconds']['geometry'])
            geometry_rows.append(row)
    # Explicit negative controls; output must remain a failure rather than identity.
    from cv_final.geometry import stitch
    for method in ['orb', 'sift']:
        cfg = dict(config['geometry'], method=method)
        status = stitch([np.zeros((100, 140, 3), np.uint8)] * 2, cfg)
        controls.append({'case': 'two_textureless_frames', 'method': method,
                         'status': status['status'], 'reason': status['reason']})
        assert status['status'] == 'failed' and status['panorama'] is None
    seg_summary = {m: aggregate([r for r in rows if r['method'] == m]) for m in ['hsv', 'exg']}
    geo_summary = {}
    for method in ['orb', 'sift']:
        subset = [r for r in geometry_rows if r['method'] == method]
        success = [r for r in subset if r['status'] == 'success']
        geo_summary[method] = {'attempted': len(subset), 'succeeded': len(success),
                               'success_rate': len(success) / len(subset),
                               'mean_corner_error_pixels': float(np.mean([r['corner_error_pixels'] for r in success])) if success else None,
                               'mean_geometry_seconds': float(np.mean([r['geometry_seconds'] for r in success])) if success else None,
                               'conditional_segmentation': aggregate([r['evaluation'] for r in success if r['evaluation']])}
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'environment': {'python': sys.version, 'platform': platform.platform(), 'opencv': cv2.__version__,
                               'numpy': np.__version__, 'opencv_threads': 1, 'seed': 2310959},
               'dataset': {'frames': len(entries), 'dev_frames': len(dev), 'test_frames': len(test),
                           'dev_sequences': sorted({e['sequence'] for e in dev}),
                           'test_sequences': sorted({e['sequence'] for e in test}),
                           'positive_class': 'Tree=5', 'ignored_class': 'Unlabelled=11'},
               'config_sha256': config_hash, 'dev_best': best, 'selected_method': winner,
               'segmentation': seg_summary, 'geometry_controlled_crops': geo_summary,
               'negative_controls': controls,
               'limitations': ['Only two held-out video sequences; frames are correlated, not 20 independent environments.',
                               'Panorama tests use overlapping crops from the same real frame, not separate camera views.',
                               'Tree class proxy, not all vegetation; no geographical area estimate.',
                               'No trained semantic model. Threshold color baselines can confuse green objects and miss shaded/yellow trees.']}
    save_json(results_dir / 'summary.json', summary)
    save_json(results_dir / 'segmentation_per_scene.json', rows)
    save_json(results_dir / 'geometry_per_scene.json', geometry_rows)
    with (results_dir / 'segmentation_per_scene.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Illustrations chosen post evaluation, never excluded from aggregate metrics.
    ordered = sorted([r for r in rows if r['method'] == winner], key=lambda r: r['iou'])
    examples = [('worst', ordered[0]), ('median', ordered[len(ordered)//2]), ('best', ordered[-1])]
    for tag, row in examples:
        sid = row['scene_id']
        hsv = ROOT / 'outputs/experiments/single/hsv' / sid
        exg = ROOT / 'outputs/experiments/single/exg' / sid
        panel([hsv/'panorama.png', hsv/'ground_truth.png', hsv/'overlay.png', exg/'overlay.png'],
              [sid+' | Input', 'Reference Tree (ignore Void)', 'HSV prediction', 'ExG prediction'], figures_dir / (tag+'.png'))
    first = test[0]['scene_id']
    for method in ['orb', 'sift']:
        out = ROOT / 'outputs/experiments/crops' / method / first
        if (out / 'panorama.png').exists():
            panel([out/'inputs.png', out/'panorama.png', out/'overlay.png'],
                  ['Controlled crops (one real frame)', method.upper()+' panorama', 'Vegetation overlay'],
                  figures_dir / ('panorama_'+method+'.png'))
    single = ROOT / 'outputs/experiments/single/hsv' / first
    panel([single/'panorama.png', single/'edges.png', single/'lines.png'],
          ['Input real scene', 'Canny edges', 'Hough lines + corners'], figures_dir / 'geometry.png')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
