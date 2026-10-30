import importlib.util
import json
import numpy as np
from PIL import Image


def test_single_scene_exports_strict_json_and_outputs(tmp_path):
    assert importlib.util.find_spec('cv_final.runner') is not None, 'runner missing'
    from cv_final.runner import run_scene
    path = tmp_path / 'cảnh xanh.png'
    image = np.full((48, 64, 3), [20, 170, 30], np.uint8)
    Image.fromarray(image).save(path)
    result = run_scene({'scene_id': 'test', 'view_paths': [str(path)]}, {}, tmp_path / 'out')
    assert result['status'] == 'success'
    for name in ['panorama', 'valid_mask', 'vegetation_mask', 'overlay', 'edges', 'lines']:
        assert (tmp_path / 'out' / (name + '.png')).is_file()
    status = json.loads((tmp_path / 'out/status.json').read_text())
    assert status['status'] == 'success'
    assert status['vegetation_pixel_ratio'] == 1.0
    json.dumps(status, allow_nan=False)


def test_json_artifact_bytes_are_platform_independent_lf(tmp_path):
    from cv_final.runner import save_json
    path = tmp_path / 'config.json'
    save_json(path, {'method': 'hsv', 'value': 1})
    expected = json.dumps({'method': 'hsv', 'value': 1}, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    assert path.read_bytes() == expected
    assert b'\r\n' not in path.read_bytes()


def test_missing_image_is_explicit_failure_without_png(tmp_path):
    from cv_final.runner import run_scene
    out = tmp_path / 'failed'
    result = run_scene({'scene_id': 'missing', 'view_paths': [str(tmp_path / 'missing.png')]}, {}, out)
    assert result['status'] == 'failed'
    assert not list(out.glob('*.png'))
    assert json.loads((out / 'status.json').read_text())['reason']


def test_rerun_failure_removes_stale_success_outputs(tmp_path):
    from cv_final.runner import run_scene
    image = tmp_path / 'image.png'
    Image.fromarray(np.full((32, 32, 3), [0, 180, 0], np.uint8)).save(image)
    out = tmp_path / 'run'
    assert run_scene({'view_paths': [str(image)]}, {}, out)['status'] == 'success'
    result = run_scene({'view_paths': [str(tmp_path / 'missing.png')]}, {}, out)
    assert result['status'] == 'failed'
    assert not list(out.glob('*.png'))
    assert not (out / 'evaluation.json').exists()


def test_output_directory_does_not_delete_unowned_files(tmp_path):
    import pytest
    from cv_final.runner import run_scene
    out = tmp_path / 'important'
    out.mkdir()
    (out / 'my-note.txt').write_text('keep')
    with pytest.raises(ValueError, match='unowned'):
        run_scene({'view_paths': ['missing.png']}, {}, out)
    assert (out / 'my-note.txt').read_text() == 'keep'


def test_annotation_tree_mapping_ignores_void(tmp_path):
    from cv_final.runner import run_scene
    image = np.full((32, 32, 3), [20, 170, 30], np.uint8)
    labels = np.full((32, 32), 5, np.uint8)
    labels[:16] = 11
    Image.fromarray(image).save(tmp_path / 'image.png')
    Image.fromarray(labels).save(tmp_path / 'label.png')
    result = run_scene({'view_paths': [str(tmp_path / 'image.png')],
                        'annotation_path': str(tmp_path / 'label.png')}, {}, tmp_path / 'run')
    assert result['status'] == 'success'
    assert result['evaluation']['valid_pixels'] == 512
    assert result['evaluation']['tp'] == 512
    assert result['evaluation']['iou'] == 1


def test_hough_visualization_handles_detected_lines(tmp_path):
    from cv_final.runner import run_scene
    image = np.zeros((100, 150, 3), np.uint8)
    image[48:52, :] = 255
    Image.fromarray(image).save(tmp_path / 'line.png')
    result = run_scene({'view_paths': [str(tmp_path / 'line.png')]}, {}, tmp_path / 'out')
    assert result['status'] == 'success'
    assert result['line_count'] > 0
    assert (tmp_path / 'out/lines.png').exists()


def test_controlled_crops_have_known_origins():
    from cv_final.runner import controlled_views
    image = np.zeros((30, 100, 3), np.uint8)
    views, origins = controlled_views(image)
    assert origins == [0, 18, 35]
    assert all(v.shape == (30, 65, 3) for v in views)
