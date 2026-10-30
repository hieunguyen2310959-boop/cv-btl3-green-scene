import importlib
import numpy as np
import pytest


def test_preprocess_preserves_rgb_and_aspect_ratio():
    spec = importlib.util.find_spec('cv_final.preprocess')
    assert spec is not None, 'preprocess module missing'
    from cv_final.preprocess import preprocess_views
    image = np.full((40, 80, 3), [10, 20, 30], dtype=np.uint8)
    output = preprocess_views([image], {'max_side': 40})
    assert output[0].shape == (20, 40, 3)
    np.testing.assert_array_equal(output[0][0, 0], [10, 20, 30])
