"""Geometry contracts; deterministic synthetic texture is a geometry fixture only."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'src' / 'cv_final' / 'geometry.py'

def geometry():
    if not MODULE.exists():
        raise AssertionError('geometry module has not been implemented')
    spec = importlib.util.spec_from_file_location('geometry_under_test', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class GeometryTests(unittest.TestCase):
    def test_identity_warp_preserves_every_pixel_including_black_and_edges(self):
        image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        image[1, 1] = 0
        warped, valid = geometry().warp_image(image, np.eye(3), (4, 5))
        np.testing.assert_array_equal(warped, image)
        self.assertEqual(valid.dtype, np.bool_)
        self.assertTrue(valid.all())

    def test_textureless_views_fail_without_identity_fallback(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        module = geometry()
        self.assertTrue(hasattr(module, 'stitch'), 'stitch is not implemented')
        result = module.stitch([image, image.copy()])
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['reason'])
        self.assertIsNone(result['panorama'])
        self.assertIsNone(result['valid_mask'])
        self.assertEqual(result['transforms_to_canvas'], [])

    def test_textured_crops_register_with_source_to_canvas_transforms(self):
        # Synthetic geometry fixture; not a camera panorama or scene benchmark.
        rng = np.random.default_rng(7103)
        scene = rng.integers(0, 256, (260, 680, 3), dtype=np.uint8)
        views = [scene[:, :360].copy(), scene[:, 150:510].copy(),
                 scene[15:245, 300:660].copy()]
        result = geometry().stitch(views)
        self.assertEqual(result['status'], 'success', result['reason'])
        self.assertEqual(result['panorama'].dtype, np.uint8)
        self.assertEqual(result['valid_mask'].dtype, np.bool_)
        self.assertEqual(result['panorama'].shape[:2], result['valid_mask'].shape)
        self.assertEqual(len(result['transforms_to_canvas']), 3)
        self.assertEqual(len(result['pair_metrics']), 2)
        origin = result['transforms_to_canvas'][0]
        for index, offset in enumerate(((0, 0), (150, 0), (300, 15))):
            relative = np.linalg.inv(origin) @ result['transforms_to_canvas'][index]
            points = np.array([[40., 40., 1.], [180., 180., 1.]])
            projected = (relative @ points.T).T
            projected = projected[:, :2] / projected[:, 2:]
            np.testing.assert_allclose(projected, points[:, :2] + offset, atol=1.5)
        for metric in result['pair_metrics']:
            self.assertGreaterEqual(metric['match_count'], metric['inlier_count'])
            self.assertGreaterEqual(metric['inlier_count'], 8)
            self.assertGreater(metric['inlier_ratio'], 0)
            self.assertLessEqual(metric['inlier_ratio'], 1)
            self.assertTrue(np.isfinite(metric['inlier_rmse']))
        self.assertGreater(result['valid_mask'].sum(), 160000)

    def test_canvas_pixel_limit_rejects_before_canvas_allocation(self):
        from unittest.mock import patch
        image = np.random.default_rng(1).integers(0, 256, (160, 200, 3), dtype=np.uint8)
        module = geometry()
        # Allocation guard is instrumentation, not a substitute for real matching.
        original = np.zeros
        def guarded(shape, *args, **kwargs):
            if isinstance(shape, tuple) and len(shape) >= 2 and shape[0] * shape[1] > 1000:
                raise AssertionError('canvas allocated before checking limit')
            return original(shape, *args, **kwargs)
        with patch.object(module.np, 'zeros', side_effect=guarded):
            result = module.stitch([image, image.copy()], {'max_canvas_pixels': 1000})
        self.assertEqual(result['status'], 'failed')
        self.assertIn('canvas', result['reason'])
        self.assertIsNone(result['panorama'])

    def test_stitch_bad_inputs_fail_with_explicit_reasons(self):
        valid = np.zeros((32, 40, 3), dtype=np.uint8)
        cases = [(None, None), ([], None), ([valid[:, :, 0]], None),
                 ([valid.astype(float)], None), ([valid[:15]], None),
                 ([None], None), ([valid], []), ([valid], {'method': 'bogus'}),
                 ([valid], {'ratio_test': np.nan}), ([valid], {'ratio_test': 1}),
                 ([valid], {'max_features': 0}), ([valid], {'min_inliers': 3}),
                 ([valid], {'max_canvas_pixels': -1}),
                 ([valid], {'ransac_threshold': float('inf')}),
                 ([valid], {'edge_threshold': -1}), ([valid], {'fast_threshold': 1.5})]
        for views, config in cases:
            with self.subTest(config=config, views_type=type(views)):
                try:
                    result = geometry().stitch(views, config)
                except Exception as exc:
                    self.fail(f'invalid stitch input escaped as {type(exc).__name__}: {exc}')
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['reason'])
                self.assertIsNone(result['panorama'])

    def test_sift_uses_real_sift_detector_for_translated_views(self):
        from unittest.mock import patch
        module = geometry()
        scene = np.random.default_rng(28).integers(0, 256, (180, 300, 3), dtype=np.uint8)
        with patch.object(module.cv2, 'SIFT_create', wraps=module.cv2.SIFT_create) as factory:
            result = module.stitch([scene[:, :220], scene[:, 70:]], {'method': 'sift'})
        self.assertTrue(factory.called, 'method=sift did not select SIFT')
        self.assertEqual(result['status'], 'success', result['reason'])
        self.assertTrue(np.isfinite(result['pair_metrics'][0]['inlier_rmse']))

    def test_orb_small_view_thresholds_are_configurable(self):
        from unittest.mock import patch
        module = geometry()
        image = np.random.default_rng(6).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        for config, expected in (({}, (7, 7)), ({'edge_threshold': 5, 'fast_threshold': 3}, (5, 3))):
            with self.subTest(config=config):
                with patch.object(module.cv2, 'ORB_create', wraps=module.cv2.ORB_create) as factory:
                    result = module.stitch([image, image.copy()], config)
                self.assertEqual(factory.call_args.kwargs.get('edgeThreshold'), expected[0])
                self.assertEqual(factory.call_args.kwargs.get('fastThreshold'), expected[1])
                self.assertEqual(result['status'], 'success', result['reason'])

    def test_warp_rejects_unsafe_projective_geometry(self):
        image = np.zeros((24, 30, 3), dtype=np.uint8)
        bad = [np.zeros((3, 3)), np.diag([1., 0., 1.]),
               np.full((3, 3), np.nan), np.full((3, 3), np.inf),
               np.eye(2), np.array([[1., 0., 0.], [0., 1., 0.], [.1, 0., -1.]])]
        for H in bad:
            with self.subTest(H=H):
                with self.assertRaises(ValueError):
                    geometry().warp_image(image, H, (24, 30))
        for shape in ((0, 3), (3, -1), (3.5, 2), (2,), (1, 32767)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    geometry().warp_image(image, np.eye(3), shape)

    def test_stitch_rejects_pathological_estimator_outputs(self):
        from unittest.mock import patch
        image = np.random.default_rng(44).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        cases = [None, np.zeros((3, 3)), np.diag([1., 0., 1.]),
                 np.full((3, 3), np.nan), np.full((3, 3), np.inf),
                 np.array([[1., 0., 0.], [0., 1., 0.], [.1, 0., -1.]]),
                 np.array([[1., 0., 1e10], [0., 1., 0.], [0., 0., 1.]])]
        for H in cases:
            with self.subTest(H=H):
                module = geometry()
                # Only the estimator is fault-injected; detection/matching are real.
                def estimate(source, target, *args):
                    return H, np.ones((len(source), 1), dtype=np.uint8)
                with patch.object(module.cv2, 'findHomography', side_effect=estimate):
                    try:
                        result = module.stitch([image, image.copy()])
                    except Exception as exc:
                        self.fail(f'unsafe homography escaped as {type(exc).__name__}: {exc}')
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['reason'])
                self.assertIsNone(result['panorama'])
                self.assertEqual(result['transforms_to_canvas'], [])

    def test_warp_fractional_support_does_not_darken_valid_pixels(self):
        image = np.ones((1, 1, 3), dtype=np.uint8)
        H = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
        rgb, mask = geometry().warp_image(image, H, (2, 2))
        self.assertTrue(mask.all())
        np.testing.assert_array_equal(rgb, np.ones((2, 2, 3), dtype=np.uint8))

    def test_identity_warp_handles_single_row_column_and_pixel(self):
        for shape in ((1, 5, 3), (6, 1, 3), (1, 1, 3)):
            with self.subTest(shape=shape):
                image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
                rgb, mask = geometry().warp_image(image, np.eye(3) * 1e-20, shape[:2])
                np.testing.assert_array_equal(rgb, image)
                self.assertTrue(mask.all())

    def test_single_view_returns_full_support_identity_success(self):
        image = np.zeros((16, 21, 3), dtype=np.uint8)
        image[-1, -1] = [255, 20, 1]
        result = geometry().stitch([image])
        self.assertEqual(result['status'], 'success', result['reason'])
        np.testing.assert_array_equal(result['panorama'], image)
        self.assertTrue(result['valid_mask'].all())
        np.testing.assert_array_equal(result['transforms_to_canvas'][0], np.eye(3))
        self.assertEqual(result['pair_metrics'], [])

    def test_stitch_backend_failure_is_explicit_not_a_crash(self):
        from unittest.mock import patch
        module = geometry()
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        with patch.object(module.cv2, 'ORB_create', side_effect=module.cv2.error('backend failure')):
            try:
                result = module.stitch([image, image.copy()])
            except Exception as exc:
                self.fail(f'backend failure escaped: {exc}')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('backend', result['reason'])
        self.assertIsNone(result['panorama'])

    def test_reverse_order_translated_views_fit_inside_canvas(self):
        scene = np.random.default_rng(17).integers(0, 256, (180, 480, 3), dtype=np.uint8)
        views = [scene[:, 120:].copy(), scene[:, :360].copy()]
        result = geometry().stitch(views)
        self.assertEqual(result['status'], 'success', result['reason'])
        height, width = result['valid_mask'].shape
        for view, H in zip(views, result['transforms_to_canvas']):
            h, w = view.shape[:2]
            corners = np.array([[0., 0., 1.], [w - 1., 0., 1.],
                                [w - 1., h - 1., 1.], [0., h - 1., 1.]]) @ H.T
            points = corners[:, :2] / corners[:, 2:]
            self.assertTrue((points >= -1e-6).all())
            self.assertTrue((points <= [width - 1 + 1e-6, height - 1 + 1e-6]).all())
        self.assertGreaterEqual(result['transforms_to_canvas'][0][0, 2], 119)

    def test_unrelated_textures_do_not_fabricate_registration_success(self):
        rng = np.random.default_rng(203)
        views = [rng.integers(0, 256, (160, 200, 3), dtype=np.uint8) for _ in range(2)]
        result = geometry().stitch(views)
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['panorama'])

if __name__ == '__main__':
    unittest.main()
