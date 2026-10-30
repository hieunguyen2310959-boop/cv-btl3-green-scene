"""RGB vegetation segmentation and visualization contracts."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import cv2


def implementation():
    spec = importlib.util.spec_from_file_location("segmentation", Path(__file__).parents[1] / "src/cv_final/segmentation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SegmentationTests(unittest.TestCase):
    def test_default_hsv_separates_pure_green_from_red(self):
        image = np.array([[[0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
        result = implementation().segment(image, np.ones((1, 2), dtype=bool))
        np.testing.assert_array_equal(result['vegetation_mask'], [[True, False]])
        self.assertEqual(result['vegetation_mask'].dtype, np.bool_)
        self.assertEqual(result['vegetation_pixel_ratio'], .5)
        self.assertEqual(result['parameters'], dict(method='hsv', h_min=25, h_max=95,
                          s_min=35, v_min=25, min_component_area=0, morphology_radius=0))

    def test_invalid_pixels_are_excluded_from_mask_and_ratio(self):
        image = np.array([[[0, 255, 0], [0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
        valid = np.array([[True, False, True]])
        result = implementation().segment(image, valid)
        np.testing.assert_array_equal(result['vegetation_mask'], [[True, False, False]])
        self.assertEqual(result['vegetation_pixel_ratio'], .5)

    def test_hsv_configuration_controls_inclusive_bounds(self):
        hsv = np.array([[[25, 35, 25], [95, 35, 25], [24, 255, 255],
                         [96, 255, 255], [60, 20, 255], [60, 255, 20]]], dtype=np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        valid = np.ones(image.shape[:2], dtype=bool)
        config = dict(h_min=0, h_max=179, s_min=0, v_min=0)
        result = implementation().segment(image, valid, config)
        self.assertTrue(result['vegetation_mask'].all())
        self.assertEqual(config, dict(h_min=0, h_max=179, s_min=0, v_min=0))
        self.assertEqual(result['parameters']['h_max'], 179)

    def test_exg_uses_normalized_chromaticities(self):
        image = np.array([[[100, 110, 100], [200, 220, 200], [0, 255, 0],
                           [255, 0, 0], [0, 0, 0]]], dtype=np.uint8)
        valid = np.ones((1, 5), dtype=bool)
        result = implementation().segment(image, valid, {'method': 'exg'})
        np.testing.assert_array_equal(result['vegetation_mask'], [[True, True, True, False, False]])
        self.assertEqual(result['parameters']['exg_min'], .05)
        hsv = implementation().segment(image, valid)
        self.assertFalse(hsv['vegetation_mask'][0, 0])

    def test_empty_support_raises(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            implementation().segment(image, np.zeros((2, 2), dtype=bool))

    def test_bad_image_and_support_inputs_raise_value_error(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        valid = np.ones((2, 2), dtype=bool)
        for bad in (None, image.tolist(), image.astype(float), image[..., 0], image[..., :2], image[..., :1]):
            with self.subTest(image=type(bad)), self.assertRaises(ValueError):
                implementation().segment(bad, valid)
        for bad in (None, valid.tolist(), valid.astype(np.uint8), valid[..., None], valid[:1]):
            with self.subTest(valid=type(bad)), self.assertRaises(ValueError):
                implementation().segment(image, bad)

    def test_bad_configuration_is_rejected(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        valid = np.ones((2, 2), dtype=bool)
        bad_configs = [[], 'hsv', {'method': 'unknown'}, {'unknown': 1},
                       {'h_min': -1}, {'h_max': 180}, {'h_min': 96, 'h_max': 95},
                       {'s_min': 256}, {'v_min': -1}, {'h_min': float('nan')},
                       {'exg_min': float('inf')}, {'exg_min': -2}, {'exg_min': 3},
                       {'morphology_radius': -1}, {'morphology_radius': 1.5},
                       {'min_component_area': -1}, {'min_component_area': True},
                       {'s_min': '35'}, {'method': []}]
        for config in bad_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                implementation().segment(image, valid, config)

    def test_minimum_component_area_removes_only_small_components(self):
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        image[1:3, 1:3] = [0, 255, 0]
        image[3, 5] = [0, 255, 0]
        valid = np.ones((5, 7), dtype=bool)
        result = implementation().segment(image, valid, {'min_component_area': 4})
        expected = np.zeros_like(valid)
        expected[1:3, 1:3] = True
        np.testing.assert_array_equal(result['vegetation_mask'], expected)
        self.assertAlmostEqual(result['vegetation_pixel_ratio'], 4 / 35)

    def test_morphology_radius_removes_an_isolated_green_pixel(self):
        image = np.zeros((7, 9, 3), dtype=np.uint8)
        image[2:5, 1:4] = [0, 255, 0]
        image[3, 7] = [0, 255, 0]
        valid = np.ones((7, 9), dtype=bool)
        result = implementation().segment(image, valid, {'morphology_radius': 1})
        self.assertFalse(result['vegetation_mask'][3, 7])
        self.assertTrue(result['vegetation_mask'][3, 2])
        self.assertLess(np.count_nonzero(result['vegetation_mask']), 10)

    def test_overlay_tints_selected_pixels_green_without_mutation(self):
        image = np.array([[[100, 100, 100], [200, 100, 50]]], dtype=np.uint8)
        original = image.copy()
        result = implementation().overlay(image, np.array([[True, False]]))
        np.testing.assert_array_equal(result, [[[60, 162, 60], [200, 100, 50]]])
        np.testing.assert_array_equal(image, original)
        self.assertEqual(result.dtype, np.uint8)

    def test_overlay_never_paints_invalid_pixels(self):
        image = np.zeros((1, 3, 3), dtype=np.uint8)
        mask = np.ones((1, 3), dtype=bool)
        valid = np.array([[True, False, True]])
        result = implementation().overlay(image, mask, valid, alpha=1.)
        np.testing.assert_array_equal(result, [[[0, 255, 0], [0, 0, 0], [0, 255, 0]]])
        self.assertTrue(mask.all())

    def test_overlay_rejects_invalid_alpha(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)
        for alpha in (-.1, 1.1, float('nan'), float('inf'), '0.4', True, None):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                implementation().overlay(image, mask, alpha=alpha)

    def test_overlay_validates_image_and_matching_boolean_masks(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)
        for bad in (None, image.tolist(), image.astype(float), image[..., 0], image[..., :2]):
            with self.subTest(image=type(bad)), self.assertRaises(ValueError):
                implementation().overlay(bad, mask)
        for bad in (None, mask.tolist(), mask.astype(np.uint8), mask[..., None], mask[:1]):
            with self.subTest(mask=type(bad)), self.assertRaises(ValueError):
                implementation().overlay(image, bad)
        for bad in (mask.tolist(), mask.astype(np.uint8), mask[..., None], mask[:1]):
            with self.subTest(valid=type(bad)), self.assertRaises(ValueError):
                implementation().overlay(image, mask, valid_mask=bad)

    def test_method_must_be_a_string(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            implementation().segment(image, np.ones((1, 1), dtype=bool),
                                     {'method': np.array(['hsv'])})

    def test_no_vegetation_returns_zero_for_both_methods(self):
        image = np.array([[[255, 0, 0], [0, 0, 0], [100, 100, 100]]], dtype=np.uint8)
        valid = np.ones((1, 3), dtype=bool)
        for method in ('hsv', 'exg'):
            with self.subTest(method=method):
                result = implementation().segment(image, valid, {'method': method})
                self.assertFalse(result['vegetation_mask'].any())
                self.assertEqual(result['vegetation_pixel_ratio'], 0.)

    def test_exg_threshold_controls_output(self):
        image = np.array([[[100, 110, 100], [0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
        valid = np.ones((1, 3), dtype=bool)
        result = implementation().segment(image, valid, {'method': 'exg', 'exg_min': 1.})
        np.testing.assert_array_equal(result['vegetation_mask'], [[False, True, False]])
        self.assertEqual(result['parameters']['exg_min'], 1.)

    def test_cleanup_cannot_paint_invalid_support(self):
        image = np.full((9, 9, 3), [0, 255, 0], dtype=np.uint8)
        valid = np.ones((9, 9), dtype=bool)
        valid[4, 4] = False
        result = implementation().segment(image, valid, {'morphology_radius': 1, 'min_component_area': 2})
        self.assertFalse(result['vegetation_mask'][4, 4])
        self.assertFalse((result['vegetation_mask'] & ~valid).any())
        self.assertTrue(np.isfinite(result['vegetation_pixel_ratio']))
        self.assertGreater(result['vegetation_pixel_ratio'], 0.)

    def test_overlay_all_invalid_support_is_unchanged(self):
        image = np.full((2, 2, 3), 127, dtype=np.uint8)
        result = implementation().overlay(image, np.ones((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool))
        np.testing.assert_array_equal(result, image)

    def test_default_hsv_bounds_are_inclusive(self):
        hsv = np.array([[[25, 255, 255], [95, 255, 255], [24, 255, 255],
                         [96, 255, 255], [60, 0, 255], [60, 255, 20]]], dtype=np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        result = implementation().segment(image, np.ones((1, 6), dtype=bool))
        np.testing.assert_array_equal(result['vegetation_mask'], [[True, True, False, False, False, False]])

    def test_out_of_range_large_numbers_raise_value_error(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        valid = np.ones((1, 1), dtype=bool)
        with self.assertRaises(ValueError):
            implementation().segment(image, valid, {'h_min': 10 ** 400})
        with self.assertRaises(ValueError):
            implementation().overlay(image, valid, alpha=10 ** 400)


if __name__ == '__main__':
    unittest.main()
