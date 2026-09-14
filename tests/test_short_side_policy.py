"""CPU regression checks for the requested thin-object eligibility change."""
import unittest

from src.regions.consensus import size_assessment


class ShortSidePolicyTests(unittest.TestCase):
    rules = {
        "min_area_ratio": 0.00001,
        "max_area_ratio": 0.10,
        "preferred_min_area_ratio": 0.001,
        "preferred_max_area_ratio": 0.10,
    }

    def test_thin_objects_pass_in_both_orientations_without_threshold(self):
        for box in ((10, 10, 11, 110), (10, 10, 110, 11), (10, 10, 18, 310)):
            with self.subTest(box=box):
                result = size_assessment(box, 1000, 1000, self.rules)
                self.assertTrue(result["size_pass"])
                self.assertIsNone(result["size_reject_reason"])
                self.assertLess(result["bbox_short_side_px"], 32)

    def test_legacy_short_side_setting_does_not_reintroduce_filter(self):
        result = size_assessment((10, 10, 18, 310), 1000, 1000,
                                 {**self.rules, "min_short_side_px": 32})
        self.assertTrue(result["size_pass"])

    def test_minimum_area_still_applies(self):
        result = size_assessment((0, 0, 1, 1), 1000, 1000, self.rules)
        self.assertFalse(result["size_pass"])
        self.assertEqual(result["size_reject_reason"], "bbox_area_below_0_001_percent")

    def test_maximum_area_still_applies(self):
        result = size_assessment((0, 0, 800, 800), 1000, 1000, self.rules)
        self.assertFalse(result["size_pass"])
        self.assertEqual(result["size_reject_reason"], "bbox_area_above_10_percent")


if __name__ == "__main__":
    unittest.main()
