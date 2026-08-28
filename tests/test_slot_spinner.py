"""Vision and catalog checks for the automatic slot module."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from food_catalog import SECRET_CAT_INDICES, SECRET_RECIPES
from kisiki.core import CATS, COMING_SOON_CATS
from kisiki.modules.slot_spinner import (
    SLOT_BET_PANEL_RATIO, SLOT_CONTROLS_RATIO, SLOT_TOAST_ACCENT_RATIO,
    SlotSpinnerModule, can_start_slot_spin, classify_slot_result, parse_win_target,
    slot_interface_visible, slot_win_visible,
)


def paint_ratio(
    frame: np.ndarray, ratio: tuple[float, float, float, float], color: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x, y, region_width, region_height = ratio
    x1, x2 = round(width * x), round(width * (x + region_width))
    y1, y2 = round(height * y), round(height * (y + region_height))
    frame[y1:y2, x1:x2] = color
    return x1, y1, x2, y2


def synthetic_slot(result: str | None = None) -> np.ndarray:
    frame = np.full((1440, 2560, 3), 110, dtype=np.uint8)
    for ratio in (SLOT_BET_PANEL_RATIO, SLOT_CONTROLS_RATIO):
        x1, y1, x2, y2 = paint_ratio(frame, ratio, (24, 27, 31))
        frame[y1 + 8:min(y2, y1 + 24), max(x1, x2 - 150):x2 - 10] = (230, 230, 230)
    if result is not None:
        x1, y1, x2, y2 = paint_ratio(frame, SLOT_TOAST_ACCENT_RATIO, (20, 20, 20))
        hsv_color = np.uint8([[[65 if result == "win" else 175, 220, 220]]])
        color = tuple(int(value) for value in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0])
        centre = (x1 + x2) // 2
        frame[y1 + 18:y2 - 12, centre:centre + 5] = color
    return frame


class SlotVisionTests(unittest.TestCase):
    def test_first_f9_is_not_blocked_by_an_obscured_capture(self) -> None:
        obscured = np.zeros((1080, 1920, 3), dtype=np.uint8)

        self.assertTrue(can_start_slot_spin(obscured, first_spin=True))
        self.assertFalse(can_start_slot_spin(obscured, first_spin=False))

    def test_slot_ui_is_required_before_any_result(self) -> None:
        self.assertTrue(slot_interface_visible(synthetic_slot()))
        self.assertFalse(slot_interface_visible(np.zeros((1080, 1920, 3), dtype=np.uint8)))
        self.assertIsNone(classify_slot_result(np.zeros((1080, 1920, 3), dtype=np.uint8)))

    def test_green_toast_is_a_win(self) -> None:
        frame = synthetic_slot("win")

        self.assertEqual(classify_slot_result(frame), "win")
        self.assertTrue(slot_win_visible(frame))

    def test_red_toast_is_not_a_win(self) -> None:
        frame = synthetic_slot("loss")

        self.assertEqual(classify_slot_result(frame), "loss")
        self.assertFalse(slot_win_visible(frame))

    def test_plain_slot_screen_has_no_result(self) -> None:
        self.assertIsNone(classify_slot_result(synthetic_slot()))


class SlotInputTests(unittest.TestCase):
    def test_target_counts_wins_not_spins(self) -> None:
        self.assertEqual(parse_win_target("4"), 4)
        self.assertEqual(parse_win_target(" 15 "), 15)
        self.assertIsNone(parse_win_target("0"))
        self.assertIsNone(parse_win_target("1000"))

    def test_next_spin_delay_is_random_but_compact(self) -> None:
        minimum, maximum = SlotSpinnerModule.NEXT_SPIN_DELAY_SECONDS

        self.assertGreaterEqual(minimum, 1.5)
        self.assertLessEqual(maximum, 2.5)
        self.assertLessEqual(maximum - minimum, 0.8)

    def test_slow_slot_has_a_long_result_safety_window(self) -> None:
        self.assertGreaterEqual(SlotSpinnerModule.SPIN_RESULT_TIMEOUT_SECONDS, 15.0)


class SlotCatalogTests(unittest.TestCase):
    def test_seven_replaces_spin_placeholder(self) -> None:
        self.assertEqual(CATS[7][0], "Семёрка")
        self.assertTrue(CATS[7][2].endswith("08_slot_cat.png"))
        self.assertNotIn("Спин", {cat[0] for cat in COMING_SOON_CATS})

    def test_slot_recipe_is_registered_for_seven(self) -> None:
        recipes = {secret_id: ingredients for secret_id, _title, ingredients in SECRET_RECIPES}
        self.assertEqual(SECRET_CAT_INDICES["slot_spinner"], 7)
        self.assertEqual(recipes["slot_spinner"], ("pizza", "donut", "ice_cream"))


if __name__ == "__main__":
    unittest.main()
