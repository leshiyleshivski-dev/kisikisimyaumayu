"""Vision checks for the Inside Track result notification."""

from __future__ import annotations

import cv2
import numpy as np
import unittest
from pathlib import Path

from food_catalog import SECRET_CAT_INDICES, SECRET_RECIPES
from kisiki.core import CATS, COMING_SOON_CATS
from kisiki.modules.race_bettor import (
    BET_BUTTON_RATIO, BET_DIALOG_RATIO, ODDS_FIRST_Y_RATIO, ODDS_X_RATIO,
    ODDS_ROW_STEP_RATIO, RACE_TOAST_ACCENT_RATIO, RaceBettorModule,
    bet_dialog_visible, classify_race_toast,
    classify_race_result, find_two_to_one_horse, parse_win_target,
    race_result_board_visible,
)


PROJECT_ROOT = Path(__file__).parents[1]
ODDS_TEMPLATE = PROJECT_ROOT / "assets" / "vision" / "race_odds_2_1.png"


def accent_image(hue: int) -> np.ndarray:
    hsv = np.zeros((120, 44, 3), dtype=np.uint8)
    hsv[30:100, 18:23] = (hue, 220, 220)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def synthetic_horse_list(width: int = 2559, height: int = 1439) -> np.ndarray:
    frame = np.full((height, width, 3), 24, dtype=np.uint8)
    template = cv2.imread(str(ODDS_TEMPLATE))
    row = 4
    centre_y = round(height * (ODDS_FIRST_Y_RATIO + row * ODDS_ROW_STEP_RATIO))
    x = round(width * ODDS_X_RATIO[0]) + max(2, round(width * 0.006))
    y = centre_y - template.shape[0] // 2
    frame[y:y + template.shape[0], x:x + template.shape[1]] = template
    return frame


def paint_dialog(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    x, y, region_width, region_height = BET_DIALOG_RATIO
    x1, x2 = round(width * x), round(width * (x + region_width))
    y1, y2 = round(height * y), round(height * (y + region_height))
    frame[y1:y2, x1:x2] = (28, 28, 28)
    line_height = max(5, round((y2 - y1) * 0.035))
    frame[y1 + line_height:y1 + line_height * 4, x1 + line_height:x2 - line_height] = (150, 150, 150)


def synthetic_win(*, with_dialog: bool = False) -> np.ndarray:
    height, width = 1439, 2559
    frame = np.full((height, width, 3), 20, dtype=np.uint8)
    frame[round(height * 0.05):round(height * 0.27), round(width * 0.27):round(width * 0.61)] = (150, 35, 135)
    x, y, region_width, region_height = RACE_TOAST_ACCENT_RATIO
    x1, x2 = round(width * x), round(width * (x + region_width))
    y1, y2 = round(height * y), round(height * (y + region_height))
    accent_x = x1 + max(2, (x2 - x1) // 2)
    frame[y1 + 20:y2 - 15, accent_x:accent_x + 5] = (80, 210, 130)
    if with_dialog:
        paint_dialog(frame)
    return frame


class RaceToastVisionTests(unittest.TestCase):
    def test_green_vertical_accent_is_recognized(self) -> None:
        self.assertEqual(classify_race_toast(accent_image(65)), "green")

    def test_red_vertical_accent_is_recognized(self) -> None:
        self.assertEqual(classify_race_toast(accent_image(175)), "red")

    def test_small_coloured_noise_is_not_a_notification(self) -> None:
        hsv = np.zeros((120, 44, 3), dtype=np.uint8)
        hsv[10:16, 8:14] = (65, 255, 255)
        hsv[90:96, 30:36] = (175, 255, 255)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.assertIsNone(classify_race_toast(image))

    def test_empty_capture_is_safe(self) -> None:
        self.assertIsNone(classify_race_toast(None))
        self.assertIsNone(classify_race_toast(np.empty((0, 0, 3), dtype=np.uint8)))

    def test_green_finish_is_not_counted_as_a_loss(self) -> None:
        frame = synthetic_win()

        self.assertTrue(race_result_board_visible(frame))
        self.assertEqual(classify_race_result(frame), "win")

    def test_coloured_notification_without_finish_board_is_ignored(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[970:1045, 1150:1160] = (30, 30, 220)

        self.assertFalse(race_result_board_visible(frame))
        self.assertIsNone(classify_race_result(frame))

    def test_persistent_dialog_does_not_hide_a_real_win(self) -> None:
        win = synthetic_win(with_dialog=True)

        self.assertTrue(bet_dialog_visible(win))
        self.assertEqual(classify_race_result(win), "win")


class RaceHorseVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = synthetic_horse_list()

    def test_2k_list_finds_fifth_horse_with_two_to_one_odds(self) -> None:
        detection = find_two_to_one_horse(self.frame)

        self.assertIsNotNone(detection)
        row, click_x, click_y, score = detection
        self.assertEqual(row, 4)
        self.assertGreater(score, 0.9)
        self.assertAlmostEqual(click_x / self.frame.shape[1], 0.160, places=2)
        self.assertAlmostEqual(click_y / self.frame.shape[0], 0.758, places=2)

    def test_full_hd_scaling_still_finds_two_to_one(self) -> None:
        full_hd = synthetic_horse_list(1920, 1080)

        detection = find_two_to_one_horse(full_hd)

        self.assertIsNotNone(detection)
        self.assertEqual(detection[0], 4)
        self.assertGreater(detection[3], 0.65)

    def test_plain_frame_is_never_clicked(self) -> None:
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)

        self.assertIsNone(find_two_to_one_horse(blank))

    def test_bet_timing_covers_random_remaining_window(self) -> None:
        self.assertEqual(RaceBettorModule.BET_WINDOW_SECONDS, 28.0)
        self.assertEqual(RaceBettorModule.BET_REMAINING_SECONDS, (15.0, 27.0))
        self.assertEqual(BET_BUTTON_RATIO, (0.369, 0.405))

    def test_dialog_is_confirmed_before_button_click(self) -> None:
        dialog = np.full((1439, 2559, 3), 110, dtype=np.uint8)
        paint_dialog(dialog)

        self.assertTrue(bet_dialog_visible(dialog))
        self.assertFalse(bet_dialog_visible(self.frame))


class RaceBettorCatalogTests(unittest.TestCase):
    def test_favorite_replaces_jackpot_placeholder(self) -> None:
        self.assertEqual(CATS[6][0], "Фаворит")
        self.assertTrue(CATS[6][2].endswith("07_race_bettor_cat.png"))
        self.assertNotIn("Джекпот", {cat[0] for cat in COMING_SOON_CATS})

    def test_favorite_recipe_is_registered(self) -> None:
        recipes = {secret_id: ingredients for secret_id, _title, ingredients in SECRET_RECIPES}
        self.assertEqual(SECRET_CAT_INDICES["race_bettor"], 6)
        self.assertEqual(recipes["race_bettor"], ("carrot", "cat_treat", "fish"))

    def test_win_target_means_wins_not_attempts(self) -> None:
        self.assertEqual(parse_win_target("4"), 4)
        self.assertEqual(parse_win_target(" 12 "), 12)
        self.assertIsNone(parse_win_target("0"))
        self.assertIsNone(parse_win_target("1000"))
        self.assertIsNone(parse_win_target("четыре"))
