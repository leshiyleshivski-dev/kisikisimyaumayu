from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from food_catalog import SECRET_CAT_INDICES, SECRET_RECIPES
from kisiki.core import CATS
from kisiki.modules.miner import MinerModule
from kisiki.modules.miner_vision import (
    ORE_TYPES, find_ore_targets, is_supported_2k,
    is_supported_miner_resolution, miner_overlay_visible,
    mining_progress_visible, read_ore_notification,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "miner"
FRAME_HEIGHT, FRAME_WIDTH = 1439, 2559


def load_color(name: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURES / name))
    if image is None:
        raise RuntimeError(f"Missing miner fixture: {name}")
    return image


def overlay_frame(name: str) -> np.ndarray:
    crop = load_color(name)
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    left, top = round(FRAME_WIDTH * 0.30), round(FRAME_HEIGHT * 0.20)
    frame[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return frame


def toast_frame(name: str) -> np.ndarray:
    crop = load_color(name)
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    left, top = round(FRAME_WIDTH * 0.412), round(FRAME_HEIGHT * 0.935)
    frame[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return frame


def has_target_near(targets, expected: tuple[int, int], radius: int = 36) -> bool:
    x, y = expected
    return any((target[0] - x) ** 2 + (target[1] - y) ** 2 <= radius ** 2 for target in targets)


class MinerVisionTests(unittest.TestCase):
    def test_2k_and_full_hd_are_supported(self) -> None:
        self.assertTrue(is_supported_2k(2560, 1440))
        self.assertTrue(is_supported_2k(2559, 1439))
        self.assertFalse(is_supported_2k(1920, 1080))
        self.assertTrue(is_supported_miner_resolution(1920, 1080))
        self.assertTrue(is_supported_miner_resolution(1919, 1079))
        self.assertFalse(is_supported_miner_resolution(1600, 900))

    def test_full_hd_frames_are_normalized_for_detection_and_ocr(self) -> None:
        overlay = cv2.resize(
            overlay_frame("overlay-123608.jpg"), (1920, 1080),
            interpolation=cv2.INTER_AREA,
        )
        self.assertTrue(miner_overlay_visible(overlay))
        targets = find_ore_targets(overlay)
        # Resampling a 2K fixture through Full HD softens the tiniest sprites,
        # but the stable inclusions must survive normalization and map back to
        # Full-HD client coordinates.
        for point in ((1092, 471), (972, 507), (1144, 525)):
            self.assertTrue(has_target_near(targets, point, radius=28), (point, targets))

        toast = cv2.resize(
            toast_frame("toast-silicon.png"), (1920, 1080),
            interpolation=cv2.INTER_AREA,
        )
        present, ore_key, _score = read_ore_notification(toast)
        self.assertTrue(present)
        self.assertEqual(ore_key, "silicon")

    def test_overlay_and_colored_inclusions_are_detected(self) -> None:
        frame = overlay_frame("overlay-123608.jpg")

        self.assertTrue(miner_overlay_visible(frame))
        targets = find_ore_targets(frame)

        for point in ((1455, 628), (1296, 676), (1471, 762)):
            self.assertTrue(has_target_near(targets, point), point)

    def test_mining_progress_bar_is_a_distinct_start_signal(self) -> None:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        left = round(FRAME_WIDTH * 0.82)
        top = round(FRAME_HEIGHT * 0.89)
        cv2.rectangle(frame, (left + 180, top + 52), (left + 405, top + 58), (170, 95, 27), -1)

        self.assertTrue(mining_progress_visible(frame))
        self.assertFalse(mining_progress_visible(np.zeros_like(frame)))

    def test_neutral_inclusions_are_kept_as_fallback_targets(self) -> None:
        frame = overlay_frame("overlay-123454.jpg")

        targets = find_ore_targets(frame)

        self.assertTrue(has_target_near(targets, (1368, 502)))
        self.assertTrue(has_target_near(targets, (1304, 567)))
        self.assertTrue(any(target[3] == "neutral" for target in targets))

    def test_large_black_inclusions_from_live_bug_report_are_detected(self) -> None:
        frame = load_color("overlay-live-black.png")

        targets = find_ore_targets(frame)

        for point in ((1013, 489), (1133, 584), (994, 645)):
            self.assertTrue(has_target_near(targets, point, radius=20), (point, targets))
        self.assertEqual(len(targets), 3)

    def test_final_inclusion_at_right_rock_edge_is_detected(self) -> None:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        def hsv_color(hue: int, saturation: int, value: int) -> tuple[int, int, int]:
            pixel = np.uint8([[[hue, saturation, value]]])
            blue, green, red = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
            return int(blue), int(green), int(red)

        cv2.rectangle(
            frame, (760, 280), (1840, 1120), hsv_color(160, 120, 135), -1,
        )
        cv2.ellipse(
            frame, (1530, 640), (170, 210), 0, 0, 360,
            hsv_color(105, 100, 145), -1,
        )
        expected = (1655, 640)
        cv2.circle(frame, expected, 16, hsv_color(20, 210, 220), -1)

        targets = find_ore_targets(frame)

        self.assertEqual(len(targets), 1, targets)
        self.assertTrue(has_target_near(targets, expected, radius=12), targets)

    def test_known_notification_words_are_classified(self) -> None:
        for filename, expected in (
            ("toast-chrome.png", "chrome"),
            ("toast-silicon.png", "silicon"),
            ("toast-manganese.png", "manganese"),
        ):
            present, ore_key, score = read_ore_notification(toast_frame(filename))
            self.assertTrue(present, filename)
            self.assertEqual(ore_key, expected, (filename, score))
            self.assertLess(score, 0.18)


class MinerIntegrationTests(unittest.TestCase):
    def test_strikes_use_a_human_sized_random_interval(self) -> None:
        minimum, maximum = MinerModule.STRIKE_INTERVAL_RANGE_SECONDS

        self.assertGreaterEqual(minimum, 0.9)
        self.assertLessEqual(maximum, 1.3)
        self.assertGreater(maximum, minimum)

    def test_visible_target_can_be_retried_but_not_spammed_forever(self) -> None:
        module = MinerModule.__new__(MinerModule)
        module.target_history = []

        point = (1100, 600)
        self.assertTrue(module._target_ready(point, 10.0))
        self.assertEqual(module._remember_target_attempt(point, 10.0), 1)
        self.assertFalse(module._target_ready(point, 10.2))
        self.assertTrue(module._target_ready(point, 10.6))
        self.assertEqual(module._remember_target_attempt(point, 10.6), 2)
        self.assertEqual(module._remember_target_attempt(point, 11.2), 3)
        self.assertFalse(module._target_ready(point, 20.0))

    def test_successful_cycle_returns_to_auto_watch_instead_of_stopping(self) -> None:
        module = MinerModule.__new__(MinerModule)
        module.running = True
        module.phase = "waiting_result"
        module.restore_cursor = Mock()
        module.stage = Mock()
        module.status = Mock()
        module.main_button = Mock()

        module.finish("Железная руда записана.")

        self.assertTrue(module.running)
        self.assertEqual(module.phase, "watching")
        module.restore_cursor.assert_called_once()
        module.stage.set.assert_called_once_with("АВТОДЕТЕКТ ВКЛЮЧЁН")

    def test_e_arms_an_unbounded_strike_cycle_while_watching(self) -> None:
        module = MinerModule.__new__(MinerModule)
        module.running = True
        module.phase = "watching"
        module._foreground_is_game = Mock(return_value=True)
        module.saved_cursor = None
        module.strike_count = 17
        module.target_attempts = 4
        module.target_history = [(1, 2, 1, 0.0)]
        module.overlay_missing_frames = 3
        module.stage = Mock()
        module.status = Mock()

        with patch("kisiki.modules.miner.cursor_position", return_value=(25, 40)):
            module._arm_cycle()

        self.assertEqual(module.phase, "arming")
        self.assertEqual(module.strike_count, 0)
        self.assertEqual(module.target_history, [])
        self.assertEqual(module.saved_cursor, (25, 40))

    def test_arming_waits_for_progress_bar_before_striking(self) -> None:
        module = MinerModule.__new__(MinerModule)
        module.phase = "arming"
        module.phase_started_at = 10.0
        module.next_action_at = 10.0
        module.game_window = 99
        module._foreground_is_game = Mock(return_value=True)
        module.capture_game_image = Mock(return_value=(
            np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8),
            (0, 0, FRAME_WIDTH, FRAME_HEIGHT),
        ))
        module.stop = Mock()
        module.stage = Mock()
        module.status = Mock()

        module._arming_step(10.5)

        self.assertEqual(module.phase, "arming")
        module.stop.assert_not_called()

    def test_arming_enters_striking_only_after_progress_bar(self) -> None:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        left = round(FRAME_WIDTH * 0.82)
        top = round(FRAME_HEIGHT * 0.89)
        cv2.rectangle(frame, (left + 180, top + 52), (left + 405, top + 58), (170, 95, 27), -1)
        module = MinerModule.__new__(MinerModule)
        module.phase = "arming"
        module.phase_started_at = 10.0
        module.next_action_at = 10.0
        module.game_window = 99
        module._foreground_is_game = Mock(return_value=True)
        module.capture_game_image = Mock(return_value=(
            frame, (0, 0, FRAME_WIDTH, FRAME_HEIGHT),
        ))
        module.stop = Mock()
        module.stage = Mock()
        module.status = Mock()

        module._arming_step(10.5)

        self.assertEqual(module.phase, "striking")
        self.assertEqual(module.next_action_at, 10.5)
        module.stop.assert_not_called()

    def test_every_ore_has_a_transparent_statistics_icon(self) -> None:
        for index, (ore_key, _title) in enumerate(ORE_TYPES, start=1):
            path = ROOT / "assets" / "ores" / f"{index:02d}_{ore_key}.png"
            icon = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(icon, path)
            self.assertEqual(icon.shape, (64, 64, 4), path)
            self.assertEqual(int(icon[:, :, 3].min()), 0, path)
            self.assertGreater(int(icon[:, :, 3].max()), 0, path)

    def test_fifth_cat_and_recipe_are_registered(self) -> None:
        self.assertEqual(CATS[4][0], "Кварц")
        self.assertEqual(CATS[4][4], "Шахтёр")
        self.assertEqual(SECRET_CAT_INDICES["miner"], 4)
        recipe = next(item for item in SECRET_RECIPES if item[0] == "miner")
        self.assertEqual(recipe[2], ("kibble", "burger", "milk"))

    def test_daily_miner_statistics_reset_on_a_new_date(self) -> None:
        module = MinerModule.__new__(MinerModule)
        module.daily_stats = {
            "date": "2000-01-01", "total": 12, "unknown": 2,
            "ores": {"iron": 10},
        }
        module.on_stats_change = None

        changed = module.ensure_daily_stats(notify=False)

        self.assertTrue(changed)
        self.assertEqual(module.daily_stats["total"], 0)
        self.assertEqual(module.daily_stats["unknown"], 0)
        self.assertTrue(all(value == 0 for value in module.daily_stats["ores"].values()))


if __name__ == "__main__":
    unittest.main()
