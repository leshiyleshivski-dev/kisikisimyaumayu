from __future__ import annotations

import importlib.machinery
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import kisiki.modules.volt.tumblers as TUMBLERS_MODULE


ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader("kiski_clicker_tumblers", str(ROOT / "kiski_clicker.pyw"))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
MODULE = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(MODULE)


class TumblerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = cv2.imread(str(ROOT / "bug-fixes" / "screenshots" / "07-tumblers.png"))
        if cls.image is None:
            raise RuntimeError("Tumbler fixture is missing")

    def test_detects_all_red_switches_and_their_states(self) -> None:
        module = MODULE.ElectricianModule.__new__(MODULE.ElectricianModule)
        module.capture_game_image = lambda: (
            self.image,
            (0, 0, self.image.shape[1], self.image.shape[0]),
        )

        detected = module.detect_tumbler_layout()

        self.assertIsNotNone(detected)
        positions, states = detected
        self.assertEqual(len(positions), 6)
        self.assertEqual(states, [False] * 6)

        module.tumbler_positions = positions
        self.assertEqual(module.read_tumbler_switch_states(), [False] * 6)

    def test_reads_zero_needles_and_separate_red_targets(self) -> None:
        detected = MODULE.ElectricianModule.analyze_tumbler_gauges(self.image)

        self.assertIsNotNone(detected)
        values, targets = detected
        self.assertLess(values[0], 0.03)
        self.assertLess(values[1], 0.03)
        self.assertGreater(targets[0][0], 0.5)
        self.assertGreater(targets[1][0], 0.5)

    def test_layout_signature_requires_a_regular_central_panel(self) -> None:
        module = MODULE.ElectricianModule.__new__(MODULE.ElectricianModule)
        bounds = (0, 0, self.image.shape[1], self.image.shape[0])
        detected = module.detect_tumbler_layout((self.image, bounds))
        self.assertIsNotNone(detected)
        positions, _states = detected

        signature = module.tumbler_layout_signature(positions, bounds)
        jittered = [(x + (index % 2), y - (index % 3)) for index, (x, y) in enumerate(positions)]

        self.assertIsNotNone(signature)
        self.assertEqual(module.tumbler_layout_signature(jittered, bounds), signature)
        self.assertIsNone(module.tumbler_layout_signature(
            [(500, 650), (900, 670), (1180, 820), (1610, 930)], bounds,
        ))

    def test_solver_must_place_both_needles_in_their_own_ranges(self) -> None:
        selection, score = MODULE.ElectricianModule.choose_tumbler_solution(
            (0.05, 0.05),
            [(0.35, 0.0), (0.0, 0.35), (0.10, 0.10)],
            ((0.38, 0.42), (0.38, 0.42)),
        )

        self.assertEqual(selection, [0, 1])
        self.assertEqual(score, 0.0)

    def test_numeric_equality_outside_red_ranges_is_not_success(self) -> None:
        targets = ((0.38, 0.42), (0.58, 0.62))

        self.assertGreater(MODULE.ElectricianModule.tumbler_distance(0.20, targets[0]), 0.0)
        self.assertGreater(MODULE.ElectricianModule.tumbler_distance(0.20, targets[1]), 0.0)

    def test_solver_prefers_center_of_red_zones_over_fewer_switches(self) -> None:
        selection, score = MODULE.ElectricianModule.choose_tumbler_solution(
            (0.10, 0.10),
            [(0.28, 0.28), (0.15, 0.15), (0.15, 0.15)],
            ((0.38, 0.42), (0.38, 0.42)),
        )

        self.assertEqual(selection, [1, 2])
        self.assertEqual(score, 0.0)

    def test_measurement_uses_clean_baseline_and_turns_switch_back_off(self) -> None:
        module = MODULE.ElectricianModule.__new__(MODULE.ElectricianModule)
        module.running = True
        module.active = True
        module.game_window = 17
        module.process = SimpleNamespace(get=lambda: "GTA5.exe")
        module.action_job = None
        module.tumbler_positions = [(100, 100), (200, 100)]
        module.tumbler_phase = "measure_confirm"
        module.tumbler_index = 0
        module.tumbler_pending_values = (0.31, 0.42)
        module.tumbler_baseline = (0.10, 0.20)
        module.tumbler_previous = (0.29, 0.39)
        module.tumbler_effects = []
        module.tumbler_settle_retries = 0
        module.read_tumbler_gauges = lambda: (
            (0.32, 0.41), ((0.5, 0.6), (0.5, 0.6)),
        )
        clicked: list[int] = []
        module.click_tumbler = lambda index: clicked.append(index) or True
        module.after = lambda delay, callback: f"after-{delay}"

        fake_user32 = SimpleNamespace(GetForegroundWindow=lambda: 17)
        with patch.object(TUMBLERS_MODULE, "find_game_window", return_value=17), \
             patch.object(TUMBLERS_MODULE, "user32", fake_user32):
            module.run_tumbler_step()

        self.assertAlmostEqual(module.tumbler_effects[0][0], 0.215)
        self.assertAlmostEqual(module.tumbler_effects[0][1], 0.215)
        self.assertEqual(clicked, [0])
        self.assertEqual(module.tumbler_phase, "measure_off")


if __name__ == "__main__":
    unittest.main()
