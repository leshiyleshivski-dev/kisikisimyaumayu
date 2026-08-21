from __future__ import annotations

import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader("kiski_clicker_maze", str(ROOT / "kiski_clicker.pyw"))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
MODULE = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(MODULE)


class MazeTests(unittest.TestCase):
    def read_fixture(self, name: str):
        image = cv2.imread(str(ROOT / "bug-fixes" / "screenshots" / name))
        self.assertIsNotNone(image)
        return image

    def assert_no_immediate_reverse(self, route: list[int]) -> None:
        opposite = {
            MODULE.VK_W: MODULE.VK_S,
            MODULE.VK_S: MODULE.VK_W,
            MODULE.VK_A: MODULE.VK_D,
            MODULE.VK_D: MODULE.VK_A,
        }
        for first, second in zip(route, route[1:]):
            self.assertNotEqual(opposite[first], second)

    def test_yellow_traversed_wires_remain_part_of_graph(self) -> None:
        route = MODULE.ElectricianModule.analyze_maze_route(self.read_fixture("01-maze-stops.png"))

        self.assertIsNotNone(route)
        self.assertEqual(len(route), 11)
        self.assert_no_immediate_reverse(route)

    def test_complex_sparse_layout_uses_only_real_contacts(self) -> None:
        route = MODULE.ElectricianModule.analyze_maze_route(self.read_fixture("03-maze-variant.png"))

        self.assertIsNotNone(route)
        self.assertEqual(len(route), 41)
        self.assert_no_immediate_reverse(route)

    def test_current_grid_is_not_misdetected_as_maze(self) -> None:
        route = MODULE.ElectricianModule.analyze_maze_route(self.read_fixture("05-current-grid.png"))

        self.assertIsNone(route)

    def test_fast_classifier_separates_maze_from_other_volt_games(self) -> None:
        self.assertTrue(
            MODULE.ElectricianModule.is_maze_screen(self.read_fixture("03-maze-variant.png"))
        )
        self.assertFalse(
            MODULE.ElectricianModule.is_maze_screen(self.read_fixture("05-current-grid.png"))
        )
        self.assertFalse(
            MODULE.ElectricianModule.is_maze_screen(self.read_fixture("07-tumblers.png"))
        )


if __name__ == "__main__":
    unittest.main()
