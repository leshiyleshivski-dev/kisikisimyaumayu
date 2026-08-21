from __future__ import annotations

import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader("kiski_clicker", str(ROOT / "kiski_clicker.pyw"))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
MODULE = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(MODULE)


class CurrentGridTests(unittest.TestCase):
    def test_fixture_has_two_sources_and_one_target(self) -> None:
        image = cv2.imread(str(ROOT / "bug-fixes" / "screenshots" / "05-current-grid.png"))
        self.assertIsNotNone(image)
        module = MODULE.ElectricianModule.__new__(MODULE.ElectricianModule)
        module.capture_game_image = lambda: (image, (0, 0, image.shape[1], image.shape[0]))

        captured = module.capture_current_grid()

        self.assertIsNotNone(captured)
        _masks, terminals, _selection = captured
        self.assertEqual(
            terminals,
            {
                (1, 0, MODULE.PORT_LEFT),
                (6, 0, MODULE.PORT_LEFT),
                (5, 7, MODULE.PORT_RIGHT),
            },
        )

    def test_solver_connects_every_source(self) -> None:
        # Тройник допускает слияние двух входных веток в общий путь к выходу.
        masks = [MODULE.PORT_UP | MODULE.PORT_RIGHT | MODULE.PORT_DOWN] * 64
        terminals = {
            (1, 0, MODULE.PORT_LEFT),
            (6, 0, MODULE.PORT_LEFT),
            (4, 7, MODULE.PORT_RIGHT),
        }

        solved = MODULE.ElectricianModule.solve_current_path(masks, terminals)

        self.assertIsNotNone(solved)
        self.assertTrue(MODULE.ElectricianModule.current_grid_complete(solved, terminals))

    def test_one_connected_source_is_not_success(self) -> None:
        masks = [0] * 64
        for column in range(8):
            masks[1 * 8 + column] = MODULE.PORT_LEFT | MODULE.PORT_RIGHT
        terminals = {
            (1, 0, MODULE.PORT_LEFT),
            (6, 0, MODULE.PORT_LEFT),
            (1, 7, MODULE.PORT_RIGHT),
        }

        self.assertFalse(MODULE.ElectricianModule.current_grid_complete(masks, terminals))

    def test_complex_board_does_not_treat_cell_border_as_a_wire(self) -> None:
        image = cv2.imread(
            str(ROOT / "bug-fixes" / "screenshots" / "09-current-complex-false-branch.png")
        )
        self.assertIsNotNone(image)
        module = MODULE.ElectricianModule.__new__(MODULE.ElectricianModule)
        module.capture_game_image = lambda: (image, (0, 0, image.shape[1], image.shape[0]))

        captured = module.capture_current_grid()

        self.assertIsNotNone(captured)
        masks, terminals, selection = captured
        # На этом углу справа нет провода. Старое окно захватывало
        # металлическую границу клетки и превращало угол в тройник.
        self.assertFalse(masks[4 * 8 + 3] & MODULE.PORT_RIGHT)
        self.assertFalse(module.current_grid_complete(masks, terminals))

        solved = module.solve_current_path(masks, terminals)

        self.assertIsNotNone(solved)
        self.assertTrue(module.current_grid_complete(solved, terminals))
        self.assertGreater(len(module.build_key_plan(masks, solved, selection) or []), 0)


if __name__ == "__main__":
    unittest.main()
