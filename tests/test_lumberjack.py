"""Регрессия TIMBER CUT: разбор кадра, ритм ударов и обрубка веток.

Кадры лежат в ``tests/fixtures/lumberjack/`` кропами и вставляются обратно в
чёрный кадр 2560x1440 по тому смещению, с которого сняты. Так фикстуры весят
мегабайты вместо десятков: разбор всё равно смотрит только в свои области.

Столы хранятся PNG, а не JPEG, и это не расточительность. Ветка — тонкий
отросток в десяток пикселей поперёк, и артефакты сжатия склеивают её со
стволом: на том же кадре при качестве 92 три цели превращались в две, а при
100 — всё ещё сдвигались на пиксель. Фикстура, которая ловит не тот случай,
хуже отсутствующей.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from food_catalog import SECRET_CAT_INDICES, SECRET_RECIPES
from kisiki.core import CATS, CAT_CATEGORIES, VK_F9, VK_F11
from kisiki.modules.lumberjack import LumberjackModule
from kisiki.modules.lumberjack import vision
from kisiki.modules.lumberjack import ml_vision


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "lumberjack"
FRAME_WIDTH, FRAME_HEIGHT = 2560, 1440

# Смещения, с которых сняты кропы.
TABLE_ORIGIN = (600, 300)
BAR_ORIGIN = (2202, 1318)


def load(name: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURES / name))
    if image is None:
        raise RuntimeError(f"Нет фикстуры лесоруба: {name}")
    return image


def frame_with(name: str, origin: tuple[int, int]) -> np.ndarray:
    crop = load(name)
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    left, top = origin
    frame[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return frame


def table(name: str) -> np.ndarray:
    return frame_with(name, TABLE_ORIGIN)


def bar(name: str) -> np.ndarray:
    return frame_with(name, BAR_ORIGIN)


def empty_frame() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


class Text:
    """Заглушка ``ctk.StringVar``: экран тестам не нужен, а текст — нужен."""

    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class LearnedBranchDetectorTests(unittest.TestCase):
    """The shipped ONNX cascade must remain loadable and return mask points."""

    def test_models_are_shipped_and_detect_branches(self) -> None:
        self.assertTrue(ml_vision.PRIMARY_MODEL.is_file())
        self.assertTrue(ml_vision.SENSITIVE_MODEL.is_file())
        targets = ml_vision.find_branch_targets_ml(table("day-table.png"))
        self.assertIsNotNone(targets)
        self.assertGreaterEqual(len(targets), 4)
        for target in targets:
            with self.subTest(point=(target.x, target.y)):
                self.assertEqual(target.kind, "сучок")
                self.assertTrue(0 <= target.x < FRAME_WIDTH)
                self.assertTrue(0 <= target.y < FRAME_HEIGHT)

    def test_empty_frame_has_no_learned_targets(self) -> None:
        self.assertEqual(ml_vision.find_branch_targets_ml(empty_frame()), [])


def bare_module() -> LumberjackModule:
    """Модуль без окна: логика фаз проверяется без CustomTkinter."""
    module = LumberjackModule.__new__(LumberjackModule)
    module.active = True
    module.watching = True
    module.phase = "watching"
    module.game_window = 1
    module.saved_cursor = None
    module.phase_started_at = 0.0
    module.next_action_at = 0.0
    module.last_watch_at = 0.0
    module.chops = 0
    module.chop_clicks = 0
    module.clicks_since_step = 0
    module.rhythm_paused = False
    module.paused_chops = 0
    module.ready_floor = LumberjackModule.CHOP_READY_RANGE[0]
    module.clean_streak = 0
    module.click_to_step = LumberjackModule.CLICK_TO_STEP_SECONDS
    module.last_click_at = 0.0
    module.last_step_at = 0.0
    module.step_gaps = []
    module.progress_fill = None
    module.bar_missing = 0
    module.table_missing = 0
    module.target_clicks = 0
    module.target_rounds = 0
    module.last_target_at = 0.0
    module.attempts = []
    module.last_look_at = 0.0
    module.sightings = []
    module.trees = 0
    module.branches = 0
    module.keys = {VK_F9: False, VK_F11: False}
    module.process = Text("GTA5.exe")
    module.connection = Text()
    module.stage = Text()
    module.status = Text()
    module.rhythm = Text()
    module.table_text = Text()
    module.session_text = Text()
    return module


class ChoppingBarTests(unittest.TestCase):
    """Полоса «Рубка дерева» — тот же виджет игры, что и у шахтёра."""

    def test_missing_bar_reads_as_nothing(self) -> None:
        self.assertIsNone(vision.chopping_bar_fill(bar("bar-missing.png")))
        self.assertFalse(vision.chopping_bar_visible(bar("bar-missing.png")))

    def test_empty_frame_reads_as_nothing(self) -> None:
        self.assertIsNone(vision.chopping_bar_fill(empty_frame()))

    def test_fill_grows_with_the_tree(self) -> None:
        low = vision.chopping_bar_fill(bar("bar-low.png"))
        half = vision.chopping_bar_fill(bar("bar-half.png"))
        full = vision.chopping_bar_fill(bar("bar-full.png"))
        self.assertIsNotNone(low)
        self.assertLess(low, half)
        self.assertLess(half, full)

    def test_fill_matches_the_recording(self) -> None:
        # Замерено по записи: первое дерево записи проходит эти три отметки.
        self.assertAlmostEqual(vision.chopping_bar_fill(bar("bar-low.png")), 0.106, places=2)
        self.assertAlmostEqual(vision.chopping_bar_fill(bar("bar-half.png")), 0.496, places=2)
        self.assertAlmostEqual(vision.chopping_bar_fill(bar("bar-full.png")), 0.965, places=2)

    def test_bar_is_visible_on_every_chopping_frame(self) -> None:
        for name in ("bar-low.png", "bar-half.png", "bar-full.png"):
            with self.subTest(name):
                self.assertTrue(vision.chopping_bar_visible(bar(name)))

    def test_step_of_the_bar_clears_the_threshold(self) -> None:
        """Ступенька на записи — 0,086-0,110; порог обязан быть ниже неё."""
        self.assertLess(LumberjackModule.PROGRESS_STEP, 0.086)

    def test_the_bar_lights_up_empty(self) -> None:
        """Полоса зажигается, когда игра приняла дерево, а не когда ударили.

        На дневной записи она стоит на нуле 1,3 секунды: столько проходит от
        подсказки «Рубить дерево ЛКМ» до первого засчитанного удара.
        """
        self.assertTrue(vision.chopping_bar_visible(bar("day-bar-empty.png")))
        self.assertEqual(vision.chopping_bar_fill(bar("day-bar-empty.png")), 0.0)

    def test_the_first_step_is_a_step(self) -> None:
        """Первая ступенька дневной записи — те же 8-11%, что и у ночной."""
        first = vision.chopping_bar_fill(bar("day-bar-step.png"))
        self.assertAlmostEqual(first, 0.106, places=2)
        self.assertGreater(first, LumberjackModule.PROGRESS_STEP)


class LogTableTests(unittest.TestCase):
    """Стол с бревном виден по зелёному ковру в середине экрана."""

    def test_forest_without_the_table(self) -> None:
        self.assertFalse(vision.log_table_visible(table("table-absent.png")))

    def test_empty_frame_is_not_a_table(self) -> None:
        self.assertFalse(vision.log_table_visible(empty_frame()))

    def test_every_table_fixture_is_recognised(self) -> None:
        for name in ("table-three.png", "table-fork.png", "table-last.png",
                     "table-hard.png", "table-many.png",
                     "day-table.png", "day-table-late.png"):
            with self.subTest(name):
                self.assertTrue(vision.log_table_visible(table(name)))

    def test_the_daylight_forest_is_not_a_table(self) -> None:
        """Днём зелени в кадре полно, а стола нет.

        Порог на насыщенности, а не на «зелёном вообще»: у ковра S=246, у
        листвы ивы 111. Прежний порог S=70 днём брал весь лес.
        """
        self.assertFalse(vision.log_table_visible(table("day-forest.png")))


class BranchTests(unittest.TestCase):
    """Ветка — тонкий тёплый отросток силуэта бревна."""

    def test_nothing_on_an_empty_frame(self) -> None:
        self.assertEqual(vision.find_branch_targets(empty_frame()), [])

    def test_nothing_in_the_forest(self) -> None:
        self.assertEqual(vision.find_branch_targets(table("table-absent.png")), [])

    def test_branches_are_found_on_every_table(self) -> None:
        for name in ("table-three.png", "table-fork.png", "table-last.png",
                     "table-hard.png", "table-many.png", "day-table.png"):
            with self.subTest(name):
                self.assertTrue(vision.find_branch_targets(table(name)))

    def test_counts_match_the_recordings(self) -> None:
        expected = {
            "table-three.png": 6,
            "table-fork.png": 6,
            "table-last.png": 3,
            "table-hard.png": 4,
            "table-many.png": 6,
            "day-table.png": 8,
            "day-table-late.png": 3,
        }
        for name, count in expected.items():
            with self.subTest(name):
                self.assertEqual(len(vision.find_branch_targets(table(name))), count)

    def test_the_daylight_forest_holds_no_branches(self) -> None:
        """Самая дорогая ошибка разбора: лес, принятый за стол.

        Днём под прежний порог зелени попадала листва ивы над камерой, помощник
        считал стол открытым всю прогулку и щёлкал по «веткам» в лесу. Клик по
        лесу не бесплатен: он же и открывает рубку дерева, у которого стоишь, —
        так помощник сам начинал рубку и бил в воздух.
        """
        self.assertEqual(vision.find_branch_targets(table("day-forest.png")), [])

    def test_targets_land_inside_the_table(self) -> None:
        """Клик по ветке обязан попасть в стол, а не в подсказки по краям."""
        left = round(FRAME_WIDTH * vision.TABLE_ZONE[0])
        top = round(FRAME_HEIGHT * vision.TABLE_ZONE[1])
        right = left + round(FRAME_WIDTH * vision.TABLE_ZONE[2])
        bottom = top + round(FRAME_HEIGHT * vision.TABLE_ZONE[3])
        for name in ("table-three.png", "table-many.png"):
            for target in vision.find_branch_targets(table(name)):
                with self.subTest(name=name, target=target):
                    self.assertTrue(left <= target.x < right)
                    self.assertTrue(top <= target.y < bottom)

    def test_the_lamp_is_not_a_branch(self) -> None:
        """Нога лампы стоит на ковре сама по себе, а ветка растёт из бревна.

        Держит её не порог, а само устройство разбора: дырок в ковре много,
        бревно — самая большая из них, и отростки ищутся только у неё. Нога
        лампы — своя отдельная дырка, и целью она не станет ни при каком
        пороге. Проверяется поэтому и то, и другое: что цели там нет и что
        нога в силуэт бревна не входит.
        """
        frame = table("table-three.png")
        lamp = (1002, 432)
        for target in vision.find_branch_targets(frame):
            with self.subTest(target=target):
                self.assertGreater(
                    (target.x - lamp[0]) ** 2 + (target.y - lamp[1]) ** 2, 30 ** 2,
                )
        zone, left, top = vision._zone(frame)
        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        mat = vision._mat_mask(hsv)
        log = vision._log_mask(hsv, mat, vision._mat_hull(mat))
        self.assertEqual(log[lamp[1] - top, lamp[0] - left], 0)

    def test_the_biggest_branch_comes_first(self) -> None:
        targets = vision.find_branch_targets(table("table-many.png"))
        self.assertEqual([target.score for target in targets],
                         sorted((target.score for target in targets), reverse=True))

    def test_stout_and_slim_branches_are_told_apart(self) -> None:
        kinds = {target.kind for target in vision.find_branch_targets(table("table-many.png"))}
        self.assertEqual(kinds, {"сук", "прутик"})

    def test_targets_do_not_crowd_one_branch(self) -> None:
        """Одна ветка — одна цель: развилку порог режет на два-три куска."""
        for name in ("table-three.png", "table-many.png", "table-fork.png"):
            targets = vision.find_branch_targets(table(name))
            for index, one in enumerate(targets):
                for other in targets[index + 1:]:
                    with self.subTest(name=name):
                        self.assertGreater(
                            (one.x - other.x) ** 2 + (one.y - other.y) ** 2,
                            vision.TWIG_MERGE_DISTANCE ** 2,
                        )

    def test_half_resolution_frame_gives_the_same_places(self) -> None:
        """1920x1080 приводится к эталону и отвечает теми же точками."""
        full = table("table-many.png")
        small = cv2.resize(full, (1920, 1080), interpolation=cv2.INTER_AREA)
        scaled = vision.find_branch_targets(small)
        self.assertTrue(scaled)
        for target in scaled:
            closest = min(
                vision.find_branch_targets(full),
                key=lambda other: (other.x - target.x * 4 / 3) ** 2,
            )
            self.assertLess(abs(closest.x - target.x * 4 / 3), 40)


class RhythmTests(unittest.TestCase):
    """Ритм ударов задаёт игра: считаются ступеньки полосы, а не клики."""

    def test_step_of_the_bar_counts_as_a_chop(self) -> None:
        module = bare_module()
        module.phase = "chopping"
        module.phase_started_at = 100.0
        module.progress_fill = 0.20
        with patch.object(LumberjackModule, "capture_client",
                          return_value=(bar("bar-half.png"), (0, 0, 2560, 1440))), \
             patch.object(LumberjackModule, "send_chop"):
            module.chop_step(100.0)
        self.assertEqual(module.chops, 1)
        self.assertEqual(module.stage.get(), "УДАР 1")

    def test_a_click_without_a_step_is_not_a_chop(self) -> None:
        """Клик — заявка, а не удар: ранний игра молча выбрасывает."""
        module = bare_module()
        module.phase = "chopping"
        module.phase_started_at = 100.0
        module.progress_fill = 0.49
        with patch.object(LumberjackModule, "capture_client",
                          return_value=(bar("bar-half.png"), (0, 0, 2560, 1440))), \
             patch.object(LumberjackModule, "send_chop"):
            module.chop_step(100.0)
        self.assertEqual(module.chops, 0)

    def test_missing_bar_ends_the_tree(self) -> None:
        module = bare_module()
        module.phase = "chopping"
        module.phase_started_at = 100.0
        with patch.object(LumberjackModule, "capture_client",
                          return_value=(bar("bar-missing.png"), (0, 0, 2560, 1440))):
            for tick in range(LumberjackModule.BAR_MISSING_FRAMES):
                module.last_watch_at = 0.0
                module.chop_step(100.0 + tick)
        self.assertEqual(module.phase, "awaiting_table")
        self.assertEqual(module.trees, 1)

    def test_a_missed_click_raises_the_floor(self) -> None:
        module = bare_module()
        module.chops = 3
        module.clicks_since_step = 1
        before = module.ready_floor
        with patch.object(LumberjackModule, "game_is_foreground", return_value=True), \
             patch("kisiki.modules.lumberjack.module.send_left_click", return_value=True):
            module.send_chop(100.0)
        self.assertGreater(module.ready_floor, before)

    def test_the_floor_never_climbs_past_its_ceiling(self) -> None:
        module = bare_module()
        module.chops = 3
        with patch.object(LumberjackModule, "game_is_foreground", return_value=True), \
             patch("kisiki.modules.lumberjack.module.send_left_click", return_value=True):
            for _ in range(40):
                module.clicks_since_step = 1
                module.send_chop(100.0)
        self.assertLessEqual(module.ready_floor, LumberjackModule.CHOP_READY_FLOOR_MAX)

    def test_pace_stays_under_the_ceiling_measured_on_the_recording(self) -> None:
        """Задержка плюс путь «клик -> ступенька» не должны выходить за 1,20 с."""
        top = LumberjackModule.CHOP_READY_FLOOR_MAX + LumberjackModule.CHOP_READY_SPREAD
        self.assertLessEqual(LumberjackModule.CHOP_READY_RANGE[1], 0.88)
        self.assertLessEqual(top + LumberjackModule.CLICK_TO_STEP_SECONDS, 1.35)

    def test_the_helper_does_not_click_while_gta_is_behind(self) -> None:
        module = bare_module()
        with patch.object(LumberjackModule, "game_is_foreground", return_value=False), \
             patch("kisiki.modules.lumberjack.module.send_left_click") as click:
            module.send_chop(100.0)
        click.assert_not_called()
        self.assertTrue(module.rhythm_paused)

    def test_a_paused_gap_does_not_go_into_the_rhythm(self) -> None:
        module = bare_module()
        module.last_step_at = 100.0
        module.rhythm_paused = True
        module.credit_chop(103.0)
        self.assertEqual(module.step_gaps, [])
        self.assertEqual(module.paused_chops, 1)


class CollectingTests(unittest.TestCase):
    """Обрубка веток: одна цель — один клик, а промах ничего не стоит."""

    def test_the_table_opens_the_collecting_phase(self) -> None:
        module = bare_module()
        with patch.object(LumberjackModule, "capture_client",
                          return_value=(table("table-many.png"), (0, 0, 2560, 1440))), \
             patch("kisiki.modules.lumberjack.module.cursor_position", return_value=(10, 10)):
            module.watch_step(100.0)
        self.assertEqual(module.phase, "collecting")

    def test_the_bar_opens_the_chopping_phase(self) -> None:
        """Видно полосу — рубка идёт, ровно как у Кварца в карьере."""
        module = bare_module()
        with patch.object(LumberjackModule, "capture_client",
                          return_value=(bar("bar-half.png"), (0, 0, 2560, 1440))):
            module.watch_step(100.0)
        self.assertEqual(module.phase, "chopping")

    def test_an_empty_bar_opens_the_chopping_phase_too(self) -> None:
        """Пустая полоса — тоже дерево: человеку бить самому не надо.

        Первую секунду игра эти клики выбрасывает, и это осознанная плата:
        ждать ступеньку значит просить у человека удар, которого он не просил.
        Пол задержки от выброшенных кликов не растёт — `send_chop` трогает его
        только после первой засчитанной ступеньки.
        """
        module = bare_module()
        with patch.object(LumberjackModule, "capture_client",
                          return_value=(bar("day-bar-empty.png"), (0, 0, 2560, 1440))):
            module.watch_step(100.0)
        self.assertEqual(module.phase, "chopping")
        self.assertEqual(module.chops, 0)

    def test_cold_clicks_do_not_raise_the_floor(self) -> None:
        """Клики до первой ступеньки в пустоту — не промахи по ритму."""
        module = bare_module()
        module.chops = 0
        module.clicks_since_step = 1
        before = module.ready_floor
        with patch.object(LumberjackModule, "game_is_foreground", return_value=True),              patch("kisiki.modules.lumberjack.module.send_left_click", return_value=True):
            module.send_chop(100.0)
        self.assertEqual(module.ready_floor, before)

    def test_a_clicked_branch_is_not_clicked_again_at_once(self) -> None:
        module = bare_module()
        module.remember_attempt((1000, 500), 100.0)
        self.assertFalse(module.target_ready((1000, 500), 100.1))
        self.assertTrue(module.target_ready((1000, 500), 100.6))

    def test_three_attempts_retire_a_branch(self) -> None:
        module = bare_module()
        for _ in range(LumberjackModule.MAX_ATTEMPTS_PER_TARGET):
            module.remember_attempt((1000, 500), 100.0)
        self.assertFalse(module.target_ready((1000, 500), 200.0))

    def test_a_branch_seen_and_lost_is_worth_one_more_visit(self) -> None:
        module = bare_module()
        seen = vision.BranchTarget(1500, 520, 900.0, "сук")
        module.remember_sightings([seen], 100.0)
        module.remember_sightings([seen], 100.3)
        lost = module.lost_sighting([], 101.0)
        self.assertIsNotNone(lost)
        self.assertEqual((lost.x, lost.y), (1500, 520))

    def test_one_frame_of_flicker_is_not_an_address(self) -> None:
        module = bare_module()
        module.remember_sightings([vision.BranchTarget(1500, 520, 900.0, "сук")], 100.0)
        self.assertIsNone(module.lost_sighting([], 101.0))

    def test_a_branch_gone_after_a_click_counts_as_cut(self) -> None:
        module = bare_module()
        seen = vision.BranchTarget(1500, 520, 900.0, "сук")
        module.remember_sightings([seen], 100.0)
        module.remember_sightings([seen], 100.3)
        module.remember_attempt((1500, 520), 100.4)
        self.assertIsNone(module.lost_sighting([], 101.0))

    def test_a_closed_table_sends_the_helper_back_to_waiting(self) -> None:
        module = bare_module()
        module.phase = "collecting"
        with patch.object(LumberjackModule, "capture_client",
                          return_value=(table("table-absent.png"), (0, 0, 2560, 1440))), \
             patch.object(LumberjackModule, "restore_cursor"):
            for _ in range(LumberjackModule.TABLE_MISSING_FRAMES):
                module.collect_step(100.0)
        self.assertEqual(module.phase, "watching")

    def test_a_clean_log_does_not_hand_the_table_over(self) -> None:
        """Чистое бревно — не заклинивший стол: ветки видны и не сразу."""
        module = bare_module()
        module.phase = "collecting"
        module.last_target_at = 0.0
        with patch.object(LumberjackModule, "capture_client",
                          return_value=(table("table-last.png"), (0, 0, 2560, 1440))), \
             patch.object(LumberjackModule, "game_is_foreground", return_value=True), \
             patch("kisiki.modules.lumberjack.module.find_branch_targets", return_value=[]):
            module.collect_step(1000.0)
        self.assertEqual(module.phase, "collecting")

    def test_stuck_branches_hand_the_table_over(self) -> None:
        module = bare_module()
        module.phase = "collecting"
        module.last_target_at = 0.0
        module.target_rounds = LumberjackModule.MAX_TARGET_ROUNDS
        # Все найденные цели исчерпали попытки — тогда стол возвращается человеку.
        for _ in range(LumberjackModule.MAX_ATTEMPTS_PER_TARGET):
            for target in vision.find_branch_targets(table("table-many.png")):
                module.remember_attempt((target.x, target.y), 0.0)
        with patch.object(LumberjackModule, "capture_client",
                          return_value=(table("table-many.png"), (0, 0, 2560, 1440))), \
             patch.object(LumberjackModule, "game_is_foreground", return_value=True), \
             patch("kisiki.modules.lumberjack.module.find_branch_targets",
                   return_value=vision.find_branch_targets(table("table-many.png"))), \
             patch.object(LumberjackModule, "restore_cursor"):
            module.collect_step(1000.0)
        self.assertEqual(module.phase, "manual")

    def test_no_blind_comb_clicks_when_detected_branches_end(self) -> None:
        """Без найденной маски помощник ждёт новый взгляд, а не тыкает область."""
        module = bare_module()
        module.phase = "collecting"
        module.last_target_at = 0.0
        frame = table("day-table.png")
        with patch.object(LumberjackModule, "capture_client",
                          return_value=(frame, (0, 0, 2560, 1440))),              patch.object(LumberjackModule, "game_is_foreground", return_value=True),              patch.object(LumberjackModule, "click_target") as click:
            with patch("kisiki.modules.lumberjack.module.find_branch_targets",
                       return_value=[]):
                module.collect_step(1000.0)
        click.assert_not_called()
        self.assertEqual(module.phase, "collecting")
        self.assertEqual(module.target_rounds, 1)

    def test_no_clicks_while_gta_is_behind(self) -> None:
        module = bare_module()
        module.phase = "collecting"
        with patch.object(LumberjackModule, "capture_client",
                          return_value=(table("table-many.png"), (0, 0, 2560, 1440))), \
             patch.object(LumberjackModule, "game_is_foreground", return_value=False), \
             patch.object(LumberjackModule, "click_target") as click:
            module.collect_step(100.0)
        click.assert_not_called()


class CatalogTests(unittest.TestCase):
    """Котик, категория и рецепт лесоруба."""

    def test_the_cat_sits_next_to_the_miner(self) -> None:
        name, _description, filename, _colour, category = CATS[5]
        self.assertEqual(name, "Сучок")
        self.assertTrue(filename.endswith("12_lumberjack_cat.png"))
        self.assertEqual(category, "Добывающие котики")

    def test_the_category_is_listed(self) -> None:
        self.assertIn("Добывающие котики", CAT_CATEGORIES)

    def test_the_miner_shares_the_category(self) -> None:
        # Шахтёр и лесоруб стояли категориями по одному котику, и полоса над
        # карточкой повторяла её же название. Работа у них одна, категория
        # теперь тоже — старых названий в каталоге остаться не должно.
        self.assertEqual(CATS[4][4], CATS[5][4])
        self.assertNotIn("Шахтёр", CAT_CATEGORIES)
        self.assertNotIn("Лесоруб", CAT_CATEGORIES)

    def test_the_cat_image_is_shipped(self) -> None:
        path = ROOT / "assets" / "cats" / "12_lumberjack_cat.png"
        self.assertTrue(path.exists())
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        # Котики лежат 484x484 с прозрачностью: при таком размере все экранные
        # размеры получаются кратным делением почти без остатка.
        self.assertEqual(image.shape, (484, 484, 4))

    def test_the_recipe_belongs_to_the_cat(self) -> None:
        self.assertEqual(SECRET_CAT_INDICES["lumberjack"], 5)
        recipe = next(item for item in SECRET_RECIPES if item[0] == "lumberjack")
        self.assertEqual(recipe[1], "TIMBER CUT")
        self.assertEqual(recipe[2], ("pumpkin", "salmon", "cheese"))

    def test_every_recipe_is_still_unique(self) -> None:
        pairs = [(SECRET_CAT_INDICES[secret], ingredients)
                 for secret, _title, ingredients in SECRET_RECIPES]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_every_secret_has_its_own_cat(self) -> None:
        indices = list(SECRET_CAT_INDICES.values())
        self.assertEqual(len(indices), len(set(indices)))
        for index in indices:
            self.assertLess(index, len(CATS))


class BoundaryTests(unittest.TestCase):
    """Модуль не забирает фокус и не жмёт клавиши — только мышь."""

    def test_the_module_does_not_import_keyboard_or_focus_helpers(self) -> None:
        source = (ROOT / "kisiki" / "modules" / "lumberjack" / "module.py").read_text(
            encoding="utf-8",
        )
        for forbidden in ("send_key_tap", "activate_window", "confine_cursor_to_client"):
            with self.subTest(forbidden):
                self.assertNotIn(forbidden, source)

    def test_vision_creates_no_windows_and_sends_no_input(self) -> None:
        source = (ROOT / "kisiki" / "modules" / "lumberjack" / "vision.py").read_text(
            encoding="utf-8",
        )
        for forbidden in ("customtkinter", "send_left_click", "SetCursorPos", "import tkinter"):
            with self.subTest(forbidden):
                self.assertNotIn(forbidden, source)

    def test_the_module_does_not_reach_into_other_modules(self) -> None:
        """Игровые модули видят core.py и свою папку, но не друг друга."""
        for name in ("module.py", "vision.py"):
            source = (ROOT / "kisiki" / "modules" / "lumberjack" / name).read_text(
                encoding="utf-8",
            )
            for forbidden in ("from ..miner", "from ..poker", "from ..volt",
                              "from ..blackjack"):
                with self.subTest(name=name, forbidden=forbidden):
                    self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
