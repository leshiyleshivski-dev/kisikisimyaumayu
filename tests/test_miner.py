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
    ORE_TYPES, TARGET_SEARCH_RATIO, _ratio_crop, _stones_from, find_ore_targets,
    is_supported_2k, is_supported_miner_resolution, miner_overlay_visible,
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


def overlay_frame_png(name: str) -> np.ndarray:
    """Same as :func:`overlay_frame`, for crops kept lossless.

    JPEG re-encoding of a night frame is enough to invent an extra inclusion,
    so tables that are checked for a false-target count are stored as PNG.
    """
    return overlay_frame(name)


def toast_frame(name: str) -> np.ndarray:
    crop = load_color(name)
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    left, top = round(FRAME_WIDTH * 0.412), round(FRAME_HEIGHT * 0.935)
    frame[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return frame


def progress_frame(fill: float = 0.0) -> np.ndarray:
    """Frame with only the blue «Добыча руды» bar at the bottom right."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    left, top = round(FRAME_WIDTH * 0.82), round(FRAME_HEIGHT * 0.89)
    # The unlit part of the track is a muted blue; the lit head is much lighter.
    cv2.rectangle(frame, (left + 180, top + 52), (left + 405, top + 58), (110, 70, 25), -1)
    if fill > 0:
        # The lit head of the track, drawn over the same strip.
        head = left + 180 + round(225 * fill)
        cv2.rectangle(frame, (left + 180, top + 52), (head, top + 58), (245, 190, 120), -1)
    return frame


def empty_frame() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def striking_module(frame: np.ndarray, *, now: float) -> MinerModule:
    """A module parked in the strike phase right after a swing."""
    module = MinerModule.__new__(MinerModule)
    module.phase = "striking"
    module.phase_started_at = now - 12.0
    # A strike has just been sent, so the next one is a full interval away.
    module.next_action_at = now + 1.0
    module.last_capture_at = 0.0
    module.strike_count = 12
    module.strike_clicks = 12
    module.progress_confirmed = True
    module.progress_missing_frames = 0
    module.progress_fill = None
    module.result_allows_table = False
    module.saved_cursor = None
    module.game_window = 99
    module._foreground_is_game = Mock(return_value=True)
    module.capture_game_image = Mock(return_value=(
        frame, (0, 0, FRAME_WIDTH, FRAME_HEIGHT),
    ))
    module.stop = Mock()
    module.finish = Mock()
    module.stage = Mock()
    module.status = Mock()
    return module


RUG_COLOUR = (160, 120, 135)
STONE_COLOUR = (105, 100, 145)
WARM_ORE = (20, 210, 220)


def hsv_colour(hue: int, saturation: int, value: int) -> tuple[int, int, int]:
    pixel = np.uint8([[[hue, saturation, value]]])
    blue, green, red = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(blue), int(green), int(red)


def rug_frame() -> np.ndarray:
    """Empty sorting rug at 2K, without any stone on it."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(frame, (760, 280), (1840, 1120), hsv_colour(*RUG_COLOUR), -1)
    return frame


def stone(frame: np.ndarray, center: tuple[int, int], axes: tuple[int, int],
          *, colour: tuple[int, int, int] = STONE_COLOUR) -> None:
    cv2.ellipse(frame, center, axes, 0, 0, 360, hsv_colour(*colour), -1)


def inclusion(frame: np.ndarray, center: tuple[int, int],
              colour: tuple[int, int, int], radius: int = 16) -> None:
    cv2.circle(frame, center, radius, hsv_colour(*colour), -1)


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
        frame = progress_frame()

        self.assertTrue(mining_progress_visible(frame))
        self.assertFalse(mining_progress_visible(np.zeros_like(frame)))

    def test_neutral_inclusions_are_kept_as_fallback_targets(self) -> None:
        frame = overlay_frame("overlay-123454.jpg")

        targets = find_ore_targets(frame)

        self.assertTrue(has_target_near(targets, (1368, 502)))
        self.assertTrue(has_target_near(targets, (1304, 567)))
        # Dim, colourless ore must come from a rock-relative rule, never from
        # the bright warm/cyan masks.
        self.assertTrue(all(target[3] in ("neutral", "ore") for target in targets), targets)

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

    def test_local_contrast_finds_ore_outside_absolute_hsv_ranges(self) -> None:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        def hsv_color(hue: int, saturation: int, value: int) -> tuple[int, int, int]:
            pixel = np.uint8([[[hue, saturation, value]]])
            blue, green, red = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
            return int(blue), int(green), int(red)

        cv2.rectangle(
            frame, (760, 280), (1840, 1120), hsv_color(160, 120, 135), -1,
        )
        cv2.ellipse(
            frame, (1350, 620), (250, 220), 0, 0, 360,
            hsv_color(105, 100, 145), -1,
        )
        expected = (1410, 620)
        # H/S/V deliberately misses warm, cyan and neutral absolute masks.
        cv2.circle(frame, expected, 18, hsv_color(105, 48, 96), -1)

        targets = find_ore_targets(frame)

        self.assertEqual(len(targets), 1, targets)
        self.assertEqual(targets[0][3], "adaptive")
        self.assertTrue(has_target_near(targets, expected, radius=10), targets)

    def test_adaptive_fallback_does_not_click_a_plain_rock(self) -> None:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        def hsv_color(hue: int, saturation: int, value: int) -> tuple[int, int, int]:
            pixel = np.uint8([[[hue, saturation, value]]])
            blue, green, red = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
            return int(blue), int(green), int(red)

        cv2.rectangle(
            frame, (760, 280), (1840, 1120), hsv_color(160, 120, 135), -1,
        )
        cv2.ellipse(
            frame, (1350, 620), (250, 220), 0, 0, 360,
            hsv_color(105, 100, 145), -1,
        )

        self.assertEqual(find_ore_targets(frame), [])

    def test_inclusion_on_a_nearly_finished_stone_is_still_found(self) -> None:
        """Живой баг: рядом с большим камнем мелкий переставал существовать.

        Камень стачивается по ходу сбора, и как только он становился меньше
        прежнего порога площади, его вкрапления пропадали из поиска — это и
        оставляло одну-две несобранные крупинки на столе.
        """
        frame = rug_frame()
        stone(frame, (1080, 640), (190, 200))
        inclusion(frame, (1080, 590), WARM_ORE)
        stone(frame, (1480, 660), (105, 120))
        inclusion(frame, (1480, 620), WARM_ORE)

        targets = find_ore_targets(frame)

        self.assertTrue(has_target_near(targets, (1080, 590), radius=25), targets)
        self.assertTrue(has_target_near(targets, (1480, 620), radius=25), targets)
        self.assertEqual(len(targets), 2, targets)

    def test_stone_lit_outside_the_blue_grey_window_is_still_a_stone(self) -> None:
        # На закате и в дождь камень уходит из сине-серого диапазона целиком,
        # и раньше на таком столе не находилось вообще ничего.
        frame = rug_frame()
        stone(frame, (1300, 650), (190, 200), colour=(20, 25, 120))
        inclusion(frame, (1300, 590), (20, 200, 215))

        targets = find_ore_targets(frame)

        self.assertTrue(has_target_near(targets, (1300, 590), radius=25), targets)

    def test_progress_note_is_never_taken_for_a_stone(self) -> None:
        # Бумажка «прогресс сбора руды» лежит на том же ковре и тоже компактна,
        # но её надпись не должна становиться целью для кликов.
        frame = rug_frame()
        cv2.rectangle(frame, (1130, 930), (1580, 1048), (245, 245, 240), -1)
        cv2.putText(
            frame, "PROGRESS", (1160, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (30, 30, 30), 4,
        )

        self.assertEqual(find_ore_targets(frame), [])

    def test_dark_and_bright_inclusions_are_collected_in_one_sweep(self) -> None:
        # Тёмное вкрапление рядом с ярким раньше оставалось на столе: поиск по
        # локальному контрасту включался, только когда не найдено вообще ничего.
        frame = rug_frame()
        stone(frame, (1300, 650), (200, 210))
        inclusion(frame, (1240, 590), WARM_ORE)
        inclusion(frame, (1360, 700), (105, 30, 40))

        targets = find_ore_targets(frame)

        self.assertTrue(has_target_near(targets, (1240, 590), radius=25), targets)
        self.assertTrue(has_target_near(targets, (1360, 700), radius=25), targets)

    def test_recorded_tables_are_collected_without_a_single_false_target(self) -> None:
        """Каждая цель на записанных столах — настоящая руда, лишних нет."""
        for name, expected in (
            ("overlay-123608.jpg", (
                (1296, 676), (1525, 700), (1455, 629), (1527, 504), (1472, 762),
                (1471, 573), (1276, 606), (1158, 733), (1225, 736),
            )),
            ("overlay-123454.jpg", ((1303, 566), (1367, 503))),
        ):
            targets = find_ore_targets(overlay_frame(name))
            for point in expected:
                self.assertTrue(has_target_near(targets, point, radius=30), (name, point, targets))
            self.assertEqual(len(targets), len(expected), (name, targets))

    def test_two_touching_stones_are_both_worked(self) -> None:
        """Кадр 882 из записи 21.08.2026, 23:14 — ночь, два соприкасающихся камня.

        Главный баг этой записи. Смыкание 9x9, которое лечит крапинки в
        цветовой маске, заодно закрывает узкую полоску ковра между камнями:
        два камня склеивались в один компонент 704x512, заполнение его bbox
        падало до 0.53 при пороге 0.60, и пара выбрасывалась целиком. Стол
        уезжал на аварийный проход «всё, что не ковёр» — тот держится только
        на «этот пиксель розовый?» и на живом кадре рассыпается от шума в
        цветности, которого нет в перекодированной записи. В живом прогоне
        правый камень так и не получил ни одного клика: четыре вкрапления
        пролежали нетронутыми 21 секунду, и сбор ушёл в ручную проверку.
        """
        frame = overlay_frame_png("overlay-night-two-stones.png")

        targets = find_ore_targets(frame)

        left_stone = ((1276, 775), (1231, 604), (1207, 830), (1166, 591), (1028, 656))
        right_stone = ((1475, 507), (1631, 516), (1504, 575), (1540, 645))
        for point in left_stone + right_stone:
            self.assertTrue(has_target_near(targets, point, radius=30), (point, targets))
        # Терялся именно правый камень, поэтому он проверяется отдельно.
        self.assertEqual(
            sum(1 for target in targets if target[0] >= 1400), len(right_stone), targets,
        )
        self.assertEqual(len(targets), len(left_stone) + len(right_stone), targets)

    def test_red_inclusion_is_not_erased_by_the_not_rug_rescue(self) -> None:
        """Кадр 35000 записи 10-43-16 — последнее вкрапление, красное.

        Красная руда по оттенку неотличима от розового ковра: hue уходит за
        180 и попадает в тот же диапазон. Пока стол держался на аварийном
        проходе «всё, что не ковёр», вкрапление **само вырезало дырку** в
        маске своего камня, попадало за пределы зоны поиска и не могло стать
        целью ни при каких порогах. Живьём модуль 13 секунд кликал по
        фантомам из запасного контрастного прохода, а вкрапление так и
        осталось на столе.
        """
        frame = overlay_frame_png("overlay-red-inclusion.png")

        targets = find_ore_targets(frame)

        self.assertTrue(has_target_near(targets, (1066, 675), radius=30), targets)
        # Ровно одна цель: фантомов контрастного прохода быть не должно.
        self.assertEqual(len(targets), 1, targets)
        self.assertNotEqual(targets[0][3], "adaptive", targets)

    def test_touching_stones_are_kept_off_the_not_rug_rescue(self) -> None:
        """Ночной стол должен разбираться основным проходом, а не аварийным.

        Проверяется не результат, а по какой дороге он получен: пока пара
        камней не проходила по форме, весь стол держался на запасном проходе,
        и попадание руды становилось лотереей.
        """
        frame = overlay_frame_png("overlay-night-two-stones.png")
        search, left, top = _ratio_crop(frame, TARGET_SEARCH_RATIO)
        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        candidate = cv2.morphologyEx(
            cv2.inRange(hsv, (82, 15, 25), (138, 255, 220)),
            cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8),
        )
        rug_mask = cv2.inRange(hsv, (145, 35, 65), (179, 255, 255))

        stones = _stones_from(candidate, rug_mask)

        self.assertGreater(np.count_nonzero(stones), 0, "основной проход снова пуст")
        # По одной точке в теле каждого камня — обе должны быть внутри маски.
        for centre in ((1150, 700), (1520, 560)):
            self.assertTrue(stones[centre[1] - top, centre[0] - left] > 0, centre)

    def test_merged_pair_of_stones_is_split_not_dropped(self) -> None:
        """Слипшуюся пару камней надо разнимать, а не браковать по bbox.

        Два камня по диагонали, между ними восемь пикселей ковра: смыкание их
        соединяет, и общий bbox наполовину состоит из пустоты между ними.
        """
        rug = rug_frame()
        first, second = (1003, 827), (1257, 573)
        stone(rug, first, (175, 175))
        stone(rug, second, (175, 175))
        search, left, top = _ratio_crop(rug, TARGET_SEARCH_RATIO)
        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        candidate = cv2.morphologyEx(
            cv2.inRange(hsv, (82, 15, 25), (138, 255, 220)),
            cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8),
        )
        rug_mask = cv2.inRange(hsv, (145, 35, 65), (179, 255, 255))
        self.assertEqual(
            cv2.connectedComponentsWithStats(candidate)[0] - 1, 1,
            "камни должны слипнуться, иначе тест проверяет не то",
        )

        stones = _stones_from(candidate, rug_mask)

        self.assertGreater(np.count_nonzero(stones), 0, "пара камней потеряна целиком")
        for centre in (first, second):
            self.assertTrue(stones[centre[1] - top, centre[0] - left] > 0, centre)

    def test_night_table_from_the_recorded_session(self) -> None:
        """Кадр из записи 21.08.2026, 22:20 — ночь, восемь вкраплений.

        На этом столе программа промахивалась дважды: не видела бирюзовый
        кристалл у края камня и кликала по ковру, просвечивающему в выемке
        камня. Оба места проверяются здесь на живом кадре.
        """
        frame = overlay_frame("overlay-night-table.jpg")

        targets = find_ore_targets(frame)

        expected = (
            (1131, 466), (1567, 481), (1112, 534), (1199, 470),
            (1488, 512), (1312, 651), (1065, 463), (1484, 580),
        )
        for point in expected:
            self.assertTrue(has_target_near(targets, point, radius=30), (point, targets))
        # Бирюзовый кристалл у нижнего края правого камня.
        self.assertTrue(has_target_near(targets, (1312, 651), radius=30), targets)
        # Выемка с ковром на левом камне целью быть не должна.
        self.assertFalse(has_target_near(targets, (1198, 577), radius=30), targets)
        self.assertEqual(len(targets), len(expected), targets)

    def test_mining_bar_is_not_confused_with_night_ground(self) -> None:
        # Ночью синеватая земля под игроком давала «полосу добычи» шириной в
        # два десятка пикселей, и модуль считал прогулку между камнями за
        # начатую добычу.
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        left, top = round(FRAME_WIDTH * 0.82), round(FRAME_HEIGHT * 0.89)
        cv2.rectangle(frame, (left + 180, top + 52), (left + 202, top + 57), (170, 95, 27), -1)

        self.assertFalse(mining_progress_visible(frame), "a 22 px sliver is not the bar")
        self.assertTrue(mining_progress_visible(progress_frame()))

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
        # Замеры по записям 22.08 (см. README фикстур): после засчитанной
        # ступеньки игра ~0.75 с не принимает мышь, дальше клик заводит взмах,
        # и ступенька приходит ещё через ~0.30 с. Пол — примерно 1.05 с на
        # удар, и весь вопрос в том, чтобы поставить клик сразу за глухим
        # окном. Клик раньше не просто пропадает — модуль потом высиживает
        # весь запасной интервал (так темп и упал до 2.0 с).
        ready_low, ready_high = MinerModule.STRIKE_READY_DELAY_RANGE_SECONDS
        self.assertGreaterEqual(ready_low, 0.70)
        self.assertLessEqual(ready_high, 1.00)
        self.assertGreater(ready_high, ready_low)

        # Повтор должен быть длиннее, чем путь «клик → ступенька» (~0.30 с)
        # плюс интервал наблюдения, иначе удавшийся взмах получает лишний
        # клик до того, как его ступенька успела перепланировать следующий.
        retry_low, retry_high = MinerModule.STRIKE_RETRY_RANGE_SECONDS
        self.assertGreater(
            retry_low, 0.30 + MinerModule.STRIKE_OBSERVE_INTERVAL_SECONDS,
        )
        self.assertLess(retry_high, ready_low)

        # Холодный старт: до первой ступеньки цепляться не за что.
        minimum, maximum = MinerModule.STRIKE_INTERVAL_RANGE_SECONDS
        self.assertGreater(minimum, 0.0)
        self.assertLessEqual(maximum, ready_low)

    def test_exhausted_targets_get_another_round_after_a_cooldown(self) -> None:
        """Отработанная цель — ещё не потерянная.

        На столе 10-43-16 (4:07) модуль трижды попал точно в центр золотого
        вкрапления, игра его не отдала, цель ушла в отставку — и стол простоял
        открытым 18 секунд. Через 17 секунд игрок собрал это же вкрапление
        рукой, то есть кликать было нужно дольше, а не бросать.
        """
        module = MinerModule.__new__(MinerModule)
        module.target_history = []
        module.target_rounds = 0
        module.last_target_at = 0.0

        point = (1400, 610)
        for attempt in range(MinerModule.MAX_ATTEMPTS_PER_TARGET):
            module._remember_target_attempt(point, attempt * 0.6)
        self.assertFalse(module._target_ready(point, 2.0), "цель должна быть исчерпана")

        self.assertTrue(module._retry_round(3.0))
        self.assertTrue(module._target_ready(point, 3.0), "после круга цель снова в очереди")

        for extra in range(MinerModule.MAX_TARGET_ROUNDS):
            module._retry_round(10.0 + extra)
        self.assertFalse(module._retry_round(99.0), "круги не бесконечны")
        self.assertLessEqual(module.target_rounds, MinerModule.MAX_TARGET_ROUNDS)
        # Круги дешевле, чем простой: даже все они укладываются в общий лимит.
        self.assertLessEqual(
            MinerModule.MAX_TARGET_ROUNDS * MinerModule.MAX_ATTEMPTS_PER_TARGET * 3,
            MinerModule.MAX_TARGET_ATTEMPTS,
        )
        # Ждать между кругами дольше, чем между попытками внутри круга.
        self.assertGreater(
            MinerModule.TARGET_COOLDOWN_SECONDS, MinerModule.TARGET_RETRY_SECONDS,
        )
        self.assertLess(
            MinerModule.TARGET_COOLDOWN_SECONDS, MinerModule.TARGET_STALL_SECONDS,
        )

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

    def test_failed_table_pauses_for_manual_review_without_disabling_auto_detect(self) -> None:
        module = MinerModule.__new__(MinerModule)
        module.running = True
        module.phase = "collecting"
        module.restore_cursor = Mock()
        module.stage = Mock()
        module.status = Mock()
        module.main_button = Mock()

        module._pause_for_manual_review(12.0, "Цель не распознана.")

        self.assertTrue(module.running)
        self.assertEqual(module.phase, "manual_review")
        module.restore_cursor.assert_called_once()
        module.stage.set.assert_called_once_with("НУЖНА РУЧНАЯ ПРОВЕРКА")

    def test_manual_completion_returns_to_result_check_with_auto_detect_enabled(self) -> None:
        module = MinerModule.__new__(MinerModule)
        module.running = True
        module.phase = "manual_review"
        module.last_capture_at = 0.0
        module.saved_cursor = None
        module.overlay_missing_frames = 1
        module.capture_game_image = Mock(return_value=(
            np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8),
            (0, 0, FRAME_WIDTH, FRAME_HEIGHT),
        ))
        module.stop = Mock()
        module.stage = Mock()
        module.status = Mock()

        module._manual_review_step(10.0)

        self.assertTrue(module.running)
        self.assertEqual(module.phase, "waiting_result")
        module.stop.assert_not_called()

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

    def test_e_starts_the_next_rock_during_the_result_check(self) -> None:
        # Между камнями модуль несколько секунд ждёт уведомление о руде.
        # Игрок в это время уже стоит у следующего камня и жмёт E.
        module = MinerModule.__new__(MinerModule)
        module.running = True
        module.phase = "waiting_result"
        module._foreground_is_game = Mock(return_value=True)
        module.saved_cursor = None
        module.strike_count = 12
        module.target_attempts = 0
        module.target_history = []
        module.overlay_missing_frames = 0
        module.stage = Mock()
        module.status = Mock()

        with patch("kisiki.modules.miner.cursor_position", return_value=(25, 40)):
            module._arm_cycle()

        self.assertEqual(module.phase, "arming")
        self.assertEqual(module.strike_count, 0)
        # Открытый стол игрок доигрывает сам, E его не прерывает.
        self.assertNotIn("manual_review", MinerModule.ARMABLE_PHASES)
        self.assertNotIn("collecting", MinerModule.ARMABLE_PHASES)

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
        frame = progress_frame()
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
        self.assertTrue(module.progress_confirmed)
        self.assertEqual(module.progress_missing_frames, 0)
        module.stop.assert_not_called()

    def test_strikes_stop_when_the_mining_bar_disappears(self) -> None:
        """The live bug: the rock was worked out and the pickaxe kept swinging."""
        module = striking_module(empty_frame(), now=30.0)

        with patch("kisiki.modules.miner.send_left_click", return_value=True) as click:
            for index in range(MinerModule.PROGRESS_MISSING_FRAMES):
                module._strike_step(30.0 + 0.2 * (index + 1))

        self.assertEqual(module.phase, "waiting_result")
        self.assertTrue(module.result_allows_table)
        click.assert_not_called()
        module.stop.assert_not_called()
        module.finish.assert_not_called()

    def test_strikes_continue_while_the_mining_bar_is_visible(self) -> None:
        module = striking_module(progress_frame(), now=30.0)
        # One dropped frame must not be read as a finished rock.
        module.progress_missing_frames = MinerModule.PROGRESS_MISSING_FRAMES - 1
        module.next_action_at = 30.0

        with patch("kisiki.modules.miner.send_left_click", return_value=True) as click:
            module._strike_step(30.0)

        self.assertEqual(module.phase, "striking")
        self.assertEqual(module.progress_missing_frames, 0)
        # Живой баг: в окне модуля стояло «УДАР 2», когда персонаж ударил один
        # раз. Клик — это заявка, а не удар: слишком ранний игра молча
        # выбрасывает. Считается только ступенька на полосе.
        self.assertEqual(module.strike_clicks, 13)
        self.assertEqual(module.strike_count, 12)
        click.assert_called_once()

    def test_credited_hit_counts_and_schedules_the_next_swing(self) -> None:
        """Ступенька на полосе — единственный честный признак удара.

        По ней и считается счётчик, и от неё же отсчитывается глухое окно:
        следующий клик ставится за него, а не сразу после ступеньки.
        """
        module = striking_module(progress_frame(fill=0.42), now=30.0)
        module.progress_fill = 0.31
        module.next_action_at = 31.0

        with patch("kisiki.modules.miner.send_left_click", return_value=True) as click:
            module._strike_step(30.0)

        self.assertEqual(module.strike_count, 13)
        ready_low, ready_high = MinerModule.STRIKE_READY_DELAY_RANGE_SECONDS
        self.assertGreaterEqual(module.next_action_at, 30.0 + ready_low)
        self.assertLessEqual(module.next_action_at, 30.0 + ready_high)
        self.assertAlmostEqual(module.progress_fill, 0.42, places=2)
        # Клик сразу после ступеньки игра выбрасывает — его тут быть не должно.
        click.assert_not_called()

    def test_click_without_a_step_is_retried_soon(self) -> None:
        """Клик, не давший ступеньки, был слишком ранним — повторить быстро."""
        module = striking_module(progress_frame(fill=0.42), now=30.0)
        module.progress_fill = 0.42
        module.next_action_at = 30.0

        with patch("kisiki.modules.miner.send_left_click", return_value=True) as click:
            module._strike_step(30.0)

        click.assert_called_once()
        self.assertEqual(module.strike_clicks, 13)
        self.assertEqual(module.strike_count, 12)
        low, high = MinerModule.STRIKE_RETRY_RANGE_SECONDS
        self.assertGreaterEqual(module.next_action_at, 30.0 + low)
        self.assertLessEqual(module.next_action_at, 30.0 + high)

    def test_unchanged_bar_keeps_the_pending_schedule(self) -> None:
        module = striking_module(progress_frame(fill=0.42), now=30.0)
        module.progress_fill = 0.42
        module.next_action_at = 30.7

        with patch("kisiki.modules.miner.send_left_click", return_value=True):
            module._strike_step(30.0)

        self.assertEqual(module.next_action_at, 30.7)

    def test_whole_rock_is_worked_at_the_pace_the_game_allows(self) -> None:
        """Прогон целого камня против модели игры, снятой с записей 22.08.

        Модель: после засчитанной ступеньки игра ~0.75 с не принимает мышь,
        следующий принятый клик заводит взмах, ступенька приходит ещё через
        ~0.30 с. Пол — ~1.05 с на удар. Живьём модуль выдавал 2.05 с и бросал
        камень на 33-й секунде, упёршись лимитом в собственные клики.
        """
        deaf_window, click_to_step, step_size, hits = 0.75, 0.30, 0.071, 15
        module = striking_module(progress_frame(), now=0.0)
        module.phase_started_at = 0.0
        module.next_action_at = 0.0
        module.last_capture_at = -1.0
        module.strike_count = 0
        module.strike_clicks = 0
        module.progress_fill = 0.0
        frames = {n: progress_frame(n * step_size) for n in range(hits + 1)}
        game = {"hits": 0, "last_step": 0.0, "pending": None}
        module.capture_game_image = lambda: (
            frames[game["hits"]], (0, 0, FRAME_WIDTH, FRAME_HEIGHT),
        )
        clicks, credited_at, now = [], [], 0.0

        def press(_hold: float) -> bool:
            clicks.append(now)
            if game["pending"] is None and now >= game["last_step"] + deaf_window:
                game["pending"] = now + click_to_step
            return True

        with patch("kisiki.modules.miner.send_left_click", side_effect=press):
            while now < 40.0 and game["hits"] < hits:
                if game["pending"] is not None and now >= game["pending"]:
                    game["last_step"] = game["pending"]
                    game["pending"] = None
                    game["hits"] += 1
                    credited_at.append(game["last_step"])
                module._strike_step(now)
                now += 1 / 60

        self.assertEqual(game["hits"], hits, "камень не добит")
        module.finish.assert_not_called()
        module.stop.assert_not_called()
        self.assertEqual(module.phase, "striking")

        spacing = [b - a for a, b in zip(credited_at, credited_at[1:])]
        pace = sum(spacing) / len(spacing)
        # Ручная добыча в записи 08-14-51 даёт 1.07 с на удар; сломанная
        # сборка давала 2.05 с. Держимся рядом с ручной.
        self.assertLess(pace, 1.35, f"темп {pace:.2f} с на удар")
        # И при этом без долбёжки: клик на удар, а не десять.
        self.assertLess(len(clicks) / hits, 1.6, f"{len(clicks)} кликов на {hits} ударов")

    def test_strike_limit_is_a_safety_net_above_a_real_rock(self) -> None:
        # Живой камень отрабатывается за 10–16 засчитанных ударов, а мельче
        # ~4% полоса не шагает — значит 25 ударов хватает на любой камень.
        # Пока лимит считал клики, камень на 15 ударов упирался в него на
        # 33-й секунде, и модуль бросал недобитый камень.
        self.assertGreaterEqual(MinerModule.MAX_STRIKES, 25)
        self.assertLessEqual(MinerModule.MAX_STRIKES, 34)
        # Клики ограничены отдельно: клик — не удар.
        self.assertGreater(MinerModule.MAX_STRIKE_CLICKS, MinerModule.MAX_STRIKES)
        # Времени должно хватать на самый долгий камень с запасом.
        self.assertGreaterEqual(
            MinerModule.MAX_STRIKING_SECONDS, MinerModule.MAX_STRIKES * 1.3,
        )
        # Экран проверяется намного чаще, чем отправляются удары, иначе конец
        # камня замечался бы только через один лишний взмах киркой.
        self.assertLess(
            MinerModule.STRIKE_OBSERVE_INTERVAL_SECONDS,
            MinerModule.STRIKE_INTERVAL_RANGE_SECONDS[0] / 4,
        )

    def test_table_opened_after_the_last_strike_is_still_collected(self) -> None:
        module = MinerModule.__new__(MinerModule)
        module.phase = "waiting_result"
        module.phase_started_at = 10.0
        module.last_capture_at = 0.0
        module.result_allows_table = True
        module.saved_cursor = (12, 34)
        module.target_history = []
        module.capture_game_image = Mock(return_value=(
            overlay_frame("overlay-123608.jpg"), (0, 0, FRAME_WIDTH, FRAME_HEIGHT),
        ))
        module.stop = Mock()
        module.finish = Mock()
        module.stage = Mock()
        module.status = Mock()

        module._waiting_result_step(10.5)

        self.assertEqual(module.phase, "collecting")
        self.assertFalse(module.result_allows_table)
        module.finish.assert_not_called()

    def test_finished_table_is_not_collected_a_second_time(self) -> None:
        module = MinerModule.__new__(MinerModule)
        module.phase = "waiting_result"
        module.phase_started_at = 10.0
        module.last_capture_at = 0.0
        module.result_allows_table = False
        module.saved_cursor = None
        module.target_history = []
        module.capture_game_image = Mock(return_value=(
            overlay_frame("overlay-123608.jpg"), (0, 0, FRAME_WIDTH, FRAME_HEIGHT),
        ))
        module.stop = Mock()
        module.finish = Mock()
        module.stage = Mock()
        module.status = Mock()

        module._waiting_result_step(10.5)

        self.assertEqual(module.phase, "waiting_result")
        module.finish.assert_not_called()

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
