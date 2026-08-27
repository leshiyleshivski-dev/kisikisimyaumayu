"""Регрессия ORE HUNT: разбор кадра, ритм ударов и счёт добытой руды.

Кадры лежат в ``tests/fixtures/miner/`` кропами и вставляются обратно в чёрный
кадр 2560x1440 по тому смещению, с которого сняты. Так фикстуры весят мегабайты
вместо десятков: разбор всё равно смотрит только в свои области.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from food_catalog import SECRET_CAT_INDICES, SECRET_RECIPES
from kisiki.core import VK_F9, VK_F11
from kisiki.modules.miner import MinerModule, OreTally, fresh_daily_stats
from kisiki.modules.miner import vision


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "miner"
FRAME_WIDTH, FRAME_HEIGHT = 2560, 1440

# Смещения, с которых сняты кропы (см. scratch-скрипт в разборе записи).
TABLE_ORIGIN = (768, 288)
BAR_ORIGIN = (2202, 1318)
TOAST_ORIGIN = (870, 1336)


def load(name: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURES / name))
    if image is None:
        raise RuntimeError(f"Нет фикстуры шахтёра: {name}")
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


def toast(name: str) -> np.ndarray:
    return frame_with(name, TOAST_ORIGIN)


def empty_frame() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


class Text:
    """Заглушка ``ctk.StringVar``: экран тестам не нужен, а текст — нужен."""

    def __init__(self, value: str = "") -> None:
        self.value = value

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


def bare_module(**overrides) -> MinerModule:
    """Модуль без окна CustomTkinter — только состояние и логика."""
    module = MinerModule.__new__(MinerModule)
    module.active = True
    module.watching = True
    module.phase = "watching"
    module.game_window = 42
    module.saved_cursor = None
    module.phase_started_at = 0.0
    module.next_action_at = 0.0
    module.last_watch_at = -10.0
    module.strikes = 0
    module.strike_clicks = 0
    module.clicks_since_step = 0
    module.rhythm_paused = False
    module.paused_strikes = 0
    module.ready_floor = MinerModule.STRIKE_READY_RANGE[0]
    module.clean_streak = 0
    module.click_to_step = MinerModule.CLICK_TO_STEP_SECONDS
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
    module.sightings = []
    module.on_stats_change = None
    module.tally = OreTally({})
    module.stage = Text()
    module.status = Text()
    module.rhythm = Text()
    module.session_total_text = Text()
    module.daily_total_text = Text()
    module.unknown_text = Text()
    module.leaders_text = Text()
    module.ore_text = {key: Text() for key, _title in vision.ORE_TYPES}
    module.process = Text("GTA5.exe")
    module.keys = {VK_F9: False, VK_F11: False}
    for name, value in overrides.items():
        setattr(module, name, value)
    return module


class MiningBarTests(unittest.TestCase):
    def test_bar_fill_is_read_from_the_lit_part_of_the_track(self):
        self.assertAlmostEqual(vision.mining_bar_fill(bar("bar-low.png")), 0.11, delta=0.03)
        self.assertAlmostEqual(vision.mining_bar_fill(bar("bar-half.png")), 0.60, delta=0.03)
        self.assertAlmostEqual(vision.mining_bar_fill(bar("bar-full.png")), 0.90, delta=0.03)

    def test_walking_between_rocks_shows_no_bar(self):
        self.assertIsNone(vision.mining_bar_fill(bar("bar-missing.png")))
        self.assertFalse(vision.mining_bar_visible(bar("bar-missing.png")))

    def test_bar_is_visible_whenever_it_can_be_read(self):
        self.assertTrue(vision.mining_bar_visible(bar("bar-half.png")))

    def test_a_short_bluish_sliver_is_not_the_bar(self):
        """Ночная земля давала обрывок синего; настоящая дорожка длинная."""
        frame = empty_frame()
        left, top = BAR_ORIGIN
        cv2.rectangle(frame, (left + 40, top + 16), (left + 58, top + 22), (150, 110, 40), -1)
        self.assertFalse(vision.mining_bar_visible(frame))

    def test_empty_frame_is_not_a_bar(self):
        self.assertIsNone(vision.mining_bar_fill(empty_frame()))


class SortingTableTests(unittest.TestCase):
    def test_rug_is_recognized_on_every_recorded_table(self):
        for name in (
            "table-red-and-grey.jpg", "table-many.jpg",
            "table-touching-stones.jpg", "table-two-grains.jpg",
            "table-red-at-edge.png",
        ):
            with self.subTest(name=name):
                self.assertTrue(vision.sorting_table_visible(table(name)))

    def test_walking_is_not_a_table(self):
        self.assertFalse(vision.sorting_table_visible(table("table-absent.jpg")))
        self.assertFalse(vision.sorting_table_visible(empty_frame()))


class OreTargetTests(unittest.TestCase):
    def test_no_targets_without_a_table(self):
        self.assertEqual(vision.find_ore_targets(table("table-absent.jpg")), [])
        self.assertEqual(vision.find_ore_targets(empty_frame()), [])

    def test_red_inclusion_is_not_swallowed_by_the_rug_colour(self):
        """Красная руда попадает в маску ковра и вырезает дырку в камне.

        Без заливки силуэта она не становилась целью ни при каких порогах: на
        этом столе игрок собирал её руками, а помощник кликал мимо.
        """
        targets = vision.find_ore_targets(table("table-red-and-grey.jpg"))
        self.assertGreaterEqual(len(targets), 3)
        self.assertTrue(
            any(abs(t.x - 1442) <= 24 and abs(t.y - 769) <= 24 for t in targets),
            f"красное вкрапление не найдено: {[(t.x, t.y) for t in targets]}",
        )

    def test_red_inclusion_near_the_stone_edge_is_found(self):
        targets = vision.find_ore_targets(table("table-red-at-edge.png"))
        self.assertTrue(
            any(abs(t.x - 1220) <= 26 and abs(t.y - 805) <= 26 for t in targets),
            f"красное вкрапление у кромки не найдено: {[(t.x, t.y) for t in targets]}",
        )

    def test_the_last_grain_of_the_long_table_is_seen_all_along(self):
        """Стол 5:06 записи «тест 3»: помощник собрал семь крупинок из восьми.

        Восьмая, красная, простояла на камне все пятнадцать секунд стола. На
        этом кадре она видна с запасом по каждой проверке (пик 177 при пороге
        150, площадь 748, плотность 0,97, свет 202), и стрелка мыши в двух
        шагах от неё её не заслоняет. Кадр держится тут именно поэтому: если
        правка порогов когда-нибудь её потеряет, потеря будет видна сразу.
        """
        image = table("table-last-grain.jpg")
        for cursor in (None, (1185, 444)):
            targets = vision.find_ore_targets(image, cursor=cursor)
            self.assertTrue(
                any(abs(t.x - 1114) <= 24 and abs(t.y - 502) <= 24 for t in targets),
                f"последняя крупинка не найдена (курсор {cursor}): "
                f"{[(t.x, t.y) for t in targets]}",
            )

    def test_two_touching_stones_are_both_worked(self):
        """Слипшуюся пару нельзя выбросить целиком: правый камень тоже руда."""
        targets = vision.find_ore_targets(table("table-touching-stones.jpg"))
        left = [t for t in targets if t.x < 1330]
        right = [t for t in targets if t.x >= 1330]
        self.assertTrue(left, "на левом камне не найдено ни одной цели")
        self.assertTrue(right, "на правом камне не найдено ни одной цели")

    def test_dark_bevel_of_a_stone_is_not_mistaken_for_ore(self):
        """У почти чёрного пикселя насыщенность уползает к 255 сама собой.

        Тёмная фаска по нижнему краю камня давала из-за этого отклонение 173
        при пороге 150 — «руду» убедительнее настоящей руды. На этом столе
        таких ложных целей было шесть из десяти, и одна из них своим соседством
        закрывала настоящее красное вкрапление.
        """
        targets = vision.find_ore_targets(table("table-dark-bevel.jpg"))
        for false_point in ((1108, 840), (1163, 848), (1240, 840), (1440, 876), (1481, 876)):
            nearest = min(
                ((t.x - false_point[0]) ** 2 + (t.y - false_point[1]) ** 2) ** 0.5
                for t in targets
            )
            with self.subTest(point=false_point):
                self.assertGreater(nearest, 22, f"фаска в {false_point} снова стала целью")
        self.assertTrue(
            any(abs(t.x - 1257) <= 26 and abs(t.y - 714) <= 26 for t in targets),
            "красное вкрапление рядом с фаской потерялось",
        )

    def test_a_crevice_between_facets_is_not_ore(self):
        """Щель между гранями отклоняется от камня сильнее самой руды.

        Света в щели нет вовсе, яркость там 0-2 при 91 у камня — а чем чернее
        пиксель, тем бессмысленнее его оттенок: он расходится с камнем на 56-67
        и после утроения даёт отклонение 168-200 при пороге 150. Настоящая руда
        на этом же столе набирает 190-245, то есть по одной силе отклонения щель
        от руды не отличить.

        На записи 26.08 такие щели жили весь стол и стоили дороже всего: каждая
        забирала три клика подряд, стол 8:46 не закрывался 12,8 секунды вместо
        обычных четырёх, а красная крупинка в углу камня так и уехала
        несобранной. Всего по тридцати столам записи их набралось четырнадцать.
        """
        targets = vision.find_ore_targets(table("table-shadow-crevices.jpg"))
        for crevice in ((1309, 754), (1296, 866)):
            nearest = min(
                ((t.x - crevice[0]) ** 2 + (t.y - crevice[1]) ** 2) ** 0.5
                for t in targets
            ) if targets else 1e9
            with self.subTest(point=crevice):
                self.assertGreater(nearest, 22, f"щель в {crevice} снова стала целью")
        self.assert_finds("table-shadow-crevices.jpg", [
            (1275, 691), (1258, 754), (1255, 610),
        ])

    def test_a_crevice_among_dark_ore_goes_and_the_ore_stays(self):
        """Тёмная руда и щель рядом с ней — то место, где легко срезать лишнее.

        На правом камне этого стола серые крупинки лежат кучей, и между ними
        сидит совершенно чёрная щель (яркость 0 при 102 у камня). Порог по свету
        обязан развести их: щель уходит, а сами крупинки — и серые, и красные на
        соседнем камне — остаются целями.
        """
        image = table("table-crevice-among-ore.jpg")
        targets = vision.find_ore_targets(image)
        nearest = min(
            ((t.x - 1487) ** 2 + (t.y - 699) ** 2) ** 0.5 for t in targets
        ) if targets else 1e9
        self.assertGreater(nearest, 22, "щель между серыми крупинками снова цель")
        self.assert_finds("table-crevice-among-ore.jpg", [
            (1032, 620), (1291, 733), (1186, 476), (1524, 679), (1447, 649),
        ])

    def test_light_is_what_tells_ore_from_a_shadow(self):
        """Проверка по свету смотрит на саму цель, а не на её окрестности."""
        stone = np.zeros((40, 40), dtype=np.uint8)
        blob = np.zeros((40, 40), dtype=bool)
        blob[10:20, 10:20] = True
        stone[blob] = vision.LIT_ENOUGH_FOR_ORE + 5
        self.assertTrue(vision._is_lit(stone, blob))
        stone[blob] = vision.LIT_ENOUGH_FOR_ORE - 5
        self.assertFalse(vision._is_lit(stone, blob))
        # Одинокий блик на кромке щель не спасает: берётся процентиль, не максимум.
        stone[10, 10] = 255
        self.assertFalse(vision._is_lit(stone, blob))

    def test_a_nearly_black_stone_is_worked_too(self):
        """Пара камней, слипшаяся через тень, уходила целиком в мусор.

        Полоска ковра между камнями в тени сама перестаёт быть розовой, пара
        сливается в один блоб с заполнением 0,51 при пороге 0,52 — и стол
        оставался без единой цели. Разнимать важно ещё и потому, что цвет-эталон
        берётся с камня: здесь один почти чёрный, второй сине-серый, и общая на
        двоих медиана не годится ни одному.
        """
        image = table("table-black-stone.jpg")
        stones = vision._stone_masks(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        self.assertEqual(len(stones), 2)
        targets = vision.find_ore_targets(image)
        self.assertGreaterEqual(len(targets), 5)
        self.assertTrue([t for t in targets if t.x < 1300], "на левом камне пусто")
        self.assertTrue([t for t in targets if t.x >= 1300], "на правом камне пусто")

    def assert_finds(self, name: str, truth: list[tuple[int, int]], *, slack: int = 30) -> None:
        targets = vision.find_ore_targets(table(name))
        for point in truth:
            distance = min(
                ((t.x - point[0]) ** 2 + (t.y - point[1]) ** 2) ** 0.5 for t in targets
            ) if targets else 1e9
            with self.subTest(point=point):
                self.assertLessEqual(
                    distance, slack,
                    f"{point} не найдено, ближайшая цель в {distance:.0f} px",
                )

    def test_night_table_is_read_as_well_as_a_day_one(self):
        """Ночь — то освещение, на котором отбор камня по его цвету разваливался.

        Кадр и разметка взяты из чужой записи от 21.08: восемь вкраплений,
        снятые вручную. Камень здесь ищется дыркой в ковре, а вкрапление —
        отклонением от цвета самого камня, и оба правила от освещения не
        зависят.
        """
        self.assert_finds("table-night.jpg", [
            (1131, 466), (1567, 481), (1112, 534), (1199, 470),
            (1488, 512), (1312, 651), (1065, 463), (1484, 580),
        ])

    def test_touching_stones_at_night_are_split_and_both_collected(self):
        """Смыкание маски ковра склеивало пару в один блоб, и правый терялся."""
        self.assert_finds("table-night-touching.jpg", [
            (1276, 775), (1231, 604), (1207, 830), (1166, 591), (1028, 656),
            (1475, 507), (1631, 516), (1504, 575), (1540, 645),
        ])
        image = table("table-night-touching.jpg")
        stones = vision._stone_masks(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        self.assertEqual(len(stones), 2, "камни должны разниматься, а не слипаться")

    def test_targets_are_sorted_by_confidence(self):
        targets = vision.find_ore_targets(table("table-many.jpg"))
        self.assertGreaterEqual(len(targets), 6)
        self.assertEqual([t.score for t in targets], sorted((t.score for t in targets), reverse=True))

    def test_one_grain_gives_one_target(self):
        """Блестящий спрайт режется порогом на куски — цель должна быть одна."""
        for name in ("table-many.jpg", "table-two-grains.jpg", "table-red-and-grey.jpg"):
            targets = vision.find_ore_targets(table(name))
            for index, first in enumerate(targets):
                for second in targets[index + 1:]:
                    distance = ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5
                    with self.subTest(name=name, pair=(first[:2], second[:2])):
                        self.assertGreater(distance, vision.INCLUSION_MERGE_DISTANCE)

    def test_targets_land_on_stones_not_on_the_rug(self):
        for name in ("table-many.jpg", "table-two-grains.jpg"):
            image = table(name)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            stones = vision._stone_masks(hsv)
            union = np.zeros(image.shape[:2], dtype=np.uint8)
            for mask in stones:
                union = cv2.bitwise_or(union, mask)
            for target in vision.find_ore_targets(image):
                with self.subTest(name=name, target=(target.x, target.y)):
                    self.assertTrue(union[target.y, target.x])

    def test_full_hd_frame_is_understood_and_answers_in_its_own_pixels(self):
        """Вся геометрия снята на 2K; 1920x1080 должен работать теми же числами."""
        big = table("table-many.jpg")
        small = cv2.resize(big, (1920, 1080), interpolation=cv2.INTER_AREA)
        self.assertTrue(vision.sorting_table_visible(small))
        targets = vision.find_ore_targets(small)
        self.assertTrue(targets)
        for target in targets:
            self.assertLess(target.x, 1920)
            self.assertLess(target.y, 1080)

    def test_supported_resolutions(self):
        self.assertTrue(vision.supported_resolution(2560, 1440))
        self.assertTrue(vision.supported_resolution(1920, 1080))
        self.assertFalse(vision.supported_resolution(1280, 720))

    def test_pale_crystal_is_ore_even_though_its_outline_is_hollow(self):
        """Бледный кристалл полупрозрачный, и от камня у него отличается только
        светящийся контур: середина по тону та же. После top-hat выходит не
        пятно, а полое рваное кольцо, у которого площадь вдвое меньше выпуклой
        оболочки — плотность 0,53-0,62.

        Порог стоял на 0,62 и резал ровно по этой куче: у принятых целей записи
        «тест 2» минимум 0,621, у отброшенных максимум 0,620. Столы 7:05 и 9:40
        из-за этого тянулись по 6 и 11 секунд при норме четыре, и руду с них
        игрок докапывал руками. На этом столе кристаллов два, и оба обязаны
        быть целями.
        """
        self.assert_finds("table-silver-crystal.jpg", [(1384, 709), (1452, 667)])

    def test_cursor_on_bare_stone_is_not_ore(self):
        """Стрелка мыши на камне — белое пятно с тёмной обводкой, и отклонение
        от камня у неё не хуже, чем у руды.

        На записи «тест 2» она набирала 244 против 212 у настоящего кристалла и
        вставала первой в очереди. Клик по ней уходит в никуда — мышь и так уже
        там, — и стол 3:42 простоял из-за этого пять секунд, не сдвинув курсор
        ни на пиксель, пока кристалл лежал рядом второй в списке.
        """
        image = table("table-cursor-on-stone.jpg")
        blind = vision.find_ore_targets(image)
        self.assertTrue(
            any(abs(t.x - 1368) <= 20 and abs(t.y - 506) <= 20 for t in blind),
            "фикстура бесполезна: без точки курсор целью не стал",
        )
        targets = vision.find_ore_targets(image, cursor=(1364, 496))
        for target in targets:
            with self.subTest(target=(target.x, target.y)):
                self.assertGreater(
                    ((target.x - 1368) ** 2 + (target.y - 506) ** 2) ** 0.5, 20,
                    "стрелка курсора снова стала целью",
                )
        self.assertTrue(
            any(abs(t.x - 1298) <= 24 and abs(t.y - 532) <= 24 for t in targets),
            f"кристалл рядом с курсором потерялся: {[(t.x, t.y) for t in targets]}",
        )

    def test_a_grain_under_the_cursor_stays_a_target(self):
        """Гасятся пиксели стрелки, а не цель под ней.

        Помощник сам наводит мышь на крупинку, чтобы кликнуть, и после
        неудачного клика обязан увидеть её снова — иначе повторить будет нечего.
        Крупинку видно по тому, что торчит из-под стрелки.
        """
        image = table("table-red-and-grey.jpg")
        found = vision.find_ore_targets(image, cursor=(1442, 769))
        self.assertTrue(
            any(abs(t.x - 1442) <= 26 and abs(t.y - 769) <= 26 for t in found),
            f"крупинка под курсором пропала: {[(t.x, t.y) for t in found]}",
        )

    def test_cursor_outside_the_frame_changes_nothing(self):
        """Мышь бывает и за окном игры: отрицательная точка не должна гасить
        половину камня отрицательным концом среза."""
        image = table("table-many.jpg")
        plain = vision.find_ore_targets(image)
        for point in ((-400, -400), (5000, 5000), (-10, 700), (700, -10)):
            with self.subTest(cursor=point):
                self.assertEqual(
                    [(t.x, t.y) for t in vision.find_ore_targets(image, cursor=point)],
                    [(t.x, t.y) for t in plain],
                )


class OreToastTests(unittest.TestCase):
    def test_every_recorded_ore_name_is_read(self):
        for name, ore in (
            ("toast-silicon.png", "silicon"),
            ("toast-manganese.png", "manganese"),
            ("toast-chrome.png", "chrome"),
            ("toast-nickel.png", "nickel"),
        ):
            with self.subTest(name=name):
                reading = vision.read_ore_toast(toast(name))
                self.assertTrue(reading.visible)
                self.assertEqual(reading.ore, ore)
                self.assertLessEqual(reading.distance, vision.TOAST_ACCEPT_DISTANCE)

    def test_faded_toast_is_not_guessed(self):
        """Гаснущая плашка читается плохо — молчание честнее выдуманной руды."""
        reading = vision.read_ore_toast(toast("toast-faded.png"))
        self.assertIsNone(reading.ore)

    def test_no_toast_at_all(self):
        self.assertFalse(vision.read_ore_toast(toast("toast-none.png")).visible)
        self.assertFalse(vision.read_ore_toast(empty_frame()).visible)

    def test_ore_word_is_taken_from_between_its_fixed_neighbours(self):
        """«собрали» слева и «руда!» справа — проверка, что фраза именно та."""
        words = vision._toast_words(toast("toast-silicon.png"))
        picked = vision._ore_word(words)
        self.assertIsNotNone(picked)
        word, anchor = picked
        self.assertAlmostEqual(anchor, 69.0, delta=6)
        self.assertAlmostEqual(word.shape[1], 104, delta=6)

    def test_a_foreign_notification_is_not_taken_for_ore(self):
        words = vision._toast_words(toast("toast-silicon.png"))
        self.assertIsNone(vision._ore_word(words[:2]))


class OreTallyTests(unittest.TestCase):
    def test_fresh_day_starts_empty(self):
        tally = OreTally({}, clock=lambda: "2026-08-26")
        self.assertEqual(tally.daily_total, 0)
        self.assertEqual(tally.daily["date"], "2026-08-26")
        self.assertEqual(set(tally.daily["ores"]), set(vision.ORE_KEYS))

    def test_known_ore_counts_in_both_session_and_day(self):
        daily = fresh_daily_stats("2026-08-26")
        daily["ores"]["gold"] = 4
        daily["total"] = 4
        tally = OreTally(daily, clock=lambda: "2026-08-26")
        tally.record("gold")
        self.assertEqual(tally.session_count("gold"), 1)
        self.assertEqual(tally.daily_count("gold"), 5)
        self.assertEqual(tally.session_total, 1)
        self.assertEqual(tally.daily_total, 5)

    def test_unknown_ore_counts_in_the_total_but_not_in_a_kind(self):
        """Руда добыта, вид не прочитан: приписать её кремниевой значит соврать."""
        tally = OreTally({}, clock=lambda: "2026-08-26")
        tally.record(None)
        self.assertEqual(tally.session_total, 1)
        self.assertEqual(tally.daily_total, 1)
        self.assertEqual(tally.session_unknown, 1)
        self.assertEqual(sum(tally.session.values()), 0)

    def test_a_new_day_resets_the_daily_count_but_not_the_session(self):
        day = ["2026-08-26"]
        tally = OreTally({}, clock=lambda: day[0])
        tally.record("iron")
        day[0] = "2026-08-27"
        tally.record("iron")
        self.assertEqual(tally.daily_count("iron"), 1)
        self.assertEqual(tally.session_count("iron"), 2)
        self.assertEqual(tally.daily["date"], "2026-08-27")

    def test_a_broken_save_is_repaired_rather_than_fatal(self):
        """Файл прогресса живёт долго и переживает правки руками."""
        daily = {
            "date": "2026-08-26", "total": "12", "unknown": -3,
            "ores": {"gold": "7", "iron": None, "unobtanium": 5},
        }
        tally = OreTally(daily, clock=lambda: "2026-08-26")
        self.assertEqual(tally.daily_total, 12)
        self.assertEqual(tally.daily_unknown, 0)
        self.assertEqual(tally.daily_count("gold"), 7)
        self.assertEqual(tally.daily_count("iron"), 0)
        self.assertNotIn("unobtanium", tally.daily["ores"])

    def test_daily_stats_are_written_back_into_the_same_dict(self):
        """Тот же объект лежит в ``progress.json`` — экран его и правит."""
        daily = fresh_daily_stats("2026-08-26")
        tally = OreTally(daily, clock=lambda: "2026-08-26")
        tally.record("tin")
        self.assertEqual(daily["ores"]["tin"], 1)
        self.assertEqual(daily["total"], 1)

    def test_leaders_show_the_best_of_the_session(self):
        tally = OreTally({}, clock=lambda: "2026-08-26")
        for ore in ("gold", "gold", "iron", "gold", "iron", "tin"):
            tally.record(ore)
        self.assertEqual(tally.leaders(2), [("gold", 3), ("iron", 2)])


class GameModel:
    """Модель камня: игра засчитывает удар только вовремя.

    После засчитанной ступеньки мышь какое-то время не принимается, а принятый
    клик доходит до ступеньки не мгновенно. Из этих двух чисел и складывается
    пол в 1,05 секунды на удар, снятый с живой игры.
    """

    DEAF_SECONDS = 0.75
    CLICK_TO_STEP = 0.30

    def __init__(self, hits_needed: int = 15) -> None:
        self.hits_needed = hits_needed
        self.hits = 0
        self.clicks = 0
        self.last_step_at = 0.0
        self.step_times: list[float] = []
        self.swing_lands_at: float | None = None
        self.finished_at: float | None = None

    @property
    def steady_gaps(self) -> list[float]:
        """Промежутки между ступеньками без разгона.

        Первый удар стоит дороже остальных: ритм ещё не за что зацепить, и
        клики идут вслепую. Мерить по нему темп — мерить не ритм, а старт.
        """
        return [b - a for a, b in zip(self.step_times[1:], self.step_times[2:])]

    def click(self, now: float) -> None:
        self.clicks += 1
        if self.finished_at is not None:
            return
        if now - self.last_step_at < self.DEAF_SECONDS:
            return          # игра выбросила ранний клик
        if self.swing_lands_at is None:
            self.swing_lands_at = now + self.CLICK_TO_STEP

    def fill(self, now: float) -> float | None:
        if self.swing_lands_at is not None and now >= self.swing_lands_at:
            self.swing_lands_at = None
            self.hits += 1
            self.last_step_at = now
            self.step_times.append(now)
            if self.hits >= self.hits_needed:
                self.finished_at = now
        if self.finished_at is not None and now >= self.finished_at + 0.2:
            return None     # полоса пропала: камень отбит
        return min(1.0, self.hits / self.hits_needed)


def run_rock(module: MinerModule, model: GameModel, *, limit: float = 60.0) -> float:
    """Прогнать фазу ударов по модели игры и вернуть, сколько это заняло."""
    now = 0.0
    module.phase = "striking"
    module.phase_started_at = now
    module.next_action_at = now
    with patch.object(module, "capture_client", return_value=(empty_frame(), (0, 0, 2560, 1440))), \
            patch.object(module, "game_is_foreground", return_value=True), \
            patch("kisiki.modules.miner.module.mining_bar_fill", side_effect=lambda _f: model.fill(now)), \
            patch("kisiki.modules.miner.module.send_left_click",
                  side_effect=lambda _hold: (model.click(now), True)[1]):
        while now < limit and module.phase == "striking":
            module.strike_step(now)
            now += 0.02
    return now


class StrikeRhythmTests(unittest.TestCase):
    def test_next_strike_is_scheduled_from_the_credited_step(self):
        """Отсчёт идёт от ступеньки, а не от клика: клик может быть выброшен."""
        module = bare_module(phase="striking", last_step_at=0.0)
        module.credit_strike(10.0)
        self.assertEqual(module.strikes, 1)
        # Ступенька случилась между прошлым и этим взглядом на экран, поэтому
        # отсчёт идёт от середины окна опроса, а не от момента, когда увидели.
        stepped_at = 10.0 - module.STRIKE_WATCH_SECONDS / 2
        self.assertGreaterEqual(module.next_action_at, stepped_at + module.STRIKE_READY_RANGE[0])
        self.assertLessEqual(module.next_action_at, stepped_at + module.STRIKE_READY_RANGE[1])

    def test_the_pace_stays_under_the_ceiling_the_game_shows(self):
        """Удар не должен планироваться медленнее, чем играет человек.

        По 220 промежуткам между ступеньками записи 09:50: медиана 1,10 с,
        девяностый процентиль 1,20 с. Верхняя граница задержки плюс путь
        «клик -> ступенька» обязаны укладываться в этот потолок, иначе модуль
        сам себе назначает темп ниже игрового.
        """
        slowest = MinerModule.STRIKE_READY_RANGE[1] + MinerModule.CLICK_TO_STEP_SECONDS
        self.assertLessEqual(slowest, MinerModule.STRIKE_PACE_CEILING_SECONDS)
        # И нижняя граница не должна попадать в глухое окно: клик оттуда игра
        # выбрасывает, а повтор стоит дороже, чем полученная скорость.
        self.assertGreaterEqual(MinerModule.STRIKE_READY_RANGE[0], 0.76)
        # Подбор может поднять паузу выше стартовой, но не бесконечно: даже на
        # самой медленной игре удар остаётся в пределах 1,25 с.
        self.assertLessEqual(
            MinerModule.STRIKE_READY_FLOOR_MAX + MinerModule.CLICK_TO_STEP_SECONDS, 1.25,
        )
        # Повтор обязан быть длиннее пути «клик -> ступенька» плюс окна опроса,
        # иначе удавшийся взмах получит лишний клик до собственной ступеньки.
        module = bare_module()
        for measured in (0.14, 0.30, 0.45):
            module.click_to_step = measured
            with self.subTest(measured=measured):
                self.assertGreater(
                    module.retry_window()[0], measured + MinerModule.STRIKE_WATCH_SECONDS,
                )

    def test_click_without_a_step_is_retried_sooner(self):
        module = bare_module(phase="striking")
        with patch.object(module, "game_is_foreground", return_value=True), \
                patch("kisiki.modules.miner.module.send_left_click", return_value=True):
            module.send_strike(5.0)
        self.assertEqual(module.strike_clicks, 1)
        low, high = module.retry_window()
        self.assertGreaterEqual(module.next_action_at, 5.0 + low)
        self.assertLessEqual(module.next_action_at, 5.0 + high)

    def test_the_retry_shortens_once_the_swing_is_measured(self):
        """Пока путь «клик -> ступенька» считался равным 0,30, повтор после
        пропущенного клика был вдвое длиннее нужного, и удар стоил 1,46 с."""
        module = bare_module()
        slow = module.retry_window()[0]
        module.click_to_step = 0.17
        self.assertLess(module.retry_window()[0], slow)

    def test_the_swing_is_measured_from_a_hit_taken_by_one_click(self):
        module = bare_module(phase="striking", strikes=3, last_step_at=0.0)
        with patch.object(module, "game_is_foreground", return_value=True),                 patch("kisiki.modules.miner.module.send_left_click", return_value=True):
            module.send_strike(10.0)
        module.credit_strike(10.18)
        self.assertLess(module.click_to_step, MinerModule.CLICK_TO_STEP_SECONDS)

    def test_a_hit_that_took_two_clicks_teaches_nothing_about_the_swing(self):
        """Какому из двух кликов принадлежит ступенька — неизвестно."""
        module = bare_module(phase="striking", strikes=3, last_step_at=0.0)
        with patch.object(module, "game_is_foreground", return_value=True),                 patch("kisiki.modules.miner.module.send_left_click", return_value=True):
            module.send_strike(10.0)
            module.send_strike(10.5)
        module.credit_strike(10.7)
        self.assertEqual(module.click_to_step, MinerModule.CLICK_TO_STEP_SECONDS)

    def test_pauses_are_a_range_and_not_one_number(self):
        """Ровный интервал — подпись робота. Тут его быть не должно."""
        module = bare_module(phase="striking")
        ready = set()
        retry = set()
        for step in range(40):
            module.credit_strike(float(step))
            ready.add(round(
                module.next_action_at - step + module.STRIKE_WATCH_SECONDS / 2, 4,
            ))
        with patch.object(module, "game_is_foreground", return_value=True), \
                patch("kisiki.modules.miner.module.send_left_click", return_value=True):
            for step in range(40):
                module.send_strike(float(step))
                retry.add(round(module.next_action_at - step, 4))
        self.assertGreater(len(ready), 30)
        self.assertGreater(len(retry), 30)
        self.assertTrue(all(module.STRIKE_READY_RANGE[0] <= v <= module.STRIKE_READY_RANGE[1] for v in ready))
        low, high = module.retry_window()
        self.assertTrue(all(low <= v <= high for v in retry))

    def test_click_hold_is_a_range_too(self):
        module = bare_module(phase="striking")
        holds = []
        with patch.object(module, "game_is_foreground", return_value=True), \
                patch("kisiki.modules.miner.module.send_left_click",
                      side_effect=lambda hold: holds.append(hold) or True):
            for step in range(30):
                module.send_strike(float(step))
        self.assertGreater(len(set(holds)), 20)
        self.assertTrue(all(module.CLICK_HOLD_RANGE[0] <= h <= module.CLICK_HOLD_RANGE[1] for h in holds))

    def test_whole_rock_is_worked_at_the_pace_the_game_allows(self):
        """Ритм должен держаться в том же коридоре, в каком играет человек.

        По записи 09:50: медиана промежутка между ступеньками 1,10 с,
        девяностый процентиль 1,20 с. Меряется установившийся ритм — разгон
        первого удара к нему не относится.
        """
        module = bare_module()
        model = GameModel(hits_needed=15)
        run_rock(module, model)
        self.assertEqual(model.hits, 15)
        gaps = model.steady_gaps
        self.assertGreaterEqual(len(gaps), 10)
        average = sum(gaps) / len(gaps)
        # Потолок плюс один тик опроса: ступеньку видно не раньше следующего
        # взгляда на экран, и этот шаг не выкинуть — он есть и у человека,
        # только называется реакцией.
        allowed = (
            MinerModule.STRIKE_PACE_CEILING_SECONDS + MinerModule.SCAN_INTERVAL_MS / 1000
        )
        self.assertLessEqual(
            average, allowed,
            f"{average:.2f} с на удар — медленнее, чем играет человек",
        )
        self.assertGreater(average, 0.95)
        self.assertLess(model.clicks / model.hits, 1.4, "лишние клики уходят в глухое окно")

    def test_strikes_stop_when_the_bar_disappears(self):
        """Конец камня — это флаг игры, а не наш счётчик ударов."""
        module = bare_module(phase="striking", progress_fill=0.9)
        with patch.object(module, "capture_client", return_value=(empty_frame(), (0, 0, 2560, 1440))), \
                patch.object(module, "game_is_foreground", return_value=True), \
                patch("kisiki.modules.miner.module.mining_bar_fill", return_value=None):
            for step in range(module.BAR_MISSING_FRAMES):
                module.strike_step(step * 0.5)
        self.assertEqual(module.phase, "awaiting_table")

    def test_one_missing_frame_does_not_end_the_rock(self):
        module = bare_module(phase="striking", progress_fill=0.5)
        with patch.object(module, "capture_client", return_value=(empty_frame(), (0, 0, 2560, 1440))), \
                patch.object(module, "game_is_foreground", return_value=True), \
                patch("kisiki.modules.miner.module.mining_bar_fill", return_value=None):
            module.strike_step(0.0)
        self.assertEqual(module.phase, "striking")

    def test_limits_count_credited_hits_not_clicks(self):
        """Кликов на камень набегает вдвое больше — лимит по ним бросал камень."""
        module = bare_module(phase="striking", strikes=10, strike_clicks=40)
        self.assertLess(module.strikes, module.MAX_STRIKES)
        self.assertLess(module.strike_clicks, module.MAX_STRIKE_CLICKS)

    def test_nothing_is_clicked_while_gta_is_not_in_front(self):
        module = bare_module(phase="striking")
        with patch.object(module, "game_is_foreground", return_value=False), \
                patch("kisiki.modules.miner.module.send_left_click") as click:
            module.send_strike(3.0)
        click.assert_not_called()
        self.assertEqual(module.strike_clicks, 0)


class DeafWindowTests(unittest.TestCase):
    """Пауза до удара подбирается по игре, а не выставляется на глаз."""

    def strike(self, module: MinerModule, now: float) -> None:
        with patch.object(module, "game_is_foreground", return_value=True),                 patch("kisiki.modules.miner.module.send_left_click", return_value=True):
            module.send_strike(now)

    def test_a_dropped_click_raises_the_floor(self):
        """Клик, после которого ступеньки не было, ушёл в глухое окно."""
        module = bare_module(phase="striking", strikes=4)
        start = module.ready_floor
        self.strike(module, 1.0)          # первый клик после ступеньки
        self.strike(module, 2.0)          # ступеньки не пришло — это повтор
        self.assertAlmostEqual(
            module.ready_floor, start + module.STRIKE_READY_FLOOR_STEP, places=6,
        )

    def test_one_clean_hit_does_not_ease_the_floor(self):
        """Сползание по чуть-чуть на каждом ударе давало пилу и пропуски."""
        module = bare_module(phase="striking", strikes=4, last_step_at=0.0)
        module.ready_floor = 0.90
        self.strike(module, 1.0)
        module.credit_strike(2.0)
        self.assertEqual(module.ready_floor, 0.90)

    def test_a_long_clean_streak_eases_the_floor_back(self):
        module = bare_module(phase="striking", strikes=4, last_step_at=0.0)
        module.ready_floor = 0.90
        for step in range(module.STRIKE_CLEAN_STREAK_TO_EASE):
            self.strike(module, float(step) + 0.5)
            module.credit_strike(float(step) + 1.0)
        self.assertAlmostEqual(
            module.ready_floor, 0.90 - module.STRIKE_READY_FLOOR_EASE, places=6,
        )

    def test_a_dropped_click_resets_the_clean_streak(self):
        module = bare_module(phase="striking", strikes=4, last_step_at=0.0)
        for step in range(5):
            self.strike(module, float(step) + 0.5)
            module.credit_strike(float(step) + 1.0)
        self.assertEqual(module.clean_streak, 5)
        self.strike(module, 10.0)
        self.strike(module, 11.0)
        self.assertEqual(module.clean_streak, 0)

    def test_the_floor_is_raised_decisively_and_eased_grudgingly(self):
        """Равновесие этих двух чисел — доля кликов, уходящих в пустоту."""
        rise_per_drop = MinerModule.STRIKE_READY_FLOOR_STEP
        ease_per_hit = (
            MinerModule.STRIKE_READY_FLOOR_EASE / MinerModule.STRIKE_CLEAN_STREAK_TO_EASE
        )
        self.assertGreater(rise_per_drop, ease_per_hit * 50)

    def test_blind_clicks_before_the_first_step_are_not_evidence(self):
        """Пока ступеньки на камне не было, клики идут вслепую.

        Считать их уликой против паузы — значит гнать её вверх на каждом камне
        и портить темп на игре, которая и так быстрая.
        """
        module = bare_module(phase="striking", strikes=0)
        start = module.ready_floor
        for step in range(4):
            self.strike(module, float(step))
        self.assertEqual(module.ready_floor, start)

    def test_the_floor_never_runs_past_its_cap(self):
        module = bare_module(phase="striking", strikes=4)
        module.ready_floor = module.STRIKE_READY_FLOOR_MAX
        self.strike(module, 1.0)
        self.strike(module, 2.0)
        self.assertEqual(module.ready_floor, module.STRIKE_READY_FLOOR_MAX)

    def test_the_delay_is_drawn_from_the_current_floor(self):
        module = bare_module(phase="striking", strikes=4, last_step_at=0.0)
        module.ready_floor = 0.86
        seen = set()
        for step in range(40):
            module.clicks_since_step = 1
            module.credit_strike(float(step))
            seen.add(round(module.next_action_at - step + module.STRIKE_WATCH_SECONDS / 2, 4))
        self.assertGreater(len(seen), 30)
        low = 0.86 - module.STRIKE_READY_FLOOR_EASE * 40
        self.assertTrue(all(low <= value <= 0.86 + module.STRIKE_READY_SPREAD for value in seen))


class ForegroundTests(unittest.TestCase):
    def test_the_game_is_recognized_by_its_process_not_by_a_handle(self):
        """У игры несколько видимых окон, и какое вернётся — меняется.

        Сравнение хэндлов давало ложное «игра не впереди»: помощник переставал
        бить посреди камня и добирал задержку четвертьсекундными кусками.
        """
        module = bare_module(game_window=111)
        with patch("kisiki.modules.miner.module.user32.GetForegroundWindow", return_value=222),                 patch("kisiki.modules.miner.module.process_for_window", return_value="GTA5.exe"):
            self.assertTrue(module.game_is_foreground())

    def test_another_application_in_front_stops_the_clicks(self):
        module = bare_module(game_window=111)
        with patch("kisiki.modules.miner.module.user32.GetForegroundWindow", return_value=222),                 patch("kisiki.modules.miner.module.process_for_window", return_value="explorer.exe"):
            self.assertFalse(module.game_is_foreground())

    def test_no_foreground_window_at_all(self):
        module = bare_module()
        with patch("kisiki.modules.miner.module.user32.GetForegroundWindow", return_value=0):
            self.assertFalse(module.game_is_foreground())


class CollectingTests(unittest.TestCase):
    def collecting(self) -> MinerModule:
        module = bare_module(phase="collecting", saved_cursor=(100, 100))
        return module

    def test_target_is_reached_by_a_glide_and_not_by_a_jump(self):
        """Прыжок курсора в точку — ровно то, чем он выглядит со стороны."""
        module = self.collecting()
        target = vision.OreTarget(1200, 700, 210.0, "цветное")
        with patch("kisiki.modules.miner.module.glide_cursor_to", return_value=True) as glide, \
                patch("kisiki.modules.miner.module.send_left_click", return_value=True):
            module.click_target(target, (10, 20, 2560, 1440), 1.0)
        glide.assert_called_once()
        self.assertEqual(glide.call_args.args, (1210, 720))
        self.assertEqual(glide.call_args.kwargs["duration_range"], module.GLIDE_RANGE)

    def test_settle_after_a_click_is_a_range(self):
        module = self.collecting()
        target = vision.OreTarget(1200, 700, 210.0, "цветное")
        pauses = set()
        with patch("kisiki.modules.miner.module.glide_cursor_to", return_value=True), \
                patch("kisiki.modules.miner.module.send_left_click", return_value=True), \
                patch("kisiki.modules.miner.module.time.monotonic", return_value=1.0):
            for _ in range(30):
                module.attempts = []
                module.click_target(target, (0, 0, 2560, 1440), 1.0)
                pauses.add(round(module.next_action_at - 1.0, 4))
        self.assertGreater(len(pauses), 20)
        self.assertTrue(all(
            module.TARGET_SETTLE_RANGE[0] <= p <= module.TARGET_SETTLE_RANGE[1] for p in pauses
        ))

    def test_a_target_is_retired_after_three_tries(self):
        module = self.collecting()
        point = (1200, 700)
        for _ in range(module.MAX_ATTEMPTS_PER_TARGET):
            module.remember_attempt(point, 0.0)
        self.assertFalse(module.target_ready(point, 1.0))

    def test_exhausted_targets_get_another_round(self):
        """Игра иногда не отдаёт крупинку сразу, а через пару секунд отдаёт."""
        module = self.collecting()
        module.remember_attempt((1200, 700), 0.0)
        module.remember_attempt((1200, 700), 0.1)
        module.remember_attempt((1200, 700), 0.2)
        self.assertTrue(module.retry_round(3.0))
        self.assertTrue(module.target_ready((1200, 700), 3.0))
        self.assertEqual(module.target_rounds, 1)

    def test_rounds_are_bounded(self):
        module = self.collecting()
        for _ in range(module.MAX_TARGET_ROUNDS):
            self.assertTrue(module.retry_round(1.0))
        self.assertFalse(module.retry_round(1.0))

    def test_a_closed_table_sends_the_module_to_wait_for_the_ore(self):
        module = self.collecting()
        with patch.object(module, "capture_client", return_value=(empty_frame(), (0, 0, 2560, 1440))), \
                patch.object(module, "restore_cursor"), \
                patch("kisiki.modules.miner.module.sorting_table_visible", return_value=False):
            for step in range(module.TABLE_MISSING_FRAMES):
                module.collect_step(step * 0.5)
        self.assertEqual(module.phase, "awaiting_result")

    def test_nothing_is_clicked_on_the_table_while_gta_is_not_in_front(self):
        module = self.collecting()
        with patch.object(module, "capture_client", return_value=(empty_frame(), (0, 0, 2560, 1440))), \
                patch.object(module, "game_is_foreground", return_value=False), \
                patch("kisiki.modules.miner.module.sorting_table_visible", return_value=True), \
                patch("kisiki.modules.miner.module.find_ore_targets") as find:
            module.collect_step(1.0)
        find.assert_not_called()

    def run_steps(self, module, targets, moments):
        seen = list(targets)
        clicks = []
        with patch.object(module, "capture_client",
                          return_value=(empty_frame(), (0, 0, 2560, 1440))), \
                patch.object(module, "game_is_foreground", return_value=True), \
                patch.object(module, "restore_cursor"), \
                patch("kisiki.modules.miner.module.sorting_table_visible", return_value=True), \
                patch("kisiki.modules.miner.module.cursor_position", return_value=(0, 0)), \
                patch("kisiki.modules.miner.module.find_ore_targets",
                      side_effect=lambda *a, **k: seen.pop(0)), \
                patch.object(module, "click_target",
                             side_effect=lambda t, b, n, **k: clicks.append((t, n, k))):
            for moment in moments:
                module.next_action_at = 0.0
                module.collect_step(moment)
        return clicks

    def test_a_grain_lost_from_sight_is_tried_at_its_last_place(self):
        """Со стола руду забирает только наш клик.

        Значит цель, пропавшая раньше первого клика по ней, не собрана — её
        потеряло зрение. Ровно это вышло на столе 5:06 записи «тест 3»:
        красная крупинка стояла на камне все пятнадцать секунд, а помощник
        перестал её видеть после седьмого клика и простоял рядом до конца.
        """
        module = self.collecting()
        grain = vision.OreTarget(1114, 502, 170.0, "цветное")
        clicks = self.run_steps(module, [[grain], [grain], [], []], [0.0, 0.2, 0.4, 0.6])
        self.assertTrue(clicks, "помощник не сходил по запомненному месту")
        target, _moment, kwargs = clicks[-1]
        self.assertEqual((target.x, target.y), (1114, 502))
        self.assertTrue(kwargs.get("remembered"))

    def test_a_grain_that_blinked_once_is_not_an_address(self):
        """Мигание одного кадра — известная ложная цель, а не потерянная руда."""
        module = self.collecting()
        blink = vision.OreTarget(1300, 600, 160.0, "тёмное")
        module.remember_sightings([blink], 0.0)
        self.assertIsNone(module.lost_sighting([], 0.5))
        module.remember_sightings([blink], 0.2)
        self.assertIsNotNone(module.lost_sighting([], 0.5))

    def test_a_grain_that_was_clicked_is_not_chased_again(self):
        """Кликнули и она пропала — значит собрали, а не потеряли."""
        module = self.collecting()
        grain = vision.OreTarget(1114, 502, 170.0, "цветное")
        module.remember_sightings([grain], 0.0)
        module.remember_sightings([grain], 0.2)
        module.remember_attempt((1114, 502), 0.4)
        self.assertIsNone(module.lost_sighting([], 1.0))

    def test_the_blind_approach_happens_once_and_not_in_a_loop(self):
        """Один заход по памяти, а не три: голый камень кликать незачем."""
        module = self.collecting()
        grain = vision.OreTarget(1114, 502, 170.0, "цветное")
        module.remember_sightings([grain], 0.0)
        module.remember_sightings([grain], 0.2)
        with patch("kisiki.modules.miner.module.glide_cursor_to", return_value=True), \
                patch("kisiki.modules.miner.module.send_left_click", return_value=True):
            module.click_target(module.lost_sighting([], 0.5), (0, 0, 2560, 1440), 0.5,
                                remembered=True)
        self.assertIsNone(module.lost_sighting([], 2.0))

    def test_an_empty_stone_does_not_hand_the_table_over(self):
        """Пустой камень — не заклинивший стол.

        Раньше пустой кадр отсчитывал ту же выдержку, что и не поддающаяся
        руда, и живой стол уходил в «ручную проверку» вместе с ней.
        """
        module = self.collecting()
        self.run_steps(module, [[], [], []], [0.0, 3.0, module.TARGET_STALL_SECONDS + 1.0])
        self.assertEqual(module.phase, "collecting")

    def test_ore_that_will_not_be_taken_still_hands_the_table_over(self):
        """А вот руда, которая видна и не даётся, стол по-прежнему отдаёт."""
        module = self.collecting()
        grain = vision.OreTarget(1114, 502, 170.0, "цветное")
        for _ in range(module.MAX_ATTEMPTS_PER_TARGET):
            module.remember_attempt((1114, 502), 0.0)
        for _ in range(module.MAX_TARGET_ROUNDS):
            module.retry_round(0.0)
        for _ in range(module.MAX_ATTEMPTS_PER_TARGET):
            module.remember_attempt((1114, 502), 0.0)
        module.last_target_at = 0.0
        self.run_steps(module, [[grain]], [module.TARGET_STALL_SECONDS + 0.1])
        self.assertEqual(module.phase, "manual")

    def test_the_pointer_is_handed_to_the_vision_in_client_coordinates(self):
        """Где стрелка — знает система, а кадр живёт в клиентских координатах.

        Без этого перевода стрелка на камне остаётся целью, а клик по ней уходит
        в никуда: мышь уже там.
        """
        module = self.collecting()
        with patch.object(module, "capture_client",
                          return_value=(empty_frame(), (100, 40, 2560, 1440))), \
                patch.object(module, "game_is_foreground", return_value=True), \
                patch.object(module, "click_target"), \
                patch("kisiki.modules.miner.module.sorting_table_visible", return_value=True), \
                patch("kisiki.modules.miner.module.cursor_position", return_value=(900, 600)), \
                patch("kisiki.modules.miner.module.find_ore_targets",
                      return_value=[]) as find:
            module.collect_step(1.0)
        self.assertEqual(find.call_args.kwargs["cursor"], (800, 560))


class ResultTests(unittest.TestCase):
    def test_a_read_toast_is_recorded_and_the_module_waits_for_the_next_rock(self):
        module = bare_module(phase="awaiting_result", phase_started_at=0.0)
        with patch.object(module, "capture_client", return_value=(empty_frame(), (0, 0, 2560, 1440))), \
                patch("kisiki.modules.miner.module.read_ore_toast",
                      return_value=vision.ToastReading(True, "gold", 0.08)):
            module.result_step(1.0)
        self.assertEqual(module.tally.session_count("gold"), 1)
        self.assertEqual(module.phase, "watching")

    def test_an_unread_toast_still_counts_the_rock(self):
        module = bare_module(phase="awaiting_result", phase_started_at=0.0)
        with patch.object(module, "capture_client", return_value=(empty_frame(), (0, 0, 2560, 1440))), \
                patch("kisiki.modules.miner.module.read_ore_toast",
                      return_value=vision.ToastReading(True, None, 0.4)):
            module.result_step(1.0)
        self.assertEqual(module.tally.session_total, 1)
        self.assertEqual(module.tally.session_unknown, 1)

    def test_no_toast_at_all_leaves_the_statistics_alone(self):
        """Не увидели уведомления — значит не знаем, что добыли. И не считаем."""
        module = bare_module(phase="awaiting_result", phase_started_at=0.0)
        with patch.object(module, "capture_client", return_value=(empty_frame(), (0, 0, 2560, 1440))), \
                patch("kisiki.modules.miner.module.read_ore_toast",
                      return_value=vision.ToastReading(False, None, 1.0)):
            module.result_step(module.RESULT_TIMEOUT_SECONDS + 1.0)
        self.assertEqual(module.tally.session_total, 0)
        self.assertEqual(module.phase, "watching")

    def test_saving_is_asked_for_only_when_something_was_counted(self):
        saves = []
        module = bare_module(phase="awaiting_result", on_stats_change=lambda: saves.append(1))
        module.record_ore("iron")
        self.assertEqual(len(saves), 1)


class WatchingTests(unittest.TestCase):
    def test_the_bar_starts_the_strike_phase(self):
        module = bare_module(phase="watching")
        with patch.object(module, "capture_client", return_value=(empty_frame(), (0, 0, 2560, 1440))), \
                patch("kisiki.modules.miner.module.sorting_table_visible", return_value=False), \
                patch("kisiki.modules.miner.module.mining_bar_fill", return_value=0.0):
            module.watch_step(1.0)
        self.assertEqual(module.phase, "striking")

    def test_the_table_starts_the_collecting_phase(self):
        module = bare_module(phase="watching")
        with patch.object(module, "capture_client", return_value=(empty_frame(), (0, 0, 2560, 1440))), \
                patch("kisiki.modules.miner.module.cursor_position", return_value=(5, 5)), \
                patch("kisiki.modules.miner.module.sorting_table_visible", return_value=True):
            module.watch_step(1.0)
        self.assertEqual(module.phase, "collecting")

    def test_an_empty_screen_keeps_waiting(self):
        module = bare_module(phase="watching")
        with patch.object(module, "capture_client", return_value=(empty_frame(), (0, 0, 2560, 1440))), \
                patch("kisiki.modules.miner.module.sorting_table_visible", return_value=False), \
                patch("kisiki.modules.miner.module.mining_bar_fill", return_value=None):
            module.watch_step(1.0)
        self.assertEqual(module.phase, "watching")


class HotkeyTests(unittest.TestCase):
    def press_f9(self, module: MinerModule, *, reset: bool = True) -> None:
        if reset:
            module.keys = {key: False for key in module.keys}
        with patch.object(MinerModule, "winfo_exists", return_value=False), \
                patch("kisiki.modules.miner.module.user32.GetAsyncKeyState",
                      side_effect=lambda key: 0x8000 if key == VK_F9 else 0):
            module.poll_hotkeys()

    def test_a_hidden_screen_ignores_f9(self):
        """Экран, спрятанный за другим модулем, не должен начать кликать.

        ``active`` — это «экран сверху», а не «идёт работа». Пока они были
        одним флагом, F9 в открытом покере запускал добычу на скрытом экране
        шахтёра, и он бил по чужой мини-игре.
        """
        module = bare_module(active=False, watching=False, phase="idle")
        self.press_f9(module)
        self.assertFalse(module.watching)
        self.assertEqual(module.phase, "idle")

    def test_the_screen_on_top_starts_on_f9(self):
        module = bare_module(active=True, watching=False, phase="idle")
        self.press_f9(module)
        self.assertTrue(module.watching)
        self.assertEqual(module.phase, "watching")

    def test_leaving_the_screen_stops_the_work(self):
        module = bare_module(active=True, watching=True, phase="striking")
        with patch("kisiki.modules.miner.module.user32.GetAsyncKeyState", return_value=0):
            module.deactivate("Остановлено: открыт модуль покера.")
        self.assertFalse(module.watching)
        self.assertFalse(module.active)
        self.assertEqual(module.phase, "idle")

    def test_a_key_held_across_the_switch_is_not_a_fresh_press(self):
        module = bare_module(active=False, watching=False)
        with patch("kisiki.modules.miner.module.user32.GetAsyncKeyState", return_value=0x8000):
            module.activate()
        self.assertTrue(module.active)
        self.assertTrue(all(module.keys.values()))
        # Клавиша так и остаётся зажатой — это не новое нажатие.
        self.press_f9(module, reset=False)
        self.assertFalse(module.watching)


class ModuleContractTests(unittest.TestCase):
    def test_the_module_never_sends_a_keystroke(self):
        """Клавиши здесь ни при чём: `E` жмёт человек, а помощник только кликает."""
        source = (ROOT / "kisiki" / "modules" / "miner" / "module.py").read_text(encoding="utf-8")
        self.assertNotIn("send_key_tap", source)
        self.assertNotIn("activate_window", source)

    def test_vision_opens_no_windows_and_sends_no_input(self):
        source = (ROOT / "kisiki" / "modules" / "miner" / "vision.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("customtkinter", imported)
        for forbidden in ("send_left_click", "SetCursorPos", "CTkFrame"):
            self.assertNotIn(forbidden, source)

    def test_the_ore_hunt_recipe_points_at_the_quartz_cat(self):
        recipes = {key: sequence for key, _title, sequence in SECRET_RECIPES}
        self.assertEqual(recipes["miner"], ("kibble", "burger", "milk"))
        self.assertEqual(SECRET_CAT_INDICES["miner"], 4)

    def test_every_ore_kind_has_an_icon_file(self):
        for index, (key, _title) in enumerate(vision.ORE_TYPES):
            path = ROOT / "assets" / "ores" / f"{index + 1:02d}_{key}.png"
            with self.subTest(ore=key):
                self.assertTrue(path.exists(), f"нет иконки {path.name}")


if __name__ == "__main__":
    unittest.main()
