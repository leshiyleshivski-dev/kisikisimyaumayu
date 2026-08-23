"""Vision, strategy and catalog checks for the automatic blackjack module."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from food_catalog import SECRET_CAT_INDICES, SECRET_RECIPES
from kisiki.core import CATS, COMING_SOON_CATS
from kisiki.modules.blackjack import BlackjackModule, parse_win_target
from kisiki.modules.blackjack_vision import (
    GLYPH_HEIGHT, GLYPH_WIDTH, HAND_LABEL_RATIO, HandReader,
    PROMPT_BOTTOM_ROW_RATIO, PROMPT_TOP_ROW_RATIO, TOAST_BAND_RATIO,
    classify_round_toast, decide_move, digit_templates, hand_totals, table_phase,
)

FELT = 96
# Настоящее сукно стола: тот же зелёный тон, что и полоска победы.
FELT_BGR = (69, 113, 88)
PANEL = (26, 29, 34)
LABEL_INK = (235, 235, 235)
DIGIT_HEIGHT = 19


def blank_frame() -> np.ndarray:
    return np.full((1440, 2560, 3), FELT, dtype=np.uint8)


def green_felt_frame() -> np.ndarray:
    return np.full((1440, 2560, 3), FELT_BGR, dtype=np.uint8)


def paint_ratio(
    frame: np.ndarray, ratio: tuple[float, float, float, float], color: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x, y, region_width, region_height = ratio
    x1, x2 = round(width * x), round(width * (x + region_width))
    y1, y2 = round(height * y), round(height * (y + region_height))
    frame[y1:y2, x1:x2] = color
    return x1, y1, x2, y2


def paint_white_button(frame: np.ndarray, ratio: tuple[float, float, float, float]) -> None:
    """Кнопка «СЪЕСТЬ ЗА» занимает правую часть строки подсказок."""
    x1, y1, x2, y2 = (
        round(frame.shape[1] * ratio[0]), round(frame.shape[0] * ratio[1]),
        round(frame.shape[1] * (ratio[0] + ratio[2])),
        round(frame.shape[0] * (ratio[1] + ratio[3])),
    )
    frame[y1 + 4:y2 - 4, x1 + (x2 - x1) // 3:x2 - 6] = (245, 245, 245)


def draw_number(frame: np.ndarray, value: int, left: int, top: int) -> int:
    """Нарисовать число шрифтом из эталонной полосы и вернуть правый край."""
    templates = digit_templates()
    assert templates is not None
    width = round(GLYPH_WIDTH * DIGIT_HEIGHT / GLYPH_HEIGHT)
    for digit in str(value):
        glyph = cv2.resize(
            templates[int(digit)].astype(np.uint8) * 255, (width, DIGIT_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )
        area = frame[top:top + DIGIT_HEIGHT, left:left + width]
        area[glyph > 127] = LABEL_INK
        left += width + 4
    return left


def synthetic_table(
    phase: str, *, toast: str | None = None, totals: tuple[int, ...] = (),
) -> np.ndarray:
    frame = blank_frame()
    if phase in {"bet", "action"}:
        paint_ratio(frame, PROMPT_BOTTOM_ROW_RATIO, PANEL)
    if phase == "bet":
        paint_ratio(frame, PROMPT_TOP_ROW_RATIO, PANEL)
    elif phase == "action":
        paint_white_button(frame, PROMPT_TOP_ROW_RATIO)
    elif phase == "wait":
        paint_white_button(frame, PROMPT_BOTTOM_ROW_RATIO)
    if toast is not None:
        paint_toast(frame, toast)
    left = 1150
    for value in totals:
        left = draw_number(frame, value, left, 1040) + 210
    return frame


def paint_toast(frame: np.ndarray, result: str) -> None:
    """Тёмная плашка уведомления с цветной полоской на правом краю."""
    frame[1336:1396, 1100:1542] = PANEL
    frame[1352:1368, 1290:1350] = (235, 235, 235)
    hsv_color = np.uint8([[[65 if result == "win" else 175, 220, 220]]])
    color = tuple(int(value) for value in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0])
    frame[1340:1392, 1542:1548] = color


class BlackjackPhaseTests(unittest.TestCase):
    def test_two_stake_panels_mean_the_bet_screen(self) -> None:
        self.assertEqual(table_phase(synthetic_table("bet")), "bet")

    def test_timer_panel_with_a_white_button_means_the_move_screen(self) -> None:
        self.assertEqual(table_phase(synthetic_table("action")), "action")

    def test_bare_prompts_mean_the_table_is_still_dealing(self) -> None:
        self.assertEqual(table_phase(synthetic_table("wait")), "wait")

    def test_unknown_screen_is_reported_instead_of_guessed(self) -> None:
        self.assertIsNone(table_phase(blank_frame()))
        self.assertIsNone(table_phase(None))


class BlackjackToastTests(unittest.TestCase):
    def test_green_notification_is_a_win(self) -> None:
        self.assertEqual(classify_round_toast(synthetic_table("wait", toast="win")), "win")

    def test_red_notification_is_a_loss(self) -> None:
        self.assertEqual(classify_round_toast(synthetic_table("wait", toast="loss")), "loss")

    def test_plain_table_has_no_result(self) -> None:
        self.assertIsNone(classify_round_toast(synthetic_table("wait")))

    def test_win_is_read_even_when_the_felt_touches_the_stripe(self) -> None:
        # Зелёная полоска победы вплотную примыкает к зелёному сукну справа.
        # Пока полоску искали отдельным пятном, они слипались в одно широкое,
        # и каждая победа уходила в «раздачи без выигрыша».
        frame = green_felt_frame()
        paint_toast(frame, "win")

        self.assertEqual(classify_round_toast(frame), "win")

    def test_felt_around_a_red_stripe_stays_a_loss(self) -> None:
        frame = green_felt_frame()
        paint_toast(frame, "loss")

        self.assertEqual(classify_round_toast(frame), "loss")

    def test_bare_felt_without_a_plate_has_no_result(self) -> None:
        self.assertIsNone(classify_round_toast(green_felt_frame()))

    def test_green_felt_alone_is_not_a_win(self) -> None:
        # Стол блэкджека зелёный, а фишки красные. Без тёмной плашки слева
        # цветная полоса не должна засчитываться как результат раздачи.
        frame = blank_frame()
        band_top = round(frame.shape[0] * TOAST_BAND_RATIO[1])
        frame[band_top:band_top + 70, 1500:1508] = (60, 210, 60)
        frame[band_top:band_top + 70, 1600:1608] = (60, 60, 210)

        self.assertIsNone(classify_round_toast(frame))


class BlackjackHandTests(unittest.TestCase):
    def test_both_totals_are_read_from_the_label(self) -> None:
        self.assertEqual(hand_totals(synthetic_table("action", totals=(17, 6))), (17, 6))
        self.assertEqual(hand_totals(synthetic_table("action", totals=(11, 10))), (11, 10))
        self.assertEqual(hand_totals(synthetic_table("wait", totals=(9, 0))), (9, 0))

    def test_half_read_label_is_refused(self) -> None:
        self.assertIsNone(hand_totals(synthetic_table("action", totals=(17,))))
        self.assertIsNone(hand_totals(synthetic_table("action")))
        self.assertIsNone(hand_totals(None))

    def test_label_band_stays_above_the_prompt_column(self) -> None:
        label_bottom = HAND_LABEL_RATIO[1] + HAND_LABEL_RATIO[3]

        self.assertLess(label_bottom, PROMPT_TOP_ROW_RATIO[1])
        self.assertLess(HAND_LABEL_RATIO[0] + HAND_LABEL_RATIO[2], PROMPT_TOP_ROW_RATIO[0])


class BlackjackStrategyTests(unittest.TestCase):
    def test_seventeen_and_above_always_stands(self) -> None:
        for dealer in range(0, 12):
            self.assertEqual(decide_move(17, dealer), "stand")
            self.assertEqual(decide_move(20, dealer), "stand")

    def test_eleven_and_below_always_takes_a_card(self) -> None:
        for dealer in range(0, 12):
            self.assertEqual(decide_move(11, dealer), "hit")
            self.assertEqual(decide_move(5, dealer), "hit")

    def test_stiff_hands_follow_the_dealer_card(self) -> None:
        self.assertEqual(decide_move(16, 6), "stand")
        self.assertEqual(decide_move(16, 7), "hit")
        self.assertEqual(decide_move(13, 2), "stand")
        self.assertEqual(decide_move(13, 10), "hit")

    def test_twelve_only_stands_against_four_to_six(self) -> None:
        self.assertEqual(decide_move(12, 3), "hit")
        self.assertEqual(decide_move(12, 4), "stand")
        self.assertEqual(decide_move(12, 6), "stand")
        self.assertEqual(decide_move(12, 7), "hit")

    def test_hidden_dealer_card_plays_by_the_dealer_rule(self) -> None:
        self.assertEqual(decide_move(16, 0), "hit")
        self.assertEqual(decide_move(12, 0), "hit")


class SoftHandStrategyTests(unittest.TestCase):
    def test_soft_seventeen_always_takes_a_card(self) -> None:
        # A+6 нельзя перебрать, а 17 — слабейшая рука в игре.
        for dealer in range(2, 12):
            self.assertEqual(decide_move(17, dealer, soft=True), "hit")

    def test_soft_twelve_never_stands(self) -> None:
        # Два туза: жёсткая таблица встала бы против 4-6.
        for dealer in range(2, 12):
            self.assertEqual(decide_move(12, dealer, soft=True), "hit")

    def test_soft_eighteen_stands_only_against_a_weak_dealer(self) -> None:
        for dealer in (2, 3, 6, 7, 8):
            self.assertEqual(decide_move(18, dealer, soft=True), "stand")
        for dealer in (9, 10, 11):
            self.assertEqual(decide_move(18, dealer, soft=True), "hit")

    def test_soft_nineteen_and_above_stands(self) -> None:
        for dealer in range(2, 12):
            self.assertEqual(decide_move(19, dealer, soft=True), "stand")
            self.assertEqual(decide_move(21, dealer, soft=True), "stand")

    def test_dealer_ace_is_read_as_eleven(self) -> None:
        # Подпись показывает туза дилера как «11», а не как «1».
        self.assertEqual(decide_move(16, 11), "hit")
        self.assertEqual(decide_move(18, 11, soft=True), "hit")


class HandReaderTests(unittest.TestCase):
    def test_ace_as_the_first_card_makes_the_hand_soft(self) -> None:
        reader = HandReader()

        self.assertFalse(reader.update(0))
        self.assertTrue(reader.update(11))
        self.assertTrue(reader.update(16))

    def test_ace_as_the_second_card_makes_the_hand_soft(self) -> None:
        reader = HandReader()
        reader.update(0)
        reader.update(9)

        self.assertTrue(reader.update(20))

    def test_two_plain_cards_stay_hard(self) -> None:
        reader = HandReader()
        for total in (0, 9, 11):
            self.assertFalse(reader.update(total))

    def test_pair_of_aces_reads_as_a_soft_twelve(self) -> None:
        # Второй туз приходит единицей: 11 -> 12, но первый ещё за 11.
        reader = HandReader()
        reader.update(0)
        reader.update(11)

        self.assertTrue(reader.update(12))

    def test_ten_on_a_soft_hand_makes_it_hard(self) -> None:
        # A+5 это мягкие 16; десятка роняет туза, и снова показано 16.
        reader = HandReader()
        reader.update(0)
        reader.update(11)
        reader.update(16)
        reader.draw()

        self.assertTrue(reader.update(16), "карта ещё не доехала — рука мягкая")
        self.assertFalse(reader.update(16, settled=True))

    def test_growing_total_keeps_a_soft_hand_soft(self) -> None:
        reader = HandReader()
        reader.update(0)
        reader.update(11)
        reader.update(14)
        reader.draw()

        self.assertTrue(reader.update(17, settled=True))

    def test_ace_drawn_to_a_hard_hand_makes_it_soft(self) -> None:
        reader = HandReader()
        reader.update(0)
        reader.update(4)
        reader.update(9)
        reader.draw()

        self.assertTrue(reader.update(20, settled=True))

    def test_new_round_forgets_the_previous_ace(self) -> None:
        reader = HandReader()
        reader.update(0)
        reader.update(11)
        reader.reset()
        reader.update(0)

        self.assertFalse(reader.update(15))


class BlackjackInputTests(unittest.TestCase):
    def test_target_counts_wins_not_rounds(self) -> None:
        self.assertEqual(parse_win_target("15"), 15)
        self.assertEqual(parse_win_target(" 4 "), 4)
        self.assertIsNone(parse_win_target("0"))
        self.assertIsNone(parse_win_target("1000"))
        self.assertIsNone(parse_win_target("две"))

    def test_hand_is_confirmed_by_several_frames(self) -> None:
        self.assertGreaterEqual(BlackjackModule.STABLE_READS, 2)

    def test_round_has_a_long_safety_window(self) -> None:
        self.assertGreaterEqual(BlackjackModule.ROUND_TIMEOUT_SECONDS, 30.0)
        self.assertGreaterEqual(BlackjackModule.MOVE_ACCEPT_TIMEOUT_SECONDS, 8.0)

    def test_next_bet_fits_into_the_open_bet_window(self) -> None:
        # Стол открывает ставку сразу после раздачи, а плашка результата
        # висит ещё около пяти секунд. Пауза должна быть заметно короче
        # окна ставки, иначе цикл упирается в тайм-аут «экран ставки».
        self.assertLessEqual(max(BlackjackModule.NEXT_ROUND_DELAY_SECONDS), 3.0)


class BlackjackCatalogTests(unittest.TestCase):
    def test_ace_replaces_bonus_placeholder(self) -> None:
        self.assertEqual(CATS[7][0], "Туз")
        self.assertTrue(CATS[7][2].endswith("10_blackjack_cat.png"))
        self.assertEqual(CATS[7][4], "Лудоманы")
        self.assertNotIn("Бонус", {cat[0] for cat in COMING_SOON_CATS})

    def test_blackjack_recipe_is_registered_for_the_ace(self) -> None:
        recipes = {secret_id: ingredients for secret_id, _title, ingredients in SECRET_RECIPES}
        self.assertEqual(SECRET_CAT_INDICES["blackjack"], 7)
        self.assertEqual(recipes["blackjack"], ("sushi", "watermelon", "strawberry"))

    def test_every_recipe_stays_unique(self) -> None:
        sequences = [ingredients for _id, _title, ingredients in SECRET_RECIPES]

        self.assertEqual(len(sequences), len(set(sequences)))

    def test_digit_templates_ship_with_the_assets(self) -> None:
        templates = digit_templates()

        self.assertIsNotNone(templates)
        self.assertEqual(len(templates), 10)
        self.assertEqual(templates[0].shape, (GLYPH_HEIGHT, GLYPH_WIDTH))


if __name__ == "__main__":
    unittest.main()
