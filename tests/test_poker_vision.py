"""Разбор покерного стола на сохранённых кадрах записи.

Кадры лежат в ``tests/fixtures/poker`` и описаны в ``poker-plan/README.md``:
стол на двоих, стол на шестерых, вскрытие и кнопка хода под курсором мыши.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from kisiki.modules.poker_vision import (
    BOARD_SLOTS, RING_ORDER, SEAT_SLOTS, all_in_only, bet_offer,
    bet_slider_at_minimum, bets_on_felt, blinds_from_bets, board_cards,
    call_amount, card_at, card_crop, dealer_seat, digit_templates, due_from_bets,
    hero_seat, is_face_down, is_face_up, is_my_turn, minimum_bet, pot_size,
    rank_templates, read_card, sane_minimum, seat_cards, seat_positions,
    seats_in_order, seats_with_cards, side_pots, stack_size, suit_templates,
    table_state, event_log, winning_seats, word_templates,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "poker"

TOP_LEFT, TOP_RIGHT, LEFT, RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT = range(6)


def frame(name: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURES / f"{name}.jpg"))
    if image is None:
        raise RuntimeError(f"Нет опорного кадра: {name}")
    return image


class TemplateTests(unittest.TestCase):
    def test_all_three_strips_load(self) -> None:
        self.assertEqual(len(rank_templates()), 13)
        self.assertEqual(len(suit_templates()), 4)
        self.assertEqual(len(digit_templates()), 10)


class CardReadingTests(unittest.TestCase):
    def test_board_is_read_street_by_street(self) -> None:
        self.assertEqual(board_cards(frame("preflop-hole-cards")), ())
        self.assertEqual(board_cards(frame("flop-opponent-turn")), ("6h", "Qs", "8h"))
        self.assertEqual(board_cards(frame("turn-four-board-cards")), ("6h", "Qs", "8h", "6c"))
        self.assertEqual(
            board_cards(frame("river-five-board-cards")), ("Ah", "Kc", "Tc", "5c", "2h")
        )
        self.assertEqual(board_cards(frame("six-max-flop")), ("9d", "2c", "Ks"))

    def test_hole_cards_are_read_at_two_different_seats(self) -> None:
        # В первой записи игрок сидел внизу справа, во второй — слева.
        self.assertEqual(seat_cards(frame("flop-opponent-turn"), BOTTOM_RIGHT), ("Kh", "3h"))
        self.assertEqual(seat_cards(frame("hole-cards-visible"), LEFT), ("6h", "Jc"))
        self.assertEqual(seat_cards(frame("my-turn-with-hole-cards"), LEFT), ("7h", "6s"))

    def test_showdown_cards_are_read_although_the_game_dims_them(self) -> None:
        # Карты, не вошедшие в победную комбинацию, игра гасит до серого.
        # Порог чернил берётся от белизны самой карты, поэтому они читаются.
        showdown = frame("six-max-showdown")

        self.assertEqual(seat_cards(showdown, TOP_LEFT), ("As", "7s"))
        self.assertEqual(seat_cards(showdown, BOTTOM_LEFT), ("Ks", "Qd"))
        self.assertEqual(seat_cards(showdown, BOTTOM_RIGHT), ("8c", "Tc"))

    def test_hole_cards_survive_the_shift_when_the_board_opens(self) -> None:
        # Одна и та же рука до флопа и после: на средних местах карта уезжает
        # вверх примерно на 50 пикселей, когда в панели появляется строка
        # комбинации.
        self.assertEqual(seat_cards(frame("right-seat-preflop"), RIGHT), ("9c", "Ks"))
        self.assertEqual(seat_cards(frame("right-seat-flop"), RIGHT), ("9c", "Ks"))

    def test_fixed_rectangle_alone_would_lose_the_hand(self) -> None:
        # Ради этого случая карту и ищут в полосе: по жёсткому прямоугольнику
        # своя рука пропадала, как только открывался борд.
        flop = frame("right-seat-flop")
        slot = SEAT_SLOTS[RIGHT][0]

        self.assertIsNone(read_card(card_crop(flop, slot)))
        self.assertEqual(read_card(card_at(flop, slot)), "9c")

    def test_face_down_card_is_not_guessed(self) -> None:
        preflop = frame("preflop-hole-cards")
        back = card_crop(preflop, BOARD_SLOTS[0])

        self.assertTrue(is_face_down(back))
        self.assertFalse(is_face_up(back))
        self.assertIsNone(read_card(back))

    def test_empty_slot_reads_as_nothing(self) -> None:
        idle = frame("table-idle")

        self.assertIsNone(read_card(card_crop(idle, BOARD_SLOTS[0])))
        self.assertIsNone(read_card(card_crop(idle, SEAT_SLOTS[BOTTOM_RIGHT][0])))


class SeatTests(unittest.TestCase):
    def test_own_seat_is_the_one_with_open_cards(self) -> None:
        self.assertEqual(hero_seat(frame("flop-opponent-turn")), BOTTOM_RIGHT)
        self.assertEqual(hero_seat(frame("my-turn-with-hole-cards")), LEFT)

    def test_showdown_hides_which_seat_is_mine(self) -> None:
        # Лицом лежит несколько рук, и своё место по кадру уже не отличить —
        # экран обязан держать место, найденное раньше.
        self.assertIsNone(hero_seat(frame("six-max-showdown")))

    def test_players_in_hand_are_counted_by_cards(self) -> None:
        # Сбросивший игрок остаётся за столом, но карты у него забирают.
        self.assertEqual(len(seats_with_cards(frame("table-idle"))), 0)
        self.assertEqual(seats_with_cards(frame("flop-opponent-turn")), (LEFT, BOTTOM_RIGHT))
        self.assertEqual(
            seats_with_cards(frame("my-turn-with-hole-cards")),
            (TOP_LEFT, TOP_RIGHT, LEFT, BOTTOM_LEFT),
        )


class NumberTests(unittest.TestCase):
    def test_pot_is_read_from_the_plate(self) -> None:
        self.assertEqual(pot_size(frame("flop-opponent-turn")), 100)
        self.assertEqual(pot_size(frame("my-turn-call-raise-fold")), 250)
        self.assertEqual(pot_size(frame("six-max-flop")), 17_250)
        self.assertEqual(pot_size(frame("river-five-board-cards")), 25_994)
        self.assertEqual(pot_size(frame("six-max-showdown")), 244_000)

    def test_empty_pot_has_no_plate(self) -> None:
        # Плашки нет — значит банк пуст, и это честный ноль, а не «не разобрал».
        # На префлопе всё поставленное лежит ставками на сукне.
        self.assertEqual(pot_size(frame("table-idle")), 0)
        self.assertEqual(pot_size(frame("preflop-hole-cards")), 0)

    def test_call_amount_is_read_from_the_button(self) -> None:
        self.assertEqual(call_amount(frame("my-turn-call-raise-fold")), 50)
        self.assertEqual(call_amount(frame("my-turn-with-hole-cards")), 500)

    def test_check_button_means_nothing_to_call(self) -> None:
        # На своём ходу без доплаты игра пишет «CHECK» и «BET»: малиновой
        # суммы на левой кнопке нет, и это ноль, а не «не разобрал».
        self.assertEqual(call_amount(frame("right-seat-flop")), 0)

    def test_button_under_the_cursor_is_still_read(self) -> None:
        # Кнопка под курсором заливается малиновым, а текст на ней белеет.
        # Заливка читалась как одно сплошное пятно суммы, и помощник замолкал
        # ровно тогда, когда игрок тянулся к кнопке.
        self.assertEqual(call_amount(frame("hover-call-button")), 25)
        self.assertEqual(call_amount(frame("hover-check-button")), 0)

    def test_cursor_on_top_of_a_digit_stops_the_reading(self) -> None:
        # Здесь курсор накрыл двойку в «CALL 25». Читать оставшуюся пятёрку
        # нельзя: помощник посоветовал бы колл впятеро дешевле настоящего.
        self.assertIsNone(call_amount(frame("cursor-over-call-amount")))

    def test_bets_are_read_from_the_felt(self) -> None:
        # Плашка «Общий банк» ставки этой улицы не считает: в six-max-flop на
        # ней 17 250, а перед игроками лежат ещё 2 500 и 16 500.
        self.assertEqual(
            bets_on_felt(frame("six-max-flop")), (2_500, 16_500, 0, 0, 0, 0)
        )
        self.assertEqual(
            bets_on_felt(frame("my-turn-call-raise-fold")), (0, 0, 100, 0, 0, 50)
        )
        self.assertEqual(
            bets_on_felt(frame("right-seat-preflop")), (0, 0, 0, 25, 0, 50)
        )
        self.assertEqual(bets_on_felt(frame("table-idle")), (0,) * 6)

    def test_buy_menu_is_not_a_bet(self) -> None:
        # Под панелью игрока игра рисует меню «ПОКУПКА ЗА ⊙», и его золотая
        # кнопка ложится ровно туда, где у места «верх-слева» лежит ставка.
        # Помощник принимал её за плашку: на первом кадре своя ставка 250
        # читалась как 2501, на втором пустое место — как «не разобрал».
        self.assertEqual(
            bets_on_felt(frame("top-seat-buy-menu")), (250, 500, 0, 0, 0, 0)
        )
        self.assertEqual(
            bets_on_felt(frame("top-seat-no-bet")), (0, 250, 0, 0, 500, 0)
        )

    def test_own_bet_at_the_top_seat_gives_the_same_call_as_the_button(self) -> None:
        # Ради этого счёт и ведётся дважды: пока чужое меню считалось ставкой,
        # два ответа расходились, и экран молчал всю раздачу.
        state = table_state(frame("top-seat-buy-menu"))

        self.assertEqual(call_amount(frame("top-seat-buy-menu")), 250)
        self.assertEqual(due_from_bets(state.bets, TOP_LEFT), 250)
        self.assertEqual(state.to_call, 250)

    def test_stack_is_read_from_the_panel(self) -> None:
        # Стек нужен, чтобы не советовать ставку больше своих фишек.
        self.assertEqual(stack_size(frame("top-seat-buy-menu"), TOP_LEFT), 520)
        self.assertEqual(stack_size(frame("check-and-bet"), LEFT), 4_250)
        self.assertEqual(stack_size(frame("my-turn-call-raise-fold"), BOTTOM_RIGHT), 875)
        self.assertIsNone(stack_size(frame("table-idle"), None))

    def test_bet_button_shows_what_the_slider_will_put_in(self) -> None:
        # Правая кнопка называется «BET», пока доплаты нет, и «RAISE», когда
        # она есть; сумма на ней — то, что стоит на ползунке.
        self.assertEqual(bet_offer(frame("check-and-bet")), 50)
        self.assertEqual(bet_offer(frame("right-seat-flop")), 50)
        self.assertEqual(bet_offer(frame("hover-call-button")), 100)

    def test_thousands_are_read_as_one_number(self) -> None:
        # Разряды игра делит пробелом, и по одному последнему куску «1 000»
        # читалась тысяча как ноль, а «CALL 1 250» — как 250.
        self.assertEqual(bet_offer(frame("my-turn-with-hole-cards")), 1_000)
        self.assertEqual(bet_offer(frame("top-seat-buy-menu")), 1_000)

    def test_untouched_slider_means_the_button_shows_the_minimum(self) -> None:
        # Ниже минимума ползунок не опускается: пока бегунок у левого края,
        # на кнопке написана наименьшая ставка, которую игра примет.
        self.assertTrue(bet_slider_at_minimum(frame("check-and-bet")))
        self.assertEqual(minimum_bet(frame("check-and-bet")), 50)
        self.assertEqual(minimum_bet(frame("top-seat-buy-menu")), 1_000)

    def test_no_slider_means_no_minimum(self) -> None:
        # Ползунок появляется только на своём ходу — судить о минимуме между
        # ходами не по чему.
        self.assertFalse(bet_slider_at_minimum(frame("table-idle")))
        self.assertIsNone(minimum_bet(frame("table-idle")))

    def test_moved_slider_no_longer_shows_the_minimum(self) -> None:
        # Тот же стол, что и в check-and-bet, но ползунок утянут вправо: на
        # кнопке теперь 1 150 вместо минимальных 50. Принять это за минимум
        # значило бы посоветовать ставку в двадцать раз крупнее нужной.
        dragged = frame("slider-dragged")

        self.assertEqual(bet_offer(dragged), 1_150)
        self.assertFalse(bet_slider_at_minimum(dragged))
        self.assertIsNone(minimum_bet(dragged))

    def test_call_is_also_counted_from_the_felt(self) -> None:
        # Старшая ставка минус своя — та же доплата, что игра пишет на кнопке.
        # Второй счёт нужен, когда кнопку закрыл курсор.
        preflop = table_state(frame("hover-call-button"), known_seat=BOTTOM_RIGHT)
        self.assertEqual(due_from_bets(preflop.bets, BOTTOM_RIGHT), 25)
        self.assertEqual(due_from_bets(preflop.bets, BOTTOM_LEFT), 0)

    def test_call_falls_back_to_the_felt_when_the_button_is_covered(self) -> None:
        # На этом кадре курсор съел двойку в «CALL 25», зато ставки на сукне
        # целы: 50 у соперника против наших 25.
        state = table_state(frame("cursor-over-call-amount"), known_seat=BOTTOM_RIGHT)

        self.assertIsNone(call_amount(frame("cursor-over-call-amount")))
        self.assertEqual(state.to_call, 25)

    def test_unknown_seat_leaves_the_felt_count_silent(self) -> None:
        # Не зная своего места, вычесть свою ставку не из чего.
        self.assertIsNone(due_from_bets((0, 0, 0, 0, 50, 25), None))
        self.assertIsNone(due_from_bets((0, None, 0, 0, 50, 25), 5))

    def test_money_in_play_adds_the_felt_to_the_plate(self) -> None:
        # Префлоп: плашки банка нет вовсе, блайнды лежат перед игроками.
        preflop = table_state(frame("hover-call-button"), known_seat=BOTTOM_RIGHT)
        self.assertEqual(preflop.pot, 0)
        self.assertEqual(preflop.bets, (0, 0, 0, 0, 50, 25))
        self.assertEqual(preflop.money_in_play, 75)

        flop = table_state(frame("six-max-flop"))
        self.assertEqual(flop.money_in_play, 17_250 + 2_500 + 16_500)

    def test_unreadable_bet_hides_the_whole_sum(self) -> None:
        # Заниженный банк тихо превращает выгодный колл в фолд, поэтому
        # неразобранная ставка обнуляет весь ответ, а не выпадает из суммы.
        state = table_state(frame("six-max-flop"))
        broken = replace(state, bets=(None, 16_500, 0, 0, 0, 0))

        self.assertIsNone(broken.money_in_play)


class SidePotTests(unittest.TestCase):
    """Полоса горшков под бордом: появляется, когда кто-то ушёл в олл-ин."""

    WITH_ROW = {
        "six-max-showdown": (121_500, 57_000, 65_500),
        "side-pots-in-hand": (2_000, 5_000),
        "pot-mid-sweep": (6_420, 5_000),
    }

    def test_row_is_read_left_to_right(self) -> None:
        for name, expected in self.WITH_ROW.items():
            with self.subTest(name):
                self.assertEqual(side_pots(frame(name)), expected)

    def test_row_is_absent_on_every_other_frame(self) -> None:
        # Полоса стоит по центру сукна, и рамка под неё широкая: важно, что в
        # неё не попадают ни карты борда, ни жёлтая разметка стола.
        for path in sorted(FIXTURES.glob("*.jpg")):
            if path.stem in self.WITH_ROW:
                continue
            with self.subTest(path.stem):
                self.assertEqual(side_pots(frame(path.stem)), ())

    def test_plate_above_the_board_already_counts_the_pots(self) -> None:
        # Полоса — не прибавка к «Общему банку», а его разбор. Проверено на
        # записях: в покое сумма сходится с плашкой знак в знак, и складывать
        # их значило бы посчитать банк дважды.
        for name in ("six-max-showdown", "side-pots-in-hand"):
            with self.subTest(name):
                self.assertEqual(pot_size(frame(name)), sum(side_pots(frame(name))))

    def test_pot_is_hidden_while_the_plate_catches_up(self) -> None:
        # Пока анимация сгребает фишки, плашка отстаёт от полосы: 9 842 при
        # настоящих 13 000. Такой банк занижен на четверть, и считать по нему
        # шансы банка нельзя.
        state = table_state(frame("pot-mid-sweep"))

        self.assertTrue(state.split_pot)
        self.assertEqual(state.pot, 9_842)
        self.assertNotEqual(state.pot, sum(state.side_pots))
        self.assertIsNone(state.money_in_play)

    def test_settled_pot_is_trusted(self) -> None:
        state = table_state(frame("side-pots-in-hand"))

        self.assertTrue(state.split_pot)
        self.assertEqual(state.pot, 7_000)
        self.assertEqual(state.money_in_play, 13_000)

    def test_unreadable_row_is_not_a_split_pot(self) -> None:
        # Игрок встал из-за стола, и в рамку попали фишки чужого стола: числа
        # не сложились. Без второго счёта банк остаётся как был — молчать
        # из-за нечитаемого украшения не за чем.
        state = replace(table_state(frame("six-max-flop")), side_pots=(None, None))

        self.assertFalse(state.split_pot)
        self.assertEqual(state.money_in_play, 17_250 + 2_500 + 16_500)

    def test_split_pot_counts_the_due_by_the_button_alone(self) -> None:
        # Перед ушедшим в олл-ин лежит вся его ставка, а уравнивать её не
        # надо: лишнее уже сложено в горшок. Счёт по сукну даёт 1 000 доплаты
        # там, где игра пишет «CHECK», и прежде экран молчал от расхождения.
        state = table_state(frame("side-pots-in-hand"))

        self.assertTrue(state.my_turn)
        self.assertEqual(due_from_bets(state.bets, state.hero_seat), 1_000)
        self.assertEqual(call_amount(frame("side-pots-in-hand")), 0)
        self.assertEqual(state.to_call, 0)
        self.assertIsNone(state.faced_bet)


class TurnTests(unittest.TestCase):
    def test_my_turn_is_seen_by_the_quick_bet_row(self) -> None:
        self.assertTrue(is_my_turn(frame("my-turn-call-raise-fold")))
        self.assertTrue(is_my_turn(frame("my-turn-with-hole-cards")))

    def test_highlighted_preaction_is_not_my_turn(self) -> None:
        # В этом кадре преактив CHECK подсвечен красным, а ход чужой: таймер
        # идёт у соперника. Ловушка, из-за которой цвет кнопки не годится.
        self.assertFalse(is_my_turn(frame("turn-four-board-cards")))
        self.assertFalse(is_my_turn(frame("flop-opponent-turn")))
        self.assertFalse(is_my_turn(frame("table-idle")))


class TableStateTests(unittest.TestCase):
    def test_full_reading_of_a_playable_spot(self) -> None:
        state = table_state(frame("my-turn-call-raise-fold"))

        self.assertEqual(state.board, ("Kh", "Ad", "Tc"))
        self.assertEqual(state.hole, ("3h", "Qd"))
        self.assertEqual(state.hero_seat, BOTTOM_RIGHT)
        self.assertEqual(state.players, 2)
        self.assertEqual(state.pot, 250)
        self.assertEqual(state.to_call, 50)
        self.assertTrue(state.my_turn)
        self.assertFalse(state.showdown)

    def test_idle_table_reads_as_nothing(self) -> None:
        state = table_state(frame("table-idle"))

        self.assertEqual(state.board, ())
        self.assertEqual(state.hole, ())
        self.assertIsNone(state.hero_seat)
        self.assertEqual(state.players, 0)
        self.assertEqual(state.pot, 0)
        self.assertEqual(state.money_in_play, 0)
        self.assertFalse(state.my_turn)

    def test_right_seat_flop_is_read_whole(self) -> None:
        state = table_state(frame("right-seat-flop"))

        self.assertEqual(state.board, ("7d", "As", "Kh"))
        self.assertEqual(state.hole, ("9c", "Ks"))
        self.assertEqual(state.hero_seat, RIGHT)
        self.assertEqual(state.players, 4)
        self.assertEqual(state.pot, 200)
        self.assertEqual(state.to_call, 0)
        self.assertTrue(state.my_turn)

    def test_check_and_bet_spot_is_read_whole(self) -> None:
        # Доплаты нет: игра показывает «CHECK» и «BET 50», а не «RAISE».
        state = table_state(frame("check-and-bet"))

        self.assertEqual(state.board, ("7d", "4c", "6d", "5c"))
        self.assertEqual(state.hole, ("8d", "6s"))
        self.assertEqual(state.hero_seat, LEFT)
        self.assertEqual(state.pot, 150)
        self.assertEqual(state.money_in_play, 150)
        self.assertEqual(state.to_call, 0)
        self.assertEqual(state.min_bet, 50)
        self.assertEqual(state.stack, 4_250)
        self.assertTrue(state.my_turn)

    def test_own_bet_comes_from_the_seat(self) -> None:
        # Своя ставка этой улицы нужна, чтобы назвать размер рейза: игра
        # просит итоговую ставку, а не доплату сверх чужой.
        self.assertEqual(table_state(frame("top-seat-buy-menu")).my_bet, 250)
        self.assertEqual(table_state(frame("check-and-bet")).my_bet, 0)
        self.assertIsNone(table_state(frame("table-idle")).my_bet)

    def test_known_seat_survives_the_showdown(self) -> None:
        state = table_state(frame("six-max-showdown"), known_seat=LEFT)

        self.assertTrue(state.showdown)
        self.assertEqual(state.hero_seat, LEFT)
        self.assertEqual(state.pot, 244_000)


if __name__ == "__main__":
    unittest.main()


class RingOrderTests(unittest.TestCase):
    """Порядок мест по кругу — чистая логика, кадры для неё не нужны."""

    def test_every_seat_appears_once(self) -> None:
        self.assertEqual(sorted(RING_ORDER), list(range(6)))

    def test_the_round_starts_after_the_button(self) -> None:
        self.assertEqual(
            seats_in_order((0, 1, 3, 5, 4, 2), dealer=BOTTOM_RIGHT),
            (BOTTOM_LEFT, LEFT, TOP_LEFT, TOP_RIGHT, RIGHT, BOTTOM_RIGHT),
        )

    def test_empty_seats_drop_out_of_the_round(self) -> None:
        # Свободные места круг пропускает, но порядок оставшихся не меняет:
        # после «слева-середина» идёт «справа-середина», и только за ней —
        # «низ-справа».
        self.assertEqual(
            seats_in_order((LEFT, RIGHT, BOTTOM_RIGHT), dealer=LEFT),
            (RIGHT, BOTTOM_RIGHT, LEFT),
        )

    def test_without_a_button_there_is_no_order(self) -> None:
        self.assertEqual(seats_in_order((LEFT, RIGHT), dealer=None), ())


class SeatPositionTests(unittest.TestCase):
    def test_blinds_sit_right_after_the_button(self) -> None:
        places = seat_positions((0, 1, 3, 5, 4, 2), dealer=BOTTOM_RIGHT)

        self.assertEqual(places[BOTTOM_RIGHT], "BTN")
        self.assertEqual(places[BOTTOM_LEFT], "SB")
        self.assertEqual(places[LEFT], "BB")

    def test_two_handed_play_puts_the_big_blind_on_the_button(self) -> None:
        # Правило одно и то же на любом числе игроков: малый блайнд — первое
        # живое место после кнопки, большой — второе. Когда игроков двое,
        # «второе» заворачивается обратно на кнопку. Отдельного случая для
        # стола на двоих не нужно — проверено по записи на шести раскладах.
        places = seat_positions((RIGHT, BOTTOM_RIGHT), dealer=RIGHT)

        self.assertEqual(places[BOTTOM_RIGHT], "SB")
        self.assertEqual(places[RIGHT], "BB")

    def test_the_rest_are_counted_back_from_the_button(self) -> None:
        places = seat_positions((0, 1, 3, 5, 4, 2), dealer=TOP_LEFT)

        self.assertEqual(places[TOP_LEFT], "BTN")
        self.assertEqual(places[LEFT], "CO")
        self.assertEqual(places[BOTTOM_LEFT], "MP")

    def test_a_lone_player_has_no_position(self) -> None:
        self.assertEqual(seat_positions((LEFT,), dealer=LEFT), {})

    def test_nobody_shares_a_position(self) -> None:
        for dealer in range(6):
            places = seat_positions((0, 1, 3, 5, 4, 2), dealer=dealer)
            self.assertEqual(len(set(places.values())), len(places), dealer)


class BlindSizeTests(unittest.TestCase):
    def test_a_clean_deal_gives_both_blinds(self) -> None:
        bets = [0] * 6
        bets[BOTTOM_LEFT], bets[LEFT] = 250, 500

        self.assertEqual(
            blinds_from_bets(tuple(bets), (LEFT, BOTTOM_LEFT, BOTTOM_RIGHT),
                             dealer=BOTTOM_RIGHT),
            (250, 500),
        )

    def test_a_raise_already_made_hides_the_blinds(self) -> None:
        # После чужого повышения по сукну размер блайнда не определить, а
        # ошибиться нельзя: глубина стека в блайндах выбирает режим совета.
        bets = [0] * 6
        bets[BOTTOM_LEFT], bets[LEFT], bets[BOTTOM_RIGHT] = 250, 500, 1500

        self.assertIsNone(
            blinds_from_bets(tuple(bets), (LEFT, BOTTOM_LEFT, BOTTOM_RIGHT),
                             dealer=BOTTOM_RIGHT)
        )

    def test_sizes_that_are_not_half_and_whole_are_refused(self) -> None:
        bets = [0] * 6
        bets[BOTTOM_LEFT], bets[LEFT] = 300, 500

        self.assertIsNone(
            blinds_from_bets(tuple(bets), (LEFT, BOTTOM_LEFT), dealer=BOTTOM_RIGHT)
        )

    def test_an_unread_bet_refuses_the_answer(self) -> None:
        bets = [0] * 6
        bets[BOTTOM_LEFT], bets[LEFT] = 250, None

        self.assertIsNone(
            blinds_from_bets(tuple(bets), (LEFT, BOTTOM_LEFT), dealer=BOTTOM_RIGHT)
        )

    def test_a_bet_and_a_raise_after_the_flop_are_not_blinds(self) -> None:
        # Кадр 122205 записи `8.mp4`: на борде три карты, на сукне 3 500 и
        # 7 000, чужих ставок больше нет — картинка неотличима от чистого
        # начала раздачи. Так блайнды и прочитались 3 500 / 7 000 при
        # настоящих 250 / 500, глубина стека вышла 4 ставки вместо шестидесяти,
        # и помощник три раздачи подряд советовал ва-банк.
        bets = [0] * 6
        bets[BOTTOM_LEFT], bets[LEFT] = 3_500, 7_000

        self.assertIsNone(
            blinds_from_bets(tuple(bets), (LEFT, BOTTOM_LEFT), dealer=BOTTOM_RIGHT,
                             board=("7s", "Ah", "Kd"))
        )
        self.assertEqual(
            blinds_from_bets(tuple(bets), (LEFT, BOTTOM_LEFT), dealer=BOTTOM_RIGHT),
            (3_500, 7_000),
            "до флопа те же ставки — честные блайнды",
        )


class MinimumSanityTests(unittest.TestCase):
    """Минимум с кнопки, который игра показать не могла."""

    def test_a_minimum_above_the_stack_is_refused(self) -> None:
        # Кадр 155505 записи `8.mp4`: на кнопке `RAISE 4 000`, курсор стоит на
        # четвёрке, прочиталось 654 000. Минимум работает полом для всех
        # четырёх быстрых кнопок, и они схлопнулись в одну — `ALL IN` на
        # 84 000 при банке 12 000.
        self.assertIsNone(sane_minimum(654_000, my_bet=0, to_call=3_500, stack=84_000))

    def test_a_minimum_above_a_double_raise_is_refused(self) -> None:
        # Минимальное повышение — чужая ставка плюс её же последний шаг:
        # вдвое больше старшей ставки оно не бывает. Стек тут ни при чём —
        # 65 400 в него укладываются, а в правила игры нет.
        self.assertIsNone(sane_minimum(65_400, my_bet=0, to_call=3_500, stack=84_000))

    def test_an_honest_minimum_goes_through(self) -> None:
        self.assertEqual(sane_minimum(4_000, my_bet=0, to_call=3_500, stack=84_000), 4_000)
        self.assertEqual(sane_minimum(500, my_bet=0, to_call=0, stack=84_000), 500)

    def test_a_minimum_equal_to_the_stack_is_the_shove(self) -> None:
        # Когда минимальное повышение упирается в стек, игра так и пишет —
        # и это честный минимум, а не мусор.
        self.assertEqual(
            sane_minimum(84_000, my_bet=0, to_call=42_000, stack=84_000), 84_000
        )

    def test_nothing_read_stays_nothing(self) -> None:
        self.assertIsNone(sane_minimum(None, my_bet=0, to_call=500, stack=1_000))


class DealerButtonTests(unittest.TestCase):
    def test_the_button_is_found_on_a_live_table(self) -> None:
        self.assertEqual(dealer_seat(frame("my-turn-with-hole-cards")), TOP_LEFT)
        self.assertEqual(dealer_seat(frame("cursor-over-call-amount")), BOTTOM_LEFT)
        self.assertEqual(dealer_seat(frame("right-seat-preflop")), LEFT)

    def test_a_finished_hand_has_no_button_to_show(self) -> None:
        # Раздача сыграна, панели приглушены, метки на экране нет. Ответ
        # «не вижу» тут честнее выдуманного места.
        self.assertIsNone(dealer_seat(frame("river-five-board-cards")))

    def test_the_state_carries_the_button_and_the_position(self) -> None:
        state = table_state(frame("my-turn-with-hole-cards"))

        self.assertEqual(state.dealer, TOP_LEFT)
        self.assertEqual(state.hero_seat, LEFT)
        self.assertEqual(state.position, "CO")
        self.assertEqual(state.blinds, (250, 500))

    def test_the_position_is_none_without_a_button(self) -> None:
        state = table_state(frame("river-five-board-cards"))

        self.assertIsNone(state.dealer)
        self.assertIsNone(state.position)


class AllInScreenTests(unittest.TestCase):
    """Экран без повышения: игра рисует только «ALL IN» и «FOLD».

    На нём помощник молчал семь раз из сорока в записи ``6.mp4`` — и это были
    ровно те решения, где на кону стоял весь стек.
    """

    def test_the_two_button_screen_is_my_turn(self) -> None:
        for name in ("allin-fold", "allin-fold-hover", "allin-fold-small"):
            frame_data = frame(name)
            self.assertTrue(all_in_only(frame_data), name)
            self.assertTrue(is_my_turn(frame_data), name)

    def test_the_usual_screens_are_not_mistaken_for_it(self) -> None:
        # Одной горящей кнопки «FOLD» мало: она горит и на обычном ходу, и у
        # преактива. Отличает пустая правая нижняя клетка.
        for name in ("my-turn-call-raise-fold", "check-and-bet", "slider-dragged",
                     "flop-opponent-turn", "turn-four-board-cards", "table-idle"):
            self.assertFalse(all_in_only(frame(name)), name)

    def test_the_amount_is_read_off_both_button_styles(self) -> None:
        # Белая кнопка пишет сумму малиновым, а под курсором заливается сама и
        # белит текст. Раньше на залитой отбрасывалось первое слово, и из
        # «ALL IN 4 500» первым уходило «ALL», а «IN» в число не складывалось.
        self.assertEqual(call_amount(frame("allin-fold")), 4500)
        self.assertEqual(call_amount(frame("allin-fold-hover")), 3000)
        self.assertEqual(call_amount(frame("allin-fold-small")), 500)

    def test_the_button_names_what_we_pay_not_what_he_bet(self) -> None:
        # Соперник поставил 2 500, а фишек у нас 500: игра пишет «ALL IN 500».
        # Раньше два счёта доплаты расходились — кнопка давала 500, сукно
        # 2 500, — и экран молчал.
        state = table_state(frame("allin-fold-small"))

        self.assertTrue(state.all_in_only)
        self.assertEqual(state.to_call, 500)
        self.assertEqual(state.stack, 500)
        self.assertEqual(state.faced_bet, 2500)

    def test_the_state_says_raising_is_impossible(self) -> None:
        self.assertTrue(table_state(frame("allin-fold")).all_in_only)
        self.assertFalse(table_state(frame("my-turn-call-raise-fold")).all_in_only)


class WinnerTests(unittest.TestCase):
    """Метка «WIN»: по ней журнал считает раздачи и их исход."""

    def test_the_winner_is_read_at_the_showdown(self) -> None:
        self.assertEqual(winning_seats(frame("six-max-showdown")), (TOP_RIGHT,))

    def test_the_yellow_felt_is_not_a_winner(self) -> None:
        # Сукно расписано жёлтым, и логотип «PLAY» лежит ровно под нижним
        # правым местом — то же золото, что и у метки. Отличается метка тем,
        # что залита целиком, а разметка — тонкие линии.
        for name in (
            "table-idle", "six-max-flop", "hand-finished", "my-turn-call-raise-fold",
            "right-seat-flop", "side-pots-in-hand", "allin-fold", "top-seat-buy-menu",
        ):
            with self.subTest(name=name):
                self.assertEqual(winning_seats(frame(name)), ())

    def test_the_state_carries_the_winner(self) -> None:
        state = table_state(frame("six-max-showdown"))

        self.assertEqual(state.winners, (TOP_RIGHT,))
        self.assertTrue(state.showdown)

    def test_the_showdown_shows_what_everyone_held(self) -> None:
        # Единственный случай, когда чужие карты вообще видны, — и первое,
        # что понадобится статистике по соперникам.
        shown = table_state(frame("six-max-showdown")).shown

        self.assertEqual(shown[TOP_LEFT], ("As", "7s"))
        self.assertEqual(shown[BOTTOM_LEFT], ("Ks", "Qd"))
        self.assertEqual(shown[BOTTOM_RIGHT], ("8c", "Tc"))

    def test_the_seat_is_not_re_read_while_the_mark_burns(self) -> None:
        # Кадр выплаты: банк уже уехал к соседу, метка «WIN» горит, и лицом на
        # столе лежит ровно одна рука. Раньше по такому кадру помощник
        # объявлял её место своим — а через кадр записывал чужие карты в
        # журнал отдельной раздачей. Место с прошлых кадров тут честнее того,
        # что видно: чьи карты открыты во время выплаты, по кадру не понять.
        state = table_state(frame("payout-win-mark"), known_seat=RIGHT)

        self.assertEqual(state.winners, (LEFT,))
        self.assertEqual(state.hero_seat, RIGHT)
        self.assertEqual(state.hole, (), "чужая рука своей не станет")

    def test_the_own_hand_is_still_read_at_the_payout(self) -> None:
        # Обратная сторона того же правила: если место уже известно, выплата
        # ничего не ломает — своя рука на нём и читается.
        state = table_state(frame("payout-win-mark"), known_seat=TOP_RIGHT)

        self.assertEqual(state.hero_seat, TOP_RIGHT)
        self.assertEqual(state.hole, ("Jd", "5s"))
        self.assertEqual(state.board, ("Ah", "Qs", "2h", "6s", "5c"))
        self.assertEqual(state.pot, 2_500)

    def test_before_the_showdown_only_our_own_hand_is_open(self) -> None:
        # До вскрытия у соперников рубашки, и читать там нечего: разбор карты
        # стоит своих миллисекунд на каждом кадре.
        state = table_state(frame("preflop-hole-cards"))

        self.assertEqual(state.shown[BOTTOM_RIGHT], ("Kh", "3h"))
        self.assertEqual(
            [seat for seat, cards in enumerate(state.shown) if cards], [BOTTOM_RIGHT]
        )


def clock(seconds: int | None) -> str:
    """Секунды от полуночи обратно в «11:59:13» — так их видно в кадре."""
    if seconds is None:
        return "—"
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


class EventLogTests(unittest.TestCase):
    """Строка событий: игра сама пишет, кто что сделал.

    Это надёжнее, чем сравнивать кадры между собой: при четырёх кадрах в
    секунду быстрый фолд теряется, а дырявый журнал хуже, чем никакого.
    """

    def test_the_strip_holds_every_word(self) -> None:
        self.assertEqual(len(word_templates()), 13)

    def test_the_log_is_read_line_by_line(self) -> None:
        events = event_log(frame("event-log-hand"))

        self.assertEqual(
            [event.action for event in events],
            ["конец", "кнопка", "начало", "блайнд", "блайнд",
             "рейз", "колл", "чек", "чек"],
        )

    def test_the_amount_stands_where_the_game_wrote_it(self) -> None:
        # «повысил до 100 фишек» — сумма есть; «пропустил ход» — нет, и
        # выдумывать её неоткуда.
        amounts = {event.action: event.amount for event in event_log(frame("event-log-hand"))}

        self.assertEqual(amounts["рейз"], 100)
        self.assertIsNone(amounts["чек"])
        self.assertIsNone(amounts["блайнд"], "блайнд игра суммой не подписывает")

    def test_the_time_tells_lines_apart(self) -> None:
        # Одни и те же строки приходят с каждым кадром, пока журнал не
        # прокрутится. Отличаются они отметкой времени — по ней и считаются.
        stamps = [clock(event.at) for event in event_log(frame("event-log-hand"))]

        self.assertEqual(stamps[0], "11:59:08")
        self.assertEqual(stamps[2], "11:59:13")
        self.assertEqual(stamps[-1], "11:59:25")

    def test_hand_borders_come_from_the_log(self) -> None:
        # «Началась новая игра» и «Игра закончена» игра пишет сама, без имени.
        actions = [event.action for event in event_log(frame("event-log-hand"))]

        self.assertIn("начало", actions)
        self.assertIn("конец", actions)

    def test_the_name_is_kept_as_a_fingerprint(self) -> None:
        # Имя читать незачем — нужно лишь узнавать, что это тот же человек.
        events = event_log(frame("event-log-hand"))
        named = [event for event in events if event.name is not None]

        self.assertTrue(named)
        for event in named:
            self.assertEqual(event.name.shape, (16, 128))
        # Строки, которые игра пишет ни от кого, отпечатка не несут.
        self.assertIsNone(next(e for e in events if e.action == "начало").name)

    def test_a_folded_log_stays_silent(self) -> None:
        # Свёрнутый журнал показывает одну строку, да и ту гасит через пару
        # секунд: разобрать по ней ход нельзя, и молчание тут честнее.
        self.assertEqual(event_log(frame("event-log-closed")), ())

    def test_no_log_no_events(self) -> None:
        for name in ("table-idle", "six-max-flop", "allin-fold"):
            with self.subTest(name=name):
                self.assertEqual(event_log(frame(name)), ())

    def test_the_state_carries_the_log(self) -> None:
        self.assertEqual(len(table_state(frame("event-log-hand")).events), 9)
        self.assertEqual(table_state(frame("table-idle")).events, ())
