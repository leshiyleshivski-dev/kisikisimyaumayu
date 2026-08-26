"""Проверки считалки холдема: сила руки, эквити, шансы банка, совет."""

from __future__ import annotations

import random
import unittest
from itertools import combinations

import numpy as np

from kisiki.modules.poker_math import (
    BET_POT_SHARE, FLUSH, FULL_HOUSE, HIGH_CARD, PAIR, QUADS, STRAIGHT,
    STRAIGHT_FLUSH, TRIPS, TWO_PAIR, bet_size, card_code, card_text, equity,
    equity_vs_range, evaluate, hand_category, hand_score, parse_cards, pot_odds,
    quick_bet_for, quick_bets, round_bet,
)
from kisiki.modules.poker_ranges import top_share


def slow_best_five(codes: list[int]) -> tuple:
    """Медленный эталон: перебрать все пятёрки и вернуть сравнимый ключ.

    Нужен ровно для одной проверки — что быстрый векторный оценщик
    расставляет руки в том же порядке, что и прямой перебор.
    """
    best = None
    for combo in combinations(codes, 5):
        ranks = sorted((code // 4 for code in combo), reverse=True)
        flush = len({code % 4 for code in combo}) == 1
        unique = sorted(set(ranks), reverse=True)
        straight = 0
        if len(unique) == 5:
            if unique[0] - unique[4] == 4:
                straight = unique[0]
            elif unique == [12, 3, 2, 1, 0]:
                straight = 3
        groups = sorted(((ranks.count(rank), rank) for rank in set(ranks)), reverse=True)
        shape = [count for count, _rank in groups]
        if straight and flush:
            key = (STRAIGHT_FLUSH, (straight,))
        elif shape[0] == 4:
            key = (QUADS, tuple(rank for _count, rank in groups))
        elif shape[:2] == [3, 2]:
            key = (FULL_HOUSE, tuple(rank for _count, rank in groups))
        elif flush:
            key = (FLUSH, tuple(ranks))
        elif straight:
            key = (STRAIGHT, (straight,))
        else:
            category = {3: TRIPS, 1: HIGH_CARD}.get(shape[0])
            if category is None:
                category = TWO_PAIR if shape[:2] == [2, 2] else PAIR
            key = (category, tuple(rank for _count, rank in groups))
        best = key if best is None or key > best else best
    return best


class CardParsingTests(unittest.TestCase):
    def test_card_survives_a_round_trip(self) -> None:
        for text in ("Ah", "2c", "Td", "Ks"):
            self.assertEqual(card_text(card_code(text)), text)

    def test_ten_is_accepted_in_both_spellings(self) -> None:
        self.assertEqual(card_code("10c"), card_code("Tc"))

    def test_repeated_card_is_rejected(self) -> None:
        # Дубль означает, что зрение прочитало один слот дважды: считать по
        # такому столу нельзя, эквити получится выдуманным.
        with self.assertRaises(ValueError):
            parse_cards(["Ah", "Ah"])


class HandRankingTests(unittest.TestCase):
    def category(self, *cards: str) -> int:
        return hand_category(cards)

    def test_every_category_is_recognised(self) -> None:
        cases = (
            (HIGH_CARD, ("Ah", "Kd", "9c", "7s", "3h", "2d", "5c")),
            (PAIR, ("Ah", "Ad", "9c", "7s", "3h", "2d", "5c")),
            (TWO_PAIR, ("Ah", "Ad", "9c", "9s", "3h", "2d", "5c")),
            (TRIPS, ("Ah", "Ad", "Ac", "9s", "3h", "2d", "5c")),
            (STRAIGHT, ("6h", "5d", "4c", "3s", "2h", "Kd", "Qc")),
            (FLUSH, ("Ah", "Kh", "9h", "5h", "2h", "5c", "5d")),
            (FULL_HOUSE, ("7c", "7d", "7h", "3c", "3d", "3h", "2s")),
            (QUADS, ("5c", "5d", "5h", "5s", "2c", "2d", "Ah")),
            (STRAIGHT_FLUSH, ("9h", "8h", "7h", "6h", "5h", "As", "Ks")),
        )
        for expected, cards in cases:
            self.assertEqual(self.category(*cards), expected, cards)

    def test_five_and_six_card_hands_are_scored_too(self) -> None:
        self.assertEqual(self.category("Ah", "Kh", "Qh", "Jh", "Th"), STRAIGHT_FLUSH)
        self.assertEqual(self.category("Ah", "Ad", "Kh", "Kd", "2c", "2d"), TWO_PAIR)

    def test_wheel_is_the_weakest_straight(self) -> None:
        # Туз в A-2-3-4-5 играет снизу, поэтому колесо проигрывает шестёрочному
        # стриту, а не выигрывает как «стрит от туза».
        wheel = hand_score(("Ac", "2d", "3h", "4s", "5c", "Kd", "Qh"))
        six_high = hand_score(("6c", "2d", "3h", "4s", "5c", "Kd", "Qh"))
        self.assertEqual(wheel // 13 ** 5, STRAIGHT)
        self.assertLess(wheel, six_high)

    def test_quad_kicker_ignores_the_side_pair(self) -> None:
        # Ради этого случая кикер выбирается только по старшинству: при
        # сортировке «сначала по числу карт» кикером стала бы двойка.
        with_ace = hand_score(("5c", "5d", "5h", "5s", "2c", "2d", "Ah"))
        with_king = hand_score(("5c", "5d", "5h", "5s", "2c", "2d", "Kh"))
        self.assertGreater(with_ace, with_king)

    def test_third_pair_never_becomes_a_kicker(self) -> None:
        with_ace = hand_score(("Kc", "Kd", "9c", "9d", "5c", "5d", "Ah"))
        with_three = hand_score(("Kc", "Kd", "9c", "9d", "5c", "5d", "3h"))
        self.assertGreater(with_ace, with_three)

    def test_full_house_takes_the_higher_trips(self) -> None:
        score = hand_score(("7c", "7d", "7h", "3c", "3d", "3h", "2s"))
        self.assertEqual(score, hand_score(("7c", "7d", "7h", "3c", "3d", "3h", "2d")))
        self.assertGreater(score, hand_score(("3c", "3d", "3h", "2c", "2d", "2h", "7s")))

    def test_categories_are_ordered(self) -> None:
        ladder = (
            ("Ah", "Kd", "9c", "7s", "3h", "2d", "5c"),
            ("Ah", "Ad", "9c", "7s", "3h", "2d", "5c"),
            ("Ah", "Ad", "9c", "9s", "3h", "2d", "5c"),
            ("Ah", "Ad", "Ac", "9s", "3h", "2d", "5c"),
            ("6h", "5d", "4c", "3s", "2h", "Kd", "Qc"),
            ("Ah", "Kh", "9h", "5h", "2h", "5c", "5d"),
            ("7c", "7d", "7h", "3c", "3d", "3h", "2s"),
            ("5c", "5d", "5h", "5s", "2c", "2d", "Ah"),
            ("9h", "8h", "7h", "6h", "5h", "As", "Ks"),
        )
        scores = [hand_score(hand) for hand in ladder]
        self.assertEqual(scores, sorted(scores))

    def test_fast_evaluator_matches_brute_force(self) -> None:
        # Главная проверка оценщика: на случайных руках он обязан давать тот
        # же порядок, что и прямой перебор всех пятёрок.
        rng = random.Random(7)
        hands = [rng.sample(range(52), 7) for _ in range(400)]
        fast = evaluate(np.array(hands))
        slow = [slow_best_five(hand) for hand in hands]

        for index, hand in enumerate(hands):
            self.assertEqual(int(fast[index]) // 13 ** 5, slow[index][0], hand)
        for left in range(0, len(hands), 2):
            right = left + 1
            fast_order = (int(fast[left]) > int(fast[right])) - (int(fast[left]) < int(fast[right]))
            slow_order = (slow[left] > slow[right]) - (slow[left] < slow[right])
            self.assertEqual(fast_order, slow_order, (hands[left], hands[right]))


class EquityTests(unittest.TestCase):
    def test_known_preflop_numbers(self) -> None:
        # Табличные значения: пара тузов против одной случайной руки берёт
        # около 85 %, две семёрки-разномастки против четверых — около 11 %.
        self.assertAlmostEqual(equity(("Ah", "As"), trials=20_000), 0.85, delta=0.02)
        self.assertAlmostEqual(
            equity(("2c", "7d"), opponents=4, trials=20_000), 0.11, delta=0.03
        )

    def test_unbeatable_hand_wins_always(self) -> None:
        self.assertEqual(equity(("Ah", "Kh"), ("Qh", "Jh", "Th"), trials=2_000), 1.0)

    def test_shared_board_splits_the_pot(self) -> None:
        # Роял-флеш на столе: у всех одна и та же рука, ничья — половина.
        self.assertEqual(equity(("2c", "3d"), ("Ah", "Kh", "Qh", "Jh", "Th")), 0.5)

    def test_more_opponents_lower_the_equity(self) -> None:
        heads_up = equity(("Ah", "As"), trials=8_000)
        crowd = equity(("Ah", "As"), opponents=4, trials=8_000)
        self.assertGreater(heads_up, crowd)

    def test_river_is_exact_and_matches_sampling(self) -> None:
        hole, board = ("Ah", "Kd"), ("Ac", "7d", "2s", "9h", "3c")
        exact = equity(hole, board)
        sampled = equity(hole, board, trials=20_000, seed=5)

        self.assertAlmostEqual(exact, sampled, delta=0.02)
        self.assertEqual(exact, equity(hole, board))

    def test_same_seed_gives_the_same_answer(self) -> None:
        # Совет не должен дрожать от кадра к кадру на одном и том же столе.
        first = equity(("Jh", "Th"), ("9h", "2c", "5d"), trials=4_000, seed=3)
        second = equity(("Jh", "Th"), ("9h", "2c", "5d"), trials=4_000, seed=3)
        self.assertEqual(first, second)

    def test_the_game_percentage_would_have_been_wrong(self) -> None:
        # Тот самый случай из разбора: игра показывала 18 %, потому что видит
        # только старшую карту и не считает флеш-дро.
        chance = equity(("Kh", "3h"), ("6h", "Qs", "8h"), trials=20_000)

        self.assertGreater(chance, 0.5)

    def test_impossible_tables_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            equity(("Ah",))
        with self.assertRaises(ValueError):
            equity(("Ah", "Ks"), ("2c", "3c", "4c", "5c", "6c", "7c"))
        with self.assertRaises(ValueError):
            equity(("Ah", "Ks"), ("Ah", "3c", "4c"))
        with self.assertRaises(ValueError):
            equity(("Ah", "Ks"), opponents=0)


class PotOddsTests(unittest.TestCase):
    def test_free_look_costs_nothing(self) -> None:
        self.assertEqual(pot_odds(500, 0), 0.0)

    def test_half_pot_call_needs_a_third(self) -> None:
        self.assertAlmostEqual(pot_odds(1000, 500), 1 / 3)

    def test_bigger_bet_demands_more_equity(self) -> None:
        self.assertGreater(pot_odds(1000, 2000), pot_odds(1000, 500))


class BetSizeTests(unittest.TestCase):
    def test_bet_is_two_thirds_of_the_grown_pot(self) -> None:
        self.assertEqual(bet_size(300), round_bet(300 * BET_POT_SHARE))

    def test_raise_adds_the_call_to_the_bet(self) -> None:
        # Поднять до = уравнять и сверху поставить свою долю банка.
        self.assertEqual(bet_size(900, 100), round_bet(100 + 1000 * BET_POT_SHARE))

    def test_raise_is_never_below_the_minimum_the_game_allows(self) -> None:
        # Игра не примет рейз меньше чужой ставки, а свой минимум пишет прямо
        # на кнопке — совет ниже него игрок просто не сможет выставить.
        self.assertEqual(bet_size(100, 500), 500 + 500)
        self.assertEqual(bet_size(150, 0, minimum=50), 100)
        self.assertEqual(bet_size(10, 0, minimum=50), 50)

    def test_stack_caps_everything(self) -> None:
        self.assertEqual(bet_size(9000, 100, my_bet=200, stack=300), 500)

    def test_sizes_are_round_enough_to_set_by_hand(self) -> None:
        # Ползунок двигают рукой на таймере: «поставь 1 617» бесполезно.
        self.assertEqual(round_bet(99), 100)
        self.assertEqual(round_bet(1617), 1600)
        self.assertEqual(round_bet(24), 25)
        self.assertEqual(round_bet(0), 0)


class QuickBetTests(unittest.TestCase):
    """Ряд «MIN / 3 BB / BANK / ALL IN» — четыре готовых размера в одно нажатие."""

    def named(self, bets) -> dict[str, int]:
        return {bet.name: bet.amount for bet in bets}

    def test_each_button_is_counted_from_what_the_screen_reads(self) -> None:
        # Кадр `my-turn-with-hole-cards`: банк 750, блайнд 500, доплата 500,
        # минимальное повышение 1 000 с кнопки, стек 4 500.
        bets = self.named(quick_bets(
            pot=750, to_call=500, my_bet=0, stack=4500, big_blind=500, minimum=1000,
        ))

        self.assertEqual(bets["MIN"], 1000)
        self.assertEqual(bets["3 BB"], 1500)
        # Банк — уравнять чужие 500 и поставить сверху столько, сколько станет
        # в банке после уравнивания.
        self.assertEqual(bets["BANK"], 500 + 1250)
        self.assertEqual(bets["ALL IN"], 4500)

    def test_buttons_are_capped_by_the_stack_and_renamed(self) -> None:
        # Игрок должен видеть, что ход за весь стек, какой бы кнопкой он его
        # ни поставил: «BANK» на коротком стеке — это и есть олл-ин.
        bets = quick_bets(
            pot=9000, to_call=0, my_bet=0, stack=2000, big_blind=500, minimum=500,
        )

        self.assertEqual(self.named(bets), {"MIN": 500, "3 BB": 1500, "ALL IN": 2000})

    def test_the_same_sum_is_offered_once(self) -> None:
        # Кадр `check-and-bet`: банк 150 при блайнде 50 — «3 BB» и «BANK»
        # ставят одно и то же, и предлагать выбор из двух одинаковых незачем.
        bets = quick_bets(
            pot=150, to_call=0, my_bet=0, stack=4250, big_blind=50, minimum=50,
        )

        self.assertEqual([bet.amount for bet in bets], [50, 150, 4250])

    def test_a_button_without_numbers_behind_it_is_dropped(self) -> None:
        # Блайнды помощник знает не всегда, минимум — тоже: он читается с
        # кнопки только пока ползунок не тронут. Выдумывать их нельзя.
        bets = self.named(quick_bets(pot=1000, to_call=0, my_bet=0, stack=5000))

        self.assertNotIn("MIN", bets)
        self.assertNotIn("3 BB", bets)
        self.assertEqual(bets["BANK"], 1000)

    def test_the_game_will_not_take_less_than_the_minimum(self) -> None:
        # Против чужого повышения три блайнда меньше минимального рейза: игра
        # подтянет такую кнопку до минимума, а значит поставит ровно то же,
        # что «MIN», и отдельной кнопкой она быть перестаёт.
        bets = self.named(quick_bets(
            pot=5000, to_call=2000, my_bet=0, stack=50_000,
            big_blind=500, minimum=4000,
        ))

        self.assertEqual(bets["MIN"], 4000)
        self.assertNotIn("3 BB", bets)

    def test_a_minimum_above_the_stack_does_not_swallow_the_buttons(self) -> None:
        # Минимум работает полом: все четыре суммы подтягиваются к нему, а
        # потолок стека обрезает их обратно — и от выбора остаётся одна
        # кнопка «ALL IN». Ровно так один кадр с курсором на цифре превращал
        # совет «жми MIN, 4 000» в «жми ALL IN, 84 000» при банке 12 000.
        bets = self.named(quick_bets(
            pot=12_000, to_call=3_500, my_bet=0, stack=84_000,
            big_blind=500, minimum=654_000,
        ))

        self.assertNotIn("MIN", bets, "такого минимума игра показать не могла")
        self.assertEqual(bets["BANK"], 19_000)
        self.assertEqual(bets["ALL IN"], 84_000)


class QuickBetChoiceTests(unittest.TestCase):
    BETS = quick_bets(
        pot=750, to_call=500, my_bet=0, stack=10_000, big_blind=500, minimum=1000,
    )

    def test_the_open_goes_to_the_three_blind_button(self) -> None:
        # Таблица открывает в два с половиной блайнда, а кнопки ставят два и
        # три. Округляем вверх: мельчить с открытием — звать в раздачу лишних
        # людей задёшево.
        self.assertEqual(quick_bet_for(self.BETS, 1250).name, "3 BB")

    def test_the_shove_goes_to_the_all_in_button(self) -> None:
        self.assertEqual(quick_bet_for(self.BETS, 10_000).name, "ALL IN")

    def test_a_size_between_buttons_falls_back_to_the_slider(self) -> None:
        # 3 000 — это вдвое больше «3 BB» и вдвое меньше «BANK» ни туда ни
        # сюда: такую ставку игрок ведёт ползунком, и врать про кнопку нельзя.
        self.assertIsNone(quick_bet_for(self.BETS, 3000))
        self.assertIsNone(quick_bet_for(self.BETS, 0))
        self.assertIsNone(quick_bet_for((), 1250))

    def test_a_button_slightly_below_still_counts(self) -> None:
        # Сверху ничего близкого нет — берём самую крупную из тех, что ниже.
        self.assertEqual(quick_bet_for(self.BETS, 1900).name, "BANK")


if __name__ == "__main__":
    unittest.main()
