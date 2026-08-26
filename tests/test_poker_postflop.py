"""Проверки постфлопа: диапазон соперника на борде и совет по нему."""

from __future__ import annotations

import unittest

import numpy as np

from kisiki.modules.poker_math import (
    BET_POT_SHARE, Advice, equity, equity_vs_range, quick_bets,
)
from kisiki.modules.poker_postflop import (
    BIG_BET_KEEP, CALLER_KEEP, CHECK_KEEP, LIVE_RANGE_SHARE, MAX_BET_SHARE,
    PLAN_TRIALS, SHOVE_BLUFF, SHOVE_KEEP, SMALL_BET_KEEP, VALUE_BET_EDGE,
    aggression_bluff, aggression_keep, bet_share, caller_keep, choose_quick_bet,
    facing_bet, keep_share, narrow_to_board, opponent_range, plan_line,
    postflop_advice, raise_plan, range_combos, ranked_on_board, reply_sizes,
    strength_on_board, value_bet_equity,
)
from kisiki.modules.poker_ranges import hand_class, top_share


def classes_of(combos: np.ndarray) -> set[str]:
    """Какие классы рук остались в отфильтрованном диапазоне."""
    return {hand_class([int(first), int(second)]) for first, second in combos}


class AggressionTests(unittest.TestCase):
    def test_a_bigger_bet_means_a_narrower_range(self) -> None:
        # Банк приходит уже вместе со ставкой, поэтому из тысячи ставка 400 —
        # это 400 в банк 600, то есть две трети банка.
        keeps = [aggression_keep(1000, call) for call in (0, 200, 400, 600)]

        self.assertEqual(keeps, [CHECK_KEEP, SMALL_BET_KEEP, BIG_BET_KEEP, SHOVE_KEEP])
        self.assertEqual(keeps, sorted(keeps, reverse=True))

    def test_it_is_measured_in_pot_shares_and_not_in_chips(self) -> None:
        # Банки за этими столами отличаются на порядки: 500 фишек — это и
        # половина банка, и его сороковая часть.
        self.assertEqual(aggression_keep(1000, 500), aggression_keep(100_000, 50_000))

    def test_our_own_stack_does_not_change_what_he_bet(self) -> None:
        # Соперник выбирает размер, не зная, сколько осталось у нас. Ставка в
        # треть банка не превращается в олл-ин оттого, что нам её нечем
        # доплатить, — а пока в счёт шёл обрезанный стеком остаток, огромная
        # ставка читалась как маленькая.
        self.assertEqual(aggression_keep(4000, 1000), SMALL_BET_KEEP)
        self.assertEqual(bet_share(4000, 1000), 1000 / 3000)

    def test_the_bet_is_measured_against_the_pot_before_it(self) -> None:
        self.assertAlmostEqual(bet_share(2000, 1000), 1.0, msg="ставка в банк")
        self.assertAlmostEqual(bet_share(1500, 500), 0.5, msg="половина банка")
        self.assertEqual(bet_share(1000, 0), 0.0)

    def test_nothing_bet_says_nothing_about_the_hand(self) -> None:
        self.assertEqual(aggression_keep(1000, 0), 1.0)


class StrengthOnBoardTests(unittest.TestCase):
    FLOP = ("6h", "Qs", "8h")

    def test_a_made_hand_outranks_air(self) -> None:
        pair, air = strength_on_board(
            np.array([
                [self.code("Qc"), self.code("Qd")],
                [self.code("2c"), self.code("3d")],
            ], dtype=np.int64),
            self.FLOP,
        )

        self.assertGreater(pair, air)

    def test_a_draw_is_not_buried_at_the_bottom(self) -> None:
        # Флеш-дро сейчас не собрало ничего, но повышают с ним постоянно.
        # Считать силу по готовой комбинации значило бы выкинуть его из
        # диапазона соперника — и завысить своё эквити.
        draw, air = strength_on_board(
            np.array([
                [self.code("Ah"), self.code("3h")],
                [self.code("2c"), self.code("3d")],
            ], dtype=np.int64),
            self.FLOP,
        )

        self.assertGreater(draw, air)

    def test_the_answer_is_a_share_between_zero_and_one(self) -> None:
        combos = range_combos(top_share(0.4), np.zeros(52, dtype=bool))
        strength = strength_on_board(combos, ("Kd", "7c", "2h"))

        self.assertEqual(strength.shape, (combos.shape[0],))
        self.assertTrue(np.all((strength >= 0) & (strength <= 1)))

    def test_the_river_needs_no_guessing(self) -> None:
        # Доборов нет, значит ответ точный и повторяемый до последнего знака.
        board = ("Ah", "Kc", "Tc", "5c", "2h")
        combos = np.array([
            [self.code("Ad"), self.code("As")],
            [self.code("7d"), self.code("3s")],
        ], dtype=np.int64)

        first = strength_on_board(combos, board, seed=1)
        second = strength_on_board(combos, board, seed=99)

        np.testing.assert_array_equal(first, second)
        self.assertGreater(first[0], first[1])

    @staticmethod
    def code(text: str) -> int:
        from kisiki.modules.poker_math import card_code
        return card_code(text)


class NarrowingTests(unittest.TestCase):
    BOARD = ("Kd", "7c", "2h")

    def test_keeping_everything_changes_nothing(self) -> None:
        whole = narrow_to_board(top_share(0.5), self.BOARD, 1.0)
        combos = range_combos(top_share(0.5), self.zero_blocked())

        self.assertEqual(whole.shape[0], combos.shape[0])

    def test_a_narrower_keep_leaves_fewer_hands(self) -> None:
        wide = narrow_to_board(top_share(0.5), self.BOARD, 0.6)
        tight = narrow_to_board(top_share(0.5), self.BOARD, 0.2)

        self.assertLess(tight.shape[0], wide.shape[0])
        self.assertAlmostEqual(tight.shape[0] / wide.shape[0], 0.2 / 0.6, places=1)

    def test_the_narrow_range_is_the_one_that_hit_the_board(self) -> None:
        # На борде K-7-2 у того, кто пошёл ва-банк, должны остаться короли и
        # старшие пары, а не связки мимо.
        tight = classes_of(narrow_to_board(top_share(0.5), self.BOARD, 0.15))

        self.assertTrue({"KK", "77", "22"} & tight, "сет обязан остаться")
        self.assertNotIn("54s", tight)

    def test_cards_already_on_the_table_are_not_dealt_again(self) -> None:
        combos = narrow_to_board(
            top_share(0.6), self.BOARD, 0.5, hole=("Ah", "Kh")
        )
        used = {self.code(card) for card in self.BOARD + ("Ah", "Kh")}

        self.assertFalse(used & set(combos.ravel().tolist()))

    def test_a_bigger_bet_narrows_the_opponent(self) -> None:
        small = opponent_range(self.BOARD, hole=("Ah", "Qd"), keep=SMALL_BET_KEEP)
        shove = opponent_range(self.BOARD, hole=("Ah", "Qd"), keep=SHOVE_KEEP)

        self.assertLess(shove.shape[0], small.shape[0])

    def zero_blocked(self) -> np.ndarray:
        blocked = np.zeros(52, dtype=bool)
        blocked[[self.code(card) for card in self.BOARD]] = True
        return blocked

    @staticmethod
    def code(text: str) -> int:
        from kisiki.modules.poker_math import card_code
        return card_code(text)


class CallerKeepTests(unittest.TestCase):
    """Кто ответит на нашу ставку — та же модель, только развёрнутая на себя."""

    def keep(self, pot_before: int, bet: int) -> float:
        return caller_keep(pot_before + bet, bet)

    def test_the_bigger_the_bet_the_fewer_answer(self) -> None:
        keeps = [self.keep(1000, bet) for bet in (250, 500, 1000, 2000)]

        self.assertEqual(keeps, sorted(keeps, reverse=True))
        self.assertAlmostEqual(self.keep(1000, 1000), 0.5, places=2)
        self.assertAlmostEqual(self.keep(1000, 250), 0.8, places=2)

    def test_a_checked_street_keeps_everyone(self) -> None:
        self.assertEqual(caller_keep(1000, 0), CHECK_KEEP)

    def test_an_overbet_keeps_narrowing(self) -> None:
        # Ради этого модель и стала непрерывной: на ступеньках ставка в банк и
        # олл-ин на десять банков попадали в одну и ту же долю отвечающих, а
        # денег на олл-ине уходило больше — и помощник всегда выбирал олл-ин.
        self.assertLess(self.keep(1000, 10_000), self.keep(1000, 1000) / 4)


class QuickBetChoiceTests(unittest.TestCase):
    BOARD = ("9d", "2c", "Ks")

    def ranking(self, hole) -> np.ndarray:
        return ranked_on_board(top_share(0.55), self.BOARD, hole=hole)

    def test_the_biggest_button_is_not_always_the_answer(self) -> None:
        # Сет на сухом борде: олл-ин в десять банков — способ остаться без
        # соперника, а не выиграть больше. Модель обязана выбрать размер, на
        # который отвечают.
        hole = ("9h", "9c")
        bets = quick_bets(
            pot=1000, to_call=0, my_bet=0, stack=10_000, big_blind=500, minimum=500,
        )
        chosen = choose_quick_bet(
            hole, self.BOARD, bets, ranked=self.ranking(hole), pot=1000, opponents=1,
        )

        self.assertNotEqual(chosen.name, "ALL IN")
        self.assertLessEqual(chosen.amount, MAX_BET_SHARE * 1000)

    def test_a_short_stack_goes_all_in(self) -> None:
        # Стек меньше банка: любая ставка тут и есть олл-ин, и мельчить нечем.
        hole = ("9h", "9c")
        bets = quick_bets(
            pot=3000, to_call=0, my_bet=0, stack=2000, big_blind=500, minimum=500,
        )
        chosen = choose_quick_bet(
            hole, self.BOARD, bets, ranked=self.ranking(hole), pot=3000, opponents=1,
        )

        self.assertEqual(chosen.name, "ALL IN")

    def test_without_a_pot_no_button_is_named(self) -> None:
        hole = ("9h", "9c")
        bets = quick_bets(pot=0, to_call=0, my_bet=0, stack=2000, big_blind=500)

        self.assertIsNone(choose_quick_bet(
            hole, self.BOARD, bets, ranked=self.ranking(hole), pot=0, opponents=1,
        ))

    def test_the_advice_names_the_button_it_would_press(self) -> None:
        tip = postflop_advice(
            ("Ah", "Kd"), self.BOARD, opponents=1, pot=1000, to_call=0,
            stack=10_000, min_bet=500, big_blind=500, trials=4_000,
        )

        self.assertEqual(tip.action, "бет")
        self.assertIn(tip.button, ("MIN", "3 BB", "BANK", "ALL IN"))
        self.assertEqual(
            tip.raise_to,
            {bet.name: bet.amount for bet in quick_bets(
                pot=1000, to_call=0, my_bet=0, stack=10_000,
                big_blind=500, minimum=500,
            )}[tip.button],
        )

    def test_without_the_numbers_behind_the_row_the_slider_stays(self) -> None:
        # Ни блайндов, ни минимума, ни стека: считать «BANK» есть из чего, а
        # остальные кнопки выдумывать нельзя.
        tip = postflop_advice(
            ("Ah", "Kd"), self.BOARD, opponents=1, pot=1000, to_call=0, trials=4_000
        )

        self.assertEqual(tip.button, "BANK")


class RankingReuseTests(unittest.TestCase):
    """Расстановка по силе считается один раз и режется на любые доли."""

    BOARD = ("Kd", "7c", "2h")

    def test_a_slice_of_the_ranking_is_the_same_range(self) -> None:
        names = top_share(0.5)
        sliced = keep_share(ranked_on_board(names, self.BOARD), 0.2)
        whole = narrow_to_board(names, self.BOARD, 0.2)

        self.assertEqual(sliced.shape, whole.shape)
        self.assertEqual(
            {tuple(sorted(pair)) for pair in sliced.tolist()},
            {tuple(sorted(pair)) for pair in whole.tolist()},
        )

    def test_keeping_everything_needs_no_ranking(self) -> None:
        # Расстановка стоит дороже всего остального разбора, и когда доля
        # единица, она не нужна: набор от неё не зависит, только порядок.
        names = top_share(0.5)
        whole = narrow_to_board(names, self.BOARD, CHECK_KEEP)

        self.assertEqual(whole.shape[0], ranked_on_board(names, self.BOARD).shape[0])


class RangeEquityTests(unittest.TestCase):
    def test_the_number_that_cost_the_stack(self) -> None:
        # Тёрн 4♣5♠7♣9♥, пара девяток с кикером «тройка». Против случайной
        # руки — под три четверти раздач, против того, с чем идут ва-банк, —
        # втрое меньше. На этой разнице в записи 6.mp4 и уехал стек.
        hole, board = ("3s", "9d"), ("4c", "5s", "7c", "9h")
        random_hand = equity(hole, board, opponents=1, trials=8_000)
        shoving = equity_vs_range(
            hole, board,
            ranges=[opponent_range(board, hole=hole, keep=SHOVE_KEEP)],
            trials=8_000,
        )

        self.assertGreater(random_hand, 0.6)
        self.assertLess(shoving, random_hand / 2)

    def test_ace_high_beats_nobody_who_shoves_the_river(self) -> None:
        # Ривер, доборов нет, туз-хай не бьёт ни одной руки из тех, с которыми
        # ходят ва-банк. Старая считалка показывала тут 26 % при шансах банка
        # 23 % — то есть звала доплатить рукой, которая не выигрывает никогда.
        hole, board = ("Ac", "Qh"), ("9d", "7s", "4d", "6s", "Tc")
        chance = equity_vs_range(
            hole, board,
            ranges=[opponent_range(board, hole=hole, keep=SHOVE_KEEP)],
            trials=4_000,
        )

        self.assertLess(chance, 0.05)
        self.assertGreater(equity(hole, board, opponents=1, trials=4_000), 0.2)


class PostflopAdviceTests(unittest.TestCase):
    def test_monster_raises(self) -> None:
        advice = postflop_advice(
            ("Ah", "Kh"), ("Qh", "Jh", "Th"), pot=1000, to_call=200, trials=2_000
        )

        self.assertIsInstance(advice, Advice)
        self.assertEqual(advice.action, "рейз")
        self.assertGreater(advice.raise_to, 200)

    def test_hopeless_hand_folds_to_a_big_bet(self) -> None:
        advice = postflop_advice(
            ("2c", "7d"), ("Ah", "Kd", "Qs"), pot=500, to_call=2000, trials=4_000
        )

        self.assertEqual(advice.action, "фолд")
        self.assertLess(advice.equity, advice.odds)

    def test_free_card_is_taken_with_a_weak_hand(self) -> None:
        advice = postflop_advice(
            ("2c", "7d"), ("Ah", "Kd", "Qs"), pot=500, to_call=0, trials=4_000
        )

        self.assertEqual(advice.action, "чек")
        self.assertEqual(advice.raise_to, 0)

    def test_a_draw_calls_a_cheap_price(self) -> None:
        advice = postflop_advice(
            ("Kh", "3h"), ("6h", "Qs", "8h"), pot=1000, to_call=150, trials=8_000
        )

        self.assertIn(advice.action, {"колл", "рейз"})
        self.assertGreater(advice.equity, advice.odds)

    def test_reason_names_both_numbers(self) -> None:
        advice = postflop_advice(
            ("Ah", "As"), ("Ad", "Kd", "7c"), pot=1000, to_call=500, trials=2_000
        )

        self.assertIn("%", advice.reason)
        self.assertIn("эквити", advice.reason)
        self.assertIn("диапазон", advice.reason)

    def test_advice_names_the_made_hand(self) -> None:
        # Комбинацию видно на столе, но считать её в уме на таймере некогда.
        advice = postflop_advice(
            ("Kh", "Ks"), ("Kd", "7c", "2h"), pot=500, to_call=0, trials=2_000
        )

        self.assertEqual(advice.made, "тройка")

    def test_value_threshold_follows_the_number_of_opponents(self) -> None:
        # Против одного соперника средняя рука берёт половину раздач, против
        # пятерых — шестую часть. Один порог на все столы не годится.
        self.assertAlmostEqual(value_bet_equity(1), 0.5 + VALUE_BET_EDGE)
        self.assertAlmostEqual(value_bet_equity(5), 1 / 6 + VALUE_BET_EDGE)
        self.assertGreater(value_bet_equity(1), value_bet_equity(3))

    def test_strong_hand_bets_into_a_crowd(self) -> None:
        advice = postflop_advice(
            ("Kh", "Ks"), ("Kd", "7c", "2h"), opponents=4, pot=800, to_call=0,
            trials=4_000,
        )

        self.assertEqual(advice.action, "бет")
        self.assertGreater(advice.raise_to, 0)

    def test_nothing_to_call_is_a_bet_and_not_a_raise(self) -> None:
        # Пока доплаты нет, в игре нарисованы «CHECK» и «BET»: кнопки «RAISE»
        # там просто не существует, и советовать рейз некуда.
        hole, board = ("Kh", "Ks"), ("Kd", "7c", "2h")

        self.assertEqual(
            postflop_advice(hole, board, pot=500, to_call=0, trials=2_000).action, "бет"
        )
        self.assertEqual(
            postflop_advice(hole, board, pot=500, to_call=100, trials=2_000).action,
            "рейз",
        )

    def test_raise_size_counts_the_call_into_the_pot(self) -> None:
        advice = postflop_advice(
            ("Ah", "Ad"), ("Ac", "Kd", "7c"), pot=900, to_call=100, trials=2_000
        )

        self.assertEqual(advice.action, "рейз")
        self.assertGreaterEqual(advice.raise_to, round(1000 * BET_POT_SHARE))

    def test_raise_size_is_the_whole_bet_on_the_street(self) -> None:
        advice = postflop_advice(
            ("Ah", "Ad"), ("Ac", "Kd", "7c"), pot=900, to_call=100, my_bet=200,
            trials=2_000,
        )

        self.assertGreater(advice.raise_to, 200 + 100)

    def test_advice_never_asks_for_more_than_the_stack(self) -> None:
        advice = postflop_advice(
            ("Ah", "Ad"), ("Ac", "Kd", "7c"), pot=9000, to_call=100, stack=400,
            trials=2_000,
        )

        self.assertEqual(advice.raise_to, 400)
        self.assertTrue(advice.all_in)

    def test_the_flop_is_the_earliest_it_will_speak(self) -> None:
        # До флопа считает таблица: перебор против случайных карт там советует
        # играть всё подряд. Молча выдавать по нему совет нельзя.
        with self.assertRaises(ValueError):
            postflop_advice(("Ah", "As"), ("Kd", "7c"), pot=500, to_call=0)

    def test_the_same_table_gives_the_same_advice(self) -> None:
        # Совет пересчитывается по четыре раза в секунду и дрожать не имеет
        # права: иначе ему нельзя верить.
        first = postflop_advice(
            ("Kh", "3h"), ("6h", "Qs", "8h"), pot=1000, to_call=250, trials=4_000
        )
        second = postflop_advice(
            ("Kh", "3h"), ("6h", "Qs", "8h"), pot=1000, to_call=250, trials=4_000
        )

        self.assertEqual(first, second)

    def test_the_call_range_is_wider_than_the_shoving_range(self) -> None:
        # По первому решают, стоит ли ставить, по второму — стоит ли платить.
        # Спутать их значит либо не поставить ни разу, либо платить всегда.
        self.assertGreater(CALLER_KEEP, SHOVE_KEEP)


class SpeedTests(unittest.TestCase):
    def test_a_full_answer_fits_the_move_timer(self) -> None:
        # Игра даёт на ход около пятнадцати секунд, а экран смотрит четыре раза
        # в секунду: считать дольше четверти секунды нельзя — очередь таймеров
        # начнёт копиться.
        import time

        started = time.perf_counter()
        postflop_advice(
            ("Kh", "3h"), ("6h", "Qs", "8h"), opponents=2, pot=1000, to_call=250
        )
        spent = time.perf_counter() - started

        self.assertLess(spent, 1.0, f"совет считался {spent:.2f} с")


if __name__ == "__main__":
    unittest.main()


class BluffMixTests(unittest.TestCase):
    def test_a_checked_street_holds_no_bluffs(self) -> None:
        self.assertEqual(aggression_bluff(1000, 0), 0.0)

    def test_a_bigger_bet_carries_more_bluff(self) -> None:
        shares = [aggression_bluff(1000, call) for call in (200, 400, 600)]

        self.assertEqual(shares, sorted(shares))
        self.assertEqual(shares[-1], SHOVE_BLUFF)

    def test_the_bluffs_come_from_the_bottom_of_the_range(self) -> None:
        # Блефуют пустой рукой: ей вскрытие не выиграть, и ставка —
        # единственный способ забрать банк. Средние руки не блефуют.
        board = ("Kd", "7c", "2h")
        pure = narrow_to_board(top_share(0.5), board, 0.2)
        mixed = narrow_to_board(top_share(0.5), board, 0.2, bluff=0.25)

        self.assertGreater(mixed.shape[0], pure.shape[0])
        added = classes_of(mixed) - classes_of(pure)
        self.assertTrue(added, "блефы обязаны появиться")
        self.assertNotIn("KK", added, "сет — не блеф")

    def test_a_bluff_catcher_beats_the_bluffs_and_nothing_else(self) -> None:
        # Туз-хай на ривере бьёт ровно блефы. Без них у него выходит ноль, и
        # помощник сбрасывал бы там, где шансы банка зовут платить.
        hole, board = ("Ac", "Qh"), ("9d", "7s", "4d", "6s", "Tc")
        honest = equity_vs_range(
            hole, board,
            ranges=[opponent_range(board, hole=hole, keep=SHOVE_KEEP)],
            trials=4_000,
        )
        with_bluffs = equity_vs_range(
            hole, board,
            ranges=[opponent_range(
                board, hole=hole, keep=SHOVE_KEEP, bluff=SHOVE_BLUFF
            )],
            trials=4_000,
        )

        self.assertLess(honest, 0.05)
        self.assertGreater(with_bluffs, honest)


class ShortStackPriceTests(unittest.TestCase):
    """Когда фишек меньше чужой ставки, чисел два и они разные."""

    FLOP = ("4d", "3s", "4s")

    def test_his_bet_reads_the_range_and_ours_pays_the_price(self) -> None:
        # Соперник поставил 2 500 в банк, где было 2 000, — это ставка больше
        # банка. Фишек у нас 500. Пока в чтение диапазона шло 500, огромная
        # ставка читалась как ставка в восьмую банка.
        big = postflop_advice(
            ("Td", "9d"), self.FLOP, pot=4500, to_call=500, faced_bet=2500,
            stack=500, can_raise=False, trials=4_000,
        )
        small = postflop_advice(
            ("Td", "9d"), self.FLOP, pot=4500, to_call=500, faced_bet=500,
            stack=500, can_raise=False, trials=4_000,
        )

        self.assertGreater(big.odds, small.odds)

    def test_his_uncallable_excess_is_not_ours_to_win(self) -> None:
        # Всё, что он поставил сверх нашего стека, вернётся ему: разыгрывается
        # только та часть банка, которую мы в состоянии уравнять. Без этой
        # поправки шансы банка выходили заманчивее, чем есть.
        advice = postflop_advice(
            ("Td", "9d"), self.FLOP, pot=4500, to_call=500, faced_bet=2500,
            stack=500, can_raise=False, trials=2_000,
        )

        self.assertAlmostEqual(advice.odds, 500 / (4500 - 2000 + 500), places=3)

    def test_without_the_bet_it_falls_back_on_what_we_pay(self) -> None:
        # Ставки на сукне читаются не всегда. Тогда остаётся кнопка — она
        # занижает чужую ставку, но врать в другую сторону хуже.
        advice = postflop_advice(
            ("Td", "9d"), self.FLOP, pot=4500, to_call=500, stack=500,
            can_raise=False, trials=2_000,
        )

        self.assertAlmostEqual(advice.odds, 500 / 5000, places=3)


class NoRaiseTests(unittest.TestCase):
    def test_a_monster_calls_when_there_is_no_raise_button(self) -> None:
        # Кнопки повышения на экране нет вовсе — совет «рейз» отправлял бы
        # искать то, чего не существует.
        table = dict(pot=1000, to_call=200, trials=2_000)
        self.assertEqual(
            postflop_advice(("Ah", "Kh"), ("Qh", "Jh", "Th"), **table).action, "рейз"
        )
        self.assertEqual(
            postflop_advice(
                ("Ah", "Kh"), ("Qh", "Jh", "Th"), can_raise=False, **table
            ).action,
            "колл",
        )

    def test_folding_is_still_folding(self) -> None:
        advice = postflop_advice(
            ("2c", "7d"), ("Ah", "Kd", "Qs"), pot=500, to_call=2000,
            can_raise=False, trials=4_000,
        )

        self.assertEqual(advice.action, "фолд")


class MultiwayTests(unittest.TestCase):
    """Ставил один — узкий диапазон достаётся только ему."""

    BOARD = ("4d", "3s", "4s")

    def test_only_the_bettor_gets_the_bettor_range(self) -> None:
        # Раздать узкий диапазон всем за столом значило бы посадить против
        # себя четверых с готовой рукой сразу. Остальные просто ещё держат
        # карты, и их руки шире.
        table = dict(pot=4500, to_call=500, faced_bet=2500, stack=500,
                     can_raise=False, trials=4_000)
        alone = postflop_advice(("Td", "9d"), self.BOARD, opponents=1, **table)
        crowd = postflop_advice(("Td", "9d"), self.BOARD, opponents=3, **table)

        self.assertGreater(alone.equity, crowd.equity)
        self.assertGreater(crowd.equity, 0.0)

    def test_the_price_does_not_move_with_the_crowd(self) -> None:
        # Шансы банка считаются по деньгам, а не по числу голов: столько же
        # платим, столько же выигрываем.
        table = dict(pot=4500, to_call=500, faced_bet=2500, stack=500,
                     can_raise=False, trials=2_000)
        alone = postflop_advice(("Td", "9d"), self.BOARD, opponents=1, **table)
        crowd = postflop_advice(("Td", "9d"), self.BOARD, opponents=3, **table)

        self.assertAlmostEqual(alone.odds, crowd.odds, places=6)

    def test_an_unbet_street_treats_everyone_the_same(self) -> None:
        # Никто не ставил — значит и выделять некого: все одинаково те, кто
        # ответит на нашу ставку.
        advice = postflop_advice(
            ("Kh", "Ks"), ("Kd", "7c", "2h"), opponents=3, pot=800, to_call=0,
            trials=2_000,
        )

        self.assertEqual(advice.action, "бет")


class PlanTests(unittest.TestCase):
    """План на ответ соперника: что делать, когда он поставит или повысит.

    Ради этого пункт и брался. «Чек» ничего не говорил о том, что делать,
    когда соперник поставит, — а фишки уходят не на чеке, а сразу после него.
    """

    def test_a_hopeless_hand_folds_to_anything(self) -> None:
        advice = postflop_advice(
            ("2c", "7d"), ("Ah", "Kd", "Qs"), pot=1000, stack=4000, trials=2_000
        )

        self.assertEqual(advice.action, "чек")
        self.assertEqual(advice.plan, "если поставит — фолд")

    def test_a_draw_names_the_price_it_pays_up_to(self) -> None:
        # Флеш-дро на чеке доигрывают, но не любой ценой: план и называет ту
        # цену, до которой доплата ещё окупается.
        advice = postflop_advice(
            ("Kh", "3h"), ("6h", "Qs", "8h"), pot=1000, stack=4000, trials=2_000
        )

        self.assertEqual(advice.action, "чек")
        self.assertIn("колл", advice.plan)
        self.assertTrue(advice.plan.endswith("крупнее — фолд"), advice.plan)

    def test_a_bet_plans_for_the_raise(self) -> None:
        # Ставим сами — отвечать будет он же, только повышением.
        advice = postflop_advice(
            ("9d", "9s"), ("9h", "7c", "2h"), pot=1000, stack=4000,
            min_bet=100, big_blind=50, trials=2_000,
        )

        self.assertEqual(advice.action, "бет")
        self.assertTrue(advice.plan.startswith("если повысит"), advice.plan)

    def test_only_our_own_move_gets_a_plan(self) -> None:
        # После колла улица закрыта, а сбросив карты, мы из раздачи вышли:
        # отвечать в обоих случаях уже не на что.
        called = postflop_advice(
            ("Ad", "Ks"), ("Ah", "7c", "2h"), pot=1500, to_call=500, stack=4000,
            can_raise=False, trials=2_000,
        )
        folded = postflop_advice(
            ("2c", "7d"), ("Ah", "Kd", "Qs"), pot=500, to_call=2000, trials=2_000
        )

        self.assertEqual((called.action, called.plan), ("колл", ""))
        self.assertEqual((folded.action, folded.plan), ("фолд", ""))

    def test_a_bet_for_the_whole_stack_has_nothing_left_to_plan(self) -> None:
        # Над своим олл-ином не повышают: дальше решает соперник, а не мы.
        board = ("9h", "7c", "2h")
        ranked = ranked_on_board(top_share(LIVE_RANGE_SHARE), board, hole=("9d", "9s"))

        self.assertEqual(
            raise_plan(
                ("9d", "9s"), board, ranked=ranked, opponents=1, pot=1000,
                bet_to=4000, stack=4000,
            ),
            "",
        )

    def test_the_plan_agrees_with_the_advice_that_comes_next(self) -> None:
        # Обещать одно, а через кадр советовать другое — значит не годиться
        # ни на что. Правило у плана и у живого совета одно и то же, и это
        # проверяется на всей лесенке размеров.
        hole, board, pot, stack = ("9d", "9s"), ("Ah", "7c", "2h"), 1000, 4000
        ranked = ranked_on_board(top_share(LIVE_RANGE_SHARE), board, hole=hole)
        for size in reply_sizes(pot, stack):
            with self.subTest(size=size):
                planned, _, _ = facing_bet(
                    hole, board, ranked=ranked, opponents=1, pot=pot + size,
                    to_call=min(size, stack), faced=size, can_raise=size < stack,
                    trials=PLAN_TRIALS,
                )
                live = postflop_advice(
                    hole, board, pot=pot + size, to_call=min(size, stack),
                    stack=stack, can_raise=size < stack, trials=PLAN_TRIALS,
                )

                self.assertEqual(planned, live.action)


class ReplyLadderTests(unittest.TestCase):
    def test_the_ladder_stops_at_our_own_stack(self) -> None:
        # Ставка крупнее нашего стека — тот же олл-ин: излишек вернётся ему.
        self.assertEqual(reply_sizes(1000, 4000), (330, 660, 1000, 4000))
        self.assertEqual(reply_sizes(1000, 200), (200,))
        self.assertEqual(reply_sizes(1000), (330, 660, 1000), "стек неизвестен")

    def test_the_ladder_is_read_to_the_first_fold(self) -> None:
        # Заплатить крупную ставку, сбросив мелкую, — не план, а путаница.
        line = plan_line(
            "если поставит",
            [("колл", 300, False), ("фолд", 900, False), ("колл", 4000, True)],
        )

        self.assertEqual(line, "если поставит: доплата до 300 — колл, крупнее — фолд")

    def test_same_answers_in_a_row_become_one_boundary(self) -> None:
        # Игроку нужна граница, а не перечень размеров.
        line = plan_line("если поставит", [("колл", 300, False), ("колл", 900, False)])

        self.assertEqual(line, "если поставит: доплата до 900 — колл")

    def test_the_whole_stack_is_named_by_words(self) -> None:
        # Число тут ничего не добавит: за весь стек и так платят целиком.
        line = plan_line("если повысит", [("рейз", 900, False), ("колл", 4000, True)])

        self.assertEqual(line, "если повысит: доплата до 900 — рейз, весь стек — колл")

    def test_nothing_to_answer_stays_silent(self) -> None:
        self.assertEqual(plan_line("если поставит", []), "")
        self.assertEqual(
            plan_line("если поставит", [("фолд", 300, False)]), "если поставит — фолд"
        )
