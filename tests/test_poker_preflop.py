"""Проверки префлопа: таблица открытия и режим «олл-ин или пас»."""

from __future__ import annotations

import unittest

from kisiki.modules.poker.preflop import (
    CALL_RANGES, HEADS_UP_CALL, HEADS_UP_OPEN, OPEN_RANGES, PUSH_FOLD_BB,
    PUSH_RANGES, THREE_BET_RANGES, blinds_word, hand_word, in_hundred,
    preflop_advice, push_range, raise_reply, shove_ev,
)
from kisiki.modules.poker.ranges import range_share

BIG_BLIND = 500
DEEP = BIG_BLIND * 100
SHORT = BIG_BLIND * 9


def deep(hole, **changes):
    """Совет на глубоком стеке: по умолчанию соперник повысил до двух ставок."""
    fields = {
        "position": "BB", "opponents": 3, "big_blind": BIG_BLIND, "stack": DEEP,
        "pot": BIG_BLIND * 3, "to_call": BIG_BLIND, "my_bet": BIG_BLIND,
    }
    fields.update(changes)
    return preflop_advice(hole, **fields)


def short(hole, **changes):
    """Совет на коротком стеке: девять блайндов, никто ещё не повышал."""
    fields = {
        "position": "BTN", "opponents": 1, "big_blind": BIG_BLIND, "stack": SHORT,
        "pot": BIG_BLIND + BIG_BLIND // 2, "to_call": BIG_BLIND, "my_bet": 0,
    }
    fields.update(changes)
    return preflop_advice(hole, **fields)


class OpenRangeTests(unittest.TestCase):
    def test_later_seats_open_wider(self) -> None:
        widths = [range_share(OPEN_RANGES[place]) for place in ("UTG", "MP", "CO", "BTN")]
        self.assertEqual(widths, sorted(widths))

    def test_the_big_blind_never_opens(self) -> None:
        # Большой блайнд закрывает торговлю, а не начинает её: открывать
        # оттуда попросту не в кого — все уже спасовали.
        self.assertEqual(OPEN_RANGES["BB"], frozenset())

    def test_every_open_range_holds_the_best_hand(self) -> None:
        for place, hands in OPEN_RANGES.items():
            if place == "BB":
                continue
            self.assertIn("AA", hands, place)

    def test_no_open_range_plays_the_worst_hand(self) -> None:
        for place, hands in OPEN_RANGES.items():
            self.assertNotIn("32o", hands, place)

    def test_raising_hands_are_never_merely_called(self) -> None:
        # Рука не может быть одновременно «слишком слаба, чтобы повышать» и
        # «достаточно сильна, чтобы повышать»: перекрытие означало бы, что
        # порядок проверок в совете решает исход, а не таблица.
        for place in OPEN_RANGES:
            self.assertFalse(
                THREE_BET_RANGES[place] & CALL_RANGES[place],
                f"{place}: рука и повышает, и уравнивает",
            )


class ChartAdviceTests(unittest.TestCase):
    def test_the_hands_that_lost_the_session_are_folded(self) -> None:
        # Ровно те руки, с которыми старый помощник советовал колл на записи
        # 6.mp4, потому что перебор против случайных карт давал им сорок
        # процентов. Разбор — в poker-plan/README.md.
        for hole in (["3s", "9d"], ["2c", "9h"], ["9s", "3s"], ["Jc", "6d"]):
            self.assertEqual(deep(hole).action, "фолд", hole)

    def test_premium_hands_raise_and_name_a_size(self) -> None:
        for hole in (["Ah", "As"], ["Ah", "Ks"], ["Qh", "Qd"]):
            advice = deep(hole)
            self.assertEqual(advice.action, "рейз", hole)
            self.assertGreater(advice.raise_to, 0, hole)

    def test_a_free_look_is_taken_when_nothing_is_due(self) -> None:
        advice = deep(["7c", "2d"], to_call=0, my_bet=BIG_BLIND)

        self.assertEqual(advice.action, "чек")

    def test_the_open_names_a_size_of_a_few_blinds(self) -> None:
        advice = deep(["Ah", "Ks"], position="BTN", to_call=0, my_bet=0)

        self.assertEqual(advice.action, "рейз")
        self.assertGreaterEqual(advice.raise_to, BIG_BLIND * 2)
        self.assertLessEqual(advice.raise_to, BIG_BLIND * 4)

    def test_limpers_make_the_open_bigger(self) -> None:
        # За каждым, кто уже влез уравниванием, лежит лишняя ставка: открывать
        # в те же деньги значит звать их всех задёшево.
        alone = deep(["Ah", "Ks"], position="BTN", to_call=0, my_bet=0, limpers=0)
        crowd = deep(["Ah", "Ks"], position="BTN", to_call=0, my_bet=0, limpers=2)

        self.assertGreater(crowd.raise_to, alone.raise_to)

    def test_the_big_blind_raises_the_limpers_out(self) -> None:
        # Таблица открытия у большого блайнда пустая — торговлю он закрывает,
        # а не начинает. Из-за этого на тузов против четверых лимперов совет
        # был «чек»: формально «открывать не стоит», а по делу лучшая рука
        # уходила в многосторонний флоп даром.
        for hole in (["Ah", "As"], ["Kh", "Kd"], ["Ah", "Qs"]):
            advice = deep(hole, position="BB", to_call=0, my_bet=BIG_BLIND, limpers=3)
            self.assertEqual(advice.action, "рейз", hole)
            self.assertGreater(advice.raise_to, BIG_BLIND * 3, hole)
            self.assertIn("лимперов", advice.reason, hole)

    def test_without_limpers_the_big_blind_still_checks(self) -> None:
        # Никто не влез — забирать нечего, и лишний рейз из-под всех только
        # раздувает банк там, где после флопа ходишь первым.
        advice = deep(["Ah", "As"], position="BB", to_call=0, my_bet=BIG_BLIND, limpers=0)

        self.assertEqual(advice.action, "чек")

    def test_the_big_blind_does_not_isolate_with_junk(self) -> None:
        for hole in (["7c", "2d"], ["9h", "5s"], ["Jc", "6d"]):
            advice = deep(hole, position="BB", to_call=0, my_bet=BIG_BLIND, limpers=3)
            self.assertEqual(advice.action, "чек", hole)

    def test_the_raise_never_asks_for_more_than_the_stack(self) -> None:
        # Размер — это ставка улицы целиком, вместе с уже выставленным. Выше
        # своих денег она подняться не может: дальше только олл-ин.
        advice = deep(["Ah", "As"], stack=BIG_BLIND * 20, to_call=BIG_BLIND * 8)

        self.assertLessEqual(advice.raise_to, BIG_BLIND + BIG_BLIND * 20)

    def test_deep_stacks_use_the_chart(self) -> None:
        self.assertEqual(deep(["Ah", "Ks"]).mode, "таблица")

    def test_the_hand_class_is_reported_back(self) -> None:
        self.assertEqual(deep(["Ah", "Kd"]).hand, "AKo")


class FoldReasonTests(unittest.TestCase):
    """Фолд обязан говорить, когда же играть.

    Подряд идущие фолды читаются как поломка помощника, хотя двадцать рук из
    ста — обычная плотность игры. Вопрос «а играть-то когда» задаётся ровно в
    ту секунду, когда на экране стоит «ФОЛД», — там же он и отвечается.
    """

    def test_an_unopened_fold_names_how_wide_the_seat_opens(self) -> None:
        reason = deep(
            ["8d", "4s"], position="CO", my_bet=0, to_call=BIG_BLIND,
            pot=BIG_BLIND * 3,
        ).reason

        self.assertIn("не открывают", reason)
        self.assertIn(in_hundred(OPEN_RANGES["CO"]), reason)

    def test_a_fold_against_a_raise_names_the_answering_share(self) -> None:
        # Против повышения играется свой, куда более узкий набор — и назвать
        # надо именно его, иначе число обещает вчетверо больше игры, чем есть.
        reason = deep(
            ["8d", "4s"], position="CO", my_bet=0, to_call=BIG_BLIND * 3,
            pot=BIG_BLIND * 5,
        ).reason

        self.assertIn("против повышения", reason)
        self.assertIn(
            in_hundred(THREE_BET_RANGES["CO"] | CALL_RANGES["CO"]), reason
        )

    def test_a_free_look_is_never_a_fold(self) -> None:
        # Доплаты нет — сбрасывать нечего и незачем: флоп смотрится даром.
        tip = deep(["8d", "4s"], position="BB", my_bet=BIG_BLIND, to_call=0)

        self.assertEqual(tip.action, "чек")


class HeadsUpTests(unittest.TestCase):
    def test_two_handed_play_is_much_wider(self) -> None:
        # Вдвоём после малого блайнда говорит один человек, а не четверо.
        self.assertGreater(range_share(HEADS_UP_OPEN), range_share(OPEN_RANGES["SB"]))
        self.assertGreater(range_share(HEADS_UP_CALL), range_share(CALL_RANGES["BB"]))

    def test_a_hand_folded_six_handed_is_played_two_handed(self) -> None:
        hole = ["Ah", "7c"]
        self.assertEqual(preflop_advice(
            hole, position="BB", opponents=3, big_blind=BIG_BLIND, stack=DEEP,
            pot=BIG_BLIND * 3, to_call=BIG_BLIND, my_bet=BIG_BLIND,
        ).action, "фолд")
        self.assertEqual(preflop_advice(
            hole, position="BB", opponents=1, big_blind=BIG_BLIND, stack=DEEP,
            pot=BIG_BLIND * 3, to_call=BIG_BLIND, my_bet=BIG_BLIND,
        ).action, "колл")

    def test_the_reason_reads_as_russian(self) -> None:
        reason = preflop_advice(
            ["3s", "9d"], position="BB", opponents=1, big_blind=BIG_BLIND,
            stack=DEEP, pot=BIG_BLIND * 3, to_call=BIG_BLIND, my_bet=BIG_BLIND,
        ).reason

        self.assertIn("за столом на двоих", reason)
        self.assertNotIn("с стол", reason)


class PushRangeTests(unittest.TestCase):
    def test_more_opponents_means_a_tighter_push(self) -> None:
        for depth in (7, 12, 15):
            widths = [
                range_share(PUSH_RANGES[opponents][depth])
                for opponents in sorted(PUSH_RANGES)
            ]
            self.assertEqual(widths, sorted(widths, reverse=True), depth)

    def test_more_chips_means_a_tighter_push(self) -> None:
        for opponents in sorted(PUSH_RANGES):
            widths = [
                range_share(PUSH_RANGES[opponents][depth])
                for depth in sorted(PUSH_RANGES[opponents])
            ]
            self.assertEqual(widths, sorted(widths, reverse=True), opponents)

    def test_aces_are_pushed_from_everywhere(self) -> None:
        for opponents in sorted(PUSH_RANGES):
            for depth in sorted(PUSH_RANGES[opponents]):
                self.assertIn("AA", PUSH_RANGES[opponents][depth])

    def test_a_crowded_table_falls_back_to_the_tightest_column(self) -> None:
        self.assertEqual(push_range(9, 7), PUSH_RANGES[max(PUSH_RANGES)][7])

    def test_depth_between_steps_rounds_to_the_tighter_one(self) -> None:
        self.assertEqual(push_range(1, 9.5), PUSH_RANGES[1][12])
        self.assertEqual(push_range(1, 7.0), PUSH_RANGES[1][7])


class ShortStackTests(unittest.TestCase):
    def test_shallow_stacks_switch_the_mode(self) -> None:
        self.assertEqual(short(["Ah", "Ks"]).mode, "пуш/фолд")
        self.assertEqual(
            short(["Ah", "Ks"], stack=int(BIG_BLIND * (PUSH_FOLD_BB + 5))).mode,
            "таблица",
        )

    def test_trash_is_not_shoved(self) -> None:
        # Прямой счёт выгоды говорит, что олл-ин с 92o прибыльнее паса: три
        # раза из четырёх все спасуют. Он исходит из того, что отвечают
        # четвертью рук, а против того, кто пихает всё, отвечают вдвое шире.
        for hole in (["9h", "2c"], ["9s", "3d"], ["7c", "2d"]):
            self.assertEqual(short(hole).action, "фолд", hole)

    def test_strong_hands_are_shoved_for_the_whole_stack(self) -> None:
        for hole in (["Ah", "Ks"], ["Qh", "Qd"], ["Ah", "7c"]):
            advice = short(hole)
            self.assertEqual(advice.action, "пуш", hole)
            self.assertTrue(advice.all_in, hole)
            self.assertEqual(advice.raise_to, SHORT, hole)

    def test_nothing_due_means_a_free_look_rather_than_a_fold(self) -> None:
        advice = short(["9h", "2c"], to_call=0, my_bet=BIG_BLIND)

        self.assertEqual(advice.action, "чек")

    def test_facing_a_shove_is_decided_by_the_price(self) -> None:
        # Доплата больше стека — это ответ на чужой олл-ин, и решают шансы
        # банка против того, с чем ходят ва-банк, а не таблица пуша. Банк
        # приходит уже вместе с чужой ставкой, а излишек сверх нашего стека
        # вернётся ему: разыгрывается только та часть, которую мы покрываем.
        cheap = preflop_advice(
            ["9h", "2c"], position="BB", opponents=1, big_blind=BIG_BLIND,
            stack=BIG_BLIND * 2, pot=BIG_BLIND * 70, to_call=BIG_BLIND * 40,
            my_bet=BIG_BLIND,
        )
        dear = preflop_advice(
            ["9h", "2c"], position="BB", opponents=1, big_blind=BIG_BLIND,
            stack=BIG_BLIND * 10, pot=BIG_BLIND * 12, to_call=BIG_BLIND * 10,
            my_bet=BIG_BLIND,
        )

        self.assertEqual(cheap.action, "колл")
        self.assertTrue(cheap.all_in)
        self.assertEqual(dear.action, "фолд")

    def test_the_price_of_a_shove_is_reported(self) -> None:
        advice = preflop_advice(
            ["Ah", "Ks"], position="BB", opponents=1, big_blind=BIG_BLIND,
            stack=BIG_BLIND * 3, pot=BIG_BLIND * 10, to_call=BIG_BLIND * 40,
            my_bet=BIG_BLIND,
        )

        self.assertIsNotNone(advice.equity)
        self.assertIsNotNone(advice.odds)


class ShoveValueTests(unittest.TestCase):
    def test_the_best_hand_beats_the_worst_by_a_wide_margin(self) -> None:
        best = shove_ev(["Ah", "As"], opponents=1, pot=750, stack=4500, trials=4000)
        worst = shove_ev(["3h", "2c"], opponents=1, pot=750, stack=4500, trials=4000)

        self.assertGreater(best, worst)

    def test_a_bigger_pot_makes_a_shove_worth_more(self) -> None:
        small = shove_ev(["Ah", "Ks"], opponents=1, pot=500, stack=4500, trials=4000)
        large = shove_ev(["Ah", "Ks"], opponents=1, pot=4000, stack=4500, trials=4000)

        self.assertGreater(large, small)

    def test_a_marginal_hand_dies_as_the_table_fills_up(self) -> None:
        # Такая рука идёт ва-банк не ради победы, а ради чужого паса. Каждый
        # лишний соперник — ещё один шанс, что кто-то не спасует, и от этого
        # весь смысл хода пропадает: с четырьмя это уже прямой убыток.
        values = [
            shove_ev(["Kh", "9c"], opponents=seats, pot=750, stack=4500, trials=4000)
            for seats in (1, 2, 3, 4)
        ]

        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[0], 0)
        self.assertLess(values[-1], 0)

    def test_a_premium_hand_wants_company(self) -> None:
        # Обратная сторона того же счёта, и она тоже верна: AK берёт у
        # отвечающего диапазона больше половины раздач, поэтому чужой ответ ей
        # выгоден. Требовать «чем больше народу, тем хуже» от любой руки —
        # значит требовать неправды.
        alone = shove_ev(["Ah", "Ks"], opponents=1, pot=750, stack=4500, trials=4000)
        crowd = shove_ev(["Ah", "Ks"], opponents=3, pot=750, stack=4500, trials=4000)

        self.assertGreater(crowd, alone)


class SteadinessTests(unittest.TestCase):
    def test_the_same_table_gives_the_same_advice(self) -> None:
        # Совет пересчитывается по четыре раза в секунду: дрожать от кадра к
        # кадру он не имеет права, иначе ему нельзя верить.
        first = short(["Ah", "7c"])
        second = short(["Ah", "7c"])

        self.assertEqual(first.action, second.action)
        self.assertEqual(first.reason, second.reason)

    def test_an_unknown_position_does_not_crash(self) -> None:
        # Метку дилера может закрыть курсор. Экран в этом случае молчит, но
        # считалка обязана пережить и пустое имя позиции.
        self.assertIn(deep(["Ah", "As"], position="???").action, ("рейз", "колл", "фолд"))


if __name__ == "__main__":
    unittest.main()


class QuickBetTests(unittest.TestCase):
    """До флопа размер берётся из таблицы, а кнопка — по близости к нему."""

    def test_the_open_is_pressed_with_the_three_blind_button(self) -> None:
        # Таблица открывает в два с половиной блайнда. Кнопки ставят два (это
        # минимум) и три — берём три: мельчить с открытием нельзя.
        advice = deep(
            ["Ah", "Ks"], position="BTN", to_call=0, my_bet=0,
            min_bet=BIG_BLIND * 2,
        )

        self.assertEqual(advice.action, "рейз")
        self.assertEqual(advice.button, "3 BB")
        self.assertEqual(advice.raise_to, BIG_BLIND * 3)

    def test_the_shove_is_pressed_with_the_all_in_button(self) -> None:
        advice = short(["Ah", "Ks"], min_bet=BIG_BLIND * 2)

        self.assertEqual(advice.action, "пуш")
        self.assertEqual(advice.button, "ALL IN")
        self.assertEqual(advice.raise_to, SHORT)

    def test_a_fold_names_no_button(self) -> None:
        self.assertEqual(deep(["3s", "9d"], min_bet=BIG_BLIND * 2).button, "")

    def test_a_size_between_buttons_stays_a_number(self) -> None:
        # За каждым лимпером таблица добавляет к открытию блайнд, и с тремя
        # она просит пять с половиной — вдвое больше «3 BB» и мимо «BANK».
        # Врать про кнопку нельзя: такой размер игрок ведёт ползунком.
        advice = deep(
            ["Ah", "Ks"], position="BTN", to_call=0, my_bet=0,
            pot=BIG_BLIND * 4, limpers=3,
        )

        self.assertEqual(advice.action, "рейз")
        self.assertEqual(advice.button, "")
        self.assertGreater(advice.raise_to, BIG_BLIND * 5)


class WordingTests(unittest.TestCase):
    def test_blinds_agree_with_the_number(self) -> None:
        # Совет читают за пятнадцать секунд, и «4 блайндов» цепляет глаз ровно
        # там, где он нужен цифрам.
        self.assertEqual(blinds_word(1), "блайнд")
        self.assertEqual(blinds_word(4), "блайнда")
        self.assertEqual(blinds_word(9), "блайндов")
        self.assertEqual(blinds_word(11), "блайндов")
        self.assertEqual(blinds_word(14), "блайндов")
        self.assertEqual(blinds_word(21), "блайнд")
        self.assertEqual(blinds_word(22), "блайнда")

    def test_hands_agree_with_the_number(self) -> None:
        # Винительный падеж: «отсюда открывают 21 руку из ста».
        self.assertEqual(hand_word(1), "руку")
        self.assertEqual(hand_word(3), "руки")
        self.assertEqual(hand_word(12), "рук")
        self.assertEqual(hand_word(14), "рук")
        self.assertEqual(hand_word(21), "руку")
        self.assertEqual(hand_word(22), "руки")

    def test_the_share_is_counted_by_combinations(self) -> None:
        # `A♠K♦` и `A♠K♠` — это двенадцать раздач и четыре, а не «две руки».
        self.assertEqual(in_hundred(OPEN_RANGES["BTN"]), "39 рук из ста")
        self.assertEqual(in_hundred(OPEN_RANGES["UTG"]), "11 рук из ста")

    def test_the_short_stack_reason_reads_cleanly(self) -> None:
        self.assertIn("4 блайнда", short(["7c", "2d"], stack=BIG_BLIND * 4).reason)
        self.assertIn("9 блайндов", short(["7c", "2d"]).reason)


class RaiseReplyTests(unittest.TestCase):
    """План на чужое повышение: одним «рейз» ход не заканчивается.

    Размер чужого повышения до флопа на ответ не влияет — отвечает таблица, а
    не перебор, — поэтому и границы в плане нет: она появляется после флопа.
    """

    def open_from(self, hole, position, **changes):
        return deep(hole, position=position, to_call=0, my_bet=0,
                    pot=BIG_BLIND * 3, **changes)

    def test_the_hand_we_reraise_with_says_so_in_advance(self) -> None:
        advice = self.open_from(["As", "Ks"], "CO")

        self.assertEqual(advice.action, "рейз")
        self.assertEqual(advice.plan, "если повысит — повышаем в ответ")

    def test_a_hand_that_only_calls_says_that_too(self) -> None:
        advice = self.open_from(["7c", "7d"], "BTN")

        self.assertEqual(advice.plan, "если повысит — колл")

    def test_an_open_that_folds_to_a_raise_is_named_before_the_raise(self) -> None:
        # Ровно та рука, на которой теряют весь вечер: открыли, получили
        # повышение и полезли доигрывать. План говорит об этом заранее.
        advice = self.open_from(["Ad", "Ts"], "MP")

        self.assertEqual(advice.action, "рейз")
        self.assertEqual(advice.plan, "если повысит — фолд")

    def test_the_plan_survives_the_button_the_advice_picks(self) -> None:
        # Кнопка быстрого размера переписывает совет целиком — план при этом
        # терялся бы молча.
        advice = self.open_from(["7c", "7d"], "BTN")

        self.assertEqual(advice.button, "3 BB")
        self.assertTrue(advice.plan)

    def test_two_at_the_table_answer_by_their_own_book(self) -> None:
        # Вдвоём защищаются вдвое шире, и план обязан читать ту же книжку,
        # что и сам совет.
        self.assertEqual(raise_reply("K5o", "BTN", heads_up=True), "если повысит — колл")
        self.assertEqual(raise_reply("K5o", "BTN"), "если повысит — фолд")

    def test_no_plan_where_the_opponent_answers_last(self) -> None:
        # Чек до флопа закрывает торговлю, а олл-ин отвечать нам уже не даёт.
        self.assertEqual(self.open_from(["2c", "7d"], "BTN").plan, "")
        self.assertEqual(short(["7c", "7d"]).plan, "")
