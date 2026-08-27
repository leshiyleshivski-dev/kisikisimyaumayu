"""Замер по журналу: как за столом отвечают на ставки разного размера."""

from __future__ import annotations

import unittest

from kisiki.modules.poker.journal import Journal, new_hand
from kisiki.modules.poker.stats import (
    ENOUGH_ANSWERS, Aggression, answer_lines, answers_line, hand_aggression,
    keep_by_size, keep_share, measured, model_keep, pot_error, step_of,
    table_aggression, walk,
)

BIG_BLIND = 500


def played(events, **changes) -> object:
    """Раздача с одними лишь ходами: замеру больше ничего и не нужно.

    Ход записывается четвёркой — время, действие, сумма и номер игрока, — но
    замеру по столу игрок безразличен: он складывает всех разом. Поэтому в
    таблицах ниже пишется тройка, а номер дописывается здесь.
    """
    fields = {"hole": ("Ah", "Kd"), "big_blind": BIG_BLIND, "pot": 5750}
    fields.update(changes)
    return new_hand(
        events=tuple(move if len(move) > 3 else (*move, None) for move in events),
        **fields,
    )


# Раздача целиком: блайнды, повышение до 1 500 с коллом и пасом, флоп со
# ставкой в 1 000 и коллом. Банк по ходам — 5 750.
WHOLE_HAND = (
    (100, "начало", None),
    (100, "блайнд", None),
    (100, "блайнд", None),
    (101, "рейз", 1500),
    (102, "колл", None),
    (103, "фолд", None),
    (110, "ставка", 1000),
    (111, "колл", None),
)


class WalkTests(unittest.TestCase):
    def test_the_pot_is_counted_from_the_moves(self) -> None:
        # Банка в ходах нет, но он из них складывается: блайнды известны из
        # раздачи, ставки приходят с суммами, а колл добавляет столько же,
        # сколько стоит на столе.
        self.assertEqual(walk(played(WHOLE_HAND)).pot, 5750)

    def test_a_raise_is_measured_by_what_it_adds(self) -> None:
        # Против большого блайнда повышение до 1 500 просит доплатить тысячу
        # сверх пятисот, а банк до него — 750.
        raise_bet = hand_aggression(played(WHOLE_HAND))[0]

        self.assertAlmostEqual(raise_bet.share, 1000 / 750)
        self.assertEqual((raise_bet.called, raise_bet.folded), (1, 1))
        self.assertTrue(raise_bet.preflop)

    def test_a_bet_means_a_new_street(self) -> None:
        # «Поставил» бывает только там, где ставить ещё не начинали, а до
        # флопа на столе уже лежит большой блайнд. Значит, это постфлоп.
        bet = hand_aggression(played(WHOLE_HAND))[1]

        self.assertFalse(bet.preflop)
        self.assertAlmostEqual(bet.share, 1000 / 3750)
        self.assertEqual(bet.called, 1)

    def test_a_check_closes_the_bet(self) -> None:
        # Чекают, когда ставить не на что: значит, улица уже другая, и пас
        # после чека отвечает не той ставке.
        bets = hand_aggression(played((
            (100, "начало", None), (100, "блайнд", None), (100, "блайнд", None),
            (101, "ставка", 1000), (102, "колл", None),
            (110, "чек", None), (111, "фолд", None),
        )))

        self.assertEqual(len(bets), 1)
        self.assertEqual((bets[0].called, bets[0].folded), (1, 0))

    def test_blinds_are_not_bets(self) -> None:
        # Блайнд — долг перед раздачей, а не решение: пас против него это
        # обычный префлоп-фолд, и в замер агрессии он не идёт.
        bets = hand_aggression(played((
            (100, "начало", None), (100, "блайнд", None), (100, "блайнд", None),
            (101, "фолд", None), (102, "фолд", None),
        )))

        self.assertEqual(bets, ())

    def test_a_raise_over_a_bet_counts_as_an_answer(self) -> None:
        # Повышение — тоже не пас: тот, кто поднял, ставку оплатил.
        bets = hand_aggression(played((
            (100, "начало", None), (100, "блайнд", None), (100, "блайнд", None),
            (110, "ставка", 1000), (111, "рейз", 4000), (112, "фолд", None),
        )))

        self.assertEqual(bets[0].raised, 1)
        self.assertEqual(bets[0].kept, 1)
        self.assertEqual(bets[1].folded, 1)

    def test_a_hand_without_blinds_is_left_alone(self) -> None:
        # Без размера блайнда банк не с чего начинать, и доли выйдут выдумкой.
        self.assertEqual(hand_aggression(played(WHOLE_HAND, big_blind=None)), ())

    def test_a_hand_without_a_start_line_is_left_alone_too(self) -> None:
        # Свёрнутая строка событий гаснет через пару секунд, и «Началась новая
        # игра» теряется вместе с началом банка. Дальше банк складывается с
        # нуля с середины раздачи, и доля выходит вчетверо крупнее настоящей:
        # ставка в тысячу мерилась бы против 750, а не против 3 750.
        headless = WHOLE_HAND[1:]

        self.assertFalse(walk(played(headless)).clean)
        self.assertEqual(hand_aggression(played(headless)), ())
        self.assertTrue(walk(played(WHOLE_HAND)).clean)


class SizeTests(unittest.TestCase):
    def test_the_steps_match_the_model(self) -> None:
        self.assertEqual(step_of(0.3), "до половины банка")
        self.assertEqual(step_of(0.5), "до половины банка")
        self.assertEqual(step_of(0.9), "до банка")
        self.assertEqual(step_of(2.5), "больше банка")

    def test_answers_are_summed_by_step(self) -> None:
        bets = (
            Aggression(amount=100, share=0.3, called=2, folded=1),
            Aggression(amount=100, share=0.4, called=1, folded=3),
            Aggression(amount=100, share=1.5, called=0, folded=2),
        )

        counted = keep_by_size(bets)

        self.assertEqual(counted["до половины банка"], (3, 7))
        self.assertEqual(counted["больше банка"], (0, 2))

    def test_a_thin_sample_gives_no_share(self) -> None:
        # Десяток ходов покажет что угодно. Честнее сказать «мало», чем
        # назвать долю, по которой потом чинят константы модели.
        thin = (Aggression(amount=100, share=0.5, called=3, folded=2),)

        self.assertIsNone(keep_share(thin))

    def test_enough_answers_give_the_share(self) -> None:
        enough = tuple(
            Aggression(amount=100, share=0.5, called=1, folded=1)
            for _ in range(ENOUGH_ANSWERS)
        )

        self.assertAlmostEqual(keep_share(enough), 0.5)


class ModelComparisonTests(unittest.TestCase):
    """Измеренное рядом с тем, что ждала модель.

    Одно число само по себе не говорит, надо ли что-то менять: 52 % не
    пасовавших — это много или мало, зависит от того, сколько ждала модель.
    """

    def test_the_model_is_asked_about_the_same_bets(self) -> None:
        # Минимальная защита на ставку в половину банка — две трети диапазона.
        half = tuple(
            Aggression(amount=100, share=0.5, called=1, folded=0)
            for _ in range(ENOUGH_ANSWERS)
        )

        self.assertAlmostEqual(model_keep(half), 1 / 1.5)
        self.assertAlmostEqual(keep_share(half), 1.0)

    def test_a_thin_sample_gets_no_expectation_either(self) -> None:
        # Считать разницу между моделью и столом по трём ответам — тот же
        # обман, что и называть по ним долю.
        thin = (Aggression(amount=100, share=0.5, called=3, folded=0),)

        self.assertIsNone(model_keep(thin))

    def test_the_merged_number_speaks_before_the_steps_do(self) -> None:
        # Ступеней три, и порознь они набираются годами: за два вечера в
        # нижнюю попало два ответа. Сложенные вместе — говорят уже сейчас.
        spread = [played((
            (100, "начало", None), (100, "блайнд", None), (100, "блайнд", None),
            (110, "ставка", 300),
            *((111 + step, "колл", None) for step in range(12)),
            (130, "чек", None), (131, "ставка", 5000),
            *((132 + step, "фолд", None) for step in range(12)),
        ))]
        counted = keep_by_size(table_aggression(spread, preflop=False))
        line = answers_line(Journal(hands=spread))

        self.assertTrue(
            all(faced < ENOUGH_ANSWERS for _kept, faced in counted.values()),
            "ни одна ступень порознь ещё не набралась",
        )
        self.assertIn("не пасует", line)
        self.assertIn("модель ждёт", line)


class JournalMeasureTests(unittest.TestCase):
    def test_bets_are_collected_across_hands(self) -> None:
        hands = [played(WHOLE_HAND), played(WHOLE_HAND)]

        self.assertEqual(len(table_aggression(hands)), 4)
        self.assertEqual(len(table_aggression(hands, preflop=True)), 2)
        self.assertEqual(len(table_aggression(hands, preflop=False)), 2)

    def test_the_counted_pot_is_checked_against_the_win(self) -> None:
        # Банк считается двумя путями, и оба — из строки событий: по ставкам и
        # по строке «выиграл N фишек». Пока они рядом, ступеням размеров можно
        # верить; разошлись — считать не по чему, и это видно сразу.
        won = WHOLE_HAND + ((120, "выигрыш", 5750), (121, "конец", None))
        half = WHOLE_HAND + ((120, "выигрыш", 11_500),)

        self.assertAlmostEqual(pot_error([played(won)]), 0)
        self.assertAlmostEqual(pot_error([played(half)]), 0.5)

    def test_a_hand_without_a_start_line_is_not_checked(self) -> None:
        # Сверять с выигрышем банк, посчитанный с середины раздачи, незачем:
        # разойдётся он всегда, и медиана по журналу поедет за ним. На журнале
        # за 25–26 августа такие раздачи давали 93 % расхождения при 6 % у
        # чистых, а вместе выходило 80 % — число, по которому нельзя ни
        # доверять замеру, ни отказать ему.
        won = WHOLE_HAND + ((120, "выигрыш", 5750),)

        self.assertIsNone(pot_error([played(won[1:])]))
        self.assertAlmostEqual(pot_error([played(won)]), 0)

    def test_the_dropped_hands_are_counted_out_loud(self) -> None:
        # Молча выкинуть раздачу мало: тонкий замер тогда выглядит как редкая
        # игра за столом, а не как недочитанная строка событий.
        hands = [played(WHOLE_HAND), played(WHOLE_HAND[1:]), played(())]

        self.assertEqual(measured(hands), (1, 2))

    def test_a_hand_without_a_win_line_is_not_checked(self) -> None:
        # Строку выигрыша игра пишет последней, и в записи она есть не всегда:
        # сверять такую раздачу не с чем.
        self.assertIsNone(pot_error([played(WHOLE_HAND)]))

    def test_a_journal_without_moves_says_so(self) -> None:
        journal = Journal(hands=[played(())])

        self.assertIn("нет", answer_lines(journal)[0])

    def test_thin_steps_are_named_thin(self) -> None:
        journal = Journal(hands=[played(WHOLE_HAND)])

        self.assertTrue(any("мало" in line for line in answer_lines(journal)))


class ScreenLineTests(unittest.TestCase):
    """Строка замера на экране журнала."""

    def test_a_thin_journal_says_how_thin(self) -> None:
        # Число, посчитанное по трём раздачам, на экране хуже честного «мало»:
        # по нему потом чинят модель.
        journal = Journal(hands=[played(WHOLE_HAND), played(WHOLE_HAND[1:])])

        line = answers_line(journal)
        self.assertIn("мало", line)
        self.assertIn("в замере 1 раздача из 2", line)

    def test_a_measured_step_is_named_with_its_share(self) -> None:
        journal = Journal(hands=[played((
            (100, "начало", None), (100, "блайнд", None), (100, "блайнд", None),
            (110, "ставка", 300),
            *((111 + step, "колл", None) for step in range(ENOUGH_ANSWERS)),
        ))])

        self.assertIn("до половины банка — 100%", answers_line(journal))


if __name__ == "__main__":
    unittest.main()
