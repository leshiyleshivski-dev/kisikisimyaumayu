"""Журнал раздач: результат раздачи, BB/100 и сравнение периодов."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clean_poker_journal import clean, drop_twins, fix_blinds, round_results
from kisiki.modules.poker.journal import (
    JOURNAL_LIMIT, Hand, Journal, hand_result, hands_word, new_hand,
)

MOVES = (
    (100, "начало", None, None), (101, "блайнд", None, 0), (103, "ставка", 200, 1),
)

BIG_BLIND = 50


def played(result: int | None, big_blind: int = BIG_BLIND, **changes) -> Hand:
    """Сыгранная раздача: по умолчанию банк заведомо больше результата."""
    fields = {"hole": ("Ah", "Kd"), "pot": 100_000, "big_blind": big_blind}
    fields.update(changes)
    return new_hand(result=result, **fields)


class HandResultTests(unittest.TestCase):
    def test_the_result_is_the_stack_between_hands(self) -> None:
        # Мерка снимается в паузе: блайнды следующей раздачи ещё не
        # поставлены, а банк этой уже сгребли.
        self.assertEqual(hand_result(1000, 1400, pot=1200), 400)
        self.assertEqual(hand_result(1000, 800, pot=500), -200)
        self.assertEqual(hand_result(1000, 1000, pot=300), 0)

    def test_a_purchase_between_hands_is_not_a_win(self) -> None:
        # Выиграть больше, чем лежало в банке, нельзя. Стек вырос сильнее —
        # значит, докупили фишек, а покупку от выигрыша по стеку не отличить.
        self.assertIsNone(hand_result(1000, 6000, pot=1200))
        self.assertIsNone(hand_result(6000, 1000, pot=1200))

    def test_without_numbers_there_is_no_result(self) -> None:
        # Молчание честнее выдуманного числа — то же правило, что и у совета.
        self.assertIsNone(hand_result(None, 1400, pot=1200))
        self.assertIsNone(hand_result(1000, None, pot=1200))
        self.assertIsNone(hand_result(1000, 1400, pot=None))
        self.assertIsNone(hand_result(1000, 1400, pot=0))


class HandTests(unittest.TestCase):
    def test_the_result_is_measured_in_blinds(self) -> None:
        # Блайнды за разными столами отличаются на порядки: 500 фишек — это и
        # десять ставок, и одна пятидесятая.
        self.assertEqual(played(100).blinds_won, 2.0)
        self.assertEqual(played(100, big_blind=500).blinds_won, 0.2)

    def test_a_hand_without_numbers_is_not_counted(self) -> None:
        self.assertFalse(played(None).counted)
        self.assertFalse(played(100, big_blind=0).counted)
        self.assertTrue(played(100).counted)


class RateTests(unittest.TestCase):
    def test_bb_per_100_counts_blinds_and_not_chips(self) -> None:
        journal = Journal()
        journal.add(played(100))            # +2 блайнда
        journal.add(played(-50))            # −1 блайнд

        self.assertAlmostEqual(journal.bb_per_100(), 50.0)

    def test_tables_of_different_stakes_are_comparable(self) -> None:
        # Ровно ради этого результат и хранится в блайндах: вечер за столом
        # 25/50 и вечер за столом 250/500 иначе не сложить.
        cheap, rich = Journal(), Journal()
        cheap.add(played(100, big_blind=50))
        rich.add(played(1000, big_blind=500))

        self.assertAlmostEqual(cheap.bb_per_100(), rich.bb_per_100())

    def test_hands_without_a_result_do_not_spoil_the_rate(self) -> None:
        journal = Journal()
        journal.add(played(100))
        journal.add(played(None))

        self.assertEqual(len(journal), 2)
        self.assertEqual(len(journal.counted()), 1)
        self.assertAlmostEqual(journal.bb_per_100(), 200.0)

    def test_an_empty_journal_says_nothing(self) -> None:
        self.assertIsNone(Journal().bb_per_100())
        self.assertEqual(Journal().chips(), 0)


class TrendTests(unittest.TestCase):
    def test_the_last_hands_are_compared_with_everything_before(self) -> None:
        # Сравнение периодов — весь смысл журнала: одно число не отличает
        # «стало лучше» от «повезло».
        journal = Journal()
        for _ in range(20):
            journal.add(played(-50))        # −1 блайнд
        for _ in range(50):
            journal.add(played(100))        # +2 блайнда

        recent, earlier = journal.trend(window=50)

        self.assertAlmostEqual(recent, 200.0)
        self.assertAlmostEqual(earlier, -100.0)

    def test_a_short_journal_has_nothing_to_compare(self) -> None:
        journal = Journal()
        journal.add(played(100))

        recent, earlier = journal.trend(window=50)

        self.assertAlmostEqual(recent, 200.0)
        self.assertIsNone(earlier, "на два периода раздач ещё не набралось")


class ShowdownTests(unittest.TestCase):
    def test_wins_and_showdowns_are_counted(self) -> None:
        journal = Journal()
        journal.add(played(100, won=True, showdown=True))
        journal.add(played(-50, showdown=True))
        journal.add(played(100, won=True))

        self.assertEqual(journal.wins(), 2)
        self.assertEqual(journal.showdowns(), 2)
        self.assertEqual(journal.showdown_wins(), 1)

    def test_opponent_hands_are_kept(self) -> None:
        # Копилка на будущее: чужие карты видны только на вскрытии, и другого
        # способа узнать, с чем за этими столами доходят до конца, нет.
        journal = Journal()
        journal.add(played(100, shown=((1, ("Jc", "Js")),)))
        journal.add(played(-50, shown=((4, ("Ah", "Qd")), (2, ("7c", "7d")))))

        self.assertEqual(len(journal.opponent_hands()), 3)


class AdviceTests(unittest.TestCase):
    """Что помощник советовал — и послушали ли его.

    Единственная пара чисел, которой «совет плохой» отличается от «совет не
    послушали». Разобрать это задним числом нельзя: таблица к тому времени
    уже поменялась, и прогон старой раздачи через новую отвечает не про тот
    вечер.
    """

    def test_the_advice_before_the_flop_is_the_first_one(self) -> None:
        hand = played(-200, advice=((0, "фолд"), (3, "чек")))

        self.assertEqual(hand.advised, "фолд")

    def test_a_hand_advised_only_after_the_flop_has_none_before_it(self) -> None:
        # До флопа помощник промолчал — значит, и спрашивать с него за эту
        # раздачу нечего.
        self.assertIsNone(played(-200, advice=((3, "фолд"),)).advised)
        self.assertIsNone(played(-200).advised)

    def test_a_fold_costs_the_blind_and_not_a_chip_more(self) -> None:
        # Доплатить, сбросив карты, невозможно: пас стоит ровно блайнд, а вне
        # блайндов не стоит ничего.
        self.assertFalse(played(-BIG_BLIND, position="BB").played_on)
        self.assertFalse(played(-BIG_BLIND // 2, position="SB").played_on)
        self.assertFalse(played(0, position="BTN").played_on)

    def test_any_chip_over_the_blind_means_the_hand_was_played(self) -> None:
        # Выиграть, сбросив карты, тоже нельзя — плюс это тоже «доиграна».
        self.assertTrue(played(-BIG_BLIND - 50, position="BB").played_on)
        self.assertTrue(played(-50, position="BTN").played_on)
        self.assertTrue(played(400, position="BB").played_on)

    def test_an_uncounted_hand_says_nothing_about_discipline(self) -> None:
        # Результат не сошёлся — между раздачами покупали фишки, и по стеку не
        # понять, доигрывалась раздача или нет.
        self.assertFalse(played(None, position="BB").played_on)

    def test_ignored_folds_are_counted(self) -> None:
        journal = Journal()
        journal.add(played(-BIG_BLIND, position="BB", advice=((0, "фолд"),)))
        journal.add(played(-4000, position="CO", advice=((0, "фолд"), (3, "чек"))))
        journal.add(played(-4000, position="CO", advice=((0, "рейз"),)))

        self.assertEqual(journal.ignored_folds(), (1, 2))

    def test_a_journal_without_advice_counts_nothing(self) -> None:
        # Старый журнал советов не помнит, и выдумывать их задним числом
        # нельзя.
        journal = Journal(hands=[played(-4000, position="CO")])

        self.assertEqual(journal.ignored_folds(), (0, 0))


class DiskTests(unittest.TestCase):
    def test_the_journal_survives_a_restart(self) -> None:
        # Сравнивать периоды по раздачам одного вечера бессмысленно: журнал
        # обязан пережить выход из приложения.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "poker_journal.json"
            journal = Journal(path=path)
            journal.add(played(100, won=True, shown=((1, ("Jc", "Js")),)))
            journal.add(played(None))

            read = Journal.load(path)

            self.assertEqual(len(read), 2)
            self.assertEqual(read.hands[0].hole, ("Ah", "Kd"))
            self.assertEqual(read.hands[0].shown, ((1, ("Jc", "Js")),))
            self.assertTrue(read.hands[0].won)
            self.assertIsNone(read.hands[1].result)
            self.assertAlmostEqual(read.bb_per_100(), 200.0)

    def test_the_advice_survives_a_restart(self) -> None:
        # Спрашивают журнал уже следующим вечером — после выхода из
        # приложения, а не в ту же минуту.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "poker_journal.json"
            Journal(path=path).add(played(-200, advice=((0, "фолд"), (3, "чек"))))

            self.assertEqual(
                Journal.load(path).hands[0].advice, ((0, "фолд"), (3, "чек"))
            )

    def test_a_broken_file_is_an_empty_journal_and_not_a_crash(self) -> None:
        # Статистика не стоит упавшего экрана посреди раздачи.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "poker_journal.json"
            path.write_text("{не json", encoding="utf-8")
            self.assertEqual(len(Journal.load(path)), 0)

            path.write_text(json.dumps([{"hole": "мусор"}, 5, None]), encoding="utf-8")
            self.assertEqual(len(Journal.load(path).counted()), 0)

    def test_a_missing_file_is_an_empty_journal(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(len(Journal.load(Path(folder) / "нет.json")), 0)

    def test_the_journal_does_not_grow_forever(self) -> None:
        journal = Journal()
        for index in range(JOURNAL_LIMIT + 5):
            journal.add(played(index))

        self.assertEqual(len(journal), JOURNAL_LIMIT)
        self.assertEqual(journal.hands[-1].result, JOURNAL_LIMIT + 4)


class WordingTests(unittest.TestCase):
    def test_hands_agree_with_the_number(self) -> None:
        self.assertEqual(hands_word(1), "раздача")
        self.assertEqual(hands_word(3), "раздачи")
        self.assertEqual(hands_word(7), "раздач")
        self.assertEqual(hands_word(11), "раздач")
        self.assertEqual(hands_word(14), "раздач")
        self.assertEqual(hands_word(21), "раздача")
        self.assertEqual(hands_word(22), "раздачи")


if __name__ == "__main__":
    unittest.main()


class MoveTests(unittest.TestCase):
    """Чужие ходы из строки событий — то, ради чего журнал и заводился.

    Числа `LIVE_RANGE_SHARE`, `CALLER_KEEP` и остальные выставлены на глаз, а
    поправить их можно только по тому, как за этими столами играют на самом
    деле.
    """

    def test_moves_are_counted_across_hands(self) -> None:
        journal = Journal()
        journal.add(played(100, events=MOVES))
        journal.add(played(-50, events=MOVES[:2]))

        self.assertEqual(journal.moves(), 5)
        self.assertEqual(journal.actions()["блайнд"], 2)
        self.assertEqual(journal.actions()["ставка"], 1)

    def test_moves_survive_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "poker_journal.json"
            journal = Journal(path=path)
            journal.add(played(100, events=MOVES))

            read = Journal.load(path)

            self.assertEqual(read.hands[0].events, MOVES)
            self.assertEqual(read.actions()["ставка"], 1)

    def test_a_journal_without_moves_still_counts_hands(self) -> None:
        # Свёрнутая строка событий — обычное дело: раздачи считаются и по
        # своему стеку, просто чужих ходов в них нет.
        journal = Journal()
        journal.add(played(100))

        self.assertEqual(journal.moves(), 0)
        self.assertEqual(len(journal), 1)


class PlayerTests(unittest.TestCase):
    """Кто сделал ход: номер в списке отпечатков имён."""

    def test_players_are_counted(self) -> None:
        journal = Journal()
        journal.add(played(100, events=(
            (100, "начало", None, None), (101, "колл", None, 0),
            (102, "фолд", None, 1),
        )))
        journal.add(played(-50, events=((110, "чек", None, 0),)))

        self.assertEqual(journal.players(), 2, "строки самой игры ничьи")

    def test_moves_written_before_players_are_still_read(self) -> None:
        # Журнал пишется с первого запуска, а игрок в ходе завёлся позже.
        # Выбросить старые ходы значило бы выбросить весь замер по столу
        # заодно: ходы в них настоящие, просто ничьи.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "poker_journal.json"
            path.write_text(json.dumps([{
                "hole": ["Ah", "Kd"], "big_blind": 50, "pot": 500, "result": 100,
                "events": [[100, "начало", None], [103, "ставка", 200]],
            }]), encoding="utf-8")

            journal = Journal.load(path)

            self.assertEqual(journal.hands[0].events, (
                (100, "начало", None, None), (103, "ставка", 200, None),
            ))
            self.assertEqual(journal.players(), 0)
            self.assertEqual(journal.moves(), 2)


class CleanupTests(unittest.TestCase):
    """Починка журнала, записанного до того, как разбор кадра исправили.

    Сами ошибки закрыты в зрении и в памяти экрана, но всё, что они успели
    записать, так и лежит в файле — и портит ту самую пару чисел, ради которой
    журнал заводился.
    """

    BOARD = ("6c", "3s", "As", "7d", "9d")

    def hand(self, **changes) -> Hand:
        fields = {
            "hole": ("Ah", "Kd"), "board": self.BOARD, "position": "BTN",
            "big_blind": 500, "pot": 20_000, "result": -500, "played_at": 1000.0,
            "events": MOVES,
        }
        fields.update(changes)
        return new_hand(**fields)

    def test_the_showdown_twin_is_dropped(self) -> None:
        # Карты вскрытия лежат ещё секунду после того, как банк уехал, и на них
        # начиналась «новая раздача»: тот же борд, чужая рука, ни позиции, ни
        # ходов. В журнале записи `8.mp4` таких пар две.
        real = self.hand()
        twin = self.hand(
            hole=("Qc", "Jd"), position=None, result=0, events=(),
            played_at=real.played_at + 3,
        )

        self.assertEqual(drop_twins([real, twin]), [real])

    def test_a_real_hand_after_it_stays(self) -> None:
        real = self.hand()
        following = self.hand(
            board=("2c", "7d", "Qs", "4h", "8s"), played_at=real.played_at + 60
        )

        self.assertEqual(drop_twins([real, following]), [real, following])

    def test_a_hand_without_moves_is_not_a_twin(self) -> None:
        # Свёрнутая строка событий не делает раздачу двойником: борд у неё
        # свой, а позиция на месте.
        real = self.hand()
        quiet = self.hand(board=("Kd", "2h", "5s"), events=(),
                          played_at=real.played_at + 40)

        self.assertEqual(drop_twins([real, quiet]), [real, quiet])

    def test_a_short_run_of_wrong_blinds_is_fixed(self) -> None:
        # Ставка и ответное повышение после флопа читались как блайнды, и
        # 7 000 держались три раздачи — пока не попалось чистое начало.
        hands = [self.hand(big_blind=blind)
                 for blind in (500, 500, 7000, 7000, 7000, 500, 500)]

        self.assertEqual(
            [hand.big_blind for hand in fix_blinds(hands)], [500] * 7
        )

    def test_moving_to_another_table_is_left_alone(self) -> None:
        # Смена стола выглядит иначе: новый блайнд остаётся до конца журнала,
        # и «починить» его значило бы соврать про все следующие раздачи.
        hands = [self.hand(big_blind=blind) for blind in (500, 500, 5000, 5000, 5000)]

        self.assertEqual(
            [hand.big_blind for hand in fix_blinds(hands)],
            [500, 500, 5000, 5000, 5000],
        )

    def test_a_result_caught_mid_animation_is_rounded_to_a_chip(self) -> None:
        # За столом 250 / 500 фишка — 250, и −43 879 такой стол выдать не мог:
        # это стек, прочитанный посреди перелёта фишек.
        hands = [self.hand(result=-43_879, pot=88_500), self.hand(result=-232),
                 self.hand(result=-500)]

        self.assertEqual(
            [hand.result for hand in round_results(hands)], [-44_000, -250, -500]
        )

    def test_an_uncounted_hand_keeps_its_empty_result(self) -> None:
        hands = [self.hand(result=None), self.hand(result=-500, big_blind=None)]

        self.assertEqual([hand.result for hand in round_results(hands)], [None, -500])

    def test_the_whole_cleanup_runs_in_order(self) -> None:
        # Округлять надо уже починенным блайндом: с 7 000 фишка вышла бы
        # 3 500, и честные 250 округлились бы в ноль.
        real = self.hand(big_blind=7000, result=-1_455)
        twin = self.hand(big_blind=7000, position=None, result=0, events=(),
                         played_at=real.played_at + 2)
        hands = [self.hand(), real, twin, self.hand()]

        cleaned = clean(hands)

        self.assertEqual(len(cleaned), 3)
        self.assertEqual([hand.big_blind for hand in cleaned], [500, 500, 500])
        self.assertEqual(cleaned[1].result, -1_500)
