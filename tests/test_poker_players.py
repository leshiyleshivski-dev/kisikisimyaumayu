"""Узнавание игрока по отпечатку имени из строки событий."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from kisiki.modules.poker.players import (
    NAME_MATCH_FLOOR, ROSTER_LIMIT, Roster, packed, players_word, unpacked,
)
from kisiki.modules.poker.vision import NAME_HEIGHT, NAME_WIDTH, event_log, similarity

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "poker"


def name(seed: int, fill: float = 0.4) -> np.ndarray:
    """Отпечаток имени: случайные точки на холсте той же формы, что у зрения."""
    return np.random.default_rng(seed).random((NAME_HEIGHT, NAME_WIDTH)) < fill


def dimmed(mask: np.ndarray, drop: float, seed: int = 7) -> np.ndarray:
    """Тот же отпечаток, но часть точек погасла — плашка журнала притухла."""
    faded = mask.copy()
    lit = np.flatnonzero(faded)
    rng = np.random.default_rng(seed)
    faded.flat[rng.choice(lit, int(len(lit) * drop), replace=False)] = False
    return faded


class RecognitionTests(unittest.TestCase):
    def test_the_same_name_is_the_same_player(self) -> None:
        roster = Roster()
        first = name(1)

        self.assertEqual(roster.number_of(first), 0)
        self.assertEqual(roster.number_of(first), 0)
        self.assertEqual(len(roster), 1)

    def test_a_different_name_is_a_different_player(self) -> None:
        roster = Roster()

        self.assertEqual(roster.number_of(name(1)), 0)
        self.assertEqual(roster.number_of(name(2)), 1)
        self.assertEqual(len(roster), 2)

    def test_a_dimmed_name_is_still_the_same_player(self) -> None:
        # Ради этого сравнение и идёт площадями, а не точным равенством:
        # плашка журнала то светлеет, то гаснет, и на хеше один человек
        # рассыпался бы на десяток мнимых.
        roster = Roster()
        first = name(1)
        roster.number_of(first)
        faded = dimmed(first, 0.2)

        self.assertGreater(similarity(first, faded), NAME_MATCH_FLOOR)
        self.assertEqual(roster.number_of(faded), 0)
        self.assertEqual(len(roster), 1, "тот же человек, а не новый")

    def test_a_line_without_a_name_has_no_player(self) -> None:
        # «Началась новая игра» и «Игра закончена» пишет сама игра, и имени в
        # них нет вовсе. Приписать такую строку человеку — выдумка.
        roster = Roster()

        self.assertIsNone(roster.number_of(None))
        self.assertEqual(len(roster), 0)

    def test_the_roster_does_not_grow_forever(self) -> None:
        # Список без края означал бы не полный стол, а рассыпавшееся
        # узнавание — и каждый ход сравнивался бы с тысячей картинок на
        # пятнадцатисекундном таймере.
        roster = Roster()
        for seed in range(ROSTER_LIMIT):
            roster.number_of(name(seed))

        self.assertEqual(len(roster), ROSTER_LIMIT)
        self.assertIsNone(roster.number_of(name(ROSTER_LIMIT + 1)))
        self.assertEqual(len(roster), ROSTER_LIMIT)


class DiskTests(unittest.TestCase):
    def test_the_roster_survives_a_restart(self) -> None:
        # Узнавание тем и ценно, что помнит вчерашних соседей: список,
        # начинающийся заново каждый запуск, не узнаёт никого.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "poker_players.json"
            roster = Roster(path=path)
            roster.number_of(name(1))
            roster.number_of(name(2))

            read = Roster.load(path)

            self.assertEqual(len(read), 2)
            self.assertEqual(read.number_of(name(2)), 1, "тот же номер, что вчера")
            self.assertEqual(len(read), 2)

    def test_the_mask_survives_packing_exactly(self) -> None:
        # Восемь точек в байте: маска весит четверть килобайта, и хранить её
        # строкой из двух тысяч нулей и единиц незачем.
        mask = name(3)

        self.assertTrue(np.array_equal(unpacked(packed(mask)), mask))
        self.assertLess(len(packed(mask)), NAME_HEIGHT * NAME_WIDTH // 4)

    def test_a_broken_file_is_an_empty_roster_and_not_a_crash(self) -> None:
        # Список имён — статистика, а не прогресс игрока: уронить из-за него
        # экран посреди раздачи хуже, чем потерять.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "poker_players.json"
            path.write_text("{не json", encoding="utf-8")
            self.assertEqual(len(Roster.load(path)), 0)

            path.write_text(json.dumps(["мусор", 5, None]), encoding="utf-8")
            self.assertEqual(len(Roster.load(path)), 0)

    def test_a_missing_file_is_an_empty_roster(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(len(Roster.load(Path(folder) / "нет.json")), 0)


class LiveFrameTests(unittest.TestCase):
    """Узнавание на настоящем кадре, а не на выдуманных масках."""

    def test_a_heads_up_log_resolves_to_two_players(self) -> None:
        # Кадр из записи `7.mp4`: стол на двоих, журнал развёрнут, девять
        # строк. Имён на нём ровно два, и ходы идут через одного — как и
        # положено раздаче на двоих. Разошлись бы отпечатки — вышло бы семь
        # игроков за столом на двоих, и вся статистика по ним стала бы ложью.
        frame = cv2.imread(str(FIXTURES / "event-log-hand.jpg"))
        self.assertIsNotNone(frame, "нет опорного кадра event-log-hand")
        roster = Roster()

        rows = [(event.action, roster.number_of(event.name))
                for event in event_log(frame)]

        self.assertEqual(len(roster), 2)
        self.assertEqual(rows, [
            ("конец", None),
            ("кнопка", 0),
            ("начало", None),
            ("блайнд", 1),
            ("блайнд", 0),
            ("рейз", 1),
            ("колл", 0),
            ("чек", 1),
            ("чек", 0),
        ])


class WordingTests(unittest.TestCase):
    def test_players_agree_with_the_number(self) -> None:
        self.assertEqual(players_word(1), "игрока")
        self.assertEqual(players_word(2), "игроков")
        self.assertEqual(players_word(5), "игроков")
        self.assertEqual(players_word(11), "игроков")
        self.assertEqual(players_word(21), "игрока")


if __name__ == "__main__":
    unittest.main()
