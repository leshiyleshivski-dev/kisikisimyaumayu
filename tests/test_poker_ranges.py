"""Проверки стартовых рук: классы, их сила и запись диапазона строкой."""

from __future__ import annotations

import unittest

from kisiki.modules.poker_ranges import (
    KIND_COMBOS, STRENGTH_INDEX, STRENGTH_ORDER, TOTAL_COMBOS, class_combos,
    class_kind, combos_of, hand_class, parse_range, range_share, top_share,
)


class HandClassTests(unittest.TestCase):
    def test_two_cards_turn_into_the_usual_notation(self) -> None:
        self.assertEqual(hand_class(["Ah", "Kd"]), "AKo")
        self.assertEqual(hand_class(["Ah", "Kh"]), "AKs")
        self.assertEqual(hand_class(["Ah", "As"]), "AA")

    def test_order_of_the_two_cards_does_not_matter(self) -> None:
        self.assertEqual(hand_class(["Kd", "Ah"]), hand_class(["Ah", "Kd"]))
        self.assertEqual(hand_class(["2c", "7s"]), "72o")

    def test_ten_is_written_the_way_the_rest_of_the_code_writes_it(self) -> None:
        self.assertEqual(hand_class(["Td", "Ts"]), "TT")
        self.assertEqual(hand_class(["10d", "9d"]), "T9s")

    def test_a_hand_needs_exactly_two_cards(self) -> None:
        with self.assertRaises(ValueError):
            hand_class(["Ah"])
        with self.assertRaises(ValueError):
            hand_class(["Ah", "Kd", "Qc"])


class CombosTests(unittest.TestCase):
    def test_each_kind_has_the_number_of_suits_it_should(self) -> None:
        self.assertEqual(len(combos_of("AA")), 6)
        self.assertEqual(len(combos_of("AKs")), 4)
        self.assertEqual(len(combos_of("AKo")), 12)

    def test_every_combo_reads_back_as_its_own_class(self) -> None:
        # Если сочетание вдруг окажется чужим классом, диапазон соперника
        # молча наберётся не теми руками, и эквити будет считаться не про то.
        for name in STRENGTH_ORDER:
            for first, second in combos_of(name):
                self.assertEqual(hand_class([first, second]), name)

    def test_combos_inside_a_class_never_repeat(self) -> None:
        for name in ("AA", "AKs", "AKo", "72o"):
            pairs = combos_of(name)
            self.assertEqual(len(set(pairs)), len(pairs), name)

    def test_class_combos_matches_the_listed_combos(self) -> None:
        for name in STRENGTH_ORDER:
            self.assertEqual(class_combos(name), len(combos_of(name)), name)

    def test_kind_is_read_off_the_name(self) -> None:
        self.assertEqual(class_kind("AA"), "p")
        self.assertEqual(class_kind("AKs"), "s")
        self.assertEqual(class_kind("AKo"), "o")


class StrengthTests(unittest.TestCase):
    def test_all_hundred_sixty_nine_classes_are_listed_once(self) -> None:
        self.assertEqual(len(STRENGTH_ORDER), 169)
        self.assertEqual(len(set(STRENGTH_ORDER)), 169)

    def test_the_classes_cover_every_deal(self) -> None:
        self.assertEqual(sum(class_combos(name) for name in STRENGTH_ORDER), TOTAL_COMBOS)
        self.assertEqual(TOTAL_COMBOS, 52 * 51 // 2)

    def test_the_order_starts_and_ends_where_poker_says(self) -> None:
        self.assertEqual(STRENGTH_ORDER[0], "AA")
        self.assertEqual(STRENGTH_ORDER[-1], "32o")

    def test_pairs_are_ordered_by_rank(self) -> None:
        pairs = [name for name in STRENGTH_ORDER if class_kind(name) == "p"]
        self.assertEqual(pairs[:5], ["AA", "KK", "QQ", "JJ", "TT"])

    def test_suited_beats_the_same_hand_offsuit(self) -> None:
        # Одна масть добавляет флеш-дро, и слабее от этого рука быть не может.
        for high, low in (("A", "K"), ("K", "Q"), ("7", "2"), ("J", "T")):
            suited, offsuit = f"{high}{low}s", f"{high}{low}o"
            self.assertLess(
                STRENGTH_INDEX[suited], STRENGTH_INDEX[offsuit], f"{suited}/{offsuit}"
            )


class ShareTests(unittest.TestCase):
    def test_everything_is_the_whole_deck(self) -> None:
        self.assertAlmostEqual(range_share(STRENGTH_ORDER), 1.0)

    def test_share_counts_combos_and_not_class_names(self) -> None:
        # Одна пара — шесть сочетаний, одна разномастная рука — двенадцать.
        # Считать классы поштучно значило бы завысить вес пар вдвое.
        self.assertAlmostEqual(range_share(["AA"]), 6 / TOTAL_COMBOS)
        self.assertAlmostEqual(range_share(["AKo"]), 12 / TOTAL_COMBOS)
        self.assertAlmostEqual(range_share(["AA", "AA"]), 6 / TOTAL_COMBOS)

    def test_top_share_grows_with_the_share_asked_for(self) -> None:
        previous: frozenset[str] = frozenset()
        for share in (0.05, 0.1, 0.2, 0.4, 0.8):
            picked = top_share(share)
            self.assertTrue(previous <= picked, share)
            self.assertGreaterEqual(range_share(picked), share * 0.9)
            previous = picked

    def test_top_share_takes_the_best_hands_first(self) -> None:
        self.assertIn("AA", top_share(0.02))
        self.assertNotIn("32o", top_share(0.5))

    def test_asking_for_everything_returns_everything(self) -> None:
        self.assertEqual(top_share(1.0), frozenset(STRENGTH_ORDER))


class RangeNotationTests(unittest.TestCase):
    def test_a_single_hand_is_itself(self) -> None:
        self.assertEqual(parse_range("AKs"), {"AKs"})
        self.assertEqual(parse_range("77"), {"77"})

    def test_pairs_with_a_plus_run_up_to_aces(self) -> None:
        self.assertEqual(
            parse_range("TT+"), {"TT", "JJ", "QQ", "KK", "AA"}
        )

    def test_pair_spans_read_in_either_direction(self) -> None:
        self.assertEqual(parse_range("55-88"), parse_range("88-55"))
        self.assertEqual(parse_range("55-88"), {"55", "66", "77", "88"})

    def test_the_plus_walks_the_kicker_and_not_the_connector(self) -> None:
        # Запись «65s+» в разных программах читают по-разному: то как все
        # связки от 65s, то как все одномастные шестёрки. Здесь плюс всегда
        # наращивает младшую карту — и связки поэтому пишутся перечнем.
        self.assertEqual(parse_range("KTo+"), {"KTo", "KJo", "KQo"})
        self.assertEqual(len(parse_range("A2s+")), 12)
        self.assertIn("A2s", parse_range("A2s+"))
        self.assertIn("AKs", parse_range("A2s+"))
        self.assertNotIn("AA", parse_range("A2s+"))

    def test_kicker_spans_stay_inside_their_bounds(self) -> None:
        self.assertEqual(parse_range("A5s-A2s"), {"A2s", "A3s", "A4s", "A5s"})

    def test_commas_and_spaces_are_ignored(self) -> None:
        self.assertEqual(
            parse_range("77+,  AKs ,\n AKo"), parse_range("77+, AKs, AKo")
        )

    def test_nonsense_is_an_error_and_not_an_empty_range(self) -> None:
        # Молча выкинуть половину таблицы хуже, чем упасть на тесте: тихо
        # обрезанный диапазон открытия — это тихо проигранные деньги.
        for token in ("ZZ", "AKx", "A", "AK", "10s", "AKs+o"):
            with self.assertRaises(ValueError, msg=token):
                parse_range(token)

    def test_every_parsed_name_is_a_real_class(self) -> None:
        for text in ("22+", "A2s+", "K2o+", "32s", "A5s-A2s", "22-AA"):
            for name in parse_range(text):
                self.assertIn(name, STRENGTH_INDEX, text)


class KindTableTests(unittest.TestCase):
    def test_the_three_kinds_are_the_only_ones(self) -> None:
        self.assertEqual(set(KIND_COMBOS), {"p", "s", "o"})
        self.assertEqual(KIND_COMBOS, {"p": 6, "s": 4, "o": 12})


if __name__ == "__main__":
    unittest.main()
