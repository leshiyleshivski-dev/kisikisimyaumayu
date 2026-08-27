"""Кот «Блеф», рецепт POKER ADVISOR и память экрана между кадрами."""

from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

import cv2
import numpy as np

from food_catalog import FOOD_NAMES, SECRET_CAT_INDICES, SECRET_RECIPES
from kisiki.core import CATS, COMING_SOON_CATS, resource_path
from kisiki.modules.poker import module as poker
from kisiki.modules.poker.hand_math import Advice
from kisiki.modules.poker.journal import Journal, new_hand
from kisiki.modules.poker.module import STEPS, HandMemory, PokerModule, cards_text
from kisiki.modules.poker.players import Roster
from kisiki.modules.poker.vision import (
    NAME_HEIGHT, NAME_WIDTH, LogEvent, TableState,
)

GLYPH_WIDTH, GLYPH_HEIGHT = 24, 32


def table(**changes) -> TableState:
    """Кадр стола с полями по умолчанию: меняем только нужное.

    По умолчанию за столом сидят все шестеро, кнопка у нижнего правого места,
    блайнды 25 / 50 и тысяча фишек в стеке. Это нужно, чтобы у любого
    ``hero_seat`` находилась позиция: без неё до флопа помощник молчит, и
    проверки про доплату и банк проверяли бы совсем не то. Тесты про саму
    позицию задают эти поля сами.
    """
    fields = {
        "board": (), "hole": (), "hero_seat": None, "players": 0,
        "pot": None, "bets": (0, 0, 0, 0, 0, 0), "to_call": None,
        "my_turn": False, "showdown": False, "stack": 1000, "min_bet": None,
        "dealer": 5, "blinds": (25, 50), "live_seats": (0, 1, 2, 3, 4, 5),
    }
    fields.update(changes)
    return TableState(**fields)


def tip(
    action: str, raise_to: int = 0, all_in: bool = False, button: str = ""
) -> Advice:
    """Совет с одними лишь полями размера: остальное строке размера не нужно."""
    return Advice(action, 0.5, 0.3, raise_to, "", "", all_in, button)


def empty_frames(memory: HandMemory, count: int) -> None:
    """Столько кадров подряд без карт на столе, сколько попросили."""
    for _ in range(count):
        memory.update(table(players=0))


def verdict(memory: HandMemory, state) -> str | None:
    """Кадр так же, как его смотрит экран: сперва запомнить, потом спросить."""
    memory.update(state)
    return memory.blocker(state)


class PokerCatalogTests(unittest.TestCase):
    def test_bluff_replaces_the_placeholder(self) -> None:
        self.assertEqual(CATS[8][0], "Блеф")
        self.assertTrue(CATS[8][2].endswith("11_poker_cat.png"))
        self.assertEqual(CATS[8][4], "Лудоманы")
        self.assertNotIn("Занос", {cat[0] for cat in COMING_SOON_CATS})

    def test_poker_recipe_is_registered_for_the_bluff(self) -> None:
        recipes = {secret_id: ingredients for secret_id, _title, ingredients in SECRET_RECIPES}
        self.assertEqual(SECRET_CAT_INDICES["poker"], 8)
        self.assertEqual(recipes["poker"], ("tuna_can", "donut", "mystery_meal"))

    def test_recipe_uses_only_known_food(self) -> None:
        # Рецепт рисуется иконками из каталога: незнакомый ключ уронил бы
        # книгу рецептов, а не только кормилку.
        for _secret_id, _title, ingredients in SECRET_RECIPES:
            for food_id in ingredients:
                self.assertIn(food_id, FOOD_NAMES)

    def test_every_cat_index_points_at_a_real_cat(self) -> None:
        for secret_id, index in SECRET_CAT_INDICES.items():
            self.assertLess(index, len(CATS), secret_id)


class PokerModuleTests(unittest.TestCase):
    def test_module_never_reaches_for_the_input_helpers(self) -> None:
        # Главное обещание экрана: он советует, а нажимает человек. Проверяем
        # структурой, а не словами — отправлялки ввода сюда не импортированы.
        for helper in (
            "send_key_tap", "send_left_click", "send_relative_move",
            "move_relative_and_bet", "activate_window", "confine_cursor_to_client",
        ):
            self.assertFalse(hasattr(poker, helper), helper)

    def test_scan_interval_leaves_room_for_the_maths(self) -> None:
        # Совет вместе с планом на ответ соперника стоит до 300 мс: кадры не
        # должны идти чаще, чем он считается, иначе очередь таймеров начнёт
        # копиться. Считается он только на изменившемся столе, а следующий
        # кадр заводится после разбора предыдущего, а не по расписанию.
        self.assertGreaterEqual(PokerModule.SCAN_INTERVAL_MS, 200)

    def test_ten_is_written_the_way_the_card_writes_it(self) -> None:
        # Внутри модулей десятка зовётся «T», но на экране это читается как
        # «туз»: на карте игра рисует «10», и мы пишем так же.
        self.assertEqual(cards_text(("Ts", "8d")), "10♠  8♦")
        self.assertEqual(cards_text(("Ah", "Kc")), "A♥  K♣")
        self.assertEqual(cards_text(()), "—")

    def test_growing_pot_animation_does_not_redo_the_advice(self) -> None:
        # Пока анимация сгребает фишки, плашка банка растёт числами 150, 168,
        # 190, 199. Совет от такого прироста не меняется, а подсказка на
        # экране дёргалась бы каждую четверть секунды.
        self.assertFalse(PokerModule.pot_moved(155, 150))
        self.assertFalse(PokerModule.pot_moved(199, 190))
        self.assertTrue(PokerModule.pot_moved(300, 150), "ставка соперника — это уже другое")
        self.assertTrue(PokerModule.pot_moved(75, None), "первый счёт всегда считается")

    def test_the_plan_is_shown_and_cleared_along_with_the_advice(self) -> None:
        # Устаревший план хуже отсутствующего: «доплата до 1 500 — колл» от
        # прошлого кадра игрок прочитает как совет на этот. Проверяем
        # структурой: окно CustomTkinter в тестах не поднимается.
        for method in (PokerModule.show_advice, PokerModule.show_silence):
            self.assertIn("self.plan_text.set", inspect.getsource(method), method.__name__)

    def test_every_advice_shown_is_also_written_down(self) -> None:
        # Журнал без совета отвечает только на «сколько проиграли», а
        # спрашивают его о другом: совет был плохой или совет не послушали.
        # Проверяем структурой — окно CustomTkinter в тестах не поднимается, а
        # пропустить одну из двух считалок легко: их ровно две, до флопа и
        # после, и записываться обязаны обе.
        source = inspect.getsource(PokerModule.update_advice)

        self.assertEqual(
            source.count("self.memory.remember_advice("),
            source.count("self.show_advice("),
        )

    def test_size_line_answers_how_much_to_put_in(self) -> None:
        # «Рейз» без числа и есть тот вопрос, ради которого экран заводился:
        # сколько именно ставить. Игра просит итоговую ставку улицы.
        self.assertEqual(PokerModule.size_line(tip("бет", 100), 0), "поставить 100")
        self.assertEqual(PokerModule.size_line(tip("рейз", 1200), 250), "поднять до 1 200")
        self.assertEqual(PokerModule.size_line(tip("колл"), 250), "доплатить 250")
        self.assertEqual(PokerModule.size_line(tip("фолд"), 0), "")
        self.assertEqual(PokerModule.size_line(tip("чек"), 0), "")

    def test_size_line_says_when_the_bet_is_the_whole_stack(self) -> None:
        line = PokerModule.size_line(tip("рейз", 770, all_in=True), 250)

        self.assertIn("770", line)
        self.assertIn("ALL IN", line)

    def test_size_line_admits_it_cannot_name_a_size(self) -> None:
        # Без банка размер считать не от чего, а молча писать «бет» без числа
        # — ровно то, на что жаловались.
        self.assertIn("не подскажу", PokerModule.size_line(tip("бет", 0), 0))

    def test_size_line_names_the_button_that_sets_the_size(self) -> None:
        # Ради этого пункт и брался: игра ставит четыре готовых размера одним
        # нажатием, а число игрок ведёт ползунком на таймере и промахивается.
        line = PokerModule.size_line(tip("рейз", 1200, button="BANK"), 250)

        self.assertTrue(line.startswith("жми BANK"), line)
        self.assertIn("1 200", line, "число остаётся — есть с чем сверить")

    def test_the_shove_button_is_named_once(self) -> None:
        # «жми ALL IN … , ALL IN» — то же слово дважды в одной строке.
        line = PokerModule.size_line(tip("пуш", 3000, button="ALL IN"), 500)

        self.assertEqual(line.count("ALL IN"), 1, line)
        self.assertIn("3 000", line)

    def test_without_a_button_the_line_stays_as_it_was(self) -> None:
        # Кнопка находится не всегда: размер между «3 BB» и «BANK» игрок
        # выставляет ползунком, и тогда строка должна остаться прежней.
        self.assertEqual(PokerModule.size_line(tip("рейз", 1200), 250), "поднять до 1 200")
        self.assertIn("ALL IN", PokerModule.size_line(tip("пуш", 3000), 500))

    def test_pots_line_names_the_pots_and_keeps_quiet_otherwise(self) -> None:
        # Какой из горшков наш, помощник не считает — для этого надо знать,
        # сколько внёс каждый. Зато сказать, что банк разложен, обязан:
        # шансы банка он считает по всему банку целиком.
        line = PokerModule.pots_line((2_000, 5_000))

        self.assertIn("2 000", line)
        self.assertIn("5 000", line)
        self.assertIn("олл-ин", line)
        self.assertEqual(PokerModule.pots_line(()), "")

    def test_steps_are_shown(self) -> None:
        self.assertEqual(len(STEPS), 2)
        for number, heading, description in STEPS:
            self.assertTrue(number and heading and description)


class HandMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = HandMemory()

    def test_hole_cards_survive_a_blind_frame(self) -> None:
        # Свои карты может на кадр перекрыть рука дилера. Забывать руку из-за
        # одного кадра нельзя: подсказка погаснет посреди хода.
        self.memory.update(table(hole=("Kh", "3h"), hero_seat=5, players=2))
        self.memory.update(table(players=2))

        self.assertEqual(self.memory.hole, ("Kh", "3h"))

    def test_hand_ends_when_the_table_is_empty(self) -> None:
        self.memory.update(table(hole=("Kh", "3h"), hero_seat=5, players=2, pot=250))
        empty_frames(self.memory, HandMemory.EMPTY_FRAMES_TO_FORGET)

        self.assertEqual(self.memory.hole, ())
        self.assertIsNone(self.memory.pot)
        self.assertEqual(self.memory.seat, 5, "место за столом от раздачи не зависит")

    def test_one_empty_frame_does_not_end_the_hand(self) -> None:
        # Пустой кадр посреди раздачи — это анимация или чужая рука над
        # столом. Раньше от одного такого кадра гасла вся подсказка.
        self.memory.update(
            table(hole=("Kh", "3h"), board=("2c", "7d", "9s"), hero_seat=5,
                  players=2, pot=250)
        )
        empty_frames(self.memory, HandMemory.EMPTY_FRAMES_TO_FORGET - 1)

        self.assertEqual(self.memory.hole, ("Kh", "3h"))
        self.assertEqual(self.memory.board, ("2c", "7d", "9s"))
        self.assertEqual(self.memory.pot, 250)

    def test_pot_is_remembered_while_the_plate_hides(self) -> None:
        # Плашки банка нет, пока в банке пусто, и она пропадает между улицами.
        self.memory.update(table(players=2, pot=17_250))
        self.memory.update(table(players=2))

        self.assertEqual(self.memory.pot, 17_250)

    def test_pot_counts_the_bets_lying_on_the_felt(self) -> None:
        # На префлопе плашки банка нет вовсе: блайнды лежат ставками перед
        # игроками. Считать банк по одной плашке — занижать его.
        self.memory.update(table(players=2, pot=0, bets=(0, 0, 0, 0, 50, 25)))

        self.assertEqual(self.memory.pot, 75)

    def test_pot_does_not_shrink_inside_a_hand(self) -> None:
        # Пока улица меняется, ставки с сукна уже убрали, а плашка банка ещё
        # не выросла: на этих кадрах банк выглядит меньше, чем он есть.
        self.memory.update(table(players=2, pot=100, bets=(0, 0, 0, 0, 50, 50)))
        self.memory.update(table(players=2, pot=100, bets=(0, 0, 0, 0, 0, 0)))

        self.assertEqual(self.memory.pot, 200)

    def test_pot_holds_while_the_plate_catches_up_to_the_pots(self) -> None:
        # Банк разложен по горшкам, и плашка над бордом догоняет их анимацией:
        # 9 842 при настоящих 13 000. Заниженный на четверть банк завышает
        # шансы банка, поэтому такой кадр банка не касается.
        settled = table(players=3, pot=7_000, bets=(0, 0, 0, 6_000, 0, 0),
                        side_pots=(2_000, 5_000))
        self.memory.update(settled)
        self.assertEqual(self.memory.pot, 13_000)

        self.memory.update(replace(settled, pot=9_842, bets=(0, 0, 0, 0, 0, 0)))

        self.assertEqual(self.memory.pot, 13_000)

    def test_pots_are_remembered_while_the_row_stays(self) -> None:
        # Сложенный посреди раздачи горшок никуда не денется: полоса, которую
        # не удалось прочесть, — плохой кадр, а не «горшков больше нет».
        self.memory.update(table(players=3, pot=7_000, side_pots=(2_000, 5_000)))
        self.memory.update(table(players=3, pot=7_000, side_pots=(None, 5_000)))

        self.assertEqual(self.memory.side_pots, (2_000, 5_000))

    def test_pots_are_forgotten_when_the_row_is_gone(self) -> None:
        self.memory.update(table(players=3, pot=7_000, side_pots=(2_000, 5_000)))
        self.memory.update(table(players=3, pot=7_000))

        self.assertEqual(self.memory.side_pots, ())

    def test_unreadable_bet_does_not_touch_the_pot(self) -> None:
        # Плашка ставки есть, а число не сложилось: сумма неизвестна целиком.
        self.memory.update(table(players=2, pot=100, bets=(0, 0, 0, 0, None, 0)))

        self.assertIsNone(self.memory.pot)

    def test_board_survives_a_blind_frame(self) -> None:
        self.memory.update(table(board=("2c", "7d", "9s", "Ah"), players=2))
        self.memory.update(table(board=("2c", "7d"), players=2))

        self.assertEqual(self.memory.board, ("2c", "7d", "9s", "Ah"))

    def test_half_seen_hand_is_not_a_new_hand(self) -> None:
        # Одну карту из двух может закрыть рука дилера. Раньше такой кадр
        # считался новой раздачей и стирал руку вместе с банком.
        self.memory.update(
            table(hole=("Kh", "3h"), board=("2c", "7d", "9s"), hero_seat=5,
                  players=2, pot=250)
        )
        self.memory.update(table(hole=("Kh",), board=("2c", "7d", "9s"), players=2))

        self.assertEqual(self.memory.hole, ("Kh", "3h"))
        self.assertEqual(self.memory.pot, 250)

    def test_a_different_board_means_a_new_hand(self) -> None:
        # Стол успел перетасоваться, пока помощник моргал: держаться за старую
        # руку нельзя, иначе совет пойдёт по картам прошлой раздачи.
        self.memory.update(
            table(hole=("Kh", "3h"), board=("2c", "7d", "9s"), hero_seat=5,
                  players=2, pot=250)
        )
        self.memory.update(table(board=("As", "Kd", "4c"), players=2))

        self.assertEqual(self.memory.hole, ())
        self.assertEqual(self.memory.board, ("As", "Kd", "4c"))

    def test_seat_survives_the_showdown(self) -> None:
        self.memory.update(table(hero_seat=2, players=3))
        self.memory.update(table(players=3, showdown=True))

        self.assertEqual(self.memory.seat, 2)

    def test_seat_does_not_jump_to_a_lone_opponent(self) -> None:
        # Своя рука сброшена, а до вскрытия дошёл один соперник: его карты
        # лежат лицом, и по кадру место не отличить от своего. Пересесть
        # посреди раздачи нельзя — иначе чужая рука станет своей.
        self.memory.update(table(hero_seat=2, players=3))
        self.memory.update(table(hero_seat=5, players=2, hole=("Ah", "Kh")))

        self.assertEqual(self.memory.seat, 2)

    def test_new_hand_lets_the_player_change_seats(self) -> None:
        self.memory.update(table(hero_seat=2, players=3))
        empty_frames(self.memory, HandMemory.EMPTY_FRAMES_TO_FORGET)
        self.memory.update(table(hero_seat=4, players=3))

        self.assertEqual(self.memory.seat, 4)

    def test_own_bet_and_stack_are_remembered(self) -> None:
        # Размер рейза считается от своей ставки этой улицы, а стек не даёт
        # посоветовать больше, чем есть фишек.
        self.memory.update(
            table(hero_seat=5, players=2, bets=(0, 0, 0, 0, 0, 250), stack=520)
        )

        self.assertEqual(self.memory.my_bet, 250)
        self.assertEqual(self.memory.stack, 520)

    def test_minimum_bet_is_dropped_with_the_call(self) -> None:
        # Кто-то поставил — и минимальный рейз стал другим, как и доплата.
        for _ in range(2):
            self.memory.update(
                table(hero_seat=5, players=2, pot=250, to_call=50, min_bet=100)
            )
        self.memory.update(
            table(hero_seat=5, players=2, pot=250, to_call=None, bets=(0, 0, 0, 0, 500, 0))
        )

        self.assertIsNone(self.memory.min_bet)

    def test_the_minimum_is_taken_from_the_second_matching_frame(self) -> None:
        # Курсор поверх цифры даёт число на один кадр: на кнопке `RAISE 4 000`
        # прочиталось 654 000. Минимум работает полом для всех четырёх быстрых
        # кнопок, и по такому числу совет уходил в олл-ин на весь стек.
        # Настоящий минимум стоит на кнопке всю улицу — второго кадра ждать
        # четверть секунды, а стек он бережёт целиком.
        state = table(hero_seat=5, players=2, pot=12_000, to_call=3_500, min_bet=4_000)
        self.memory.update(state)

        self.assertIsNone(self.memory.min_bet, "одного кадра для минимума мало")

        self.memory.update(replace(state, min_bet=654_000))

        self.assertIsNone(self.memory.min_bet, "мигнувшее число не минимум")

        self.memory.update(state)
        self.memory.update(state)

        self.assertEqual(self.memory.min_bet, 4_000)

    def test_the_stack_in_the_pause_waits_for_the_animation(self) -> None:
        # Пока банк едет к победителю, панель показывает промежуточные числа —
        # 19 988, 14 855, 10 618, — и каждое годится в результат раздачи не
        # больше, чем стрелка часов посреди оборота. Берём то, что
        # повторилось: досчитанный стек стоит до самой раздачи.
        for stack in (19_988, 14_855, 10_618, 15_000, 15_000):
            self.memory.update(table(players=0, stack=stack))

        self.assertEqual(self.memory.pause_stack, 15_000)
        self.assertEqual(self.memory.stack, 15_000, "совету идёт свежее число")

    def test_silent_until_the_move_is_mine(self) -> None:
        self.memory.update(table(hole=("Kh", "3h"), hero_seat=5, players=2, pot=250))

        self.assertIn("хода", verdict(self.memory, table(players=2)))
        self.assertIsNone(
            verdict(self.memory, table(players=2, my_turn=True, to_call=50, pot=250))
        )

    def test_silent_without_own_cards(self) -> None:
        self.memory.update(table(players=2, pot=250))

        self.assertIn("карт", verdict(self.memory, table(players=2, my_turn=True)))

    def test_silent_without_the_pot_when_a_call_is_due(self) -> None:
        # Без банка шансы банка выходят стопроцентными, и совет всегда был бы
        # «фолд» — уверенная чушь вместо честного молчания.
        self.memory.update(table(hole=("Kh", "3h"), hero_seat=5, players=2))
        blocker = verdict(self.memory, table(players=2, my_turn=True, to_call=500))

        self.assertIn("банке", blocker)

    def test_silent_when_the_call_button_is_unreadable(self) -> None:
        # None на кнопке — «сумма есть, но не разобралась». Считать её нулём
        # нельзя: непрочитанный колл выглядел бы бесплатным чеком.
        self.memory.update(table(hole=("Kh", "3h"), hero_seat=5, players=2, pot=250))
        blocker = verdict(
            self.memory, table(players=2, my_turn=True, to_call=None, pot=900)
        )

        self.assertIn("кнопке", blocker)

    def test_call_survives_the_cursor_over_the_button(self) -> None:
        # Курсор накрывает цифру ровно тогда, когда игрок решает, что делать.
        # Пока на столе ничего не менялось, доплата остаётся прежней.
        self.memory.update(
            table(hole=("Kh", "3h"), hero_seat=5, players=2, pot=250, to_call=50)
        )
        blocker = verdict(
            self.memory, table(players=2, my_turn=True, to_call=None, pot=250)
        )

        self.assertIsNone(blocker)
        self.assertEqual(self.memory.to_call, 50)

    def test_new_money_on_the_table_drops_the_remembered_call(self) -> None:
        # Кто-то поставил — значит доплата уже другая, и старую держать нельзя.
        self.memory.update(
            table(hole=("Kh", "3h"), hero_seat=5, players=2, pot=250, to_call=50)
        )
        blocker = verdict(
            self.memory,
            table(players=2, my_turn=True, to_call=None, pot=250,
                  bets=(0, 0, 0, 0, 500, 0)),
        )

        self.assertIn("кнопке", blocker)

    def test_free_check_needs_no_pot(self) -> None:
        self.memory.update(table(hole=("Kh", "3h"), hero_seat=5, players=2))

        self.assertIsNone(
            verdict(self.memory, table(players=2, my_turn=True, to_call=0))
        )

    def test_silent_at_the_showdown(self) -> None:
        self.memory.update(table(hole=("Kh", "3h"), hero_seat=5, players=2, pot=250))
        blocker = verdict(
            self.memory, table(players=2, my_turn=True, to_call=50, showdown=True)
        )

        self.assertIn("Вскрытие", blocker)


class PokerTemplateTests(unittest.TestCase):
    """Эталоны знаков карт вырезаны из игры и должны попадать в сборку."""

    def strip(self, name: str):
        return cv2.imread(
            str(resource_path("assets", "vision", name)), cv2.IMREAD_GRAYSCALE
        )

    def test_rank_strip_holds_thirteen_glyphs(self) -> None:
        strip = self.strip("poker_ranks.png")

        self.assertIsNotNone(strip)
        self.assertEqual(strip.shape, (GLYPH_HEIGHT, GLYPH_WIDTH * 13))

    def test_suit_strip_holds_four_glyphs(self) -> None:
        strip = self.strip("poker_suits.png")

        self.assertIsNotNone(strip)
        self.assertEqual(strip.shape, (GLYPH_HEIGHT, GLYPH_WIDTH * 4))

    def test_digit_strip_holds_ten_glyphs(self) -> None:
        strip = self.strip("poker_digits.png")

        self.assertIsNotNone(strip)
        self.assertEqual(strip.shape, (26, 18 * 10))

    def test_every_glyph_cell_has_ink(self) -> None:
        # Пустая клетка означала бы, что знак при нарезке потерялся.
        for name, count, width in (
            ("poker_ranks.png", 13, GLYPH_WIDTH),
            ("poker_suits.png", 4, GLYPH_WIDTH),
            ("poker_digits.png", 10, 18),
        ):
            strip = self.strip(name)
            for index in range(count):
                cell = strip[:, index * width:(index + 1) * width]
                self.assertGreater((cell > 127).sum(), 60, f"{name}[{index}]")


if __name__ == "__main__":
    unittest.main()


class SeatingMemoryTests(unittest.TestCase):
    """Кнопка, позиция и блайнды: что помнится и как долго."""

    def setUp(self) -> None:
        self.memory = HandMemory()

    def test_the_position_is_read_from_the_button(self) -> None:
        self.memory.update(table(
            hole=("Kh", "3h"), hero_seat=5, players=2,
            dealer=4, live_seats=(4, 5),
        ))

        self.assertEqual(self.memory.dealer, 4)
        self.assertEqual(self.memory.position, "SB")

    def test_the_position_survives_an_opponent_folding(self) -> None:
        # Позиция считается по тем, у кого ещё есть карты, а люди из раздачи
        # выбывают. Пересчитанная после чужого фолда, она превратила бы
        # большой блайнд в малый — и таблица выдала бы совет не про эту руку.
        self.memory.update(table(
            hole=("Kh", "3h"), hero_seat=2, players=3,
            dealer=5, live_seats=(2, 4, 5),
        ))
        before = self.memory.position
        self.memory.update(table(
            hole=("Kh", "3h"), hero_seat=2, players=2,
            dealer=5, live_seats=(2, 5),
        ))

        self.assertEqual(before, "BB")
        self.assertEqual(self.memory.position, "BB")

    def test_a_new_hand_looks_at_the_button_again(self) -> None:
        self.memory.update(table(
            hole=("Kh", "3h"), hero_seat=5, players=2, dealer=4, live_seats=(4, 5),
        ))
        empty_frames(self.memory, HandMemory.EMPTY_FRAMES_TO_FORGET)
        self.memory.update(table(
            hole=("Ah", "Qd"), hero_seat=5, players=2, dealer=5, live_seats=(4, 5),
        ))

        self.assertEqual(self.memory.dealer, 5)
        self.assertEqual(self.memory.position, "BB")

    def test_blinds_outlive_the_hand(self) -> None:
        # Ставки за столом не меняются, а чистое начало раздачи видно не
        # каждый кадр. Узнали один раз — помним до конца сеанса.
        self.memory.update(table(
            hole=("Kh", "3h"), hero_seat=5, players=2, dealer=4,
            live_seats=(4, 5), blinds=(250, 500),
        ))
        empty_frames(self.memory, HandMemory.EMPTY_FRAMES_TO_FORGET)

        self.assertEqual(self.memory.big_blind, 500)

    def test_silent_before_the_flop_without_a_button(self) -> None:
        # До флопа решает таблица, а ей нужна позиция: одна и та же рука с
        # кнопки открывается, а с ранней позиции сбрасывается.
        self.memory.update(table(
            hole=("Kh", "3h"), hero_seat=5, players=2, dealer=None, live_seats=(4, 5),
        ))
        blocker = verdict(self.memory, table(
            players=2, my_turn=True, to_call=50, pot=250,
            dealer=None, live_seats=(4, 5),
        ))

        self.assertIn("дилера", blocker)

    def test_silent_before_the_flop_without_the_blinds(self) -> None:
        self.memory.update(table(
            hole=("Kh", "3h"), hero_seat=5, players=2, dealer=4,
            live_seats=(4, 5), blinds=None,
        ))
        blocker = verdict(self.memory, table(
            players=2, my_turn=True, to_call=50, pot=250,
            dealer=4, live_seats=(4, 5), blinds=None,
        ))

        self.assertIn("блайнды", blocker)

    def test_after_the_flop_the_button_is_not_required(self) -> None:
        # На флопе считает эквити, а ему позиция не нужна: молчать из-за
        # закрытой курсором метки там было бы молчанием на пустом месте.
        self.memory.update(table(
            hole=("Kh", "3h"), board=("6h", "Qs", "8h"), hero_seat=5, players=2,
            pot=250, dealer=None, live_seats=(4, 5), blinds=None,
        ))
        blocker = verdict(self.memory, table(
            board=("6h", "Qs", "8h"), players=2, my_turn=True, to_call=50, pot=250,
            dealer=None, live_seats=(4, 5), blinds=None,
        ))

        self.assertIsNone(blocker)


class JournalCollectionTests(unittest.TestCase):
    """Как раздача попадает из кадров в журнал.

    Считать помощник умеет, а отличить «стало лучше» от «повезло» без журнала
    нельзя — и спор об этом повторялся после каждой записи.
    """

    def setUp(self) -> None:
        self.journal = Journal()
        self.memory = HandMemory(on_hand=self.journal.add)

    def idle(self, stack: int, frames: int = 3) -> None:
        """Стол между раздачами: карт нет, панель со стеком на месте."""
        for _ in range(frames):
            self.memory.update(table(players=0, stack=stack))

    def play(self, **changes) -> None:
        fields = {
            "players": 3, "hero_seat": 5, "hole": ("Ah", "Kd"), "stack": 950,
            "pot": 1200, "live_seats": (0, 4, 5),
        }
        fields.update(changes)
        self.memory.update(table(**fields))

    def test_a_played_hand_lands_in_the_journal(self) -> None:
        self.idle(1000)
        self.play()
        self.play(board=("2c", "7d", "Qs"), stack=800)
        self.idle(1400)

        self.assertEqual(len(self.journal), 1)
        hand = self.journal.hands[0]
        self.assertEqual(hand.hole, ("Ah", "Kd"))
        self.assertEqual(hand.board, ("2c", "7d", "Qs"))
        self.assertEqual(hand.position, "BTN")
        self.assertEqual(hand.big_blind, 50)

    def test_the_result_is_measured_between_hands(self) -> None:
        # Внутри раздачи блайнды уже поставлены: мерка от первого кадра
        # потеряла бы их, а это четверть ставки на раздачу — больше, чем весь
        # выигрыш, который тут вообще меряют.
        self.idle(1000)
        self.play(stack=950)
        self.idle(1400)

        self.assertEqual(self.journal.hands[0].result, 400)

    def test_a_purchase_between_hands_leaves_the_hand_uncounted(self) -> None:
        # Стек вырос сильнее банка — значит, докупили фишек. Соврать про
        # BB/100 хуже, чем посчитать его по меньшему числу раздач.
        self.idle(1000)
        self.play(pot=1200)
        self.idle(9000)

        self.assertEqual(len(self.journal), 1)
        self.assertIsNone(self.journal.hands[0].result)

    def test_a_hand_joined_midway_is_not_counted(self) -> None:
        # Севший смотреть посреди раздачи не знает, сколько уже ушло из стека
        # в банк, и его «выигрыш» был бы чистой выдумкой.
        self.play()
        self.play(board=("2c", "7d", "Qs"))
        self.idle(1400)

        self.assertEqual(len(self.journal), 0)

    def test_the_hand_is_written_once(self) -> None:
        # Стол пустеет на несколько секунд, и «раздача кончилась» случается на
        # каждом кадре паузы.
        self.idle(1000)
        self.play()
        self.idle(1400, frames=12)

        self.assertEqual(len(self.journal), 1)

    def test_the_win_mark_survives_a_blind_frame(self) -> None:
        # Метку показывают считаные кадры: банк отдают в самом конце раздачи и
        # почти сразу убирают карты со стола.
        self.idle(1000)
        self.play()
        self.play(showdown=True, winners=(5,), stack=800)
        self.play(showdown=True, stack=800)
        self.idle(1400)

        hand = self.journal.hands[0]
        self.assertTrue(hand.won)
        self.assertTrue(hand.showdown)

    def test_the_win_mark_is_caught_after_the_hand_closed(self) -> None:
        # Раздача закрывается через три пустых кадра — за три четверти
        # секунды, — а метку игра зажигает вместе с уезжающим банком, и банк
        # едет дольше. Раз уж раздача всё равно ждёт устоявшегося стека, пусть
        # дождётся и метки: иначе выигранная раздача уходит в журнал
        # проигранной.
        self.idle(1000)
        self.play(stack=1000, pot=8000)
        for stack in (2_137, 3_982, 4_610):
            self.memory.update(table(players=0, stack=stack))

        self.assertEqual(len(self.journal), 0, "стек ещё не досчитал")

        for _ in range(2):
            self.memory.update(table(players=0, stack=5000, winners=(5,)))

        self.assertEqual(len(self.journal), 1)
        self.assertTrue(self.journal.hands[0].won)
        self.assertEqual(self.journal.hands[0].result, 4000)

    def test_the_win_mark_does_not_carry_over_to_the_next_hand(self) -> None:
        # Гаснет метка в начале следующей раздачи. Оставленная гореть, она
        # записывала бы выигранной каждую раздачу после выигранной.
        self.idle(1000)
        self.play(showdown=True, winners=(5,), stack=800)
        self.idle(1400)
        self.play()
        self.play(board=("2c", "7d", "Qs"), stack=700)
        self.idle(1400)

        self.assertEqual(len(self.journal), 2)
        self.assertTrue(self.journal.hands[0].won)
        self.assertFalse(self.journal.hands[1].won)

    def test_the_advice_lands_in_the_journal_with_its_street(self) -> None:
        # Улица тут важнее самого совета: до флопа «фолд» стоит раздачи
        # целиком, а тот же «фолд» на флопе — одной ставки.
        self.idle(1000)
        self.play()
        self.memory.remember_advice("фолд")
        self.memory.remember_advice("фолд")
        self.play(board=("2c", "7d", "Qs"), stack=800)
        self.memory.remember_advice("чек")
        self.idle(1400)

        hand = self.journal.hands[0]
        self.assertEqual(hand.advice, ((0, "фолд"), (3, "чек")))
        self.assertEqual(hand.advised, "фолд")

    def test_the_advice_does_not_leak_into_the_next_hand(self) -> None:
        self.idle(1000)
        self.play()
        self.memory.remember_advice("фолд")
        self.idle(1400)
        self.play()
        self.idle(1400)

        self.assertEqual(self.journal.hands[0].advice, ((0, "фолд"),))
        self.assertEqual(self.journal.hands[1].advice, ())

    def test_hands_shown_at_the_showdown_are_kept(self) -> None:
        # Чужие карты видны только здесь — и это первое, что понадобится
        # статистике по соперникам.
        self.idle(1000)
        self.play()
        self.play(
            showdown=True, winners=(4,),
            shown=((), (), (), (), ("Jc", "Js"), ("Ah", "Kd")),
        )
        self.idle(900)

        hand = self.journal.hands[0]
        self.assertEqual(hand.shown, ((4, ("Jc", "Js")),), "своя рука тут лишняя")
        self.assertFalse(hand.won)

    def test_the_result_waits_for_the_chips_to_land(self) -> None:
        # Раздача закрывается через три пустых кадра, а банк едет к победителю
        # дольше. Записанная сразу, она уносила в журнал число из середины
        # анимации; теперь ждёт, пока стек устоится.
        self.idle(1000)
        self.play(stack=950, pot=8000)
        for stack in (2_137, 3_982, 4_610):
            self.memory.update(table(players=0, stack=stack))

        self.assertEqual(len(self.journal), 0, "стек ещё не досчитал")

        self.idle(5000)

        self.assertEqual(self.journal.hands[0].result, 4000)

    def test_a_lost_stack_is_a_real_zero(self) -> None:
        # Ушёл весь стек — в панели ноль, и это честный ноль. Раньше память
        # держала последнее ненулевое число из анимации, и раздача за весь
        # стек считалась вдвое дешевле, чем стоила.
        self.idle(43_750)
        self.play(stack=23_250, pot=88_500)
        self.idle(0, frames=4)

        self.assertEqual(self.journal.hands[0].result, -43_750)

    def test_cards_left_from_the_showdown_do_not_start_a_new_hand(self) -> None:
        # Стол пустеет не разом: карты со стола убирают, борд ещё лежит, и
        # через кадр открытые карты победителя появляются снова. На них
        # начиналась «новая раздача», и в журнал уходил дубль с тем же бордом
        # и чужой рукой вместо своей — в записи `8.mp4` таких пар две.
        self.idle(1000)
        self.play()
        self.play(board=("6c", "3s", "As"))
        self.idle(1400)
        for _ in range(4):
            self.memory.update(
                table(players=2, hole=("Qc", "Jd"), hero_seat=2, stack=1400,
                      board=("6c", "3s", "As"), showdown=True, winners=(2,),
                      live_seats=(2, 5))
            )
        self.idle(1400)

        self.assertEqual(len(self.journal), 1, "раздача одна, а не две")
        self.assertEqual(self.journal.hands[0].hole, ("Ah", "Kd"))

    def test_a_new_hand_closes_the_previous_one(self) -> None:
        # Между раздачами стол пустеет не всегда: следующая может начаться
        # прямо на следующем кадре.
        self.idle(1000)
        self.play()
        self.idle(1400)
        self.play(hole=("2c", "7d"))
        self.idle(1300)

        self.assertEqual(len(self.journal), 2)
        self.assertEqual(self.journal.hands[1].hole, ("2c", "7d"))
        self.assertEqual(self.journal.hands[1].result, -100)


class JournalLineTests(unittest.TestCase):
    """Строки журнала на экране: их читают между раздачами."""

    @staticmethod
    def filled(count: int, **changes) -> Journal:
        journal = Journal()
        fields = {"hole": ("Ah", "Kd"), "big_blind": 50, "pot": 100_000, "result": 100}
        fields.update(changes)
        for _ in range(count):
            journal.add(new_hand(**fields))
        return journal

    def test_an_empty_journal_says_so_plainly(self) -> None:
        journal = Journal()

        self.assertIn("пока нет", PokerModule.journal_line(journal))
        self.assertIn("не по чему", PokerModule.money_line(journal))

    def test_the_line_counts_hands_wins_and_showdowns(self) -> None:
        line = PokerModule.journal_line(self.filled(3, won=True, showdown=True))

        self.assertIn("3 раздачи", line)
        self.assertIn("выиграно 3", line)
        self.assertIn("до вскрытия дошло 3", line)

    def test_discipline_takes_the_place_of_showdowns(self) -> None:
        # Место под последнее число одно, и вскрытия уступают его счёту
        # сброшенным рукам, которые всё равно доигрывались: разница между
        # «совет плохой» и «совет не послушали» видна только по нему.
        journal = self.filled(
            2, result=-4000, position="CO", advice=((0, "фолд"),), showdown=True,
        )
        journal.add(new_hand(
            hole=("2c", "7d"), big_blind=50, pot=500, result=-50, position="BB",
            advice=((0, "фолд"),),
        ))
        line = PokerModule.journal_line(journal)

        self.assertIn("пас не послушан 2 из 3", line)
        self.assertNotIn("вскрытия", line)

    def test_the_rate_waits_for_enough_hands(self) -> None:
        # Одна выигранная раздача даёт «+800 BB/100» — число верное и
        # бессмысленное сразу.
        self.assertIn("BB/100 —", PokerModule.money_line(self.filled(1)))
        self.assertIn("+200 BB/100", PokerModule.money_line(self.filled(12)))

    def test_uncounted_hands_are_admitted_out_loud(self) -> None:
        # Раздач в счёте меньше, чем сыграно, и тихо считать по другому набору
        # нечестно.
        journal = self.filled(12)
        journal.add(new_hand(hole=("2c", "7d"), big_blind=50, pot=500))

        self.assertIn("в счёте 12 из 13", PokerModule.money_line(journal))

    def test_the_minus_is_written_the_same_way_everywhere(self) -> None:
        line = PokerModule.money_line(self.filled(12, result=-100))

        self.assertIn("−1 200 фишек", line)
        self.assertIn("−200 BB/100", line)


def moved(
    at: int, action: str, amount: int | None = None, name=None
) -> LogEvent:
    """Строка журнала событий: время, действие, сумма и отпечаток имени.

    Имени нет у строк, которые пишет сама игра, — «Началась новая игра» и
    «Игра закончена».
    """
    return LogEvent(action=action, amount=amount, at=at, name=name)


def name_mask(seed: int) -> np.ndarray:
    """Отпечаток имени той же формы, что снимает зрение со строки событий."""
    return np.random.default_rng(seed).random((NAME_HEIGHT, NAME_WIDTH)) < 0.4


class EventCollectionTests(unittest.TestCase):
    """Чужие ходы из строки событий: окно из десяти строк — не поток.

    Одни и те же строки приходят с каждым кадром, пока журнал не прокрутится,
    и записанное приходится помнить.
    """

    def setUp(self) -> None:
        self.journal = Journal()
        self.memory = HandMemory(on_hand=self.journal.add)

    def frame(self, *events, **changes) -> None:
        fields = {
            "players": 3, "hero_seat": 5, "hole": ("Ah", "Kd"), "stack": 950,
            "pot": 1200, "live_seats": (0, 4, 5), "events": events,
        }
        fields.update(changes)
        self.memory.update(table(**fields))

    def test_the_move_remembers_who_made_it(self) -> None:
        # Ходы журнал пишет с первого запуска, а кто их сделал — не знал:
        # отпечаток имени зрение снимало с каждой строки и тем же кадром
        # выбрасывало. Без номера нет ни персональной статистики, ни данных
        # под неё.
        first, second = name_mask(1), name_mask(2)
        self.frame(
            moved(100, "начало"), moved(101, "блайнд", name=first),
            moved(103, "рейз", 200, name=second), moved(104, "колл", name=first),
        )

        self.assertEqual(
            [who for _at, _action, _amount, who in self.memory.hand_events()],
            [None, 0, 1, 0],
        )

    def test_a_player_keeps_his_number_across_hands(self) -> None:
        # Узнавание тем и ценно, что помнит соседа из раздачи номер сорок семь.
        # Список отпечатков живёт дольше раздачи и потому приходит снаружи.
        roster = Roster()
        self.memory = HandMemory(on_hand=self.journal.add, roster=roster)
        first = name_mask(1)
        self.frame(moved(100, "начало"), moved(101, "колл", name=first))
        earlier = [who for _at, _a, _m, who in self.memory.hand_events()]
        for _ in range(3):
            self.memory.update(table(players=0, stack=1000))
        self.frame(moved(200, "начало"), moved(201, "рейз", 200, name=first))
        later = [who for _at, _a, _m, who in self.memory.hand_events()]

        self.assertEqual(earlier, [None, 0])
        self.assertEqual(later, [None, 0], "тот же номер, что раздачу назад")
        self.assertEqual(len(roster), 1, "тот же человек, а не новый")

    def test_the_same_window_is_written_once(self) -> None:
        window = (moved(100, "начало"), moved(101, "блайнд"), moved(103, "колл"))
        for _ in range(4):
            self.frame(*window)

        self.assertEqual([event.action for event in self.memory.events],
                         ["начало", "блайнд", "колл"])

    def test_two_identical_lines_in_one_window_are_two_moves(self) -> None:
        # Оба блайнда игра ставит одной секундой, и двое подряд могут
        # пропустить ход в ту же секунду. Отличаются такие строки только тем,
        # сколько раз они встретились в окне.
        self.frame(moved(100, "блайнд"), moved(100, "блайнд"))

        self.assertEqual(len(self.memory.events), 2)

    def test_a_line_without_time_is_not_written(self) -> None:
        # Без отметки времени строку не отличить от такой же на прошлом кадре,
        # и в журнал она пойдёт по второму разу.
        self.frame(LogEvent(action="чек"), moved(100, "чек"))

        self.assertEqual(len(self.memory.events), 1)

    def test_the_window_scrolls_and_only_the_tail_is_new(self) -> None:
        self.frame(moved(100, "начало"), moved(101, "блайнд"))
        self.frame(moved(101, "блайнд"), moved(104, "чек"))

        self.assertEqual([event.action for event in self.memory.events],
                         ["начало", "блайнд", "чек"])

    def test_the_hand_keeps_moves_from_its_own_start(self) -> None:
        # «Игра закончена» игра пишет секунд через восемь после того, как банк
        # уехал: в память она попадает уже вместе со следующей раздачей.
        self.frame(moved(90, "конец"), moved(95, "кнопка"),
                   moved(100, "начало"), moved(101, "блайнд"))
        self.frame(moved(103, "колл"))

        self.assertEqual([action for _at, action, _amount, _who in self.memory.hand_events()],
                         ["начало", "блайнд", "колл"])

    def test_moves_are_ordered_by_the_clock(self) -> None:
        # Гаснущую строку журнал иногда отдаёт с опозданием, и прошлый «конец»
        # оказывался посреди этой раздачи.
        self.frame(moved(100, "начало"), moved(103, "колл"))
        self.frame(moved(90, "конец"), moved(105, "чек"))

        self.assertEqual([action for _at, action, _amount, _who in self.memory.hand_events()],
                         ["начало", "колл", "чек"])

    def test_a_hand_without_its_start_line_keeps_its_own_moves(self) -> None:
        # Строку «Началась новая игра» игра пишет с опозданием: на кадре, где
        # карты уже розданы, её ещё нет. Пока границей была она, в журнал
        # уходили ходы прошлой раздачи — с чужими ставками и чужим банком.
        for _ in range(3):
            self.memory.update(table(players=0, stack=1000))
        self.frame(moved(100, "начало"), moved(101, "блайнд"),
                   moved(103, "ставка", 200))
        for _ in range(3):
            self.memory.update(table(players=0, stack=1000))
        self.frame(moved(110, "колл"), hole=("2c", "7d"))
        for _ in range(3):
            self.memory.update(table(players=0, stack=1000))

        self.assertEqual(len(self.journal), 2)
        self.assertEqual(
            [action for _at, action, _amount, _who in self.journal.hands[0].events],
            ["начало", "блайнд", "ставка"],
        )
        self.assertEqual(
            [action for _at, action, _amount, _who in self.journal.hands[1].events],
            ["колл"],
            "ходы прошлой раздачи достаются ей, а не этой",
        )

    def test_moves_already_read_stay_with_the_hand_that_made_them(self) -> None:
        # Ходы прошлой раздачи никуда из памяти не деваются — теперь их не
        # выбрасывают, потому что «выиграл N фишек» приходит уже над пустым
        # столом. К новой раздаче они всё равно не относятся.
        for _ in range(3):
            self.memory.update(table(players=0, stack=1000))
        self.frame(moved(100, "начало"), moved(103, "ставка", 200))
        for _ in range(3):
            self.memory.update(table(players=0, stack=1000))
        self.frame(moved(120, "чек"), hole=("2c", "7d"))

        self.assertEqual(
            [action for _at, action, _amount, _who in self.memory.hand_events()], ["чек"]
        )
        self.assertEqual(len(self.memory.events), 3, "память о ходах не стёрлась")

    def test_the_moves_land_in_the_hand(self) -> None:
        for _ in range(3):
            self.memory.update(table(players=0, stack=1000))
        self.frame(moved(100, "начало"), moved(101, "блайнд"), moved(103, "ставка", 200))
        for _ in range(3):
            self.memory.update(table(players=0, stack=1400))

        self.assertEqual(len(self.journal), 1)
        self.assertEqual(self.journal.moves(), 3)
        self.assertEqual(self.journal.actions()["ставка"], 1)

    def test_the_screen_notices_a_folded_log(self) -> None:
        # Свёрнутая строка событий гаснет через пару секунды, и чужие ходы по
        # ней не прочитать. Молчать об этом нельзя: игрок будет думать, что
        # статистика копится.
        self.frame(moved(100, "начало"))
        self.assertTrue(self.memory.log_open)

        self.frame()
        self.assertFalse(self.memory.log_open)

        line = PokerModule.log_line(self.journal, False)
        self.assertIn("свёрнута", line)
        # Развёрнутая строка даёт и ходы, и замер по столу — обе половины
        # стоят в одной строке панели: на вторую там нет места.
        open_line = PokerModule.log_line(self.journal, True)
        self.assertIn("ходов 0 от 0 игроков", open_line)
        self.assertIn("мало", open_line)
