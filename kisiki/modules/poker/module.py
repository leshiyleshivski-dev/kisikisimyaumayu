"""Экран POKER ADVISOR: читает покерный стол и подсказывает ход.

Модуль намеренно ничего не нажимает. Из ``core.py`` он берёт только поиск
окна и границы клиентской области — ни ``send_key_tap``, ни
``send_left_click``, ни ``activate_window`` сюда не импортируются, и это
проверяется тестом. Покер — игра с неполной информацией и живыми соперниками,
автопилот тут не к месту: решение принимает человек.

Разбор кадра живёт в ``vision.py``, математика — в ``hand_math.py``.
Здесь остаётся то, чего не видно на одном кадре: своё место, свои карты и
границы раздачи. Разбор задачи — в ``poker-plan/README.md``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace

import customtkinter as ctk
import mss
import numpy as np

from ...core import (
    APP_BG, MINT, MUTED, TEXT, VK_F9, VK_F11,
    client_bounds, data_path, find_game_window, user32, window_title,
)
from .journal import (
    RATE_FROM, RECENT_HANDS, Hand, Journal, hand_result, hands_word, new_hand,
)
from .hand_math import spaced
from .postflop import postflop_advice
from .players import Roster, players_word
from .preflop import preflop_advice
from .stats import answers_line
from .vision import SEAT_NAMES, seat_positions, seats_in_order, table_state
from ..ui import connection_panel, hotkey_bar, module_header, panel, step_list

ACCENT = "#75A7FF"

ACTION_COLORS = {
    "фолд": "#FF8F91",
    "чек": "#98A5B8",
    "колл": "#68D6B4",
    "бет": "#F2C66D",
    "рейз": "#F2C66D",
    "пуш": "#FFB35C",
}

# Что писать под советом. Игра ставит и повышает разными кнопками: пока
# доплаты нет, справа написано «BET», и рейза на экране не существует вовсе.
ACTION_SIZE_WORDS = {
    "бет": "поставить",
    "рейз": "поднять до",
    "колл": "доплатить",
    "пуш": "весь стек",
}

# Позиции игра никак не подписывает — их считает сам помощник по метке
# дилера. Пишем по-русски: «BTN» человеку, который в покере ноль, не говорит
# ничего, а «кнопка» хотя бы называет место за столом.
POSITION_LABELS = {
    "UTG": "ранняя",
    "MP": "средняя",
    "CO": "предпоследняя",
    "BTN": "кнопка",
    "SB": "малый блайнд",
    "BB": "большой блайнд",
}

SUIT_SIGNS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
# Десятку внутри модулей зовут «T», но на экране это читается как «туз»:
# букву T русский глаз узнаёт раньше, чем английское Ten. Игра на самой карте
# рисует «10» — показываем так же.
RANK_SIGNS = {"T": "10"}


def cards_text(cards) -> str:
    """«Kh Th» → «K♥ 10♥»: пишем ранг так же, как он написан на карте."""
    if not cards:
        return "—"
    return "  ".join(
        f"{RANK_SIGNS.get(card[0], card[0])}{SUIT_SIGNS[card[1]]}" for card in cards
    )


STEPS = (
    ("1", "Сядь за стол", "Помощник сам найдёт твоё место по открытым картам."),
    ("2", "Нажми F9", "Смотрит на экран и подсказывает на твоём ходу. Кнопки нажимаешь ты."),
)


class HandMemory:
    """Чего не видно на одном кадре: своё место, свои карты, борд и банк.

    Живёт отдельно от экрана, потому что это и есть вся хитрая часть: карты
    может на кадр перекрыть рука дилера, банк показывается не всегда, а место
    на вскрытии не определить. Отдельным классом это проверяется тестами без
    запуска окна.
    """

    # Между раздачами стол пустеет на несколько секунд, но и посреди раздачи
    # попадается кадр, где карт не видно: анимация, чужая рука над столом,
    # переход улицы. Одного пустого кадра для «раздача кончилась» мало.
    EMPTY_FRAMES_TO_FORGET = 3

    def __init__(
        self,
        on_hand: Callable[[Hand], None] | None = None,
        roster: Roster | None = None,
    ) -> None:
        # Куда отдавать сыгранную раздачу. Память знает её границы, а что с
        # ней делать дальше — уже не её дело.
        self.on_hand = on_hand
        self.seat: int | None = None
        self.seat_locked = False
        self.hole: tuple[str, ...] = ()
        self.board: tuple[str, ...] = ()
        self.pot: int | None = None
        # Горшки с полосы под бордом. Показываем их игроку: раз банк разложен,
        # значит кто-то в олл-ине на меньшую сумму — и весь банк достанется
        # не каждому, кто дойдёт до вскрытия.
        self.side_pots: tuple[int, ...] = ()
        self.to_call: int | None = None
        self.faced_bet: int | None = None
        self.to_call_pot: int | None = None
        self.my_bet = 0
        self.min_bet: int | None = None
        # Минимум, прочитанный на прошлом кадре: в дело он идёт только со
        # второго совпадения. Курсор поверх цифры даёт число на один кадр, а
        # размер ставки по такому числу уходит в олл-ин.
        self.seen_min_bet: int | None = None
        self.stack: int | None = None
        # Стек в паузе между раздачами: тот, что повторился два кадра подряд.
        # Свежий нужен совету, а журналу — устоявшийся: пока банк едет к
        # победителю, панель показывает промежуточные числа, и результат по
        # ним врёт на десятки тысяч.
        self.pause_stack: int | None = None
        self.seen_pause: int | None = None
        self.empty_frames = 0
        self.dealer: int | None = None
        self.position: str | None = None
        # Блайнды за столом не меняются, поэтому живут дольше раздачи: узнали
        # один раз в чистом начале любой раздачи — и знаем до конца сеанса.
        self.big_blind: int | None = None
        # Метка «WIN» и открытые на вскрытии руки: копятся за раздачу, потому
        # что показывают их считаные кадры.
        self.winners: tuple[int, ...] = ()
        self.shown: tuple[tuple[str, ...], ...] = ()
        self.showdown_seen = False
        # Стек в паузе перед раздачей — с ним сравнивается стек в паузе после
        # неё. Внутри раздачи мерить нельзя: блайнды уже поставлены.
        self.stack_before: int | None = None
        # Сыгранная раздача, которая ждёт своего результата. Банк едет к
        # победителю секунду-другую после того, как стол опустел, и на кадре,
        # где раздача закрывается, стек ещё не досчитал.
        self.finished: Hand | None = None
        # В журнал идут только раздачи, начало которых видели: севший смотреть
        # посреди раздачи не знает, сколько уже ушло из стека в банк.
        self.clean_start = False
        self.pending_clean = False
        # Чужие ходы из строки событий и то, что из них уже записано. На кадре
        # видно окно из десяти строк, и одни и те же события приходят снова и
        # снова, пока журнал не прокрутится.
        self.events: list = []
        self.seen_events: Counter = Counter()
        # Список отпечатков имён. Живёт он дольше сеанса и потому приходит
        # снаружи: узнавание тем и ценно, что помнит вчерашних соседей.
        self.roster = roster if roster is not None else Roster()
        # Что помощник советовал в этой раздаче. Копится здесь, а не на
        # экране: экран показывает один совет, а в журнал нужна вся раздача.
        self.advice: list[tuple[int, str]] = []
        self.log_open = False
        # С какой отметки времени идут ходы этой раздачи и открыта ли она
        # вообще. Строку «Началась новая игра» игра пишет с опозданием, и
        # брать границу по ней значило записывать чужие ходы: см. hand_events.
        self.events_from = 0
        self.hand_open = False

    def end_hand(self) -> None:
        """Закрыть раздачу: отдать её в журнал и забыть карты.

        Записываем только те раздачи, что видели с начала и в которых знаем
        свою руку. Севший смотреть посреди раздачи не знает, сколько уже ушло
        из стека в банк, и его «выигрыш» был бы чистой выдумкой.

        Звать этот метод можно сколько угодно раз подряд — так его и зовут,
        пока стол пустует: карты забыты первым вызовом, и записывать второму
        уже нечего.

        В журнал раздача уходит не отсюда: сперва ей нужен устоявшийся стек, а
        на этом кадре банк ещё едет к победителю. Ждёт она в ``finished``.
        """
        if self.on_hand is not None and self.clean_start and len(self.hole) == 2:
            self.finished = self.record()
            self.settle_hand()
        self.forget_hand()

    def settle_hand(self) -> None:
        """Отдать в журнал раздачу, дождавшуюся устоявшегося стека."""
        if self.finished is None or self.pause_stack is None:
            return
        self.write_hand()

    def write_hand(self) -> None:
        """Записать ожидающую раздачу с тем результатом, что удалось измерить.

        Не устоялся стек — результата нет, и раздача идёт в журнал без него:
        соврать про BB/100 хуже, чем посчитать его по меньшему числу раздач.
        """
        hand, self.finished = self.finished, None
        if hand is None or self.on_hand is None:
            return
        # Ходы и метку «WIN» пересобираем прямо сейчас: строку «выиграл N
        # фишек» игра пишет последней, уже над пустым столом, а метку зажигает
        # вместе с уезжающим банком. На кадре закрытия раздачи нет ни того, ни
        # другого — раздача закрывается через три пустых кадра, то есть за
        # три четверти секунды, а банк едет дольше. Раз уж раздача всё равно
        # ждёт устоявшегося стека, пусть дождётся и метки: вечером 26 августа
        # две выигранные раздачи из четырёх записаны проигранными.
        self.on_hand(replace(
            hand,
            won=hand.won or self.win_mark_seen(),
            result=hand_result(self.stack_before, self.pause_stack, hand.pot),
            events=self.hand_events(),
        ))

    def win_mark_seen(self) -> bool:
        """Горела ли метка «WIN» у нашего места — хоть на одном кадре."""
        return self.seat is not None and self.seat in self.winners

    def record(self) -> Hand:
        """Раздача так, как её видел экран; результат допишется в паузе."""
        return new_hand(
            hole=self.hole,
            board=self.board,
            position=self.position,
            big_blind=self.big_blind,
            pot=self.pot,
            won=self.seat is not None and self.seat in self.winners,
            showdown=self.showdown_seen,
            shown=tuple(
                (seat, cards) for seat, cards in enumerate(self.shown)
                if cards and seat != self.seat
            ),
            events=self.hand_events(),
            advice=tuple(self.advice),
        )

    def forget_hand(self) -> None:
        """Раздача кончилась: карты и банк больше не наши. Место помним."""
        self.hole = ()
        self.board = ()
        self.pot = None
        self.side_pots = ()
        self.to_call = None
        self.faced_bet = None
        self.to_call_pot = None
        self.my_bet = 0
        self.min_bet = None
        self.seen_min_bet = None
        self.seat_locked = False
        # Позиция считается по тому, кто ещё в раздаче, а люди из неё выбывают:
        # сбросил малый блайнд — и следующим после кнопки становится другой.
        # Поэтому позиция берётся один раз за раздачу и дальше не пересчитывается.
        self.dealer = None
        self.position = None
        self.shown = ()
        self.showdown_seen = False
        # Совет — свойство раздачи: с её концом кончается и он.
        self.advice = []
        # Метку «WIN» не выбрасываем по той же причине, по какой не выбрасываем
        # ходы: зажигается она последней, уже над пустым столом. Забытая здесь,
        # она пропадала бы ровно в тот кадр, когда её только зажгли. Гаснет она
        # в начале следующей раздачи — там же, где начинается отсчёт ходов.
        # Ходы не выбрасываем: строки этой раздачи приходят и после того, как
        # карты со стола убрали, — «выиграл N фишек» пишется последней.
        # Границей раздач служит отметка времени, а не пустой список.
        self.hand_open = False
        self.clean_start = False
        self.pending_clean = True

    def same_hand(self, state) -> bool:
        """Тот же расклад, что помним, или за столом уже новая раздача.

        Свои карты внутри раздачи не меняются, а борд только прирастает: если
        общие карты стали другими, значит колода уже перетасована, и держаться
        за старую память нельзя. Сверяем только целую руку: одна прочитанная
        карта из двух — это перекрытый кадр, а не новая раздача.
        """
        if len(state.hole) == 2 and self.hole and state.hole != self.hole:
            return False
        shared = min(len(state.board), len(self.board))
        return state.board[:shared] == self.board[:shared]

    def update(self, state) -> None:
        if state.players == 0:
            self.empty_frames += 1
            # Панель игрока стоит и между раздачами, и стек в ней к этому
            # времени уже с выигрышем — на нём и меряется результат.
            self.watch_pause(state)
            self.remember_showdown(state)
            if self.empty_frames >= self.EMPTY_FRAMES_TO_FORGET:
                self.end_hand()
            return
        self.empty_frames = 0
        # Карты на столе — пауза кончилась. Раздача, ждавшая устоявшегося
        # стека, дольше ждать не может: следующая уже началась.
        self.write_hand()
        if not self.same_hand(state):
            self.end_hand()
            self.write_hand()
        if not self.hand_open:
            # Раздача началась: её ходы — всё, что игра напишет с этой минуты.
            # Строки прошлой раздачи к этому времени уже прочитаны, и отметка
            # времени последней из них и есть граница.
            self.hand_open = True
            self.events_from = self.newest_event()
            # Раздача началась — метка прошлой больше не наша. Гаснет она
            # здесь, а не в конце раздачи: до этой строки она ещё могла
            # достаться раздаче, которая ждёт записи.
            self.winners = ()
        if self.pending_clean and not state.board:
            # Границу раздачи видели своими глазами — значит, эту раздачу
            # можно считать целиком. Пустой борд тут обязателен: раздача
            # начинается с чистого стола, а карты вскрытия лежат ещё секунду
            # после того, как банк уехал. На них начиналась «новая раздача»,
            # и в журнал уходил дубль с тем же бордом и чужой рукой.
            self.clean_start = True
            self.pending_clean = False
            # С чего началась эта раздача: стек в паузе перед ней. Тот же, с
            # которым только что сверилась предыдущая, — иначе между двумя
            # раздачами терялись бы фишки.
            self.stack_before = self.pause_stack
        self.pause_stack = None
        self.seen_pause = None
        # Место меняют между раздачами, а не посреди неё, поэтому внутри
        # раздачи оно заперто. Когда своя рука уже сброшена, а до вскрытия
        # дошёл один соперник, его открытые карты выглядят ровно как свои — и
        # помощник принимал чужую руку за свою.
        if state.hero_seat is not None and not self.seat_locked:
            self.seat = state.hero_seat
        self.seat_locked = self.seat is not None
        self.remember_seating(state)
        if len(state.hole) == 2:
            self.hole = state.hole
        if len(state.board) > len(self.board):
            self.board = state.board
        if state.money_in_play is not None:
            # Банк внутри раздачи не убывает: из него не вынимают. А вот
            # просесть на кадр он может — пока улица меняется, ставки с сукна
            # уже убраны, а плашка банка ещё не выросла.
            self.pot = max(self.pot or 0, state.money_in_play)
        if state.stack:
            self.stack = state.stack
        self.remember_pots(state)
        self.remember_call(state)
        self.remember_showdown(state)
        self.collect_events(state)

    # Дольше этого события из памяти выбрасываются: окно журнала — десять
    # строк, и вспомнить строку десятиминутной давности всё равно негде.
    EVENT_MEMORY_SECONDS = 600

    def collect_events(self, state) -> None:
        """Забрать из журнала строки, которых ещё не видели.

        Одни и те же события приходят с каждым кадром, пока журнал не
        прокрутится, поэтому записанное приходится помнить. Различаются строки
        отметкой времени — а одинаковые события одной секунды (двое подряд
        пропустили ход, оба блайнда поставлены разом) отличаются лишь тем,
        сколько раз строка встретилась в самом окне. Поэтому не сравниваем, а
        считаем: в окне их две, записана одна — значит, вторую пора записать.
        """
        self.log_open = bool(state.events)
        events = [event for event in state.events if event.at is not None]
        if not events:
            return
        counted: Counter = Counter()
        for event in events:
            key = (event.at, event.action, event.amount)
            counted[key] += 1
            if counted[key] > self.seen_events[key]:
                self.seen_events[key] = counted[key]
                self.events.append(event)
        if len(self.seen_events) > 500:
            newest = events[-1].at
            self.seen_events = Counter({
                key: count for key, count in self.seen_events.items()
                if newest - key[0] < self.EVENT_MEMORY_SECONDS
            })
            # Сами ходы копятся теперь между раздачами — граница у них по
            # времени, а не по опустевшему списку. Старые всё равно ни к
            # какой раздаче уже не приписать.
            self.events = [
                event for event in self.events
                if newest - event.at < self.EVENT_MEMORY_SECONDS
            ]

    def newest_event(self) -> int:
        """Отметка времени последней прочитанной строки журнала."""
        return max((event.at for event in self.events), default=0)

    def hand_events(self) -> tuple[tuple, ...]:
        """Ходы этой раздачи: всё, что игра написала, пока она шла.

        Граница берётся по времени, а не по строке «Началась новая игра», и
        это разбор живой ошибки. Строку начала игра пишет с опозданием, и на
        кадре, где раздача уже раздана, её ещё нет. Пока границей была она,
        в журнал уходили ходы **прошлой** раздачи: банк по ним сходился со
        строкой «выиграл N фишек», но с картами и банком своей записи не имел
        ничего общего. В журнале с записи `8.mp4` так сдвинута половина
        раздач, и ступени размеров по ним не посчитать.

        Отметка начала при этом всё равно пригождается: если она попала в
        окно, всё, что было до неё, — хвост прошлой раздачи, дописанный уже
        после того, как её карты убрали со стола. Без неё — например, когда
        журнал держали свёрнутым — такой хвост в раздаче и останется: строка
        «выиграл N фишек» приходит последней, и по одному времени её от начала
        новой раздачи не отличить.
        """
        # Порядок — по отметке времени, а не по тому, когда строку удалось
        # прочитать: гаснущую строку журнал иногда отдаёт с опозданием, и
        # «Игра закончена» прошлой раздачи оказывалась посреди этой.
        events = sorted(
            (event for event in self.events if event.at > self.events_from),
            key=lambda event: event.at,
        )
        starts = [index for index, event in enumerate(events)
                  if event.action == "начало"]
        if starts:
            events = events[starts[-1]:]
        return tuple(
            (event.at, event.action, event.amount, self.roster.number_of(event.name))
            for event in events
        )

    def remember_advice(self, action: str) -> None:
        """Запомнить совет, который сейчас на экране, — вместе с улицей.

        Улица тут важнее самого совета: до флопа решает таблица, и её «фолд»
        стоит раздачи целиком, а тот же «фолд» на ривере стоит одной ставки.
        Пишется только смена — улицы или совета: экран показывает одно и то же
        слово четыре раза в секунду, и без этого в журнал уходили бы сотни
        одинаковых строк на раздачу.
        """
        step = (len(self.board), action)
        if self.advice and self.advice[-1] == step:
            return
        self.advice.append(step)

    def remember_showdown(self, state) -> None:
        """Копить метку «WIN» и открытые на вскрытии руки.

        Показывают их считаные кадры: банк отдают в самом конце раздачи и
        почти сразу убирают карты со стола. Одного пропущенного кадра хватило
        бы, чтобы не узнать, чем раздача кончилась, — поэтому и метка, и чужие
        руки копятся за всю раздачу, а не читаются с последнего кадра.
        """
        if state.winners:
            self.winners = tuple(sorted(set(self.winners) | set(state.winners)))
        if state.showdown:
            self.showdown_seen = True
        if not state.shown:
            return
        seen = list(self.shown) or [()] * len(state.shown)
        for seat, cards in enumerate(state.shown):
            if len(cards) > len(seen[seat]):
                seen[seat] = cards
        self.shown = tuple(seen)

    def watch_pause(self, state) -> None:
        """Стек в паузе между раздачами — единственная честная мерка результата.

        Панель в паузе досчитывает анимацию: банк едет к победителю секунду с
        лишним, и каждый кадр показывает новое число. Берём то, что
        повторилось два кадра подряд: промежуточные значения не повторяются, а
        досчитанное стоит до самой раздачи. Пока стек не устоялся, сыгранная
        раздача ждёт в ``finished`` — результата у неё ещё нет.

        Ноль в паузе настоящий: у проигравшего весь стек фишек и правда не
        осталось, и ставить ему нечем. Совету при этом идёт последнее ненулевое
        число — иначе память о стеке пропадала бы вместе с рукой.
        """
        if state.stack == self.seen_pause:
            self.pause_stack = state.stack
        self.seen_pause = state.stack
        if state.stack:
            self.stack = state.stack
        self.settle_hand()

    def remember_pots(self, state) -> None:
        """Держать горшки, пока полоса под бордом на месте.

        Пустая полоса — честный ответ «горшков нет», и строку надо убрать. А
        вот полоса, которую не удалось прочесть, — это плохой кадр: сложенный
        горшок посреди раздачи никуда не денется, и последнее прочитанное
        честнее мигающей строки.
        """
        if not state.side_pots:
            self.side_pots = ()
        elif state.split_pot:
            self.side_pots = tuple(state.side_pots)

    def remember_seating(self, state) -> None:
        """Запомнить кнопку, свою позицию и размер блайндов.

        Всё трое берётся по первому кадру, где их видно, и внутри раздачи
        больше не трогается. Метку ``D`` на кадре может закрыть курсор, а
        позиция вдобавок зависит от того, кто ещё держит карты: пересчитанная
        после чужого фолда, она превратила бы большой блайнд в малый.
        """
        if state.dealer is not None and self.dealer is None:
            self.dealer = state.dealer
        if self.position is None and self.dealer is not None and self.seat is not None:
            self.position = seat_positions(state.live_seats, self.dealer).get(self.seat)
        if state.blinds is not None:
            self.big_blind = state.blinds[1]

    def remember_call(self, state) -> None:
        """Держать доплату, пока на столе ничего не менялось.

        Курсор, наведённый на кнопку, накрывает цифру, и сумма перестаёт
        читаться — а игрок как раз в этот момент и решает, что делать.
        Молчать полторы секунды, пока рука висит над кнопкой, глупо. Но
        держать сумму вечно нельзя: любая новая ставка её меняет, а любая
        ставка меняет и то, сколько денег на столе.
        """
        if state.my_bet is not None:
            self.my_bet = state.my_bet
        if state.min_bet is not None:
            # Минимум с кнопки берём со второго одинакового кадра. Он держится
            # всю улицу, пока никто не повысил, а вот курсор поверх цифры даёт
            # число на один кадр — и по нему все четыре быстрые кнопки
            # схлопывались в «ALL IN». Одного кадра для хода за весь стек мало.
            if state.min_bet == self.seen_min_bet:
                self.min_bet = state.min_bet
            self.seen_min_bet = state.min_bet
        if state.to_call is not None:
            self.to_call = state.to_call
            self.faced_bet = state.faced_bet
            self.to_call_pot = state.money_in_play
        elif state.money_in_play != self.to_call_pot:
            self.to_call = None
            self.faced_bet = None
            self.to_call_pot = None
            self.min_bet = None
            self.seen_min_bet = None

    def blocker(self, state) -> str | None:
        """Почему сейчас нельзя советовать; ``None`` — можно.

        Молчание честнее уверенной чуши: без банка шансы банка выходят
        стопроцентными, и совет всегда получался бы «фолд».
        """
        if state.showdown:
            return "Вскрытие: раздача уже сыграна."
        if not state.my_turn:
            return "Жду своего хода — подсказка появится, когда игра его отдаст."
        if len(self.hole) != 2:
            return "Не вижу своих карт. Подсказывать вслепую не буду."
        if state.players < 2:
            return "Не вижу соперников с картами."
        if self.to_call is None:
            return "Не разобрал сумму на кнопке хода."
        if self.to_call > 0 and not self.pot:
            return "Не вижу, сколько уже в банке, — считать не по чему."
        if 0 < len(self.board) < 3:
            # Флоп выкладывается анимацией, и на кадре посреди неё видно одну
            # карту из трёх. Досчитывать по ней нельзя, а отвечать как до
            # флопа — тем более: общие карты уже на столе.
            return "Общие карты ещё раздаются — жду весь флоп."
        if not self.board:
            # До флопа решает таблица, а ей нужны позиция и глубина стека в
            # блайндах. Без них советовать нечем: одна и та же рука с кнопки
            # открывается, а с ранней позиции сбрасывается.
            if self.position is None:
                return "Не вижу метку дилера — без неё не назвать позицию."
            if not self.big_blind:
                return "Не знаю блайнды — покажет начало следующей раздачи."
            if not self.stack:
                return "Не вижу свой стек — не понять, насколько глубоко играем."
        return None


class PokerModule(ctk.CTkFrame):
    """Смотрит на стол и советует ход. Ввод в игру не отправляет никогда."""

    SCAN_INTERVAL_MS = 250
    # Перебор на 20 000 раздач занимает до 150 мс, а весь совет вместе с
    # планом на ответ соперника — до 300 мс; игра даёт на ход около пятнадцати
    # секунд. Считаем только когда стол изменился, иначе один и тот же расклад
    # пересчитывался бы четыре раза в секунду впустую.
    ADVICE_TRIALS = 12_000
    # До флопа перебор нужен только на коротком стеке, где считается
    # выгода олл-ина; таблице он не нужен вовсе, поэтому раздач меньше.
    PREFLOP_TRIALS = 8_000
    # Пока анимация сгребает фишки, банк растёт числами 150, 168, 190, 199 —
    # совет от этого не меняется, а подсказка на экране дёргается. Мелкий
    # прирост банка пересчёта не стоит.
    POT_STEP_SHARE = 0.1

    def __init__(self, parent: ctk.CTkFrame, on_back) -> None:
        super().__init__(parent, fg_color=APP_BG, corner_radius=0)
        self.on_back = on_back
        self.active = False
        self.watching = False
        self.game_window: int | None = None
        self.keys = {key: False for key in (VK_F9, VK_F11)}

        # Журнал переживает выход из приложения: сравнивать периоды по
        # раздачам одного вечера бессмысленно.
        self.journal = Journal.load(data_path("poker_journal.json"))
        # Отпечатки имён живут своим файлом: журнал за вечер набирает сотни
        # раздач, а имён за столом шесть, и переписывать их вместе незачем.
        self.roster = Roster.load(data_path("poker_players.json"))
        self.memory = HandMemory(on_hand=self.remember_hand, roster=self.roster)
        self.advice_key: tuple | None = None
        self.advice_pot: int | None = None

        self.process = ctk.StringVar(value="GTA5.exe")
        self.connection = ctk.StringVar(value="Ищу GTA5.exe…")
        self.status = ctk.StringVar(value="F9 — начать смотреть на стол.")
        self.seat_text = ctk.StringVar(value="место не найдено")
        self.hole_text = ctk.StringVar(value="—")
        self.board_text = ctk.StringVar(value="—")
        self.pot_text = ctk.StringVar(value="—")
        self.pots_text = ctk.StringVar(value="")
        self.call_text = ctk.StringVar(value="—")
        self.players_text = ctk.StringVar(value="—")
        self.position_text = ctk.StringVar(value="—")
        self.depth_text = ctk.StringVar(value="—")
        self.stack_text = ctk.StringVar(value="—")
        self.action_text = ctk.StringVar(value="ЖДУ")
        self.size_text = ctk.StringVar(value="")
        self.plan_text = ctk.StringVar(value="")
        self.numbers_text = ctk.StringVar(value="эквити —   ·   шансы банка —")
        self.made_text = ctk.StringVar(value="")
        self.reason_text = ctk.StringVar(value="Помощник не запущен.")
        self.journal_text = ctk.StringVar(value="")
        self.money_text = ctk.StringVar(value="")
        self.log_text = ctk.StringVar(value="")

        self.build_ui()
        self.refresh_journal()
        self.after(40, self.poll_hotkeys)
        self.after(self.SCAN_INTERVAL_MS, self.scan_tick)
        self.after(0, self.refresh_connection)

    # --- интерфейс ---

    def build_ui(self) -> None:
        module_header(
            self, code="PKR", title="Poker Advisor",
            subtitle="читает стол и подсказывает ход, кнопки нажимаешь сам",
            accent=ACCENT, on_back=self.on_back,
        )
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=42, pady=(0, 24))
        self.indicator = connection_panel(
            body, process=self.process, connection=self.connection,
            accent=ACCENT, on_check=self.refresh_connection,
        )

        hotkey_holder = ctk.CTkFrame(body, fg_color="transparent")
        hotkey_holder.pack(side="bottom", fill="x")
        hotkey_bar(hotkey_holder, (("F9", "смотреть / пауза"), ("F11", "остановить")), ACCENT)
        ctk.CTkLabel(
            body, textvariable=self.status, font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color=ACCENT,
        ).pack(side="bottom", anchor="w", pady=(10, 6))

        workspace = ctk.CTkFrame(body, fg_color="transparent")
        workspace.pack(fill="both", expand=True)
        workspace.grid_columnconfigure(0, weight=6, uniform="poker")
        workspace.grid_columnconfigure(1, weight=5, uniform="poker")
        workspace.grid_rowconfigure(0, weight=1)

        table = panel(workspace, "СТОЛ  /  ЧТО ВИДНО", ACCENT)
        table.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        for title, variable, size in (
            ("МОЯ РУКА", self.hole_text, 26),
            ("ОБЩИЕ КАРТЫ", self.board_text, 22),
        ):
            box = ctk.CTkFrame(table, fg_color="#1D2740", corner_radius=15)
            box.pack(fill="x", padx=16, pady=(0, 9))
            ctk.CTkLabel(
                box, text=title, font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=ACCENT,
            ).pack(anchor="w", padx=14, pady=(10, 0))
            ctk.CTkLabel(
                box, textvariable=variable, font=ctk.CTkFont("Consolas", size, "bold"),
                text_color=TEXT,
            ).pack(anchor="w", padx=14, pady=(0, 10))

        facts = ctk.CTkFrame(table, fg_color="transparent")
        facts.pack(fill="x", padx=16, pady=(0, 10))
        # Позиция и глубина стоят в одном ряду со ставками не для красоты: до
        # флопа именно они решают, что советовать, а стек в фишках без блайнда
        # ни о чём не говорит — 4 500 это и много, и девять ставок.
        cells = (
            ("БАНК", self.pot_text),
            ("ДОПЛАТА", self.call_text),
            ("МОЙ СТЕК", self.stack_text),
            ("ПОЗИЦИЯ", self.position_text),
            ("ГЛУБИНА", self.depth_text),
            ("СОПЕРНИКОВ", self.players_text),
        )
        for column in range(3):
            facts.grid_columnconfigure(column, weight=1, uniform="facts")
        for index, (title, variable) in enumerate(cells):
            row, column = divmod(index, 3)
            cell = ctk.CTkFrame(facts, fg_color="#1D2740", corner_radius=13)
            cell.grid(
                row=row, column=column, sticky="ew",
                padx=(0 if column == 0 else 6, 0), pady=(0 if row == 0 else 6, 0),
            )
            ctk.CTkLabel(
                cell, text=title, font=ctk.CTkFont("Segoe UI", 8, "bold"), text_color=MUTED,
            ).pack(padx=10, pady=(9, 0))
            ctk.CTkLabel(
                cell, textvariable=variable, font=ctk.CTkFont("Segoe UI", 15, "bold"),
                text_color=TEXT,
            ).pack(padx=10, pady=(0, 9))
        ctk.CTkLabel(
            table, textvariable=self.seat_text, font=ctk.CTkFont("Segoe UI", 9, "bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=18, pady=(0, 2))
        # Строка появляется только в раздачах с олл-ином: считать, какой горшок
        # наш, помощник не умеет, но смолчать об этом было бы нечестно.
        ctk.CTkLabel(
            table, textvariable=self.pots_text, font=ctk.CTkFont("Segoe UI", 9, "bold"),
            text_color=MUTED, wraplength=430, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))


        side = ctk.CTkFrame(workspace, fg_color="transparent")
        side.grid(row=0, column=1, padx=(6, 0), sticky="nsew")

        tip = panel(side, "СОВЕТ", ACCENT)
        tip.pack(fill="x")
        self.action_label = ctk.CTkLabel(
            tip, textvariable=self.action_text, font=ctk.CTkFont("Segoe UI", 34, "bold"),
            text_color=MUTED,
        )
        self.action_label.pack(padx=18, pady=(0, 0))
        # Размер стоит отдельной строкой и крупно: «рейз» без числа — это
        # ровно тот вопрос, на который помощник и должен отвечать.
        self.size_label = ctk.CTkLabel(
            tip, textvariable=self.size_text, font=ctk.CTkFont("Segoe UI", 17, "bold"),
            text_color=TEXT, wraplength=330,
        )
        self.size_label.pack(padx=18, pady=(0, 4))
        # План на ответ соперника стоит сразу под размером, а не в причине:
        # «чек» сам по себе не говорит, что делать, когда соперник поставит, —
        # а деньги уходят именно там.
        ctk.CTkLabel(
            tip, textvariable=self.plan_text, font=ctk.CTkFont("Segoe UI", 12, "bold"),
            text_color=MINT, wraplength=330, justify="left",
        ).pack(padx=18, pady=(0, 4))
        ctk.CTkLabel(
            tip, textvariable=self.numbers_text, font=ctk.CTkFont("Segoe UI", 12, "bold"),
            text_color=ACCENT,
        ).pack(padx=18, pady=(0, 2))
        ctk.CTkLabel(
            tip, textvariable=self.made_text, font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color=MUTED,
        ).pack(padx=18, pady=(0, 6))
        ctk.CTkLabel(
            tip, textvariable=self.reason_text, font=ctk.CTkFont("Segoe UI", 11),
            text_color=MUTED, wraplength=330, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 16))

        self.watch_button = ctk.CTkButton(
            side, text="Смотреть на стол  ·  F9", command=self.toggle, height=42,
            corner_radius=13, fg_color=ACCENT, hover_color="#8FB8FF",
            text_color="#111722", font=ctk.CTkFont("Segoe UI", 11, "bold"),
        )
        self.watch_button.pack(fill="x", pady=(10, 0))

        # Журнал стоит на экране, а не в файле, потому что смотреть в него
        # будут между раздачами: одно число само по себе не говорит ничего, а
        # пара чисел показывает, стало лучше или просто повезло.
        log = panel(side, "ЖУРНАЛ", ACCENT)
        log.pack(fill="x", pady=(10, 0))
        ctk.CTkLabel(
            log, textvariable=self.journal_text, font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color=TEXT, wraplength=330, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 2))
        ctk.CTkLabel(
            log, textvariable=self.money_text, font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color=MINT, wraplength=330, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 2))
        # Строку событий игра по умолчанию держит свёрнутой, а свёрнутая она
        # гаснет через пару секунд — чужие ходы по ней не прочитать. Сказать
        # об этом обязаны: иначе игрок будет думать, что статистика копится.
        # Развёрнутая же даёт замер по столу — ради него журнал ходы и пишет.
        ctk.CTkLabel(
            log, textvariable=self.log_text, font=ctk.CTkFont("Segoe UI", 10, "bold"),
            text_color=MUTED, wraplength=330, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        guide = panel(side, "КАК ПОЛЬЗОВАТЬСЯ", ACCENT)
        guide.pack(fill="both", expand=True, pady=(10, 0))
        step_list(guide, STEPS, ACCENT)

    # --- подключение и кадр ---

    def refresh_connection(self) -> None:
        if not self.winfo_exists():
            return
        self.game_window = find_game_window(self.process.get())
        if self.game_window:
            title = window_title(self.game_window)
            self.connection.set(f"Подключено · {title[:28]}" if title else "Подключено")
            self.indicator.configure(text_color=MINT)
        else:
            self.connection.set("Не подключено")
            self.indicator.configure(text_color="#F05A67")
        self.after(2000, self.refresh_connection)

    def capture_client(self) -> np.ndarray | None:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            return None
        bounds = client_bounds(self.game_window)
        if not bounds:
            return None
        left, top, width, height = bounds
        try:
            with mss.mss() as screen:
                return np.asarray(screen.grab({
                    "left": left, "top": top, "width": width, "height": height,
                }))[:, :, :3]
        except Exception:
            return None

    # --- жизненный цикл ---

    def toggle(self) -> None:
        if self.watching:
            self.stop("Пауза. F9 — продолжить.")
            return
        if not find_game_window(self.process.get()):
            self.status.set("GTA не найдена. Запусти игру и нажми «Проверить».")
            return
        self.watching = True
        self.watch_button.configure(text="Пауза  ·  F9")
        self.status.set("Смотрю на стол.")

    def stop(self, message: str) -> None:
        self.watching = False
        if self.winfo_exists():
            self.watch_button.configure(text="Смотреть на стол  ·  F9")
            self.status.set(message)
            self.show_silence("Помощник на паузе.")

    def activate(self) -> None:
        self.active = True
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def deactivate(self, message: str) -> None:
        self.stop(message)
        self.active = False
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def poll_hotkeys(self) -> None:
        actions = {VK_F9: self.toggle, VK_F11: lambda: self.stop("Остановлено.")}
        for key, action in actions.items():
            down = bool(user32.GetAsyncKeyState(key) & 0x8000)
            if self.active and down and not self.keys[key]:
                action()
            self.keys[key] = down
        if self.winfo_exists():
            self.after(40, self.poll_hotkeys)

    # --- разбор стола ---

    def scan_tick(self) -> None:
        if not self.winfo_exists():
            return
        if self.active and self.watching:
            frame = self.capture_client()
            if frame is None:
                self.status.set("Кадр не получен: окно GTA пропало.")
            else:
                self.read_table(frame)
        self.after(self.SCAN_INTERVAL_MS, self.scan_tick)

    def read_table(self, frame: np.ndarray) -> None:
        state = table_state(frame, known_seat=self.memory.seat)
        self.memory.update(state)

        self.seat_text.set(
            f"моё место: {SEAT_NAMES[self.memory.seat]}" if self.memory.seat is not None
            else "место не найдено — карты ещё не показывали"
        )
        self.hole_text.set(cards_text(self.memory.hole))
        self.board_text.set(cards_text(self.memory.board))
        self.pot_text.set(self.spaced(self.memory.pot))
        self.pots_text.set(self.pots_line(self.memory.side_pots))
        self.call_text.set(self.spaced(self.payable()) if self.payable() else "нет")
        self.stack_text.set(self.spaced(self.memory.stack))
        self.players_text.set(str(state.players - 1) if state.players else "—")
        self.position_text.set(POSITION_LABELS.get(self.memory.position, "—"))
        self.depth_text.set(self.depth_line())
        self.log_text.set(self.log_line(self.journal, self.memory.log_open))
        self.update_advice(state)

    @classmethod
    def pots_line(cls, pots: tuple[int, ...]) -> str:
        """Строка про горшки; пусто — банк цел, и говорить не о чем.

        Какой из горшков наш, помощник не считает: для этого нужно знать,
        сколько денег внёс каждый, а на столе видны только итоги. Поэтому
        просто называем горшки — дальше игрок смотрит сам.
        """
        if not pots:
            return ""
        split = "  +  ".join(cls.spaced(pot) for pot in pots)
        return f"банк разложен по горшкам: {split} — за столом олл-ин на меньшую сумму"

    def remember_hand(self, hand: Hand) -> None:
        """Сыгранная раздача: в журнал и на экран."""
        self.journal.add(hand)
        self.refresh_journal()

    def refresh_journal(self) -> None:
        self.journal_text.set(self.journal_line(self.journal))
        self.money_text.set(self.money_line(self.journal))
        self.log_text.set(self.log_line(self.journal, self.memory.log_open))

    @classmethod
    def log_line(cls, journal: Journal, open_log: bool) -> str:
        """Что дала строка событий — или почему не дала ничего.

        Обе половины про одно и то же, поэтому и строка одна: свёрнутый журнал
        не даёт ни ходов, ни замера, а развёрнутый — и то и другое. Панели
        «ЖУРНАЛ» при окне 1040 × 860 достаётся 167 пикселей, и лишняя строка в
        ней не помещается: нижнюю обрезало ровно на ту, что просит развернуть
        журнал, — то есть на самую нужную.
        """
        if open_log:
            players = journal.players()
            return (
                f"ходов {journal.moves()} от {players} "
                f"{players_word(players)} · {answers_line(journal)}"
            )
        return "строка событий свёрнута — разверни её стрелкой"

    @staticmethod
    def signed(value: int) -> str:
        """«+2 450» и «−1 200»: без знака выигрыш от проигрыша не отличить."""
        return f"{'+' if value >= 0 else '−'}{spaced(abs(value))}"

    @staticmethod
    def signed_rate(value: float) -> str:
        """«+21» и «−7» — тем же минусом, каким рядом написаны фишки."""
        return f"{'+' if value >= 0 else '−'}{abs(value):.0f}"

    @classmethod
    def journal_line(cls, journal: Journal) -> str:
        """Сколько раздач сыграно и чем они кончались.

        Панели «ЖУРНАЛ» хватает ровно на три строки, поэтому место под
        последнее число одно — и занимает его самое полезное из двух. Пока
        советов в журнале нет, это вскрытия; как только появились — счёт
        сброшенным рукам, которые всё равно доигрывались. Им и место: разница
        между «совет плохой» и «совет не послушали» видна только по ним.
        """
        played = len(journal)
        if not played:
            return "раздач пока нет — журнал наберётся сам, пока помощник смотрит"
        line = f"{played} {hands_word(played)} · выиграно {journal.wins()}"
        ignored, advised = journal.ignored_folds()
        if advised:
            return f"{line} · пас не послушан {ignored} из {advised}"
        return f"{line} · до вскрытия дошло {journal.showdowns()}"

    @classmethod
    def money_line(cls, journal: Journal) -> str:
        """Деньги журнала: фишки, BB/100 и сравнение периодов.

        Считаются только те раздачи, где результат сошёлся с банком: между
        раздачами покупают фишки, а покупку от выигрыша по стеку не отличить.
        Поэтому раздач тут бывает меньше, чем в строке выше, — и об этом лучше
        сказать прямо, чем тихо посчитать по другому набору.
        """
        hands = journal.counted()
        if not hands:
            return "результат считать пока не по чему"
        recent, earlier = journal.trend()
        line = f"{cls.signed(journal.chips())} фишек"
        if recent is None or len(hands) < RATE_FROM:
            line += f" · BB/100 — с {RATE_FROM}-й раздачи"
        else:
            line += f" · {cls.signed_rate(recent)} BB/100"
            if earlier is not None:
                line += (
                    f" за последние {RECENT_HANDS}, "
                    f"{cls.signed_rate(earlier)} до них"
                )
        if len(hands) < len(journal):
            line += f" · в счёте {len(hands)} из {len(journal)}"
        return line

    def depth_line(self) -> str:
        """Стек в больших блайндах — та самая мера, по которой выбирается режим."""
        memory = self.memory
        if not memory.stack or not memory.big_blind:
            return "—"
        return f"{memory.stack / memory.big_blind:.0f} BB"

    def payable(self) -> int | None:
        """Сколько реально уйдёт из стека: больше своих фишек не поставить.

        Игра пишет на кнопке ровно это число — «ALL IN 500», а не всю чужую
        ставку. Показываем так же: платит игрок столько.
        """
        memory = self.memory
        if memory.to_call is None:
            return None
        if memory.stack is None:
            return memory.to_call
        return min(memory.to_call, memory.stack)

    @staticmethod
    def spaced(value: int | None) -> str:
        """Игра пишет банк с пробелом внутри — показываем так же.

        Одно и то же написание берут и советы: план на ответ соперника
        называет доплату теми же цифрами, что и клетка «ДОПЛАТА» рядом.
        """
        return spaced(value)

    def update_advice(self, state) -> None:
        blocker = self.memory.blocker(state)
        if blocker is not None:
            self.show_silence(blocker)
            return

        memory = self.memory
        opponents = state.players - 1
        # Считалке идёт чужая ставка целиком: по её размеру читается диапазон,
        # и обрезанная нашим стеком она превратила бы ставку в банк в ставку
        # в четверть банка. Сколько из неё реально уйдёт из стека, считалка
        # разберётся сама — стек она знает.
        to_call = memory.to_call
        pot = memory.pot or 0
        key = (
            memory.hole, memory.board, opponents, to_call,
            memory.my_bet, memory.min_bet, memory.stack,
            memory.position, memory.big_blind, state.all_in_only,
            memory.faced_bet,
        )
        if key == self.advice_key and not self.pot_moved(pot, self.advice_pot):
            return
        self.advice_key, self.advice_pot = key, pot
        if len(memory.board) >= 3:
            tip = postflop_advice(
                memory.hole, memory.board, opponents=opponents,
                pot=pot, to_call=to_call, my_bet=memory.my_bet,
                stack=memory.stack, min_bet=memory.min_bet,
                big_blind=memory.big_blind, faced_bet=memory.faced_bet,
                can_raise=not state.all_in_only, trials=self.ADVICE_TRIALS,
            )
            self.memory.remember_advice(tip.action)
            self.show_advice(tip, self.payable() or 0, tip.made, "по диапазону")
            return
        # До флопа перебор против случайных карт советует играть всё подряд:
        # 3♠9♦ он оценивает в сорок процентов, и это выше шансов банка. Решает
        # таблица, а на коротком стеке — олл-ин или пас.
        tip = preflop_advice(
            memory.hole, position=memory.position, opponents=opponents,
            big_blind=memory.big_blind, stack=memory.stack, pot=pot,
            to_call=to_call, faced_bet=memory.faced_bet, my_bet=memory.my_bet,
            min_bet=memory.min_bet, limpers=self.limpers(state, memory),
            can_raise=not state.all_in_only, trials=self.PREFLOP_TRIALS,
        )
        self.memory.remember_advice(tip.action)
        self.show_advice(tip, self.payable() or 0, tip.hand, tip.mode)

    def show_advice(self, tip, to_call: int, made: str, mode: str) -> None:
        """Выложить готовый совет на экран — он одинаков для обеих считалок."""
        self.action_text.set(tip.action.upper())
        self.action_label.configure(text_color=ACTION_COLORS.get(tip.action, TEXT))
        self.size_text.set(self.size_line(tip, to_call))
        self.plan_text.set(tip.plan)
        if tip.equity is None:
            self.numbers_text.set(f"совет {mode}")
        else:
            odds = f"{tip.odds:.0%}" if to_call > 0 else "доплаты нет"
            self.numbers_text.set(
                f"эквити {tip.equity:.0%}   ·   шансы банка {odds}   ·   {mode}"
            )
        self.made_text.set(f"на руках: {made}" if made else "")
        self.reason_text.set(tip.reason)

    @staticmethod
    def limpers(state, memory) -> int:
        """Сколько соперников зашли в раздачу, просто уравняв блайнд.

        Каждый такой добавляет блайнд к размеру открытия: банк, за который
        идёт борьба, уже больше, и открывать в те же две с половиной ставки
        значит звать их всех задёшево.
        """
        order = seats_in_order(state.live_seats, memory.dealer)
        big_blind_seat = order[1] if len(order) > 1 else None
        return sum(
            1 for seat in state.live_seats
            if seat not in (memory.seat, big_blind_seat)
            and state.bets[seat] == memory.big_blind
        )

    @classmethod
    def size_line(cls, tip, to_call: int) -> str:
        """Строка размера под советом: сколько именно ставить или доплачивать.

        Игра просит итоговую ставку — ту, что стоит на ползунке и написана на
        кнопке, — а не доплату сверх чужой. Так и пишем: «поднять до 1 200».
        """
        word = ACTION_SIZE_WORDS.get(tip.action)
        if word is None:
            return ""
        if tip.action == "колл":
            return f"{word} {cls.spaced(to_call)}"
        shove = tip.action == "пуш"
        if shove:
            line = f"{word}  ·  {cls.spaced(tip.raise_to)}"
        elif not tip.raise_to:
            return "размер не подскажу — не вижу банка"
        else:
            line = f"{word} {cls.spaced(tip.raise_to)}"
            if tip.all_in:
                line += "  ·  весь стек"
        # Размер ползунком выставляют мышкой на пятнадцатисекундном таймере и
        # промахиваются. Ряд «MIN / 3 BB / BANK / ALL IN» ставит те же деньги
        # одним нажатием — поэтому кнопка идёт первой, а число остаётся,
        # чтобы было с чем сверить. Кнопки не нашлось — хотя бы подсказываем,
        # что ход за весь стек.
        if tip.button:
            return f"жми {tip.button}  ·  {line}"
        return f"{line}, ALL IN" if shove or tip.all_in else line

    @classmethod
    def pot_moved(cls, pot: int, previous: int | None) -> bool:
        """Изменился ли банк настолько, чтобы пересчитывать совет."""
        if previous is None:
            return True
        return abs(pot - previous) > max(1, previous * cls.POT_STEP_SHARE)

    def show_silence(self, reason: str) -> None:
        self.advice_key = None
        self.advice_pot = None
        self.action_text.set("ЖДУ")
        self.action_label.configure(text_color=MUTED)
        self.size_text.set("")
        self.plan_text.set("")
        self.numbers_text.set("эквити —   ·   шансы банка —")
        self.made_text.set("")
        self.reason_text.set(reason)
