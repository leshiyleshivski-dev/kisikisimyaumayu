"""Журнал раздач: что сыграно, чем кончилось и сколько это стоило.

Считать помощник умеет, а отличить «стало лучше» от «повезло» — нет. Спор об
этом повторялся после каждой записи: советы поменялись, стек к концу вечера
вырос — и непонятно, из-за советов или из-за одной удачной раздачи. Журнал и
есть та мера, по которой сравниваются периоды.

Пишется он в свой файл рядом с прогрессом кликера: раздач за вечер набегают
сотни, а прогресс читается на каждом запуске игры, и мешать их незачем.

Модуль ничего не рисует и никуда не нажимает — как и остальная покерная
математика. Границы раздачи ему приносит ``poker.py``, где живёт память между
кадрами.

Результат раздачи считается по стеку **между** раздачами, а не внутри неё. В
паузе блайнды ещё не поставлены, а банк уже сгребли — обе мерки честные, и
разница между ними и есть выигрыш. Если мерить от первого кадра раздачи,
блайнд окажется потерянным до начала счёта: на шести местах это четверть
ставки на раздачу, то есть 25 BB/100 — больше, чем весь выигрыш, который тут
вообще меряют.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# Сколько раздач держим. Тысячи хватает на много вечеров, а файл при этом
# остаётся мелким: одна запись — около двухсот байт.
JOURNAL_LIMIT = 2000

# По скольку последних раздач считается «сейчас» при сравнении периодов.
# Полсотни — тот объём, на котором BB/100 хоть что-то значит, и он же
# набирается за один вечер.
RECENT_HANDS = 50

# С какой раздачи вообще показывать BB/100. Считается оно и с первой, но одна
# выигранная раздача даёт «+800 BB/100» — число верное и бессмысленное сразу.
# Десяток тоже мало, зато глаз уже не спотыкается.
RATE_FROM = 10


def hands_word(count: int) -> str:
    """«37 раздач», но «2 раздачи» и «21 раздача».

    Строку эту читают мельком, рядом с цифрами, и «37 раздача» цепляет глаз
    ровно там, где он нужен числам.
    """
    if 11 <= count % 100 <= 14:
        return "раздач"
    last = count % 10
    if last == 1:
        return "раздача"
    if last in (2, 3, 4):
        return "раздачи"
    return "раздач"


@dataclass(frozen=True)
class Hand:
    """Одна сыгранная раздача — только то, что видно с экрана.

    ``result`` — фишки: плюс, если раздача принесла. ``None`` значит, что счёт
    не сошёлся и в статистику раздача не идёт: между раздачами покупали фишки
    или стек не прочитался.
    """

    hole: tuple[str, ...] = ()
    board: tuple[str, ...] = ()
    position: str | None = None
    big_blind: int | None = None
    pot: int | None = None
    result: int | None = None
    won: bool = False
    showdown: bool = False
    # Чужие руки, открытые на вскрытии: место и карты. Единственный случай,
    # когда чужие карты вообще видны, — и первое, что понадобится, когда
    # дойдёт очередь до статистики по соперникам.
    shown: tuple[tuple[int, tuple[str, ...]], ...] = ()
    # Чужие ходы из строки событий: время, что сделали и на сколько. Своих
    # ходов помощник не отличает от чужих — имена он не читает, — но для
    # общей статистики по столу этого и не нужно.
    events: tuple[tuple, ...] = ()
    played_at: float = 0.0

    @property
    def counted(self) -> bool:
        """Идёт ли раздача в счёт: результат известен и есть чем его мерить."""
        return self.result is not None and bool(self.big_blind)

    @property
    def blinds_won(self) -> float:
        """Результат в больших блайндах — единственная мера, общая для столов.

        Блайнды за разными столами отличаются на порядки: 500 фишек — это и
        десять ставок, и одна пятидесятая. Складывать такие раздачи в фишках
        нельзя, а в блайндах можно.
        """
        if not self.counted:
            return 0.0
        return self.result / self.big_blind


def hand_result(before: int | None, after: int | None, pot: int | None) -> int | None:
    """Сколько фишек принесла раздача; ``None`` — счёт не сошёлся.

    ``before`` и ``after`` — стек в паузе до раздачи и в паузе после неё.

    Проверка одна, зато честная: выиграть больше, чем лежало в банке, нельзя,
    и проиграть больше — тоже, ведь в банк сложено и наше. Не сошлось —
    значит, между раздачами покупали фишки, а покупку от выигрыша по стеку не
    отличить. Такую раздачу в статистику не берём: соврать про BB/100 хуже,
    чем посчитать его по меньшему числу раздач.
    """
    if before is None or after is None or not pot:
        return None
    result = after - before
    return result if abs(result) <= pot else None


@dataclass
class Journal:
    """Сыгранные раздачи и числа по ним."""

    hands: list[Hand] = field(default_factory=list)
    path: Path | None = None

    def add(self, hand: Hand) -> None:
        """Записать раздачу и сохранить журнал на диск."""
        self.hands.append(hand)
        if len(self.hands) > JOURNAL_LIMIT:
            del self.hands[:-JOURNAL_LIMIT]
        self.save()

    def __len__(self) -> int:
        return len(self.hands)

    def counted(self, last: int | None = None) -> list[Hand]:
        """Раздачи с известным результатом, при желании — только последние."""
        hands = [hand for hand in self.hands if hand.counted]
        return hands[-last:] if last else hands

    def chips(self, last: int | None = None) -> int:
        return sum(hand.result for hand in self.counted(last))

    def bb_per_100(self, last: int | None = None) -> float | None:
        """Сколько больших блайндов приносит сотня раздач; ``None`` — рано.

        Число это шумное: полсотни раздач ещё ничего не доказывают, но именно
        по нему видно, что изменение хотя бы не сделало хуже. Без него спор
        «стало лучше или нет» решается на глаз и повторяется после каждой
        записи.
        """
        hands = self.counted(last)
        if not hands:
            return None
        return 100 * sum(hand.blinds_won for hand in hands) / len(hands)

    def trend(self, window: int = RECENT_HANDS) -> tuple[float | None, float | None]:
        """BB/100 за последние ``window`` раздач и за всё, что было до них.

        Сравнение периодов — весь смысл журнала: одно число само по себе не
        говорит ничего, а пара чисел уже показывает направление.
        """
        hands = self.counted()
        if len(hands) <= window:
            return self.bb_per_100(), None
        recent, earlier = hands[-window:], hands[:-window]
        return (
            100 * sum(hand.blinds_won for hand in recent) / len(recent),
            100 * sum(hand.blinds_won for hand in earlier) / len(earlier),
        )

    def wins(self) -> int:
        return sum(1 for hand in self.hands if hand.won)

    def showdowns(self) -> int:
        return sum(1 for hand in self.hands if hand.showdown)

    def showdown_wins(self) -> int:
        return sum(1 for hand in self.hands if hand.showdown and hand.won)

    def opponent_hands(self) -> tuple[tuple[int, tuple[str, ...]], ...]:
        """Все чужие руки, дошедшие до вскрытия, — копилка на будущее."""
        return tuple(shown for hand in self.hands for shown in hand.shown)

    def moves(self) -> int:
        """Сколько ходов записано из строки событий — по всем раздачам разом."""
        return sum(len(hand.events) for hand in self.hands)

    def actions(self) -> Counter:
        """Чего и сколько за столом делали: чек, колл, ставка, рейз, фолд.

        Первое, ради чего журнал и заводился: числа `LIVE_RANGE_SHARE`,
        `CALLER_KEEP` и остальные выставлены на глаз, а поправить их можно
        только по тому, как за этими столами играют на самом деле.
        """
        return Counter(
            action for hand in self.hands for _at, action, _amount in hand.events
        )

    # --- диск ---

    def save(self) -> None:
        """Записать журнал; молча пережить любую беду с диском.

        Журнал — статистика, а не прогресс игрока: потерять его обидно, но
        уронить из-за него экран посреди раздачи куда хуже.
        """
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([asdict(hand) for hand in self.hands], ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path | None) -> "Journal":
        """Прочитать журнал; сломанный файл — это пустой журнал, а не падение."""
        journal = cls(path=path)
        if path is None:
            return journal
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return journal
        if not isinstance(saved, list):
            return journal
        for item in saved:
            hand = _hand_from(item)
            if hand is not None:
                journal.hands.append(hand)
        del journal.hands[:-JOURNAL_LIMIT]
        return journal


def _cards(value) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(card) for card in value)


def _hand_from(item) -> Hand | None:
    """Запись из файла — в раздачу; чужой мусор в журнал не пускаем."""
    if not isinstance(item, dict):
        return None
    try:
        shown = tuple(
            (int(seat), _cards(cards))
            for seat, cards in item.get("shown", ())
            if _cards(cards)
        )
        events = tuple(
            (_none_or_int(at), str(action), _none_or_int(amount))
            for at, action, amount in item.get("events", ())
        )
        return Hand(
            hole=_cards(item.get("hole")),
            board=_cards(item.get("board")),
            position=item.get("position") or None,
            big_blind=_none_or_int(item.get("big_blind")),
            pot=_none_or_int(item.get("pot")),
            result=_none_or_int(item.get("result")),
            won=bool(item.get("won")),
            showdown=bool(item.get("showdown")),
            shown=shown,
            events=events,
            played_at=float(item.get("played_at", 0.0)),
        )
    except (TypeError, ValueError):
        return None


def _none_or_int(value) -> int | None:
    return None if value is None else int(value)


def new_hand(**fields) -> Hand:
    """Раздача «сейчас»: время проставляется само, остальное приходит с экрана."""
    return replace(Hand(played_at=time.time()), **fields)
