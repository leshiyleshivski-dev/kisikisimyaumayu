"""Считалка техасского холдема: сила руки, эквити, шансы банка, совет.

Модуль ничего не знает об окнах, кадрах и вводе — только карты и числа.
Такое же разделение уже работает у пары ``module.py`` / ``vision.py``: здесь
живёт математика, которую можно проверить точными числами, не запуская ни
CustomTkinter, ни игру.

Почему нельзя взять процент, который игра рисует сама: он показывает силу уже
собранной комбинации, а не шанс выиграть раздачу. Флеш-дро игра оценивает в
18 %, тогда как настоящее эквити там 61 %. Разбор — в ``poker-plan/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"

# Категории комбинаций от старшей карты (0) до стрит-флеша (8).
(
    HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT,
    FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH,
) = range(9)

CATEGORY_NAMES = (
    "старшая карта", "пара", "две пары", "тройка", "стрит",
    "флеш", "фулл-хаус", "каре", "стрит-флеш",
)

# Стриты от старшего к младшему: старший ранг комбинации и маска рангов.
# Последним идёт «колесо» A-2-3-4-5: туз играет снизу, и старшинство даёт
# пятёрка (индекс 3), а не туз.
_STRAIGHTS: tuple[tuple[int, int], ...] = tuple(
    (high, sum(1 << (high - offset) for offset in range(5)))
    for high in range(12, 3, -1)
) + ((3, (1 << 12) | 0b1111),)


def card_code(text: str) -> int:
    """«Ah», «Td», «10c» → номер карты 0..51 (ранг * 4 + масть)."""
    text = text.strip()
    body, suit = text[:-1], text[-1].lower()
    if body == "10":
        body = "T"
    return RANKS.index(body.upper()) * 4 + SUITS.index(suit)


def card_text(code: int) -> str:
    return f"{RANKS[code // 4]}{SUITS[code % 4]}"


def parse_cards(cards) -> np.ndarray:
    """Список строк или кодов → массив кодов; повтор карты считается ошибкой."""
    codes = [
        card if isinstance(card, (int, np.integer)) else card_code(card)
        for card in cards
    ]
    if len(set(codes)) != len(codes):
        raise ValueError(f"Карта повторяется: {[card_text(code) for code in codes]}")
    return np.array(codes, dtype=np.int64)


def _top_excluding(keys: np.ndarray, taken: np.ndarray, count: int) -> np.ndarray:
    """Старшие ``count`` рангов, кроме уже занятых.

    Приём «сортировать по числу карт, потом по рангу» для кикеров не годится:
    при каре 5555 + 22 + A он выдал бы кикером двойку, потому что пара стоит
    выше одиночного туза. Кикер выбирается только по старшинству.
    """
    return np.sort(np.where(taken, -1, keys), axis=1)[:, ::-1][:, :count]


def evaluate(hands: np.ndarray) -> np.ndarray:
    """Сила лучшей пятёрки для каждой строки ``hands`` (от пяти до семи карт).

    Ответ — одно число на руку: чем больше, тем сильнее, значения сравнимы
    между собой. Считается сразу для всей пачки рук, иначе перебор эквити не
    уложится в те секунды, что игра даёт на ход.
    """
    hands = np.atleast_2d(hands)
    total = hands.shape[0]
    ranks = hands // 4
    suits = hands % 4
    rows = np.arange(total)

    counts = np.bincount(
        (rows[:, None] * 13 + ranks).ravel(), minlength=total * 13
    ).reshape(total, 13)
    suit_counts = np.bincount(
        (rows[:, None] * 4 + suits).ravel(), minlength=total * 4
    ).reshape(total, 4)

    present = counts > 0
    keys = np.where(present, np.arange(13, dtype=np.int64), -1)
    bit_of_rank = (1 << np.arange(13)).astype(np.int64)
    rank_bits = np.bitwise_or.reduce(np.where(present, bit_of_rank, 0), axis=1)

    # Ранги, сгруппированные по числу карт: каре, тройки и пары оказываются
    # впереди одиночных, а внутри группы — по старшинству.
    grouped = np.argsort(-(counts * 13 + np.arange(13)), axis=1, kind="stable")
    grouped_counts = np.take_along_axis(counts, grouped, axis=1)
    first, second = grouped[:, 0:1], grouped[:, 1:2]
    first_count, second_count = grouped_counts[:, 0], grouped_counts[:, 1]
    taken_first = np.arange(13) == first
    taken_two = taken_first | (np.arange(13) == second)

    category = np.zeros(total, dtype=np.int64)
    kickers = np.zeros((total, 5), dtype=np.int64)

    def assign(mask: np.ndarray, value: int, ranks_for_score: np.ndarray) -> None:
        """Записать категорию и кикеры. Вызывается от слабых рук к сильным."""
        if not mask.any():
            return
        width = ranks_for_score.shape[1]
        category[mask] = value
        kickers[mask, :width] = ranks_for_score[mask]
        kickers[mask, width:] = 0

    # Порядок вызовов — от слабой комбинации к сильной: сильная перетирает
    # слабую там, где обе собрались из одних и тех же семи карт.
    assign(first_count == 1, HIGH_CARD, _top_excluding(keys, np.zeros_like(present), 5))
    assign(
        (first_count == 2) & (second_count == 1), PAIR,
        np.hstack([first, _top_excluding(keys, taken_first, 3)]),
    )
    assign(
        (first_count == 2) & (second_count == 2), TWO_PAIR,
        np.hstack([first, second, _top_excluding(keys, taken_two, 1)]),
    )
    assign(
        (first_count == 3) & (second_count == 1), TRIPS,
        np.hstack([first, _top_excluding(keys, taken_first, 2)]),
    )

    straight_high = np.full(total, -1, dtype=np.int64)
    for high, pattern in _STRAIGHTS:
        hit = (straight_high < 0) & ((rank_bits & pattern) == pattern)
        straight_high[hit] = high
    assign(straight_high >= 0, STRAIGHT, straight_high[:, None])

    has_flush = (suit_counts >= 5).any(axis=1)
    if has_flush.any():
        flush_suit = np.argmax(suit_counts >= 5, axis=1)
        by_suit = np.bincount(
            ((rows[:, None] * 4 + suits) * 13 + ranks).ravel(),
            minlength=total * 4 * 13,
        ).reshape(total, 4, 13) > 0
        flush_present = by_suit[rows, flush_suit]
        flush_keys = np.where(flush_present, np.arange(13, dtype=np.int64), -1)
        assign(has_flush, FLUSH, np.sort(flush_keys, axis=1)[:, ::-1][:, :5])
    else:
        flush_present = None

    # Фулл-хаус: старшая тройка плюс старшая из оставшихся пар. Вторая тройка
    # тоже играет за пару, поэтому ищем среди рангов с двумя и более картами.
    pair_keys = np.where((counts >= 2) & ~taken_first, np.arange(13, dtype=np.int64), -1)
    best_pair = np.sort(pair_keys, axis=1)[:, ::-1][:, :1]
    assign((first_count == 3) & (second_count >= 2), FULL_HOUSE, np.hstack([first, best_pair]))
    assign(first_count == 4, QUADS, np.hstack([first, _top_excluding(keys, taken_first, 1)]))

    if flush_present is not None:
        flush_bits = np.bitwise_or.reduce(np.where(flush_present, bit_of_rank, 0), axis=1)
        straight_flush_high = np.full(total, -1, dtype=np.int64)
        for high, pattern in _STRAIGHTS:
            hit = has_flush & (straight_flush_high < 0) & ((flush_bits & pattern) == pattern)
            straight_flush_high[hit] = high
        assign(straight_flush_high >= 0, STRAIGHT_FLUSH, straight_flush_high[:, None])

    score = category
    for slot in range(5):
        score = score * 13 + kickers[:, slot]
    return score


def hand_score(cards) -> int:
    """Сила руки числом; сравнивать имеет смысл только с такими же числами."""
    codes = parse_cards(cards)
    if not 5 <= codes.size <= 7:
        raise ValueError("Оценивается от пяти до семи карт")
    return int(evaluate(codes[None, :])[0])


def hand_category(cards) -> int:
    """Категория лучшей пятёрки: от ``HIGH_CARD`` до ``STRAIGHT_FLUSH``."""
    return hand_score(cards) // 13 ** 5


def category_name(cards) -> str:
    return CATEGORY_NAMES[hand_category(cards)]


def _remaining_deck(known: np.ndarray) -> np.ndarray:
    mask = np.ones(52, dtype=bool)
    mask[known] = False
    return np.flatnonzero(mask)


def equity(
    hole,
    board=(),
    opponents: int = 1,
    trials: int = 20_000,
    seed: int = 0,
) -> float:
    """Доля раздач, которые рука выигрывает против случайных соперников.

    Ничья считается половиной победы. На ривере против одного соперника ответ
    точный — перебираются все его 990 рук; в остальных случаях идёт
    Монте-Карло с фиксированным зерном, чтобы один и тот же стол давал один и
    тот же совет, а не дрожал от кадра к кадру.
    """
    hole_codes = parse_cards(hole)
    board_codes = parse_cards(board) if len(board) else np.empty(0, dtype=np.int64)
    if hole_codes.size != 2:
        raise ValueError("У руки ровно две карты")
    if board_codes.size > 5:
        raise ValueError("На столе не бывает больше пяти карт")
    if opponents < 1:
        raise ValueError("Нужен хотя бы один соперник")
    known = np.concatenate([hole_codes, board_codes])
    if np.unique(known).size != known.size:
        raise ValueError("Карта встречается и в руке, и на столе")

    deck = _remaining_deck(known)
    missing = 5 - board_codes.size

    if missing == 0 and opponents == 1:
        villain_hands = np.array(list(combinations(deck.tolist(), 2)), dtype=np.int64)
        boards = np.tile(board_codes, (villain_hands.shape[0], 1))
        hero = evaluate(np.hstack([np.tile(hole_codes, (boards.shape[0], 1)), boards]))
        villain = evaluate(np.hstack([villain_hands, boards]))
        return float(
            (np.sum(hero > villain) + np.sum(hero == villain) / 2) / hero.size
        )

    need = missing + 2 * opponents
    if need > deck.size:
        raise ValueError("В колоде не хватает карт на такое число соперников")
    rng = np.random.default_rng(seed)
    picks = deck[rng.random((trials, deck.size)).argsort(axis=1)[:, :need]]
    full_board = np.hstack([np.tile(board_codes, (trials, 1)), picks[:, :missing]])
    hero = evaluate(np.hstack([np.tile(hole_codes, (trials, 1)), full_board]))
    best_villain = None
    for index in range(opponents):
        start = missing + 2 * index
        villain = evaluate(np.hstack([picks[:, start:start + 2], full_board]))
        best_villain = villain if best_villain is None else np.maximum(best_villain, villain)
    wins = int(np.sum(hero > best_villain))
    ties = int(np.sum(hero == best_villain))
    return (wins + ties / 2) / trials


def equity_vs_range(
    hole,
    board=(),
    *,
    ranges,
    trials: int = 20_000,
    seed: int = 0,
) -> float:
    """Эквити против соперников, у которых не случайные карты, а диапазон.

    ``ranges`` — по набору классов на каждого соперника: тот, кто повысил,
    держит не что попало. Разница огромная и всегда в одну сторону: против
    случайной руки пара девяток с кикером «тройка» на борде 4♣5♠7♣9♥ берёт
    72 %, против рейзящего диапазона — 15 %. Раздача, где на этом ушёл стек,
    разобрана в ``poker-plan/README.md``.

    Считаем отбраковкой: соперникам раздаём руки из их диапазонов, доборные
    карты — из остатка колоды, и выкидываем раздачи, где карта повторилась.
    Так проще и честнее, чем вычитать диапазоны друг из друга: карт мало,
    и перекрытия редки.
    """
    from .ranges import combos_of

    hole_codes = parse_cards(hole)
    board_codes = parse_cards(board) if len(board) else np.empty(0, dtype=np.int64)
    if hole_codes.size != 2:
        raise ValueError("У руки ровно две карты")
    if not ranges:
        raise ValueError("Нужен хотя бы один соперник")
    known = np.concatenate([hole_codes, board_codes])
    if np.unique(known).size != known.size:
        raise ValueError("Карта встречается и в руке, и на столе")

    blocked = np.zeros(52, dtype=bool)
    blocked[known] = True
    tables = []
    for names in ranges:
        # Диапазон приходит либо названиями классов, либо уже готовыми
        # сочетаниями: после флопа его сужают по борду, и там от названий
        # остаются отдельные масти, которые классом уже не записать.
        if isinstance(names, np.ndarray):
            combos = names.reshape(-1, 2)
        else:
            combos = np.array(
                [pair for name in sorted(names) for pair in combos_of(name)],
                dtype=np.int64,
            ).reshape(-1, 2)
        keep = ~(blocked[combos[:, 0]] | blocked[combos[:, 1]])
        combos = combos[keep]
        if combos.size == 0:
            raise ValueError("Диапазон соперника пуст: все его руки уже на столе")
        tables.append(combos)

    deck = _remaining_deck(known)
    missing = 5 - board_codes.size
    rng = np.random.default_rng(seed)
    wins = ties = done = 0
    # Отбраковка режет примерно пятую часть раздач, поэтому берём с запасом
    # и добираем, пока не наберём заказанное число.
    batch = max(1024, int(trials * 0.7))
    while done < trials:
        villains = [table[rng.integers(len(table), size=batch)] for table in tables]
        drawn = np.hstack(villains)
        if missing:
            runout = deck[rng.random((batch, deck.size)).argsort(axis=1)[:, :missing]]
            drawn = np.hstack([drawn, runout])
        else:
            runout = np.empty((batch, 0), dtype=np.int64)
        fresh = np.ones(batch, dtype=bool)
        for left in range(drawn.shape[1]):
            for right in range(left + 1, drawn.shape[1]):
                fresh &= drawn[:, left] != drawn[:, right]
        if not fresh.any():
            continue
        take = min(int(fresh.sum()), trials - done)
        picked = np.flatnonzero(fresh)[:take]
        full_board = np.hstack([np.tile(board_codes, (take, 1)), runout[picked]])
        hero = evaluate(np.hstack([np.tile(hole_codes, (take, 1)), full_board]))
        best = None
        for villain_cards in villains:
            score = evaluate(np.hstack([villain_cards[picked], full_board]))
            best = score if best is None else np.maximum(best, score)
        wins += int(np.sum(hero > best))
        ties += int(np.sum(hero == best))
        done += take
    return (wins + ties / 2) / done


def pot_odds(pot: int, to_call: int) -> float:
    """Какую долю итогового банка приходится вложить, чтобы досмотреть раздачу.

    Банк здесь — всё, что уже поставлено: и плашка «Общий банк», и ставки,
    лежащие на сукне. Плашка их не считает, и по одной ей доля выходила
    завышенной: помощник требовал от руки больше эквити, чем нужно, а на
    префлопе плашки нет вовсе. Разбор — в ``poker-plan/README.md``.
    """
    if to_call <= 0:
        return 0.0
    return to_call / (pot + to_call)


# Размер ставки в долях банка. Живёт здесь, потому что им пользуется
# ``bet_size``; пороги самого совета переехали в ``postflop.py``.
BET_POT_SHARE = 0.66


def round_bet(value: float) -> int:
    """Округлить ставку до круглого числа: ползунок в игре двигают рукой.

    Совет «поставь 1 617» бесполезен — выставлять его дольше, чем игра даёт
    на ход. Шаг берётся от порядка числа: двузначное округляется до пятёрок,
    трёхзначное — до десятков, четырёхзначное — до сотен.
    """
    if value <= 0:
        return 0
    step = max(5, 10 ** (len(str(int(value))) - 2))
    return int(round(value / step) * step)


def spaced(value: int | None) -> str:
    """«1 250» — с пробелом внутри, ровно так, как число написано в игре.

    Живёт рядом с расчётом, а не в экране: советы называют суммы и в тексте
    причины, и в плане на ответ соперника, а писать одно и то же число двумя
    способами в одном окне нельзя.
    """
    return "—" if value is None else f"{value:,}".replace(",", " ")


def bet_size(
    pot: int, to_call: int = 0, *,
    my_bet: int = 0, stack: int | None = None, minimum: int | None = None,
) -> int:
    """До какой суммы поднимать — вся своя ставка на этой улице целиком.

    Игра просит именно это число: и ползунок, и кнопка «RAISE 1 000» считают
    итоговую ставку, а не доплату сверх чужой. Считаем две трети от банка,
    каким он станет после уравнивания, — это примерно то же, что предлагает
    кнопка «BANK».

    Ниже минимального рейза опускаться нельзя — игра такой ход не примет, —
    а выше своего стека не прыгнуть: там уже олл-ин.
    """
    highest = my_bet + to_call
    target = round_bet(highest + BET_POT_SHARE * (pot + to_call))
    if to_call > 0:
        # Меньше чем на размер чужой ставки поднимать не разрешают.
        target = max(target, highest + to_call)
    if minimum:
        target = max(target, minimum)
    if stack is not None:
        target = min(target, my_bet + stack)
    return max(0, target)


# Ряд быстрых размеров ставки — четыре кнопки справа над кнопками хода. Игра
# ставит ими за одно нажатие то, что иначе пришлось бы вести ползунком на
# пятнадцатисекундном таймере. Названия — как в игре, порядок — как на экране.
QUICK_BET_NAMES = ("MIN", "3 BB", "BANK", "ALL IN")
QUICK_BET_BLINDS = 3  # столько блайндов ставит кнопка «3 BB»
# Насколько кнопка может отличаться от нужного размера и всё ещё считаться тем
# же ходом. Открытие в два с половиной блайнда и кнопка «3 BB» — это один ход,
# а открытие в четыре с половиной — уже другой.
QUICK_BET_TOLERANCE = 0.25


@dataclass(frozen=True)
class QuickBet:
    """Кнопка быстрого размера: как называется и до какой суммы поднимает."""

    name: str
    amount: int


def quick_bets(
    *, pot: int, to_call: int = 0, my_bet: int = 0, stack: int | None = None,
    big_blind: int | None = None, minimum: int | None = None,
) -> tuple[QuickBet, ...]:
    """Что поставит каждая из четырёх кнопок, в порядке слева направо.

    Суммы — итоговая ставка улицы целиком, как их считает и игра: кнопка
    «RAISE 1 000» просит выставить тысячу, а не тысячу сверх чужой ставки.

    Кнопка, которой не из чего посчитать сумму, в список не попадает: без
    прочитанных блайндов «3 BB» неизвестна, а без минимума с кнопки — «MIN».
    Одинаковые суммы схлопываются: на банке 150 при блайнде 50 «3 BB» и
    «BANK» ставят одно и то же, и предлагать игроку выбор из двух одинаковых
    кнопок незачем. Совпавшую со стеком сумму всегда зовём «ALL IN» — игрок
    должен видеть, что этот ход за весь стек, каким бы кнопку ни нажал.
    """
    ceiling = my_bet + stack if stack is not None else None
    if minimum and ceiling is not None and minimum > ceiling:
        # Минимума больше стека игра показать не может: столько фишек просто
        # нет. Приходит такое число с кнопки под курсором — и, встав полом,
        # оно подтягивает к себе все четыре кнопки разом, а обрезка стеком
        # схлопывает их в одну: `ALL IN`. Лучше остаться без «MIN», чем с
        # единственным «весь стек».
        minimum = None
    # Ниже минимума игра ставку не примет и сама подтянет кнопку до него.
    # Минимум читается с кнопки; не прочитался — не выдумываем его, а берём
    # то, что верно всегда: повышение не бывает меньше уравнивания.
    floor = minimum or my_bet + to_call
    amounts = (
        ("MIN", minimum),
        ("3 BB", QUICK_BET_BLINDS * big_blind if big_blind else None),
        # «Банк» — это ставка размером с банк: уравнять чужую ставку и
        # поставить сверху столько, сколько станет в банке после уравнивания.
        ("BANK", my_bet + to_call + (pot + to_call)),
        ("ALL IN", ceiling),
    )
    found: list[QuickBet] = []
    seen: set[int] = set()
    for name, amount in amounts:
        if not amount:
            continue
        amount = max(amount, floor)
        if ceiling is not None:
            amount = min(amount, ceiling)
        if amount <= 0 or amount in seen:
            continue
        seen.add(amount)
        found.append(QuickBet("ALL IN" if amount == ceiling else name, amount))
    return tuple(found)


def quick_bet_for(bets, target: int) -> QuickBet | None:
    """Кнопка, которая ставит примерно ``target``; ``None`` — такой нет.

    Нужна там, где размер берётся из таблицы, а не считается перебором:
    открытие в два с половиной блайнда игра ставит кнопкой «3 BB», а олл-ин на
    коротком стеке — кнопкой «ALL IN», и это быстрее и точнее, чем вести
    ползунок руками.

    Из подходящих берётся самая дешёвая из тех, что не меньше нужного, и лишь
    когда таких нет — самая крупная из тех, что меньше. Округлять вверх, а не
    к ближайшему, тут правильнее: открытие в два с половиной блайнда стоит
    ровно между «MIN» и «3 BB», а мельчить с открытием — отдавать в раздачу
    лишних людей задёшево.
    """
    if not bets or target <= 0:
        return None
    room = target * QUICK_BET_TOLERANCE
    above = [bet for bet in bets if 0 <= bet.amount - target <= room]
    if above:
        return min(above, key=lambda bet: bet.amount)
    below = [bet for bet in bets if 0 <= target - bet.amount <= room]
    if below:
        return max(below, key=lambda bet: bet.amount)
    return None


@dataclass(frozen=True)
class Advice:
    """Готовый совет: что делать и на каких числах это основано."""

    action: str
    equity: float
    odds: float
    raise_to: int
    reason: str
    made: str = ""
    all_in: bool = False
    # Какую из четырёх кнопок быстрого размера нажать; пусто — ни одна не
    # ставит нужную сумму, и размер придётся вести ползунком.
    button: str = ""
    # Что делать, когда соперник ответит на этот ход: «если поставит — колл
    # до 1 500, крупнее фолд». Пусто — отвечать будет уже не он, а мы.
    plan: str = ""
