"""Что делать после флопа: диапазон соперника на этом борде и совет по нему.

Считать эквити против двух случайных карт после флопа — главная ошибка, из-за
которой уходили стеки. На тёрне 4♣5♠7♣9♥ пара девяток с кикером «тройка»
берёт у случайной руки 72 %, у диапазона, с которым повышают, — 15 %. Именно
на этой руке в записи ``6.mp4`` помощник посоветовал ставить 2 400 «при
эквити 73 %», соперник ответил олл-ином, и у него оказалась та же тройка
девяток с лучшим кикером.

Диапазон сужается дважды. Сперва по стартовой руке: до флопа доходят не со
всем подряд. Потом по борду — и это важнее. Соперник, который повысил на
тёрне, держит не «сильную стартовую руку», а руку, которая попала **в эти
карты**: 7-2 на борде 7-7-2 стоит дороже, чем AK мимо.

Сила на борде считается с учётом доборов, а не по готовой комбинации. Иначе
флеш-дро попало бы в самый низ диапазона, хотя повышают с ним постоянно. На
ривере доборов нет, и ответ точный; на тёрне перебираются все 44 карты; на
флопе доборов 990, и там идёт выборка.
"""

from __future__ import annotations

import numpy as np

from .hand_math import (
    Advice, QuickBet, bet_size, category_name, equity_vs_range, evaluate,
    parse_cards, pot_odds, quick_bets, round_bet, spaced,
)
from .ranges import combos_of, top_share

# С какой долей рук вообще доходят до флопа. За столами казино играют широко,
# поэтому берём больше половины: недооценить чужой диапазон опаснее, чем
# переоценить — на этом и горели.
LIVE_RANGE_SHARE = 0.55

# Насколько узок диапазон того, кто поставил столько. Ключ — размер доплаты
# в долях банка: маленькая ставка почти ничего не говорит, олл-ин говорит всё.
CHECK_KEEP = 1.0
SMALL_BET_KEEP = 0.55
BIG_BET_KEEP = 0.30
SHOVE_KEEP = 0.18
# Ставка меряется банком, каким он был до неё: половина банка и банк целиком —
# привычные покерные величины, и пороги стоят на них.
BIG_BET_SHARE = 0.5
SHOVE_SHARE = 1.0

# Кто ответит на нашу ставку. Уравнивают заметно шире, чем повышают, и
# путать эти два диапазона нельзя: по первому решают, стоит ли ставить, по
# второму — стоит ли платить. Число стоит на привычном размере в две трети
# банка; как оно меняется с размером, считает ``caller_keep``.
CALLER_KEEP = 0.55

# Какая часть диапазона ставящего — блеф. Без этой доли модель считает, что
# идущий ва-банк никогда не блефует, и туз-хай против него берёт ровно ноль.
# Так на флопе 4♦3♠4♠ выходил фолд при шансах банка 10 к 1 — ошибка в другую
# сторону от прежней, но такая же дорогая.
#
# Блефы берутся из самого низа диапазона, и это не упрощение: блефуют как раз
# пустой рукой, которой вскрытие не выиграть. Средние руки не блефуют — они
# проверяют и уравнивают. Дро в низ не попадают: сила считается с учётом
# доборов, и флеш-дро стоит высоко, среди тех, с кем ставят ради победы.
# За столами казино блефуют реже, чем в равновесии, поэтому доли скромные:
# переоценить чужой блеф — значит снова платить блеф-кетчерами, а с этого всё
# и начиналось. Числа тут подгоняются по журналу раздач, когда он появится.
SMALL_BET_BLUFF = 0.12
BIG_BET_BLUFF = 0.20
SHOVE_BLUFF = 0.25
CALLER_BLUFF = 0.0  # уравнивают ради вскрытия, а не ради чужого паса

# Доборов на флопе 990 — перебирать их для каждого сочетания дорого, и точность
# тут не нужна: по этому числу руки только расставляются по порядку.
BOARD_TRIALS = 160
EQUITY_TRIALS = 12_000
# Перебор для сравнения размеров между собой. Числа отсюда игроку не
# показываются — по ним только выбирается кнопка, — а четыре размера считать
# полным перебором значило бы вчетверо удлинить и без того самый долгий шаг.
SIZING_TRIALS = 4_000
# Насколько крупную ставку помощник вообще готов посоветовать — в долях банка.
# Выше этого счёт перестаёт быть счётом: минимальная защита говорит, сколько
# соперник обязан отвечать, чтобы наш блеф не окупался, а живой человек на
# ставку в три банка пасует почти всегда. Весь выигрыш оверставки существует
# только в модели, зато проигрыш — в стеке. Вдобавок улица не последняя: за
# флопом идут тёрн и ривер, и оставить на них фишки дороже, чем забрать банк
# сейчас. Одноуличный счёт этого не видит, а потолок — видит.
MAX_BET_SHARE = 2.0

# Пороги совета. Считаются уже от эквити против диапазона, а не против
# случайных карт, поэтому числа другие, чем были.
RAISE_EDGE = 0.20      # насколько эквити должно превышать шансы банка для рейза
CALL_EDGE = 0.03       # запас на неточность перебора и на ошибки распознавания
VALUE_BET_EDGE = 0.05  # насколько надо обгонять тех, кто ответит на ставку


def bet_share(pot: int, to_call: int) -> float:
    """Ставка соперника в долях банка, каким он был до неё.

    ``pot`` приходит уже вместе с этой ставкой — так его считает зрение, — а
    привычные «половина банка» и «банк» отсчитываются от того, что лежало
    раньше. Отсюда вычитание.

    Своим стеком тут ничего не меряется, и это принципиально. Соперник ставит
    свой размер, не зная, сколько осталось у нас: ставка в треть банка не
    становится олл-ином оттого, что нам её нечем доплатить. Пока в счёт шёл
    обрезанный стеком остаток, огромная ставка читалась как маленькая.
    """
    if to_call <= 0:
        return 0.0
    return to_call / max(1, pot - to_call)


def aggression_keep(pot: int, to_call: int) -> float:
    """Насколько узок диапазон соперника по тому, сколько он поставил.

    Ставка в четверть банка бывает и с чем угодно, а ставка в банк — почти
    всегда либо с готовой рукой, либо с пустой.
    """
    if to_call <= 0:
        return CHECK_KEEP
    share = bet_share(pot, to_call)
    if share >= SHOVE_SHARE:
        return SHOVE_KEEP
    if share >= BIG_BET_SHARE:
        return BIG_BET_KEEP
    return SMALL_BET_KEEP


def caller_keep(pot: int, bet: int) -> float:
    """Какая доля диапазона ответит на нашу ставку такого размера.

    Считается по минимальной защите: чтобы ставящему не было выгодно блефовать
    чем попало, отвечать надо долей ``банк / (банк + ставка)``. Живые люди так
    не играют, но это единственная опора, которая не выдумана из головы, — и
    она, в отличие от порогов, непрерывная.

    Непрерывность тут не украшение. Ступеньки перестают различать размеры
    выше банка: и ставка в банк, и олл-ин на десять банков попадали в одну
    ступень, а денег в банк на олл-ине уходило больше — и помощник выбирал
    олл-ин с любой рукой, которой вообще стоит ставить. Ровно тот способ
    терять стек, ради борьбы с которым модуль и переписывался.

    ``pot`` — банк уже вместе с нашей ставкой, как его считает и зрение.
    """
    return keep_for_share(bet_share(pot, bet))


def keep_for_share(share: float) -> float:
    """Та же минимальная защита, но от готовой доли банка.

    Вынесено ради замера: `stats.py` считает по журналу, какая доля на самом
    деле не пасует, и сравнивать её надо ровно с этой формулой, а не с её
    пересказом. Разойдись они — и сравнение мерило бы разницу между двумя
    записями одного и того же, а не между моделью и столом.
    """
    if share <= 0:
        return CHECK_KEEP
    return 1 / (1 + share)


def aggression_bluff(pot: int, to_call: int) -> float:
    """Какая доля диапазона ставящего — чистый блеф.

    Чем больше ставка, тем чаще за ней стоит либо готовая рука, либо ничего:
    средним рукам такой размер не по карману. Поэтому доля блефа растёт вместе
    с сужением диапазона, а не падает.
    """
    keep = aggression_keep(pot, to_call)
    if keep >= CHECK_KEEP:
        return 0.0
    if keep <= SHOVE_KEEP:
        return SHOVE_BLUFF
    if keep <= BIG_BET_KEEP:
        return BIG_BET_BLUFF
    return SMALL_BET_BLUFF


def range_combos(names, blocked: np.ndarray) -> np.ndarray:
    """Сочетания диапазона, кроме тех, чьи карты уже лежат на столе или в руке."""
    combos = np.array(
        [pair for name in sorted(names) for pair in combos_of(name)], dtype=np.int64
    ).reshape(-1, 2)
    keep = ~(blocked[combos[:, 0]] | blocked[combos[:, 1]])
    return combos[keep]


def _runouts(
    deck: np.ndarray, missing: int, trials: int, seed: int
) -> np.ndarray:
    """Карты, которых на борде ещё нет.

    На ривере добирать нечего, на тёрне их сорок с небольшим — оба случая
    перебираются целиком. На флопе сочетаний под тысячу, и там выборка: по
    этому числу руки только расставляются по порядку, точность не нужна.
    """
    if missing == 0:
        return np.empty((1, 0), dtype=np.int64)
    if missing == 1:
        return deck[:, None]
    rng = np.random.default_rng(seed)
    return deck[rng.random((trials, deck.size)).argsort(axis=1)[:, :missing]]


def strength_on_board(
    combos: np.ndarray, board, *, blocked: np.ndarray | None = None,
    trials: int = BOARD_TRIALS, seed: int = 0,
) -> np.ndarray:
    """Насколько каждое сочетание сильно **на этом борде**, с учётом доборов.

    Ответ — доля сочетаний того же диапазона, которые эта рука обгоняет: от
    нуля до единицы. Считается по готовой комбинации не на текущем борде, а в
    среднем по доборам, поэтому флеш-дро оказывается там, где ему и место, —
    высоко, а не в самом низу.
    """
    board_codes = parse_cards(board) if len(board) else np.empty(0, dtype=np.int64)
    total = combos.shape[0]
    if total == 0:
        return np.empty(0)
    # Свои карты в добор попасть не могут — они уже в руке, а не в колоде.
    taken = np.zeros(52, dtype=bool) if blocked is None else blocked.copy()
    if board_codes.size:
        taken[board_codes] = True
    deck = np.flatnonzero(~taken)
    missing = 5 - board_codes.size
    runouts = _runouts(deck, missing, trials, seed)

    scores = np.empty((total, runouts.shape[0]), dtype=np.int64)
    valid = np.ones((total, runouts.shape[0]), dtype=bool)
    for index in range(runouts.shape[0]):
        full = np.concatenate([board_codes, runouts[index]])
        hands = np.hstack([combos, np.tile(full, (total, 1))])
        scores[:, index] = evaluate(hands)
        if runouts.shape[1]:
            clash = (
                np.isin(combos[:, 0], runouts[index])
                | np.isin(combos[:, 1], runouts[index])
            )
            valid[:, index] = ~clash

    # Доля обойдённых рук в каждом доборе. Ничьи считаются половиной: на
    # спаренном борде одинаковых по силе сочетаний десятки, и отдавать им
    # случайный порядок значило бы гадать.
    shares = np.empty_like(scores, dtype=float)
    for index in range(scores.shape[1]):
        column = np.where(valid[:, index], scores[:, index], -1)
        ordered = np.sort(column)
        left = np.searchsorted(ordered, column, side="left")
        right = np.searchsorted(ordered, column, side="right")
        shares[:, index] = (left + right) / 2 / total
    seen = valid.sum(axis=1)
    seen[seen == 0] = 1
    return (shares * valid).sum(axis=1) / seen


def ranked_on_board(
    names, board, *, hole=(), trials: int = BOARD_TRIALS, seed: int = 0,
) -> np.ndarray:
    """Сочетания диапазона, расставленные по силе на этом борде: лучшие первыми.

    Расстановка — самая дорогая часть разбора: сила каждого сочетания
    считается по доборам. Зато посчитанная один раз, она даёт любую долю
    диапазона простым срезом — а долей нужно несколько, когда помощник
    выбирает между четырьмя размерами своей ставки.
    """
    blocked = np.zeros(52, dtype=bool)
    for cards in (hole, board):
        if len(cards):
            blocked[parse_cards(cards)] = True
    combos = range_combos(names, blocked)
    if combos.shape[0] == 0:
        return combos
    strength = strength_on_board(
        combos, board, blocked=blocked, trials=trials, seed=seed
    )
    return combos[np.argsort(-strength, kind="stable")]


def keep_share(ranked: np.ndarray, keep: float, bluff: float = 0.0) -> np.ndarray:
    """Верхняя доля расставленного диапазона, а при блефе — ещё и низ.

    ``keep`` — какую долю оставить по силе: единица значит «соперник ничего не
    показал, держим весь диапазон», а пятая часть — «пошёл ва-банк, случайных
    рук там почти нет». ``bluff`` — какой долей готового набора станут руки из
    самого низа: диапазон ставящего не сплошной, он с двух концов.
    """
    total = ranked.shape[0]
    if keep >= 1 or total == 0:
        return ranked
    wanted = max(1, int(round(total * keep)))
    picked = ranked[:wanted]
    if bluff > 0:
        # Столько блефов, чтобы они заняли долю ``bluff`` уже готового набора.
        bluffs = min(int(round(wanted * bluff / (1 - bluff))), total - wanted)
        if bluffs > 0:
            picked = np.concatenate([picked, ranked[-bluffs:]])
    return picked


def narrow_to_board(
    names, board, keep: float, *, bluff: float = 0.0, hole=(),
    trials: int = BOARD_TRIALS, seed: int = 0,
) -> np.ndarray:
    """Оставить в диапазоне те руки, с которыми на этом борде так ставят.

    Держать весь диапазон — ответ, который не требует расстановки: она стоит
    дороже всего остального разбора вместе взятого, и зря её не запускаем.
    """
    if keep >= 1:
        blocked = np.zeros(52, dtype=bool)
        for cards in (hole, board):
            if len(cards):
                blocked[parse_cards(cards)] = True
        return range_combos(names, blocked)
    ranked = ranked_on_board(names, board, hole=hole, trials=trials, seed=seed)
    return keep_share(ranked, keep, bluff)


def opponent_range(
    board, *, hole, keep: float, bluff: float = 0.0,
    live_share: float = LIVE_RANGE_SHARE, trials: int = BOARD_TRIALS, seed: int = 0,
) -> np.ndarray:
    """Диапазон соперника: сперва по стартовой руке, потом по борду."""
    return narrow_to_board(
        top_share(live_share), board, keep, bluff=bluff, hole=hole,
        trials=trials, seed=seed,
    )


def value_bet_equity(opponents: int) -> float:
    """С какого эквити рука стоит ставки против стольких отвечающих.

    Считается уже против тех, кто ответит, а не против случайных карт: сама
    средняя доля от этого не меняется — против одного она половина, против
    пятерых шестая часть, — а вот число, которое с ней сравнивают, теперь
    честное.
    """
    return 1 / (opponents + 1) + VALUE_BET_EDGE


def table_ranges(
    ranked: np.ndarray, *, opponents: int, keep: float, bluff: float,
    bettor: bool = True,
) -> list[np.ndarray]:
    """Кому за столом какой диапазон раздать.

    Узкий достаётся только тому, кто ставил: остальные просто ещё держат
    карты. Раздать диапазон агрессора всем значило бы посадить против себя
    четверых с готовой рукой сразу — на четверых соперниках так выходил фолд
    там, где банк предлагает десять к одному. Не ставил никто (``bettor``
    выключен) — и выделять некого: все одинаково те, кто ответит на нашу
    ставку.
    """
    hand = keep_share(ranked, keep, bluff)
    others = max(0, opponents - 1)
    if not others:
        return [hand]
    if bettor:
        return [hand] + [keep_share(ranked, CHECK_KEEP)] * others
    return [hand] * (others + 1)


def call_or_raise(
    chance: float, odds: float, *, opponents: int, can_raise: bool = True
) -> str:
    """Что делать против ставки при таком эквити и таких шансах банка.

    Правило одно и то же и для живого хода, и для плана на ход вперёд. Иначе
    план обещал бы одно, а через кадр экран советовал бы другое — и верить
    было бы нечему.
    """
    if (
        can_raise
        and chance >= odds + RAISE_EDGE
        and chance >= value_bet_equity(opponents)
    ):
        return "рейз"
    if chance >= odds + CALL_EDGE:
        return "колл"
    return "фолд"


def facing_bet(
    hole, board, *, ranked: np.ndarray, opponents: int, pot: int, to_call: int,
    faced: int | None = None, can_raise: bool = True, trials: int = EQUITY_TRIALS,
    known: dict[float, float] | None = None, seed: int = 0,
) -> tuple[str, float, float]:
    """Ответ на чужую ставку: что делать, эквити и шансы банка.

    ``to_call`` — сколько мы заплатим, ``faced`` — сколько он поставил. Когда
    фишек у нас меньше его ставки, это разные числа: по первому считаются
    шансы банка, по второму читается диапазон.

    ``known`` — уже посчитанные эквити по долям диапазона. Нужен, когда один и
    тот же расклад считается для нескольких размеров подряд: ступеней у чужой
    агрессии три, а размеров в плане четыре, и повторять перебор незачем.
    """
    faced = to_call if faced is None else max(faced, to_call)
    keep = aggression_keep(pot, faced)
    chance = None if known is None else known.get(keep)
    if chance is None:
        chance = equity_vs_range(
            hole, board,
            ranges=table_ranges(
                ranked, opponents=opponents, keep=keep,
                bluff=aggression_bluff(pot, faced),
            ),
            trials=trials, seed=seed,
        )
        if known is not None:
            known[keep] = chance
    # Чужой излишек сверх нашего стека вернётся ему: разыгрывается только та
    # часть банка, которую мы в состоянии уравнять.
    odds = pot_odds(max(0, pot - (faced - to_call)), to_call)
    action = call_or_raise(chance, odds, opponents=opponents, can_raise=can_raise)
    return action, chance, odds


def size_ev(
    *, pot: int, add: int, their_add: int, chance: float, keep: float, opponents: int
) -> float:
    """Сколько приносит ставка такого размера — в фишках, а не в процентах.

    Два исхода, и оба считаются: все спасовали — забираем банк как есть; кто-то
    ответил — играем за банк вместе с доплатами. Середины (один ответил, второй
    пас) тут нет нарочно: считать её не по чему, а на выбор между четырьмя
    кнопками она почти не влияет.

    Отсчёт идёт от того, что уже лежит в банке, а не от нуля. Само число
    поэтому ничего не значит — сравнивать его можно только с таким же числом
    для другого размера, ради чего оно и считается.
    """
    folds = (1 - keep) ** max(1, opponents)
    called = pot + add + their_add * max(1, opponents)
    return folds * pot + (1 - folds) * (chance * called - add)


def choose_quick_bet(
    hole, board, bets, *, ranked: np.ndarray, pot: int, to_call: int = 0,
    my_bet: int = 0, opponents: int = 1, trials: int = SIZING_TRIALS, seed: int = 0,
) -> QuickBet | None:
    """Какая из кнопок быстрого размера приносит больше всего; ``None`` — нет таких.

    Игра предлагает четыре готовых размера в одно нажатие, и выбирать между
    ними — не то же самое, что назвать число: число игрок ведёт ползунком на
    пятнадцатисекундном таймере и промахивается, а кнопку жмёт сразу.

    Считается каждая по одному правилу: чем больше просим доплатить, тем чаще
    все пасуют и тем уже те, кто ответит. Эквити против отвечающих поэтому
    своё у каждого размера — считать его приходится по разу на кнопку, и
    только поэтому перебор тут короче обычного. Совпали доли — совпадёт и
    эквити, и второй раз оно не считается.
    """
    if pot <= 0:
        # Без банка размер не с чем сравнивать: и «во сколько банков ставим»,
        # и выгода считаются от него. Молчим, как молчит вся остальная
        # подсказка, когда банк не прочитался.
        return None
    sane = [bet for bet in bets if 0 < bet.amount - my_bet <= MAX_BET_SHARE * pot]
    if not sane:
        # Банк меньше минимальной ставки — так бывает на первой ставке улицы,
        # когда в банке лежат одни блайнды. Выбора нет: берём самую дешёвую.
        sane = sorted(
            (bet for bet in bets if bet.amount > my_bet), key=lambda bet: bet.amount
        )[:1]
    best, best_value = None, None
    known: dict[float, float] = {}
    for bet in sane:
        add = bet.amount - my_bet
        their_add = max(0, bet.amount - my_bet - to_call)
        # Размер меряется банком, каким он был до нашей ставки: рискуем мы
        # всей доплатой, а не только той её частью, что сверх чужой ставки.
        keep = caller_keep(pot + add, add)
        chance = known.get(keep)
        if chance is None:
            chance = equity_vs_range(
                hole, board,
                ranges=[keep_share(ranked, keep)] * max(1, opponents),
                trials=trials, seed=seed,
            )
            known[keep] = chance
        value = size_ev(
            pot=pot, add=add, their_add=their_add,
            chance=chance, keep=keep, opponents=opponents,
        )
        if best_value is None or value > best_value:
            best, best_value = bet, value
    return best


# --- план на ответ соперника ---

# Чем соперник может ответить на наш чек — в долях банка. Три размера бьют
# ровно в три ступени, которыми модуль читает чужую агрессию: треть банка,
# две трети и банк целиком. Четвёртой ступенью идёт свой стек — платить
# больше нечем, и это самый дорогой ответ из возможных.
REPLY_SHARES = (0.33, 0.66, 1.0)

# Во сколько раз соперник поднимает нашу ставку. Меньше чем вдвое поднимать
# не дают, втрое — привычное повышение на борде.
REPLY_RAISE = 3.0

# Перебор для плана короче обычного нарочно: размеров в лесенке несколько, а
# показанное решение всё равно пересчитается заново — когда соперник
# действительно поставит, считать будет полный перебор.
PLAN_TRIALS = 4_000


def reply_sizes(pot: int, stack: int | None = None) -> tuple[int, ...]:
    """Ставки, которыми соперник может ответить на наш чек: от мелкой к олл-ину.

    Больше своего стека мы всё равно не заплатим: ставка крупнее — тот же
    олл-ин, только чужой излишек вернётся ему. Поэтому лесенка обрезается
    стеком, а последней ступенью встаёт он сам.
    """
    sizes = [round_bet(share * pot) for share in REPLY_SHARES]
    if stack:
        sizes = [min(size, stack) for size in sizes] + [stack]
    return tuple(sorted({size for size in sizes if size > 0}))


def plan_line(opening: str, steps) -> str:
    """Собрать план из лесенки ответов: «доплата до 1 500 — колл, крупнее — фолд».

    Лесенка идёт от мелкой ставки к олл-ину и читается до первого «фолда»:
    заплатить крупную, сбросив мелкую, — не план, а путаница. Одинаковые
    ответы подряд склеиваются: игроку нужна граница, а не перечень размеров.

    Числом называется доплата — то же слово и то же число, что стоит в клетке
    «ДОПЛАТА» на экране. Написать «рейз до 2 000» было нельзя: этими же
    словами строка размера называет **свою** ставку, и одно и то же «до»
    означало бы в одном окне две разные вещи.
    """
    taken: list[list] = []
    folded = False
    for action, amount, all_in in steps:
        if action == "фолд":
            folded = True
            break
        if taken and taken[-1][0] == action:
            taken[-1] = [action, amount, all_in]
        else:
            taken.append([action, amount, all_in])
    if not taken:
        return f"{opening} — фолд" if folded else ""
    clauses = []
    for index, (action, amount, all_in) in enumerate(taken):
        if all_in:
            clauses.append(f"весь стек — {action}")
        elif index:
            clauses.append(f"до {spaced(amount)} — {action}")
        else:
            clauses.append(f"доплата до {spaced(amount)} — {action}")
    line = f"{opening}: " + ", ".join(clauses)
    return f"{line}, крупнее — фолд" if folded else line


def check_plan(
    hole, board, *, ranked: np.ndarray, opponents: int, pot: int,
    stack: int | None = None, trials: int = PLAN_TRIALS, seed: int = 0,
) -> str:
    """Что делать, если после нашего чека соперник поставит.

    Одно слово «чек» не отвечает на вопрос, ради которого экран и заводился:
    фишки уходят не на чеке, а на том, что случается после него. Тот же движок
    прогоняется ещё раз — по размерам, которыми соперник может ответить, — и
    план называет ставку, до которой мы платим.
    """
    if pot <= 0:
        return ""
    known: dict[float, float] = {}
    steps = []
    for size in reply_sizes(pot, stack):
        to_call = min(size, stack) if stack else size
        all_in = bool(stack) and size >= stack
        action, _, _ = facing_bet(
            hole, board, ranked=ranked, opponents=opponents, pot=pot + size,
            to_call=to_call, faced=size, can_raise=not all_in,
            trials=trials, known=known, seed=seed,
        )
        steps.append((action, to_call, all_in))
    return plan_line("если поставит", steps)


def raise_plan(
    hole, board, *, ranked: np.ndarray, opponents: int, pot: int, bet_to: int,
    to_call: int = 0, my_bet: int = 0, stack: int | None = None,
    trials: int = PLAN_TRIALS, seed: int = 0,
) -> str:
    """Что делать, если соперник поднимет нашу ставку.

    Поднимают либо втрое, либо сразу в олл-ин, и это два разных решения:
    доплатить в первом случае почти всегда дешевле. Когда наша ставка и так
    забирает весь стек, поднимать над ней нечего — плана нет.
    """
    ceiling = my_bet + stack if stack else None
    if pot <= 0 or bet_to <= 0 or (ceiling is not None and bet_to >= ceiling):
        return ""
    their_bet = my_bet + to_call
    sizes = [round_bet(REPLY_RAISE * bet_to)]
    if ceiling is not None:
        sizes = [size for size in sizes if size < ceiling] + [ceiling]
    known: dict[float, float] = {}
    steps = []
    for size in sorted({size for size in sizes if size > bet_to}):
        add = size - bet_to
        all_in = ceiling is not None and size >= ceiling
        action, _, _ = facing_bet(
            hole, board, ranked=ranked, opponents=opponents,
            # Банк к тому моменту вырастет на нашу ставку и на его повышение.
            pot=pot + (bet_to - my_bet) + (size - their_bet),
            to_call=add, faced=add, can_raise=not all_in,
            trials=trials, known=known, seed=seed,
        )
        steps.append((action, add, all_in))
    return plan_line("если повысит", steps)


def postflop_advice(
    hole,
    board,
    *,
    opponents: int = 1,
    pot: int = 0,
    to_call: int = 0,
    faced_bet: int | None = None,
    my_bet: int = 0,
    stack: int | None = None,
    min_bet: int | None = None,
    big_blind: int | None = None,
    can_raise: bool = True,
    trials: int = EQUITY_TRIALS,
    board_trials: int = BOARD_TRIALS,
    seed: int = 0,
) -> Advice:
    """Совет после флопа: фолд, чек, колл, бет или рейз — по диапазону соперника.

    Диапазон выбирается по тому, что соперник только что сделал. Поставил —
    считаем против того, с чем ставят, и решаем, окупается ли доплата. Не
    поставил — считаем против того, с чем **ответят** на нашу ставку, и решаем,
    стоит ли ставить: это разные наборы рук, и путать их нельзя.

    ``can_raise`` выключается, когда доплата больше стека: игра там не рисует
    кнопку повышения вовсе, и совет «рейз» отправлял бы искать то, чего нет.

    ``to_call`` — сколько мы заплатим, ``faced_bet`` — сколько он поставил.
    Когда фишек у нас меньше его ставки, это разные числа, и путать их нельзя:
    по первому считаются шансы банка, по второму читается диапазон.

    Вместе с ходом совет несёт план на ответ соперника: чек и бет — это не
    конец улицы, а её середина. Считает его тот же движок — ``check_plan`` и
    ``raise_plan``, по той же расстановке диапазона на борде.
    """
    if len(board) < 3:
        raise ValueError("После флопа на столе минимум три общие карты")
    faced = to_call if faced_bet is None else max(faced_bet, to_call)
    # Расстановка диапазона по силе на борде считается один раз: из неё
    # нарезаются и диапазон соперника, и диапазоны отвечающих на каждый из
    # четырёх размеров нашей ставки, и весь план на его ответ. Считать её
    # заново на каждую долю значило бы удлинить разбор впятеро — а это самый
    # долгий его шаг.
    ranked = ranked_on_board(
        top_share(LIVE_RANGE_SHARE), board, hole=hole, trials=board_trials, seed=seed
    )
    worth_betting = value_bet_equity(opponents)
    if to_call > 0:
        action, chance, odds = facing_bet(
            hole, board, ranked=ranked, opponents=opponents, pot=pot,
            to_call=to_call, faced=faced, can_raise=can_raise,
            trials=trials, seed=seed,
        )
    else:
        # Никто не ставил — значит, считаем не против того, с чем ставят, а
        # против того, кто ответит на нашу ставку: диапазоны разные, и путать
        # их нельзя. Шансов банка тут нет вовсе — доплачивать нечего.
        odds = 0.0
        chance = equity_vs_range(
            hole, board,
            ranges=table_ranges(
                ranked, opponents=opponents, keep=CALLER_KEEP,
                bluff=CALLER_BLUFF, bettor=False,
            ),
            trials=trials, seed=seed,
        )
        action = "бет" if chance >= worth_betting else "чек"
    # Размер и кнопка считаются только тогда, когда ставить и правда
    # собираемся: выбор между четырьмя кнопками — перебор на каждую из них, а
    # чеку, коллу и фолду он не нужен вовсе. Раньше он считался всегда, и
    # дороже всего обходился там, где выбрасывался.
    betting = action in ("бет", "рейз")
    suggested = bet_size(
        pot, to_call, my_bet=my_bet, stack=stack, minimum=min_bet,
    ) if betting and pot + to_call > 0 else 0
    button = ""
    if betting and can_raise:
        chosen = choose_quick_bet(
            hole, board,
            quick_bets(
                pot=pot, to_call=to_call, my_bet=my_bet, stack=stack,
                big_blind=big_blind, minimum=min_bet,
            ),
            ranked=ranked, pot=pot, to_call=to_call, my_bet=my_bet,
            opponents=opponents, seed=seed,
        )
        if chosen is not None:
            suggested, button = chosen.amount, chosen.name
    shoved = stack is not None and suggested >= my_bet + stack > 0
    made = category_name(list(hole) + list(board))

    if to_call <= 0:
        if action == "бет":
            return Advice(
                "бет", chance, odds, suggested,
                f"эквити {chance:.0%} против тех, кто ответит на ставку, "
                f"при {worth_betting:.0%} у средней руки — стоит ставить самому",
                made, shoved, button,
                raise_plan(
                    hole, board, ranked=ranked, opponents=opponents, pot=pot,
                    bet_to=suggested, my_bet=my_bet, stack=stack, seed=seed,
                ),
            )
        return Advice(
            "чек", chance, odds, 0,
            f"эквити {chance:.0%} против отвечающих — на ставку не хватает, "
            "а доплачивать нечего: смотрим следующую карту бесплатно",
            made,
            plan=check_plan(
                hole, board, ranked=ranked, opponents=opponents, pot=pot,
                stack=stack, seed=seed,
            ),
        )

    if action == "рейз":
        return Advice(
            "рейз", chance, odds, suggested,
            f"эквити {chance:.0%} против его диапазона при шансах банка "
            f"{odds:.0%} — колла мало, играем на победу",
            made, shoved, button,
            raise_plan(
                hole, board, ranked=ranked, opponents=opponents, pot=pot,
                bet_to=suggested, to_call=to_call, my_bet=my_bet, stack=stack,
                seed=seed,
            ),
        )
    if action == "колл":
        return Advice(
            "колл", chance, odds, 0,
            f"эквити {chance:.0%} против его диапазона выше шансов банка "
            f"{odds:.0%} — доплата окупается",
            made,
        )
    if chance >= odds:
        # Разница есть, но она меньше запаса на неточность перебора и на
        # ошибку распознавания. Писать «ниже», когда число выше, нельзя:
        # игрок сверяет совет с теми же процентами на экране.
        return Advice(
            "фолд", chance, odds, 0,
            f"эквити {chance:.0%} вровень с шансами банка {odds:.0%} — "
            "доплата не окупается",
            made,
        )
    return Advice(
        "фолд", chance, odds, 0,
        f"эквити {chance:.0%} против его диапазона ниже шансов банка "
        f"{odds:.0%} — доплата не окупается",
        made,
    )
