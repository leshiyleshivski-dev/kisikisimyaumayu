"""Что делать до флопа: таблица стартовых рук и режим «олл-ин или пас».

Почему это не считается эквити, как остальные улицы. До флопа перебор
сравнивает руку со случайными картами соперника и выдаёт 3♠9♦ сорок процентов
— выше шансов банка, и получается «колл». На записи ``6.mp4`` помощник так
принял двадцать префлоп-решений подряд и ни разу не сказал «фолд»: играл
9♠3♠, J♣6♦, 2♣9♥. Дальше каждую такую раздачу приходится доигрывать позади, и
проигрывается она не до флопа, а после.

Правильный ответ до флопа даёт не перебор, а таблица: какие руки открывать с
какого места. Она короткая, проверяемая и не зависит от того, что соперник
покажет потом.

Вторая половина модуля — короткий стек. Когда фишек меньше пятнадцати
блайндов, доигрывать нечем: постфлопа на такой глубине не существует, и
единственные ходы — олл-ин или пас. Там таблица уступает место прямому счёту
выгоды: сколько раз соперник спасует, сколько раз ответит и с чем.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import comb

from .poker_math import (
    equity_vs_range, quick_bet_for, pot_odds, quick_bets, round_bet,
)
from .poker_ranges import hand_class, parse_range, range_share, top_share

# Места за столом в порядке хода до флопа. Первым говорит тот, кто сидит
# после большого блайнда, последним — сам большой блайнд.
POSITIONS = ("UTG", "MP", "CO", "BTN", "SB", "BB")

# Готовые обороты целиком с предлогом: «со средней позиции», но «с кнопки».
POSITION_NAMES = {
    "UTG": "с ранней позиции",
    "MP": "со средней позиции",
    "CO": "с предпоследнего места",
    "BTN": "с кнопки",
    "SB": "с малого блайнда",
    "BB": "с большого блайнда",
}

# --- глубокий стек: таблица открытия ---

# Чем позже говоришь, тем шире можно играть: после тебя осталось меньше людей,
# которые могут проснуться с сильной рукой, и после флопа ты ходишь последним.
# Большой блайнд не открывает никогда — он закрывает торговлю, а не начинает.
OPEN_RANGES = {
    "UTG": parse_range("77+, ATs+, KTs+, QTs+, JTs, T9s, AJo+, KQo"),
    "MP": parse_range("66+, A9s+, A5s-A2s, K9s+, Q9s+, J9s+, T9s, 98s, ATo+, KJo+"),
    "CO": parse_range(
        "44+, A2s+, K7s+, Q8s+, J8s+, T8s, 97s, 87s, 76s, 65s, A9o+, KTo+, QJo"
    ),
    "BTN": parse_range(
        "22+, A2s+, K2s+, Q5s+, J7s+, T7s, 96s, 86s, 75s, 65s, 54s, "
        "A2o+, K8o+, Q9o+, J9o+, T9o"
    ),
    "SB": parse_range(
        "22+, A2s+, K5s+, Q7s+, J8s+, T8s, 97s, 87s, 76s, 65s, "
        "A2o+, K9o+, QTo+, JTo"
    ),
    "BB": frozenset(),
}

# Из большого блайнда открывать и правда нечего — торговлю он закрывает, а не
# начинает. Но когда до него доходит компания лимперов, ход появляется: они
# вошли задёшево и с чем попало, а повышение забирает банк ещё до флопа. Без
# этой строки таблица отвечала «чек» на тузов против четверых лимперов —
# формально верно («открывать не стоит»), а по делу отдавала лучшую руку в
# многосторонний флоп даром.
BB_ISOLATE = parse_range("55+, A9s+, KTs+, QTs+, JTs, T9s, ATo+, KQo")

# Ответ на чужое повышение. Против соперников, которые уравнивают почти всё,
# блефовать повышением невыгодно — поднимаем ради денег, а не ради их паса.
THREE_BET_RANGES = {
    "UTG": parse_range("QQ+, AKs, AKo"),
    "MP": parse_range("QQ+, AKs, AKo"),
    "CO": parse_range("TT+, AQs+, AKo"),
    "BTN": parse_range("TT+, AQs+, AKo"),
    "SB": parse_range("TT+, AQs+, AKo"),
    "BB": parse_range("JJ+, AQs+, AKo"),
}

# Уравнивать — то, что не повышаем и не сбрасываем, поэтому с диапазоном
# повышения эти списки не пересекаются: у руки один ответ, а не два. Иначе
# исход решал бы порядок проверок в коде, а не таблица.
CALL_RANGES = {
    "UTG": parse_range("77-JJ, AQs, AJs, KQs"),
    "MP": parse_range("55-JJ, AQs-ATs, KJs+, QJs, JTs, AQo"),
    "CO": parse_range("22-99, AJs-A9s, KTs+, QTs+, JTs, T9s, AQo, AJo"),
    "BTN": parse_range(
        "22-99, AJs-A8s, A5s-A2s, K9s+, Q9s+, J9s+, T9s, 98s, AQo, AJo, KQo"
    ),
    "SB": parse_range("22-99, AJs-ATs, KJs+, QJs, JTs, AQo"),
    "BB": parse_range(
        "22-TT, AJs-A2s, K7s+, Q9s+, J9s+, T9s, 98s, 87s, AQo-ATo, KTo+, QJo, JTo"
    ),
}

# Стол на двоих — отдельная книжка, и разница огромная. За полным столом после
# малого блайнда говорят ещё четверо, и открывать треть рук уже смело. Вдвоём
# после него говорит один человек, и та же треть — это выброшенные деньги:
# половину раздач соперник просто отдаст, не заплатив ничего.
HEADS_UP_OPEN = parse_range(
    "22+, A2s+, K2s+, Q2s+, J2s+, T2s+, 95s+, 85s+, 74s+, 63s+, 53s+, 43s, "
    "A2o+, K2o+, Q4o+, J6o+, T6o+, 96o+, 86o+, 76o"
)
HEADS_UP_THREE_BET = parse_range("99+, ATs+, KQs, AJo+")
HEADS_UP_CALL = parse_range(
    "22-88, A9s-A2s, K2s-KJs, Q4s+, J6s+, T6s+, 96s+, 85s+, 75s+, 64s+, 54s, "
    "A2o-ATo, K5o+, Q8o+, J8o+, T8o+, 98o"
)

# Открывать повышением в два с половиной блайнда, добавляя блайнд за каждого,
# кто уже влез в раздачу до нас. Ответное повышение — втрое от чужой ставки.
OPEN_RAISE_BB = 2.5
LIMPER_EXTRA_BB = 1.0
THREE_BET_MULTIPLIER = 3.0

# --- короткий стек ---

# Ниже этой глубины таблица отключается: доигрывать раздачу нечем.
PUSH_FOLD_BB = 15.0

# С какой долей своих рук соперник отвечает на олл-ин. Равновесие даёт около
# пятой части; за столами казино отвечают заметно шире, поэтому берём с
# запасом — переоценить чужой пас дороже, чем недооценить.
SHOVE_CALL_SHARE = 0.25

# С какой долей рук соперник сам идёт ва-банк на коротком стеке. Нужно, когда
# ва-банк идёт он, а решать приходится нам.
SHOVE_PUSH_SHARE = 0.30

# С чем идти ва-банк первым. Ключ — сколько живых соперников за столом и на
# сколько блайндов хватает фишек.
#
# Почему это таблица, а не тот же счёт выгоды, что в ``shove_ev``. Счёт исходит
# из того, что соперник отвечает четвертью рук, — и тогда олл-ин с 92o выходит
# выгоднее паса на полторы сотни фишек, потому что три раза из четырёх все
# спасуют. Но отвечающий диапазон не стоит на месте: против того, кто пихает
# всё подряд, отвечают вдвое шире, и 92o перестаёт окупаться. Свести это в
# равновесие на лету нельзя — оно считается заранее, и результат давно
# посчитан за нас. ``shove_ev`` остаётся для объяснения, а не для решения.
PUSH_RANGES = {
    1: {
        7: parse_range(
            "22+, A2s+, A2o+, K2s+, K2o+, Q2s+, Q5o+, J4s+, J7o+, T6s+, T8o+, "
            "96s+, 98o, 85s+, 75s+, 64s+, 54s"
        ),
        12: parse_range(
            "22+, A2s+, A2o+, K2s+, K5o+, Q4s+, Q8o+, J6s+, J9o+, T7s+, T9o, "
            "97s+, 87s, 76s, 65s"
        ),
        15: parse_range(
            "22+, A2s+, A5o+, K7s+, K9o+, Q9s+, QTo+, J9s+, JTo, T9s, 98s"
        ),
    },
    2: {
        7: parse_range(
            "22+, A2s+, A2o+, K2s+, K7o+, Q5s+, Q9o+, J7s+, J9o+, T7s+, T9o, "
            "97s+, 87s, 76s"
        ),
        12: parse_range(
            "22+, A2s+, A7o+, K5s+, K9o+, Q8s+, QTo+, J8s+, JTo, T8s+, 98s, 87s"
        ),
        15: parse_range("22+, A2s+, A9o+, K9s+, KJo+, QTs+, QJo, JTs"),
    },
    3: {
        7: parse_range(
            "22+, A2s+, A5o+, K7s+, K9o+, Q8s+, QTo+, J8s+, JTo, T8s, 98s, 87s"
        ),
        12: parse_range("22+, A2s+, A9o+, K9s+, KTo+, Q9s+, QJo, J9s+, T9s"),
        15: parse_range("33+, A5s+, ATo+, K9s+, KJo+, QTs+, JTs"),
    },
    4: {
        7: parse_range("22+, A2s+, A8o+, K9s+, KTo+, Q9s+, QJo, J9s+, T9s"),
        12: parse_range("22+, A4s+, ATo+, K9s+, KJo+, QTs+, JTs"),
        15: parse_range("55+, A7s+, AJo+, KTs+, KQo, QJs"),
    },
}


def push_range(opponents: int, depth: float) -> frozenset[str]:
    """С чем идти ва-банк при таком числе соперников и такой глубине.

    Соперников больше, чем в таблице, — берём самый тесный столбец: лишний
    человек за столом руку только ослабляет. Глубина между ступенями округляется
    вверх, к более тесному диапазону.
    """
    column = PUSH_RANGES[min(max(opponents, 1), max(PUSH_RANGES))]
    for edge in sorted(column):
        if depth <= edge:
            return column[edge]
    return column[max(column)]


@dataclass(frozen=True)
class PreflopAdvice:
    """Совет до флопа и числа, на которых он держится."""

    action: str
    raise_to: int
    reason: str
    hand: str
    mode: str
    equity: float | None = None
    odds: float | None = None
    all_in: bool = False
    # Какую из четырёх кнопок быстрого размера нажать; пусто — ни одна не
    # ставит нужную сумму, и размер придётся вести ползунком.
    button: str = ""
    # Что делать, если соперник повысит в ответ на наше повышение. Пусто —
    # отвечать будет уже не он, а мы.
    plan: str = ""


def shove_ev(
    hole,
    *,
    opponents: int,
    pot: int,
    stack: int,
    call_share: float = SHOVE_CALL_SHARE,
    trials: int = 8_000,
    seed: int = 0,
) -> float:
    """На сколько фишек олл-ин выгоднее паса; меньше нуля — значит хуже.

    Складывается из двух половин: сколько раз все спасуют и банк достанется
    даром, и сколько раз кто-то ответит — тогда считаем эквити уже против его
    отвечающего диапазона, а не против случайных карт. Свой блайнд, который
    уже лежит в банке, обратно не просим: он потерян в обоих случаях и на
    сравнение не влияет.
    """
    callers = top_share(call_share)
    share = range_share(callers)
    chance = equity_vs_range(hole, ranges=[callers], trials=trials, seed=seed)
    seats = max(1, opponents)
    value = 0.0
    for answered in range(seats + 1):
        odds_of = (
            comb(seats, answered)
            * share ** answered
            * (1 - share) ** (seats - answered)
        )
        if answered == 0:
            value += odds_of * pot
            continue
        # Побить нужно каждого, кто ответил. Точная многосторонняя доля тут
        # не считается — перемножения хватает, чтобы лишний соперник делал
        # раздачу хуже, а не лучше. Иначе выходила бы чепуха: чем больше
        # народу за столом, тем выгоднее олл-ин с любой рукой.
        won = chance ** answered
        value += odds_of * (won * (pot + (answered + 1) * stack) - stack)
    return value


def blinds_word(count: int) -> str:
    """«9 блайндов», но «4 блайнда» и «21 блайнд».

    Причина строки — совет читают в спешке, за пятнадцать секунд на ход, и
    «4 блайндов» цепляет глаз ровно там, где он нужен цифрам.
    """
    if 11 <= count % 100 <= 14:
        return "блайндов"
    last = count % 10
    if last == 1:
        return "блайнд"
    if last in (2, 3, 4):
        return "блайнда"
    return "блайндов"


def _position_or_default(position: str) -> str:
    return position if position in OPEN_RANGES else "MP"


def raise_reply(name: str, position: str, *, heads_up: bool = False) -> str:
    """Что делать, если на наше повышение соперник ответит своим.

    Считать нечего: тем же таблицам, по которым выбран ход, известен и ответ
    на чужое повышение. Поэтому план тут не догадка, а ровно то, что экран
    напишет через кадр, — размер чужого повышения на ответ не влияет.

    До флопа размера в плане и не будет: таблица отвечает на само повышение, а
    не на его величину. После флопа размер решает всё, и там план называет
    границу — ``plan_line`` в ``poker_postflop.py``.
    """
    three_bets = HEADS_UP_THREE_BET if heads_up else THREE_BET_RANGES[position]
    calls = HEADS_UP_CALL if heads_up else CALL_RANGES[position]
    if name in three_bets:
        return "если повысит — повышаем в ответ"
    if name in calls:
        return "если повысит — колл"
    return "если повысит — фолд"


def _chart_advice(
    name: str, position: str, *, raised: bool, to_call: int,
    big_blind: int, highest: int, limpers: int, my_bet: int, stack: int,
    heads_up: bool = False, can_raise: bool = True,
) -> PreflopAdvice:
    """Ход по таблице: глубина позволяет доигрывать раздачу после флопа."""
    where = "за столом на двоих" if heads_up else POSITION_NAMES.get(position, position)
    # Большой блайнд против лимперов играет по своей книжке: открывать ему
    # нечего, а вот отобрать банк у зашедших задёшево — есть чем.
    isolating = not heads_up and position == "BB" and limpers > 0
    opens = HEADS_UP_OPEN if heads_up else OPEN_RANGES[position]
    if isolating:
        opens = BB_ISOLATE
    three_bets = HEADS_UP_THREE_BET if heads_up else THREE_BET_RANGES[position]
    calls = HEADS_UP_CALL if heads_up else CALL_RANGES[position]
    if raised:
        if name in three_bets and not can_raise:
            # Доплата больше стека: кнопки повышения на экране нет вовсе, и
            # весь стек уходит в банк простым ответом.
            return PreflopAdvice(
                "колл", 0,
                f"{name} стоит любых денег, но повышать нечем — "
                "доплата и так больше стека",
                name, "таблица", all_in=True,
            )
        if name in three_bets:
            target = min(my_bet + stack, round_bet(highest * THREE_BET_MULTIPLIER))
            return PreflopAdvice(
                "рейз", target,
                f"{name} — рука, ради которой стоит повышать самому: "
                "с ней платят руки слабее",
                name, "таблица", all_in=target >= my_bet + stack,
                plan=raise_reply(name, position, heads_up=heads_up),
            )
        if name in calls:
            return PreflopAdvice(
                "колл", 0,
                f"{name} {where} стоит доплаты, но повышать ей нечего",
                name, "таблица",
            )
        return PreflopAdvice(
            "фолд", 0,
            f"{name} {where} против повышения не играется. "
            "Это и есть та рука, на которой теряют весь вечер",
            name, "таблица",
        )
    if name in opens:
        target = min(
            my_bet + stack,
            round_bet(big_blind * (OPEN_RAISE_BB + LIMPER_EXTRA_BB * limpers)),
        )
        reason = (
            f"{name} {where} повышают против лимперов: они зашли задёшево и с "
            "чем попало, а банк можно забрать ещё до флопа"
            if isolating else
            f"{name} {where} открывают повышением: заходить уравниванием "
            "этой рукой — отдавать инициативу даром"
        )
        return PreflopAdvice(
            "рейз", target, reason,
            name, "таблица", all_in=target >= my_bet + stack,
            plan=raise_reply(name, position, heads_up=heads_up),
        )
    if to_call <= 0:
        return PreflopAdvice(
            "чек", 0,
            f"{name} открывать не стоит, но доплачивать нечего — смотрим флоп даром",
            name, "таблица",
        )
    return PreflopAdvice(
        "фолд", 0,
        f"{name} {where} не открывают. Сбросить сейчас дешевле, "
        "чем разбираться после флопа",
        name, "таблица",
    )


def _short_advice(
    hole, name: str, position: str, *, opponents: int, pot: int, stack: int,
    to_call: int, faced: int, my_bet: int, depth: float, trials: int, seed: int,
) -> PreflopAdvice:
    """Ход на коротком стеке: только олл-ин или пас."""
    if to_call >= stack:
        # Ва-банк идёт соперник, и решать приходится нам: считаем эквити
        # против того, с чем на такой глубине ходят ва-банк.
        chance = equity_vs_range(
            hole, ranges=[top_share(SHOVE_PUSH_SHARE)], trials=trials, seed=seed
        )
        # Чужой излишек сверх нашего стека вернётся ему: разыгрывается только
        # та часть банка, которую мы в состоянии уравнять.
        odds = pot_odds(max(0, pot - (faced - stack)), stack)
        if chance >= odds:
            return PreflopAdvice(
                "колл", 0,
                f"{name} берёт {chance:.0%} против того, с чем идут ва-банк, "
                f"а доплата стоит {odds:.0%} банка — отвечаем",
                name, "пуш/фолд", chance, odds, all_in=True,
            )
        return PreflopAdvice(
            "фолд", 0,
            f"{name} берёт {chance:.0%} против его олл-ина, а доплата стоит "
            f"{odds:.0%} банка — не окупается",
            name, "пуш/фолд", chance, odds,
        )
    if name in push_range(opponents, depth):
        value = shove_ev(
            hole, opponents=opponents, pot=pot, stack=stack, trials=trials, seed=seed
        )
        return PreflopAdvice(
            "пуш", my_bet + stack,
            f"{depth:.0f} {blinds_word(round(depth))} — доигрывать нечем, "
            "играется только олл-ин. "
            f"{name} для этого годится: выгоднее паса примерно на {value:.0f} фишек",
            name, "пуш/фолд", all_in=True,
        )
    if to_call <= 0:
        return PreflopAdvice(
            "чек", 0,
            f"{depth:.0f} {blinds_word(round(depth))}, с {name} в олл-ин не идут — "
            "но доплачивать нечего, смотрим флоп даром",
            name, "пуш/фолд",
        )
    return PreflopAdvice(
        "фолд", 0,
        f"{depth:.0f} {blinds_word(round(depth))} — играется только олл-ин или пас, "
        f"а {name} на олл-ин не тянет",
        name, "пуш/фолд",
    )


def preflop_advice(
    hole,
    *,
    position: str,
    opponents: int,
    big_blind: int,
    stack: int,
    pot: int = 0,
    to_call: int = 0,
    faced_bet: int | None = None,
    my_bet: int = 0,
    min_bet: int | None = None,
    limpers: int = 0,
    can_raise: bool = True,
    trials: int = 8_000,
    seed: int = 0,
) -> PreflopAdvice:
    """Совет до флопа: пас, чек, колл, рейз или олл-ин.

    Режим выбирается по глубине стека в блайндах, а не по нашему желанию:
    таблица открытия предполагает, что раздачу будет чем доигрывать, и на
    девяти блайндах она советует не то.
    """
    name = hand_class(hole)
    position = _position_or_default(position)
    faced = to_call if faced_bet is None else max(faced_bet, to_call)
    highest = my_bet + to_call
    raised = big_blind > 0 and highest > big_blind
    depth = stack / big_blind if big_blind > 0 else None

    if depth is not None and depth <= PUSH_FOLD_BB:
        tip = _short_advice(
            hole, name, position, opponents=opponents, pot=pot, stack=stack,
            to_call=to_call, faced=faced, my_bet=my_bet, depth=depth,
            trials=trials, seed=seed,
        )
    else:
        tip = _chart_advice(
            name, position, raised=raised, to_call=to_call, big_blind=big_blind,
            highest=highest, limpers=limpers, my_bet=my_bet, stack=stack,
            heads_up=opponents == 1, can_raise=can_raise,
        )
    return with_quick_button(
        tip, pot=pot, to_call=to_call, my_bet=my_bet, stack=stack,
        big_blind=big_blind, minimum=min_bet,
    )


def with_quick_button(
    tip: PreflopAdvice, *, pot: int, to_call: int, my_bet: int, stack: int,
    big_blind: int | None, minimum: int | None,
) -> PreflopAdvice:
    """Дописать в совет кнопку, которая ставит нужный размер.

    До флопа размер берётся из таблицы, а не из перебора, поэтому и кнопка
    выбирается не выгодой, а близостью: открытие в два с половиной блайнда
    игра ставит кнопкой «3 BB», а олл-ин на коротком стеке — кнопкой
    «ALL IN». И то и другое одно нажатие вместо ползунка на
    пятнадцатисекундном таймере.
    """
    if not tip.raise_to:
        return tip
    button = quick_bet_for(
        quick_bets(
            pot=pot, to_call=to_call, my_bet=my_bet, stack=stack,
            big_blind=big_blind, minimum=minimum,
        ),
        tip.raise_to,
    )
    if button is None:
        return tip
    return replace(tip, raise_to=button.amount, button=button.name)
