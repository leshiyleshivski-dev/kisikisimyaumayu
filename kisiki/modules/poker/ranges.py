"""Стартовые руки холдема: 169 классов, их сила и запись диапазона строкой.

Отдельный модуль, потому что на этот список опираются сразу двое: префлоп-
таблица, которая решает, играть ли руку вообще, и оценка соперника, которой
нужно раздать ему не две случайные карты, а правдоподобный диапазон.

Класс руки — привычная покерная запись: ``"AA"`` (пара), ``"AKs"`` (одной
масти), ``"AKo"`` (разной). Мастей у класса ``s`` четыре, у ``o`` двенадцать,
у пары шесть — всего 1 326 сочетаний, и доля диапазона считается по ним, а не
по числу классов: иначе одна пара весила бы столько же, сколько дюжина
разномастных рук.

Сила взята перебором: 60 000 раздач каждого класса против случайной руки на
полном борде. Сходится с известными числами (AA — 85 %, AKs — 67 %, 32o —
32 %), проверяется тестом. Порядок запечён константой, потому что считать его
на старте приложения — это лишние секунды, а меняться ему незачем.
"""

from __future__ import annotations

from .hand_math import RANKS, parse_cards

# От сильного к слабому: доля раздач, которую класс берёт у случайной руки.
# Пересчитывается скриптом из ``poker-plan/README.md``; руками не правится.
STRENGTH_ORDER: tuple[str, ...] = (
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "AKs",
    "AQs", "77", "AKo", "AJs", "ATs", "AQo", "AJo", "KQs",
    "66", "ATo", "A9s", "KJs", "A8s", "KTs", "KQo", "A9o",
    "A7s", "KJo", "55", "QJs", "A5s", "K9s", "A8o", "KTo",
    "A6s", "QTs", "A7o", "A4s", "K8s", "QJo", "A3s", "A5o",
    "K9o", "JTs", "K7s", "Q9s", "QTo", "A6o", "A2s", "44",
    "A4o", "K6s", "K8o", "A3o", "J9s", "Q8s", "K5s", "Q9o",
    "JTo", "K7o", "A2o", "K4s", "K6o", "Q7s", "J8s", "K3s",
    "T9s", "33", "Q6s", "J9o", "Q8o", "Q5s", "K2s", "K5o",
    "T8s", "K4o", "J7s", "Q4s", "Q7o", "T9o", "K3o", "Q3s",
    "J8o", "Q6o", "22", "K2o", "J6s", "T7s", "98s", "Q2s",
    "Q5o", "J5s", "J7o", "T8o", "97s", "T6s", "J4s", "Q4o",
    "J3s", "Q3o", "98o", "J6o", "T7o", "87s", "J2s", "Q2o",
    "96s", "T5s", "J5o", "J4o", "T4s", "86s", "97o", "T3s",
    "T6o", "95s", "76s", "J3o", "87o", "T2s", "85s", "96o",
    "94s", "T5o", "J2o", "T4o", "86o", "75s", "65s", "93s",
    "T3o", "95o", "92s", "84s", "76o", "74s", "T2o", "54s",
    "64s", "85o", "83s", "94o", "82s", "93o", "75o", "65o",
    "73s", "63s", "53s", "84o", "92o", "43s", "74o", "72s",
    "54o", "52s", "64o", "62s", "83o", "82o", "42s", "73o",
    "63o", "53o", "32s", "43o", "72o", "52o", "62o", "42o",
    "32o",
)

STRENGTH_INDEX = {name: place for place, name in enumerate(STRENGTH_ORDER)}

# Сколько сочетаний карт стоит за классом: пара — шесть, одномастная — четыре,
# разномастная — двенадцать.
KIND_COMBOS = {"p": 6, "s": 4, "o": 12}
TOTAL_COMBOS = 1326


def class_kind(name: str) -> str:
    """``"AA"`` → ``"p"``, ``"AKs"`` → ``"s"``, ``"AKo"`` → ``"o"``."""
    return "p" if len(name) == 2 else name[2]


def class_combos(name: str) -> int:
    return KIND_COMBOS[class_kind(name)]


def hand_class(cards) -> str:
    """Две карты → класс: ``["Ah", "Kd"]`` → ``"AKo"``.

    Принимает и строки, и коды карт — то же, что и остальная считалка.
    """
    codes = parse_cards(cards)
    if codes.size != 2:
        raise ValueError("У стартовой руки ровно две карты")
    first, second = sorted(int(code) for code in codes)
    high, low = max(first // 4, second // 4), min(first // 4, second // 4)
    if high == low:
        return RANKS[high] * 2
    suited = "s" if first % 4 == second % 4 else "o"
    return f"{RANKS[high]}{RANKS[low]}{suited}"


def combos_of(name: str) -> tuple[tuple[int, int], ...]:
    """Все сочетания карт класса — парами кодов 0..51.

    Нужно, чтобы раздать сопернику руку из диапазона: считалка эквити работает
    с картами, а не с названиями классов.
    """
    kind = class_kind(name)
    high = RANKS.index(name[0])
    low = RANKS.index(name[1])
    out: list[tuple[int, int]] = []
    if kind == "p":
        for first in range(4):
            for second in range(first + 1, 4):
                out.append((high * 4 + first, high * 4 + second))
    elif kind == "s":
        for suit in range(4):
            out.append((high * 4 + suit, low * 4 + suit))
    else:
        for first in range(4):
            for second in range(4):
                if first != second:
                    out.append((high * 4 + first, low * 4 + second))
    return tuple(out)


def range_share(names) -> float:
    """Какую долю всех раздач занимает диапазон — по сочетаниям, не по классам."""
    return sum(class_combos(name) for name in set(names)) / TOTAL_COMBOS


def top_share(share: float) -> frozenset[str]:
    """Сильнейшие руки, занимающие примерно такую долю всех раздач.

    Набираем по порядку силы, пока не наберётся нужная доля. Ровно в долю
    попасть нельзя — классы дискретны, — поэтому берём первый набор, который
    её достиг.
    """
    if share >= 1:
        return frozenset(STRENGTH_ORDER)
    picked: list[str] = []
    combos = 0
    for name in STRENGTH_ORDER:
        if combos / TOTAL_COMBOS >= share:
            break
        picked.append(name)
        combos += class_combos(name)
    return frozenset(picked)


# --- запись диапазона строкой ---

# Плюс всегда наращивает младшую карту до старшей: ``A2s+`` — все одномастные
# тузы, ``KTo+`` — KTo, KJo, KQo. Связки плюсом не записываются: ``65s+`` в
# покерных программах читают то как «все связки от 65s», то как «все
# одномастные шестёрки», и молча угадывать тут нечего. Связки пишем перечнем.
def _pair_span(low: str, high: str) -> list[str]:
    start, stop = RANKS.index(low), RANKS.index(high)
    return [RANKS[rank] * 2 for rank in range(min(start, stop), max(start, stop) + 1)]


def _kicker_span(high: str, first: str, second: str, suited: str) -> list[str]:
    top = RANKS.index(high)
    start, stop = RANKS.index(first), RANKS.index(second)
    return [
        f"{high}{RANKS[rank]}{suited}"
        for rank in range(min(start, stop), max(start, stop) + 1)
        if rank < top
    ]


def _parse_token(token: str) -> list[str]:
    token = token.strip()
    if not token:
        return []
    if "-" in token:
        left, right = (part.strip() for part in token.split("-", 1))
        if len(left) == 2 and left[0] == left[1]:
            return _pair_span(right[0], left[0])
        if left[0] != right[0] or left[2:] != right[2:]:
            raise ValueError(f"Не разобрал диапазон: {token}")
        return _kicker_span(left[0], left[1], right[1], left[2])
    if token.endswith("+"):
        body = token[:-1]
        if len(body) == 2 and body[0] == body[1]:
            return _pair_span(body[0], "A")
        return _kicker_span(body[0], body[1], RANKS[RANKS.index(body[0]) - 1], body[2])
    if len(token) == 2 and token[0] == token[1]:
        return [token]
    if len(token) == 3 and token[2] in "so" and token[0] != token[1]:
        return [token]
    raise ValueError(f"Не разобрал руку: {token}")


def parse_range(text: str) -> frozenset[str]:
    """``"77+, A2s+, KQo"`` → множество классов.

    Понимает пары (``77``, ``77+``, ``55-99``), одномастные и разномастные
    (``AKs``, ``KTo+``, ``A5s-A2s``). Неизвестная запись — ошибка, а не пустой
    диапазон: молча выкинуть половину таблицы хуже, чем упасть на тесте.
    """
    names: set[str] = set()
    for token in text.replace("\n", " ").split(","):
        for name in _parse_token(token):
            if name not in STRENGTH_INDEX:
                raise ValueError(f"Нет такого класса: {name}")
            names.add(name)
    return frozenset(names)
