"""Что журнал говорит о столе: как за ним отвечают на ставки.

Модель размера ставки держится на одном числе — какая доля соперников не
пасует, когда мы ставим столько-то банка. Взято оно из минимальной защиты
(`банк / (банк + ставка)`), то есть из теории, а не из-за этого стола: на
ставку в банк она ждёт ответа от половины диапазона. За сорок семь минут
записи `8.mp4` это дало двенадцать советов «весь стек» — единственная жалоба
на советы, которую не закрыла починка распознавания.

Здесь то же число считается из журнала. Строка событий пишет, кто что сделал и
на сколько: поставили столько-то в такой-то банк, ответили столько-то человек,
спасовали столько-то. Ничего не выдумывается — только складывается то, что
игра сама написала в левом нижнем углу.

Банк по ходам приходится восстанавливать: в самих ходах его нет. Блайнды
известны из раздачи, ставки и повышения приходят с суммами, а колл добавляет
столько же, сколько стоит на столе. Восстановление приблизительное, и
``pot_error()`` показывает насколько: сравнивает досчитанный банк с тем, что
зрение прочло с плашки. Пока эти два числа рядом, ступеням размеров можно
верить.

Префлоп и постфлоп считаются отдельно, и это не педантизм: до флопа открывают
в четыре-двадцать блайндов при банке в полторы ставки, и в долях банка такая
ставка выходит вдесятеро крупнее любой постфлопной. Сложить их вместе значит
померить одно, а починить другое. Улицу видно по самим ходам: «поставил»
бывает только там, где ставить ещё не начинали, а до флопа на столе уже лежит
большой блайнд — значит, первая же «ставка» или «чек» открывают постфлоп.

Окон модуль не создаёт и ввод не отправляет: это счёт по уже записанному.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median

from .poker_journal import Hand, Journal

# Ступени размера — те же, которыми модуль читает чужую агрессию: маленькая
# ставка, большая, олл-ин. Иначе замер не с чем было бы сравнивать.
SIZE_STEPS = ((0.5, "до половины банка"), (1.0, "до банка"), (None, "больше банка"))

# Ниже этого числа ответов ступень ничего не значит: десяток ходов покажет что
# угодно. Пишем такую ступень как «мало данных», а не как измеренную долю.
ENOUGH_ANSWERS = 20


@dataclass(frozen=True)
class Aggression:
    """Одна ставка из строки событий и что ей ответили за столом."""

    amount: int
    # Размер в долях банка, каким он был до ставки, — так же его меряет и
    # модель: `bet_share()` в `poker_postflop.py`.
    share: float
    preflop: bool = False
    called: int = 0
    folded: int = 0
    raised: int = 0

    @property
    def faced(self) -> int:
        """Сколько человек ответили на эту ставку хоть чем-нибудь."""
        return self.called + self.folded + self.raised

    @property
    def kept(self) -> int:
        """Сколько из них не спасовало. Повышение — тоже не пас."""
        return self.called + self.raised


@dataclass(frozen=True)
class HandWalk:
    """Раздача, пройденная по ходам: ставки и досчитанный банк."""

    bets: tuple[Aggression, ...] = ()
    pot: int = 0


def walk(hand: Hand) -> HandWalk:
    """Пройти раздачу по строке событий и собрать ставки с ответами.

    Ходы идут подряд, и каждый колл или фолд отвечает последней прозвучавшей
    ставке. Чек её закрывает: чекают, когда ставить не на что, — значит,
    улица уже другая. Блайнды ставками не считаются: это долг перед раздачей,
    а не решение, и пас против блайнда — обычный префлоп-фолд.
    """
    if not hand.big_blind:
        return HandWalk()
    pot = highest = blinds = 0
    preflop = True
    bets: list[Aggression] = []
    current: int | None = None
    for _at, action, amount in hand.events:
        if action == "начало":
            pot = highest = blinds = 0
            preflop, current = True, None
        elif action == "блайнд":
            blinds += 1
            stake = hand.big_blind // 2 if blinds == 1 else hand.big_blind
            pot, highest, current = pot + stake, max(highest, stake), None
        elif action in ("ставка", "рейз"):
            if action == "ставка":
                # Ставят там, где ставить ещё не начинали, а до флопа на столе
                # уже лежит большой блайнд. Значит, это новая улица.
                preflop, highest = False, 0
            if not amount or pot <= 0:
                current = None
                continue
            if current is not None:
                bets[current] = replace(bets[current], raised=bets[current].raised + 1)
            # Размер меряется прибавкой к тому, что уже стоит на столе: против
            # ставки в 1 000 повышение до 3 000 просит доплатить две тысячи, а
            # не три. В банк при этом кладут все три: у поднявшего своей ставки
            # на этой улице обычно ещё нет.
            bets.append(Aggression(
                amount=amount, share=(amount - highest) / pot, preflop=preflop
            ))
            current = len(bets) - 1
            pot, highest = pot + amount, amount
        elif action == "колл":
            pot += highest
            if current is not None:
                bets[current] = replace(bets[current], called=bets[current].called + 1)
        elif action == "фолд":
            if current is not None:
                bets[current] = replace(bets[current], folded=bets[current].folded + 1)
        elif action == "чек":
            preflop, highest, current = False, 0, None
        else:
            current = None
    return HandWalk(bets=tuple(bets), pot=pot)


def hand_aggression(hand: Hand) -> tuple[Aggression, ...]:
    """Ставки одной раздачи вместе с ответами на них."""
    return walk(hand).bets


def table_aggression(hands, *, preflop: bool | None = None) -> tuple[Aggression, ...]:
    """Ставки журнала, у которых есть хотя бы один ответ.

    Ставка, после которой никто ничего не сделал, ничего и не говорит: либо
    раздача на ней кончилась, либо строка событий показала не всё.
    """
    return tuple(
        bet for hand in hands for bet in hand_aggression(hand)
        if bet.faced and (preflop is None or bet.preflop == preflop)
    )


def step_of(share: float) -> str:
    """К какой ступени размера относится ставка."""
    for edge, name in SIZE_STEPS:
        if edge is None or share <= edge:
            return name
    return SIZE_STEPS[-1][1]


def keep_by_size(bets) -> dict[str, tuple[int, int]]:
    """Ступень размера → сколько не спасовало и сколько всего отвечало."""
    counted: dict[str, tuple[int, int]] = {name: (0, 0) for _edge, name in SIZE_STEPS}
    for bet in bets:
        kept, faced = counted[step_of(bet.share)]
        counted[step_of(bet.share)] = (kept + bet.kept, faced + bet.faced)
    return counted


def keep_share(bets) -> float | None:
    """Какая доля ответивших не спасовала; ``None`` — ответов слишком мало."""
    faced = sum(bet.faced for bet in bets)
    if faced < ENOUGH_ANSWERS:
        return None
    return sum(bet.kept for bet in bets) / faced


def won_amount(hand: Hand) -> int | None:
    """Сколько банка забрали в этой раздаче — по строке «выиграл N фишек»."""
    taken = [
        amount for _at, action, amount in hand.events
        if action == "выигрыш" and amount
    ]
    return sum(taken) if taken else None


def pot_error(hands) -> float | None:
    """Насколько досчитанный по ходам банк расходится с выигранным.

    Единственная проверка, что складывать ходы вообще имеет смысл: банк
    считается двумя путями — по ставкам и по строке «выиграл N фишек», — и оба
    берутся из той же строки событий. Сверяться с полем раздачи было бы
    неправильно: банк там прочитан с плашки своей раздачи, а ходы к раздаче
    привязаны по журналу, и лишний сдвиг на раздачу спутал бы одно с другим.
    """
    errors = [
        abs(walk(hand).pot - won_amount(hand)) / won_amount(hand)
        for hand in hands
        if hand.big_blind and hand.events and won_amount(hand)
    ]
    return median(errors) if errors else None


def answers_line(journal: Journal) -> str:
    """Одна строка для экрана: сколько за этим столом платят на ставку.

    Ступень попадает в строку, только когда ответов набралось достаточно, —
    иначе на экране стояло бы число, посчитанное по трём раздачам, и по нему
    же потом чинили бы модель.
    """
    bets = table_aggression(journal.hands, preflop=False)
    counted = keep_by_size(bets)
    ready = [
        f"{name} — {kept / faced:.0%}"
        for _edge, name in SIZE_STEPS
        for kept, faced in [counted[name]]
        if faced >= ENOUGH_ANSWERS
    ]
    if ready:
        return "платят на ставку: " + "  ·  ".join(ready)
    answers = sum(bet.faced for bet in bets)
    return f"ответов на ставки: {answers} — мало, чтобы мерить, сколько тут платят"


def answer_lines(journal: Journal) -> list[str]:
    """Строки замера: сколько за столом платят на ставку каждого размера."""
    bets = table_aggression(journal.hands, preflop=False)
    if not bets:
        return ["постфлоп-ставок в журнале пока нет — строка событий была свёрнута"]
    lines = []
    for _edge, name in SIZE_STEPS:
        kept, faced = keep_by_size(bets)[name]
        if faced < ENOUGH_ANSWERS:
            lines.append(f"{name}: {faced} ответов — мало, чтобы считать долю")
            continue
        lines.append(f"{name}: платят {kept / faced:.0%} ({kept} из {faced})")
    return lines
