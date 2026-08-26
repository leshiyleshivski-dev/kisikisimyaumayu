"""Разовая починка журнала покера: следы уже исправленных ошибок разбора.

Журнал пишется с первого запуска, а четыре ошибки разбора нашлись позже — и
всё, что они успели записать, так и лежит в файле:

* **раздачи-двойники.** Карты вскрытия начинали «новую раздачу», и следом за
  настоящей в журнал уходила пустышка с тем же бордом и чужой рукой;
* **блайнд не тот.** Ставка и ответное повышение после флопа читались как
  блайнды, и несколько раздач подряд записаны с 7 000 вместо 500 — а BB/100
  по ним занижен в четырнадцать раз;
* **результат из середины анимации.** Пока банк едет к победителю, панель
  показывает промежуточные числа, и в журнал попадали суммы вроде −43 879 и
  −232, которых за столом с фишкой в 250 просто не бывает.

Сами ошибки закрыты в `poker_vision.py` и `poker.py`, новые раздачи пишутся
уже чистыми. Этот скрипт чинит то, что записано до правок, и запускается
руками один раз:

    python clean_poker_journal.py

Рядом со старым файлом остаётся копия `poker_journal.bak.json`: журнал —
единственные данные, которых нет больше нигде.
"""

from __future__ import annotations

import shutil
from dataclasses import replace

from kisiki.core import data_path
from kisiki.modules.poker_journal import Hand, Journal

# Насколько близко к настоящей раздаче стоит её двойник. Рождался он из карт
# вскрытия, которые лежат на столе ещё секунду-другую после того, как банк
# уехал, — на записи `8.mp4` обе пары разошлись на две и четыре секунды.
TWIN_SECONDS = 30

# Сколько раздач подряд может держаться неверно прочитанный блайнд. Держится
# он, пока не попадётся чистое начало раздачи: на записи `8.mp4` — три
# раздачи. Смена стола выглядит иначе: новый блайнд остаётся до конца журнала,
# и такой отрезок трогать нельзя.
BLIND_RUN = 5


def is_twin(hand: Hand, previous: Hand) -> bool:
    """Раздача-двойник: та же самая, записанная второй раз.

    Приметы все сразу, потому что каждая по отдельности бывает и у честной
    раздачи: тот же борд до последней карты, ни позиции, ни единого хода — и
    записана в те же секунды. Настоящая раздача с тем же бордом означала бы,
    что колода легла второй раз в том же порядке.
    """
    return bool(
        hand.board
        and hand.board == previous.board
        and hand.position is None
        and not hand.events
        and abs(hand.played_at - previous.played_at) <= TWIN_SECONDS
    )


def drop_twins(hands: list[Hand]) -> list[Hand]:
    """Выбросить двойников, оставив настоящие раздачи."""
    kept: list[Hand] = []
    for hand in hands:
        if kept and is_twin(hand, kept[-1]):
            continue
        kept.append(hand)
    return kept


def blind_runs(hands: list[Hand]) -> list[tuple[int | None, int, int]]:
    """Отрезки подряд идущих раздач с одинаковым блайндом."""
    runs: list[list] = []
    for index, hand in enumerate(hands):
        if runs and runs[-1][0] == hand.big_blind:
            runs[-1][2] = index + 1
        else:
            runs.append([hand.big_blind, index, index + 1])
    return [(blind, start, stop) for blind, start, stop in runs]


def fix_blinds(hands: list[Hand]) -> list[Hand]:
    """Короткий чужой блайнд между одинаковыми соседями — ошибка чтения.

    За столом блайнд не меняется, и пересесть за другой стол посреди вечера
    можно только в одну сторону: новый размер остаётся до конца журнала.
    Поэтому чиним лишь отрезок, у которого слева и справа стоит один и тот же
    блайнд, — то есть тот, где стол был прежним, а прочиталось другое.
    """
    fixed = list(hands)
    runs = blind_runs(hands)
    for before, run, after in zip(runs, runs[1:], runs[2:]):
        blind, start, stop = run
        if not before[0] or before[0] != after[0] or blind == before[0]:
            continue
        if stop - start > BLIND_RUN:
            continue
        for index in range(start, stop):
            fixed[index] = replace(fixed[index], big_blind=before[0])
    return fixed


def round_results(hands: list[Hand]) -> list[Hand]:
    """Результат кратен фишке: за столом 250 / 500 полтинников не бывает.

    Некратное число значит одно — стек прочитан посреди перелёта фишек. Само
    число при этом почти верное, ошибка меньше фишки, поэтому округляем, а не
    выбрасываем: выбросить −43 879 значит стереть самую крупную потерю вечера
    и сделать BB/100 красивее, чем он был.
    """
    fixed = []
    for hand in hands:
        chip = (hand.big_blind or 0) // 2
        if hand.result is None or chip <= 0 or hand.result % chip == 0:
            fixed.append(hand)
            continue
        fixed.append(replace(hand, result=round(hand.result / chip) * chip))
    return fixed


def clean(hands: list[Hand]) -> list[Hand]:
    """Все три починки по порядку: сперва лишние раздачи, потом их числа."""
    return round_results(fix_blinds(drop_twins(list(hands))))


def summary(hands: list[Hand]) -> str:
    """Строка журнала так, как её видит экран: раздачи, фишки и BB/100."""
    journal = Journal(hands=list(hands))
    counted = journal.counted()
    rate = journal.bb_per_100()
    return (
        f"{len(hands)} раздач · в счёте {len(counted)} · "
        f"{journal.chips():+} фишек · "
        f"{'BB/100 не посчитать' if rate is None else f'{rate:+.0f} BB/100'}"
    )


def main() -> None:
    path = data_path("poker_journal.json")
    journal = Journal.load(path)
    if not journal.hands:
        print(f"Журнал пуст или не найден: {path}")
        return
    before = list(journal.hands)
    after = clean(before)
    print(f"Файл:  {path}")
    print(f"Было:  {summary(before)}")
    print(f"Стало: {summary(after)}")
    if after == before:
        print("Чинить нечего.")
        return
    backup = path.with_name("poker_journal.bak.json")
    shutil.copyfile(path, backup)
    journal.hands = after
    journal.save()
    print(f"Копия старого журнала: {backup}")


if __name__ == "__main__":
    main()
