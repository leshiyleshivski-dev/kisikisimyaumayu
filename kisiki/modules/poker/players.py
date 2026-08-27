"""Кто за столом: отпечатки имён из строки событий и их узнавание.

Ходы журнал пишет с первого запуска, а кто их сделал — не знал: отпечаток
имени зрение снимает с каждой строки и тем же кадром выбрасывало. Без него нет
ни персональной статистики, ни даже данных под неё — всё остальное в журнале
копится само, а это не копилось вовсе.

Имя при этом не читается. Разбирать буквы переменной ширины — отдельный
большой проект, а нужно от имени одно: узнавать, что это тот же человек, что в
раздаче номер сорок семь. Отпечаток — картинка малиновых букв, приведённая к
общему размеру, и сравнивается он совпадением площадей, как эталоны карт.

Порог взят не с потолка, а из двух записей.

На `7.mp4` (188 секунд, 2 266 строк с именем) попарное совпадение отпечатков
разложилось надвое с пустой полосой посередине: чужие имена дают 0,20–0,50,
свои — 0,60–1,00, а между 0,50 и 0,60 из 79 800 пар нет **ни одной**. Порог
0,55 стоит ровно в этой пустоте. Узналось на записи четыре имени, и каждое
живёт своим куском времени: первые двадцать четыре секунды играет одна пара,
дальше другая — за столом сменился состав.

На `8.mp4` (47 минут, 19 022 строки, стол на шестерых) полоса не пустая, но
это её самое пустое место: в 0,50–0,55 попадает 36 пар из 61 075 — шесть
сотых процента. Узнаётся шесть крупных имён ровно по числу мест, плюс горстка
мелких — те, кто присел на пару раздач.

Ошибиться можно в обе стороны, и они не равны. Высокий порог дробит человека
на мнимых игроков: плашка журнала то светлеет, то гаснет, и одно имя даёт
немного разные картинки — на 0,75 вместо шести имён выходит двадцать. Низкий
склеивает разных людей, и статистика по ним превращается в ложь. Пустая полоса
тем и хороша, что позволяет не выбирать между этими бедами.

Хранятся отпечатки своим файлом рядом с журналом и по одному на человека, а не
по одному на ход: маска весит четверть килобайта, а ходов за вечер тысяча —
класть её в каждый ход значило бы раздуть журнал впятеро. В ходе стоит номер.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .vision import NAME_HEIGHT, NAME_WIDTH, similarity

# Совпадение площадей, ниже которого имена считаются разными. Обоснование —
# в шапке модуля: столько же, сколько у знаков карт, и по той же причине.
NAME_MATCH_FLOOR = 0.55

# Больше этого числа имён за сеанс не запоминаем. За столом шесть мест, люди
# приходят и уходят, но список, растущий без края, означал бы не полный стол,
# а рассыпавшееся узнавание — и каждый новый ход сравнивался бы с тысячей
# картинок на пятнадцатисекундном таймере.
ROSTER_LIMIT = 200


def players_word(count: int) -> str:
    """«от 2 игроков», но «от 1 игрока» — строку читают мельком, рядом с числом."""
    if 11 <= count % 100 <= 14:
        return "игроков"
    return "игрока" if count % 10 == 1 else "игроков"


@dataclass
class Roster:
    """Отпечатки встреченных имён; номер игрока — это место в списке.

    Список только растёт: ушедший из-за стола человек может вернуться, и
    узнать его тогда важнее, чем сэкономить строчку в файле.
    """

    prints: list[np.ndarray] = field(default_factory=list)
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.prints)

    def number_of(self, mask: np.ndarray | None) -> int | None:
        """Номер игрока по отпечатку; ``None`` — отпечатка на строке нет.

        Не нашли — заводим нового и запоминаем на диск: список этот и есть всё
        узнавание, и потерять его значит начать счёт заново.
        """
        if mask is None:
            return None
        for number, known in enumerate(self.prints):
            if similarity(mask, known) >= NAME_MATCH_FLOOR:
                return number
        if len(self.prints) >= ROSTER_LIMIT:
            return None
        self.prints.append(mask)
        self.save()
        return len(self.prints) - 1

    # --- диск ---

    def save(self) -> None:
        """Записать отпечатки; молча пережить любую беду с диском.

        Список имён — статистика, а не прогресс игрока: потерять его обидно, но
        уронить из-за него экран посреди раздачи куда хуже.
        """
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([packed(mask) for mask in self.prints]), encoding="utf-8"
            )
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path | None) -> "Roster":
        """Прочитать отпечатки; сломанный файл — это пустой список, а не падение."""
        roster = cls(path=path)
        if path is None:
            return roster
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return roster
        if not isinstance(saved, list):
            return roster
        for item in saved[:ROSTER_LIMIT]:
            mask = unpacked(item)
            if mask is not None:
                roster.prints.append(mask)
        return roster


def packed(mask: np.ndarray) -> str:
    """Маска в строку: по биту на точку, дальше base64.

    Восемь точек в байте — иначе одна маска занимает две тысячи символов, а их
    в файле по одной на человека и все читаются целиком при запуске.
    """
    return base64.b64encode(np.packbits(mask).tobytes()).decode("ascii")


def unpacked(item) -> np.ndarray | None:
    """Строка обратно в маску; чужой мусор в список имён не пускаем."""
    if not isinstance(item, str):
        return None
    try:
        raw = np.frombuffer(base64.b64decode(item, validate=True), dtype=np.uint8)
    except (ValueError, TypeError):
        return None
    bits = np.unpackbits(raw)
    if bits.size < NAME_HEIGHT * NAME_WIDTH:
        return None
    return bits[:NAME_HEIGHT * NAME_WIDTH].reshape(NAME_HEIGHT, NAME_WIDTH).astype(bool)
