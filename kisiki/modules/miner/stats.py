"""Счёт добытой руды: сколько взято за сеанс и сколько за день.

Экрана и окон здесь нет, и это не педантизм: счёт руды — единственная часть
модуля, у которой есть последствия за пределами одной сессии. Он переживает
закрытие приложения, переезжает в ``progress.json`` и обязан вести себя
предсказуемо на битом и старом файле. Такую вещь надо проверять тестами, а не
глазами по экрану, поэтому она отдельным файлом и без CustomTkinter.

Два счётчика, а не один. **Сеанс** живёт от открытия экрана до его закрытия и
отвечает на вопрос «идёт ли дело прямо сейчас». **День** копится через все
запуски до полуночи и отвечает на вопрос «сколько сегодня накопал». Первый без
второго обнуляется на каждый запуск, второй без первого не показывает, что
происходит в эту минуту.

Отдельной строкой идёт «нераспознано»: игра руду выдала, а её вид прочитать не
удалось. Валить такие в общий счёт по видам нельзя — вышло бы, что кремниевой
добыто больше, чем на самом деле. Не считать вовсе тоже нельзя: камень отбит и
руда получена. Поэтому в общем итоге она есть, а в разбивке по видам её нет, и
экран пишет об этом прямо.
"""

from __future__ import annotations

from collections.abc import Callable

from ...clicker import today_key
from .vision import ORE_KEYS, ORE_NAMES


def fresh_daily_stats(day: str | None = None) -> dict:
    """Пустой дневной счёт — им же чинится битый или устаревший."""
    return {
        "date": day or today_key(),
        "total": 0,
        "unknown": 0,
        "ores": {key: 0 for key in ORE_KEYS},
    }


class OreTally:
    """Счёт руды за сеанс и за день поверх словаря из ``progress.json``.

    Словарь дневного счёта передаётся снаружи и правится на месте: это тот же
    объект, который лежит в сохранении игры, и сохранение не должно узнавать о
    том, как устроен экран шахтёра.
    """

    def __init__(
        self,
        daily: dict | None = None,
        *,
        clock: Callable[[], str] = today_key,
    ) -> None:
        self.clock = clock
        self.daily = daily if isinstance(daily, dict) else {}
        self.session = {key: 0 for key in ORE_KEYS}
        self.session_unknown = 0
        self.session_total = 0
        self.refresh_day()

    def refresh_day(self) -> bool:
        """Привести дневной счёт в порядок; вернуть, начался ли новый день.

        Здесь же чинится и чужое содержимое. Файл прогресса живёт долго, его
        правят руками и он переживает обновления: любое поле может оказаться
        строкой, отрицательным числом или пропасть вовсе. Экран из-за этого
        падать не должен — счёт не те данные, ради которых стоит терять
        запущенную игру.
        """
        started_new_day = self.daily.get("date") != self.clock()
        ores = self.daily.get("ores")
        if started_new_day or not isinstance(ores, dict):
            self.daily.clear()
            self.daily.update(fresh_daily_stats(self.clock()))
            return started_new_day
        for key in ORE_KEYS:
            ores[key] = _count(ores.get(key))
        for extra in [key for key in ores if key not in ORE_KEYS]:
            del ores[extra]
        self.daily["total"] = _count(self.daily.get("total"))
        self.daily["unknown"] = _count(self.daily.get("unknown"))
        return False

    def record(self, ore_key: str | None) -> str:
        """Записать одну добытую руду и вернуть строку для экрана."""
        self.refresh_day()
        self.session_total += 1
        self.daily["total"] += 1
        if ore_key in ORE_NAMES:
            self.session[ore_key] += 1
            self.daily["ores"][ore_key] += 1
            return f"{ORE_NAMES[ore_key]} руда записана в статистику."
        self.session_unknown += 1
        self.daily["unknown"] += 1
        return "Руда добыта, но её вид не распознан — записана в общий счёт."

    @property
    def daily_total(self) -> int:
        return _count(self.daily.get("total"))

    @property
    def daily_unknown(self) -> int:
        return _count(self.daily.get("unknown"))

    def daily_count(self, ore_key: str) -> int:
        ores = self.daily.get("ores")
        return _count(ores.get(ore_key)) if isinstance(ores, dict) else 0

    def session_count(self, ore_key: str) -> int:
        return self.session.get(ore_key, 0)

    def leaders(self, limit: int = 3) -> list[tuple[str, int]]:
        """Что за сеанс идёт лучше всего — для короткой строки в шапке."""
        found = [
            (key, count) for key, count in self.session.items() if count > 0
        ]
        found.sort(key=lambda pair: (-pair[1], ORE_KEYS.index(pair[0])))
        return found[:limit]


def _count(value: object) -> int:
    """Прочитать счётчик из сохранения, чем бы он там ни оказался."""
    try:
        return max(0, int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
