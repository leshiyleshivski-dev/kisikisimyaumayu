"""Pure progression rules for the cat clicker.

This module deliberately has no Tkinter or GTA dependencies.  Keeping the
economy here makes save migrations and balance changes testable without
starting the desktop application.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import floor


@dataclass(frozen=True)
class Upgrade:
    title: str
    description: str
    base_price: int
    growth: float
    color: str
    kind: str


UPGRADES: dict[str, Upgrade] = {
    "paw": Upgrade("Мягкая лапка", "+1 за клик", 25, 1.55, "#D88470", "click"),
    "laser": Upgrade("Лазерная точка", "+3 за клик", 180, 1.62, "#AF86CA", "click"),
    "crown": Upgrade("Корона любимчика", "+35% силы клика", 850, 1.78, "#D3A947", "click"),
    "treat": Upgrade("Лакомство", "+1 мяу/сек", 70, 1.58, "#CFA05C", "auto"),
    "basket": Upgrade("Тёплая лежанка", "+6 мяу/сек", 520, 1.68, "#629C88", "auto"),
    "cafe": Upgrade("Кошачье кафе", "+25 мяу/сек", 3_200, 1.78, "#4E83A6", "auto"),
}


@dataclass(frozen=True)
class Decor:
    title: str
    description: str
    base_price: int
    growth: float
    color: str


DECOR: dict[str, Decor] = {
    "scratcher": Decor("Когтеточка", "+8% ко всем кликам", 1_500, 2.05, "#FF987C"),
    "lamp": Decor("Тёплая лампа", "+10% к автодоходу", 2_500, 2.08, "#F1C96C"),
    "window": Decor("Окно во двор", "+5% к офлайн-доходу", 4_000, 2.12, "#67C7D4"),
    "radio": Decor("Мур-радио", "+5 секунд Мур-режима", 6_500, 2.15, "#A994ED"),
}


@dataclass(frozen=True)
class Adventure:
    title: str
    description: str
    duration: int
    multiplier: float


ADVENTURES: dict[str, Adventure] = {
    "yard": Adventure("Во двор", "Быстрая прогулка за мелкой добычей", 2 * 60, 1.0),
    "roofs": Adventure("По крышам", "Средняя вылазка с хорошей наградой", 10 * 60, 1.2),
    "night": Adventure("Ночной рынок", "Долгое приключение с лучшей отдачей", 30 * 60, 1.45),
}


@dataclass(frozen=True)
class Achievement:
    key: str
    title: str
    description: str
    target: int
    reward: int
    metric: str


ACHIEVEMENTS = (
    Achievement("hello", "Первое мяу", "Нажать на котика", 1, 30, "taps"),
    Achievement("busy_paws", "Шустрые лапки", "Сделать 250 нажатий", 250, 350, "taps"),
    Achievement("combo_25", "Не отрывая лап", "Собрать комбо ×25", 25, 400, "combo"),
    Achievement("auto_10", "Мур-мотор", "Разогнать доход до 10/сек", 10, 500, "passive"),
    Achievement("upgrades_20", "Уютный дом", "Купить 20 улучшений", 20, 750, "upgrades"),
    Achievement("club_10", "Клубная легенда", "Достичь 10 уровня клуба", 10, 1_200, "club_level"),
    Achievement("four_friends", "Никого не забыли", "Набрать по 1 000 мяу каждым котиком", 1_000, 1_500, "all_cats"),
    Achievement("traveler", "Хвост трубой", "Завершить 10 вылазок", 10, 2_000, "adventures"),
    Achievement("designer", "Дом полной чашей", "Купить 8 улучшений Домика", 8, 3_000, "decor"),
    Achievement("million", "Миллион мяу", "Заработать 1 000 000 мяу за всё время", 1_000_000, 25_000, "lifetime"),
)


def fresh_cat() -> dict[str, int]:
    return {
        "meows": 0,
        "paw": 0,
        "treat": 0,
        "laser": 0,
        "crown": 0,
        "basket": 0,
        "cafe": 0,
        "taps": 0,
        "best_combo": 0,
    }


def decor_price(decor: dict[str, int], key: str) -> int:
    item = DECOR[key]
    return max(1, floor(item.base_price * (item.growth ** int(decor.get(key, 0)))))


def click_multiplier(decor: dict[str, int] | None = None) -> float:
    return 1.0 + 0.08 * int((decor or {}).get("scratcher", 0))


def passive_multiplier(decor: dict[str, int] | None = None) -> float:
    return 1.0 + 0.10 * int((decor or {}).get("lamp", 0))


def offline_rate(decor: dict[str, int] | None = None) -> float:
    return min(0.95, 0.75 + 0.05 * int((decor or {}).get("window", 0)))


def purr_duration(decor: dict[str, int] | None = None) -> int:
    return 20 + 5 * int((decor or {}).get("radio", 0))


def click_power(cat: dict[str, int], decor: dict[str, int] | None = None) -> int:
    flat = 1 + cat.get("paw", 0) + cat.get("laser", 0) * 3
    return max(1, floor(flat * (1.35 ** cat.get("crown", 0)) * click_multiplier(decor)))


def passive_income(cat: dict[str, int], decor: dict[str, int] | None = None) -> int:
    flat = cat.get("treat", 0) + cat.get("basket", 0) * 6 + cat.get("cafe", 0) * 25
    return floor(flat * passive_multiplier(decor))


def upgrade_price(cat: dict[str, int], key: str) -> int:
    upgrade = UPGRADES[key]
    return max(1, floor(upgrade.base_price * (upgrade.growth ** cat.get(key, 0))))


def level_progress(total: int) -> tuple[int, int, int]:
    """Return level, XP inside the level, and XP required for the next one."""
    total = max(0, int(total))
    level = 1
    remaining = total
    needed = 100
    while remaining >= needed:
        remaining -= needed
        level += 1
        needed = floor(100 * (level ** 1.32))
    return level, remaining, needed


def format_number(value: int) -> str:
    value = max(0, int(value))
    for divisor, suffix in ((1_000_000_000, " млрд"), (1_000_000, " млн"), (1_000, " тыс.")):
        if value >= divisor:
            compact = value / divisor
            digits = 0 if compact >= 100 else 1
            return f"{compact:.{digits}f}".replace(".0", "").replace(".", ",") + suffix
    return f"{value:,}".replace(",", " ")


def today_key(day: date | None = None) -> str:
    return (day or date.today()).isoformat()


def next_daily_streak(last_claim: str, current_streak: int, current: date | None = None) -> int:
    current = current or date.today()
    if not last_claim:
        return 1
    try:
        previous = date.fromisoformat(last_claim)
    except ValueError:
        return 1
    if previous == current:
        return max(0, current_streak)
    if previous == current - timedelta(days=1):
        return max(0, current_streak) + 1
    return 1


def daily_reward(streak: int) -> int:
    return 100 + min(max(1, streak), 7) * 75


def daily_quests(club_level: int) -> tuple[dict[str, int | str], ...]:
    scale = max(1, club_level)
    tap_target = 50 + min(150, scale * 10)
    earn_target = 250 + scale * 125
    return (
        {"key": "taps", "title": "Размять лапки", "description": f"Сделать {tap_target} нажатий", "target": tap_target, "reward": 120 + scale * 25, "metric": "taps"},
        {"key": "earn", "title": "Полная миска", "description": f"Заработать {format_number(earn_target)} мяу", "target": earn_target, "reward": 180 + scale * 35, "metric": "earned"},
        {"key": "shop", "title": "Навести уют", "description": "Купить 2 улучшения", "target": 2, "reward": 240 + scale * 40, "metric": "upgrades"},
    )


def offline_income(cats: list[dict[str, int]], elapsed_seconds: float, cap_hours: int = 8, decor: dict[str, int] | None = None) -> tuple[list[int], int]:
    """Calculate 75% offline production, capped to a reasonable absence."""
    seconds = max(0, min(int(elapsed_seconds), cap_hours * 3600))
    rewards = [floor(passive_income(cat, decor) * seconds * offline_rate(decor)) for cat in cats]
    return rewards, sum(rewards)


def adventure_reward(cat: dict[str, int], adventure_key: str, decor: dict[str, int] | None = None) -> int:
    adventure = ADVENTURES[adventure_key]
    earning_rate = click_power(cat, decor) * 2 + passive_income(cat, decor)
    return max(50, floor(adventure.duration * earning_rate * adventure.multiplier))


def achievement_metric(game: dict, metric: str) -> int:
    cats = game.get("cats", [])
    if metric == "taps":
        return sum(int(cat.get("taps", 0)) for cat in cats)
    if metric == "combo":
        return max((int(cat.get("best_combo", 0)) for cat in cats), default=0)
    if metric == "passive":
        return sum(passive_income(cat, game.get("decor")) for cat in cats)
    if metric == "upgrades":
        return sum(sum(int(cat.get(key, 0)) for key in UPGRADES) for cat in cats)
    if metric == "club_level":
        return level_progress(int(game.get("lifetime_meows", 0)))[0]
    if metric == "all_cats":
        return min((int(cat.get("meows", 0)) for cat in cats), default=0)
    if metric == "adventures":
        return int(game.get("adventures_completed", 0))
    if metric == "decor":
        return sum(int(level) for level in game.get("decor", {}).values())
    if metric == "lifetime":
        return int(game.get("lifetime_meows", 0))
    raise KeyError(metric)
