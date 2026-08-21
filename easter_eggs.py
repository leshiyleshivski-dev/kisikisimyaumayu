"""Скрытые комбинации для «Кисикисимяумяу».

Этот модуль намеренно не знает ничего об интерфейсе и GTA. Он получает
обычные события котика (нажали/отпустили) и вызывает ``on_unlock`` после
правильной комбинации. Поэтому новую пасхалку можно добавить здесь, не
трогая экономику кликера или экран выбора котиков.
"""

from __future__ import annotations

import time
from collections.abc import Callable


# Индексы котиков хранятся рядом с правилом, а не размазаны по приложению.
GRUMPY_CAT = 5
BONGO_CAT = 1
BUFF_CAT = 3

GRUMPY_HOLD_MS = 1250
BONGO_TAPS_REQUIRED = 10
BONGO_TAP_WINDOW_SECONDS = 4.0
BUFF_TAPS_PER_BEAT = 3
BUFF_TAP_SPACING_SECONDS = 0.55
BUFF_PAUSE_MIN_SECONDS = 0.70
BUFF_PAUSE_MAX_SECONDS = 1.80


class SecretRegistry:
    """Отслеживает скрытые комбинации и открывает зарегистрированный модуль.

    ``schedule`` и ``cancel`` передаются из CustomTkinter, чтобы правила
    можно было проверить отдельно от окна и не создавать второй event loop.
    """

    def __init__(
        self,
        on_unlock: Callable[[str], None],
        schedule: Callable[[int, Callable[[], None]], str],
        cancel: Callable[[str], None],
    ) -> None:
        self.on_unlock = on_unlock
        self.schedule = schedule
        self.cancel = cancel
        self.hold_job: str | None = None
        self.bongo_taps = 0
        self.bongo_deadline = 0.0
        self.buff_stage = 0
        self.buff_taps = 0
        self.buff_last_tap = 0.0

    def reset_for_cat(self, cat_index: int) -> None:
        """Сбросить незавершённые комбинации при смене котика."""
        self.cancel_hold()
        if cat_index != BONGO_CAT:
            self.bongo_taps = 0
            self.bongo_deadline = 0.0
        if cat_index != BUFF_CAT:
            self.reset_buff()

    def on_press(self, cat_index: int) -> None:
        self.cancel_hold()
        if cat_index == GRUMPY_CAT:
            self.hold_job = self.schedule(GRUMPY_HOLD_MS, self._unlock_roulette)

    def on_release(self, cat_index: int) -> None:
        self.cancel_hold()
        self.track_bongo(cat_index)
        self.track_buff(cat_index)

    def cancel_hold(self) -> None:
        if self.hold_job is None:
            return
        try:
            self.cancel(self.hold_job)
        except Exception:
            # Виджет мог быть уничтожен в момент открытия пасхалки.
            pass
        self.hold_job = None

    def _unlock_roulette(self) -> None:
        self.hold_job = None
        self.on_unlock("roulette")

    def track_bongo(self, cat_index: int) -> None:
        if cat_index != BONGO_CAT:
            self.bongo_taps = 0
            self.bongo_deadline = 0.0
            return
        now = time.monotonic()
        if now > self.bongo_deadline:
            self.bongo_taps = 0
        self.bongo_taps += 1
        self.bongo_deadline = now + BONGO_TAP_WINDOW_SECONDS
        if self.bongo_taps >= BONGO_TAPS_REQUIRED:
            self.bongo_taps = 0
            self.bongo_deadline = 0.0
            self.schedule(0, lambda: self.on_unlock("bongo"))

    def reset_buff(self) -> None:
        self.buff_stage = 0
        self.buff_taps = 0
        self.buff_last_tap = 0.0

    def track_buff(self, cat_index: int) -> None:
        if cat_index != BUFF_CAT:
            self.reset_buff()
            return
        now = time.monotonic()
        if self.buff_stage == 0:
            if self.buff_taps and now - self.buff_last_tap > BUFF_TAP_SPACING_SECONDS:
                self.buff_taps = 0
            self.buff_taps += 1
            self.buff_last_tap = now
            if self.buff_taps >= BUFF_TAPS_PER_BEAT:
                self.buff_stage = 1
            return
        if self.buff_stage == 1:
            pause = now - self.buff_last_tap
            if BUFF_PAUSE_MIN_SECONDS <= pause <= BUFF_PAUSE_MAX_SECONDS:
                self.buff_stage = 2
                self.buff_taps = 1
                self.buff_last_tap = now
                return
            self.reset_buff()
            self.buff_taps = 1
            self.buff_last_tap = now
            return
        if now - self.buff_last_tap > BUFF_TAP_SPACING_SECONDS:
            self.reset_buff()
            self.buff_taps = 1
            self.buff_last_tap = now
            return
        self.buff_taps += 1
        self.buff_last_tap = now
        if self.buff_taps >= BUFF_TAPS_PER_BEAT:
            self.reset_buff()
            self.schedule(0, lambda: self.on_unlock("buff"))


class RecipeProgress:
    """Сверяет кормление с рецептами пасхалок.

    Поддерживает несколько рецептов на одного кота: последовательность
    остаётся живой, пока она является началом хотя бы одного из них.
    """

    def __init__(self, recipes, secret_cat_indices: dict[str, int]) -> None:
        self.recipes = tuple(recipes)
        self.secret_cat_indices = secret_cat_indices
        self.sequence: list[str] = []

    def reset(self) -> None:
        self.sequence.clear()

    def feed(self, cat_index: int, food_id: str) -> tuple[str | None, tuple[str, ...]]:
        """Добавить блюдо и вернуть открытый секрет либо текущую серию."""
        available = [
            (secret_id, ingredients)
            for secret_id, _title, ingredients in self.recipes
            if self.secret_cat_indices.get(secret_id) == cat_index
        ]
        self.sequence.append(food_id)
        completed = next(
            (
                secret_id
                for secret_id, ingredients in available
                if tuple(self.sequence) == tuple(ingredients)
            ),
            None,
        )
        if completed is not None:
            sequence = tuple(self.sequence)
            self.reset()
            return completed, sequence
        if any(tuple(ingredients[:len(self.sequence)]) == tuple(self.sequence) for _, ingredients in available):
            return None, tuple(self.sequence)
        # Последнее блюдо может одновременно быть началом нового рецепта.
        self.sequence = [food_id] if any(ingredients and ingredients[0] == food_id for _, ingredients in available) else []
        return None, tuple(self.sequence)
