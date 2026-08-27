"""Жизненный цикл экранов-пасхалок.

Новый модуль регистрируется фабрикой, а приложение больше не хранит
отдельные поля и одинаковую логику остановки для каждой пасхалки.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class SecretModuleManager:
    """Создаёт модули по требованию и гарантирует работу только одного."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Any]] = {}
        self._instances: dict[str, Any] = {}

    def register(self, secret_id: str, factory: Callable[[], Any]) -> None:
        if secret_id in self._factories:
            raise ValueError(f"Пасхалка «{secret_id}» уже зарегистрирована")
        self._factories[secret_id] = factory

    def is_registered(self, secret_id: str) -> bool:
        """Есть ли у рецепта экран.

        Рецепт живёт в ``food_catalog.py`` и переживает свой модуль: у
        ORE HUNT кот и последовательность остались, а экрана больше нет.
        """
        return secret_id in self._factories

    def get(self, secret_id: str) -> Any:
        if secret_id not in self._factories:
            raise KeyError(f"Неизвестная пасхалка: {secret_id}")
        module = self._instances.get(secret_id)
        if module is None or not module.winfo_exists():
            module = self._factories[secret_id]()
            self._instances[secret_id] = module
        return module

    def is_managed(self, widget: Any) -> bool:
        return any(widget is module for module in self._instances.values())

    def deactivate_others(self, active_secret: str, reason: str) -> None:
        for secret_id, module in self._instances.items():
            if secret_id != active_secret and module.winfo_exists():
                module.deactivate(reason)

    def deactivate_all(self, reason: str) -> None:
        """Остановить все модули перед полным закрытием приложения."""
        for module in self._instances.values():
            if module.winfo_exists():
                module.deactivate(reason)
