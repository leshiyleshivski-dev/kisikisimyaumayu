"""Скрытые комбинации для «Кисикисимяумяу».

Модуль намеренно не знает ничего об интерфейсе и GTA: он получает индекс
котика и съеденное блюдо, а возвращает открытый секрет. Поэтому рецепты
можно менять и тестировать, не трогая экономику кликера и экран выбора.
"""

from __future__ import annotations


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
