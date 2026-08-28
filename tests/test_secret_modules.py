"""Рецепт, переживший свой экран: кот «Кварц» и ORE HUNT.

Логику шахтёра из проекта сняли, а котик, его картинка и последовательность
кормления остались. Раньше такой рецепт уронил бы кормилку: менеджер
пасхалок отдаёт экраны по ключу и на незнакомый бросает ``KeyError``.
"""

from __future__ import annotations

import unittest

from food_catalog import FOOD_NAMES, SECRET_CAT_INDICES, SECRET_RECIPES
from kisiki.core import CATS, resource_path
from secret_modules import SecretModuleManager


class QuartzTests(unittest.TestCase):
    def test_the_cat_keeps_its_place_and_picture(self) -> None:
        name, _description, filename, _color, category = CATS[4]
        self.assertEqual(name, "Кварц")
        self.assertEqual(category, "Добывающие котики")
        self.assertTrue(resource_path(filename).exists(), filename)

    def test_the_recipe_survived_the_module(self) -> None:
        recipes = {secret_id: ingredients for secret_id, _title, ingredients in SECRET_RECIPES}
        self.assertEqual(SECRET_CAT_INDICES["miner"], 4)
        self.assertEqual(recipes["miner"], ("kibble", "burger", "milk"))
        for food_id in recipes["miner"]:
            self.assertIn(food_id, FOOD_NAMES)


class SecretModuleManagerTests(unittest.TestCase):
    def test_a_recipe_without_a_screen_is_recognised_before_it_crashes(self) -> None:
        # Кормилка спрашивает разрешения, а не ловит исключение: экран
        # открывается только у зарегистрированного рецепта.
        manager = SecretModuleManager()
        manager.register("poker", lambda: None)
        self.assertTrue(manager.is_registered("poker"))
        self.assertFalse(manager.is_registered("miner"))
        with self.assertRaises(KeyError):
            manager.get("miner")


if __name__ == "__main__":
    unittest.main()
