"""Временный каталог еды и рецептов скрытых модулей.

Иконки лежат в assets/food с теми же именами файлов. Рецепты хранятся здесь,
чтобы их можно было менять без правок интерфейса.
"""

from __future__ import annotations


FOOD_CATALOG = (
    ("fish", "Рыбка", "01_fish.png"),
    ("milk", "Молоко", "02_milk.png"),
    ("chicken", "Куриная ножка", "03_chicken.png"),
    ("tuna_can", "Тунец", "04_tuna_can.png"),
    ("cheese", "Сыр", "05_cheese.png"),
    ("sausage", "Сосиска", "06_sausage.png"),
    ("shrimp", "Креветка", "07_shrimp.png"),
    ("salmon", "Лосось", "08_salmon.png"),
    ("kibble", "Сухой корм", "09_kibble.png"),
    ("cat_treat", "Лакомство", "10_cat_treat.png"),
    ("pizza", "Пицца", "11_pizza.png"),
    ("burger", "Бургер", "12_burger.png"),
    ("sushi", "Суши", "13_sushi.png"),
    ("donut", "Пончик", "14_donut.png"),
    ("ice_cream", "Мороженое", "15_ice_cream.png"),
    ("watermelon", "Арбуз", "16_watermelon.png"),
    ("carrot", "Морковь", "17_carrot.png"),
    ("strawberry", "Клубника", "18_strawberry.png"),
    ("pumpkin", "Тыква", "19_pumpkin.png"),
    ("mystery_meal", "Секретное блюдо", "20_mystery_meal.png"),
)

FOOD_NAMES = {food_id: title for food_id, title, _filename in FOOD_CATALOG}

# Рецепты пасхалок. Кормилка передаёт эту последовательность RecipeProgress.
SECRET_RECIPES = (
    ("roulette", "RLT CONTROL", ("fish", "tuna_can", "salmon")),
    ("bongo", "BONGO BEAT", ("milk", "sausage", "cat_treat")),
    ("buff", "BUFF TIMING", ("chicken", "carrot", "pumpkin")),
    ("volt", "VOLT GRID", ("cheese", "shrimp", "mystery_meal")),
    ("miner", "ORE HUNT", ("kibble", "burger", "milk")),
)

# Рецепт срабатывает только у соответствующего котика.
SECRET_CAT_INDICES = {
    "roulette": 0,  # Крупье
    "bongo": 1,     # Звонок
    "buff": 2,      # Строитель
    "volt": 3,      # Вольт
    "miner": 4,     # Кварц
}
