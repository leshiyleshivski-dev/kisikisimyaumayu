"""Main cat-clicker application and navigation."""

from __future__ import annotations

import json
import random
import time
import tkinter as tk
from tkinter import font as tkfont
from datetime import date
from pathlib import Path

import customtkinter as ctk

from easter_eggs import RecipeProgress
from food_catalog import FOOD_CATALOG, FOOD_NAMES, SECRET_CAT_INDICES, SECRET_RECIPES
from secret_modules import SecretModuleManager

from .core import (
    APP_BG, CATS, CAT_CATEGORIES, COMING_SOON_CATS, FONT_BODY, FONT_CAPTION,
    FONT_LEAD, FONT_NOTE, GOLD, MINT, MUTED, PINK, PURPLE,
    SURFACE, SURFACE_ALT, SURFACE_HOVER, TEXT,
    cat_sound, make_dpi_aware, progress_path, release_source_images,
    resource_path, rounded_photo, winmm,
)
from .clicker import (
    ACHIEVEMENTS, ADVENTURES, DECOR, UPGRADES, achievement_metric,
    adventure_reward, click_power as cat_click_power, daily_quests, daily_reward,
    decor_price, format_number, fresh_cat, level_progress, next_daily_streak,
    offline_income, passive_income as cat_passive_income, purr_duration,
    today_key, upgrade_price as cat_upgrade_price,
)
from .modules import (
    BlackjackModule, BuilderModule, ElectricianModule, LumberjackModule,
    MinerModule, PhoneModule,
    PokerModule, RaceBettorModule, RouletteModule, SlotSpinnerModule,
    fresh_daily_stats,
)


# The clicker has its own visual system. Secret GTA modules keep the palette
# from core.py and are intentionally unaffected by these local aliases.
APP_BG = "#111722"
SURFACE = "#1A2230"
SURFACE_ALT = "#222C3C"
SURFACE_HOVER = "#2B374A"
TEXT = "#F5F7FB"
MUTED = "#98A5B8"
# MUTED годится подписи в одну строку, но не абзацу: на тёмной карточке текст
# им сливается с фоном. Размеры шрифта общие с модулями и живут в core.py.
BODY = "#C8D3E4"
GOLD = "#F2C66D"
MINT = "#68D6B4"
PINK = "#FF8F91"
PURPLE = "#A994ED"

# Каждый размер режется из оригинала отдельно, поэтому список нужен прогреву.
# Дополняй его вместе с новыми вызовами photo()/food_photo().
CAT_IMAGE_SIZES = ((125, 98), (320, 255), (54, 44), (64, 64))
FOOD_IMAGE_SIZES = ((38, 38), (48, 48))

class KisikiApp(ctk.CTk):
    # Полосу категории считаем по самому узкому окну: на 1040 px «ДОБЫВАЮЩИЕ
    # КОТИКИ» ещё помещались бы, а на 960 обрезались бы до «ОБЫВАЮЩИЕ КОТИК» —
    # и это читается как опечатка, а не как узкое окно.
    MIN_WIDTH = 960
    CAT_GRID_PADDING = 36
    CAT_CELL_PADDING = 12

    def __init__(self) -> None:
        super().__init__()
        make_dpi_aware()
        ctk.set_appearance_mode("dark")
        self.title("Кисикисимяумяу")
        self.geometry("1040x860")
        self.minsize(self.MIN_WIDTH, 800)
        self.configure(fg_color=APP_BG)
        self.closing = False
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.selected = 0
        self.current_view = "home"
        self.game = self.load_game()
        self.sound_enabled = bool(self.game["sound_enabled"])
        self.clicks = self.cat_data()["meows"]
        self.boost_until = 0.0
        self.progress_return = "home"
        self.offline_reward = self.apply_offline_progress()
        self.last_disk_save = time.monotonic()
        self.recipe_progress = RecipeProgress(SECRET_RECIPES, SECRET_CAT_INDICES)
        self.recipe_unlock_pending = False
        self.combo_count = 0
        self.combo_deadline = 0.0
        self.combo_reset_job: str | None = None
        self.recipe_title_taps = 0
        self.recipe_title_deadline = 0.0
        # Книга рецептов открывается на той же полке, что и в прошлый раз:
        # None — все рецепты, иначе название категории котиков.
        self.recipe_filter: str | None = None
        self.gift_ready_at = time.monotonic()
        self.caption_fonts: dict[int, tkfont.Font] = {}
        self.images: dict[tuple[int, int, int], tk.PhotoImage] = {}
        self.food_images: dict[tuple[str, int, int], tk.PhotoImage] = {}
        self.click_effects: list[ctk.CTkLabel] = []
        self.content_restore_job: str | None = None
        self.sound_aliases: set[str] = set()
        self.sound_sequence = 0
        self.content = ctk.CTkFrame(self, fg_color=APP_BG, corner_radius=0)
        self.content.pack(fill="both", expand=True)
        self.secret_modules = SecretModuleManager()
        self.secret_modules.register(
            "roulette",
            lambda: RouletteModule(
                self.content,
                self.show_clicker,
                alert_sound_enabled=bool(self.game.get("roulette_sound_enabled", True)),
                on_alert_sound_change=self.set_roulette_sound,
            ),
        )
        self.secret_modules.register("phone", lambda: PhoneModule(self.content, self.show_clicker))
        self.secret_modules.register("builder", lambda: BuilderModule(self.content, self.show_clicker))
        self.secret_modules.register("volt", lambda: ElectricianModule(self.content, self.show_clicker))
        self.secret_modules.register(
            "miner",
            lambda: MinerModule(
                self.content,
                self.show_clicker,
                daily_stats=self.game["miner_stats"],
                on_stats_change=self.save_game,
            ),
        )
        self.secret_modules.register(
            "lumberjack", lambda: LumberjackModule(self.content, self.show_clicker)
        )
        self.secret_modules.register(
            "race_bettor", lambda: RaceBettorModule(self.content, self.show_clicker)
        )
        self.secret_modules.register(
            "slot_spinner", lambda: SlotSpinnerModule(self.content, self.show_clicker)
        )
        self.secret_modules.register(
            "blackjack", lambda: BlackjackModule(self.content, self.show_clicker)
        )
        self.secret_modules.register(
            "poker", lambda: PokerModule(self.content, self.show_clicker)
        )
        # CTk иногда возвращает свою стандартную иконку позднее при старте.
        # Поэтому устанавливаем cat-иконку после полной инициализации окна.
        self.after(120, self.apply_window_icon)
        self.after(1000, self.passive_tick)
        self.show_home()
        self.after(200, self.prewarm_images)

    @staticmethod
    def fresh_game() -> dict:
        return {
            "version": 4,
            "sound_enabled": True,
            "roulette_sound_enabled": True,
            "coins": 0,
            "lifetime_meows": 0,
            "purr_charge": 0,
            "daily_date": today_key(),
            "daily_level": 1,
            "daily": {"taps": 0, "earned": 0, "upgrades": 0},
            "quest_claims": [],
            "daily_claim_date": "",
            "daily_streak": 0,
            "achievements": [],
            "decor": {key: 0 for key in DECOR},
            "adventure": {},
            "adventures_completed": 0,
            # Счёт добытой руды за день. Живёт здесь, а не отдельным файлом:
            # запись одна на камень, а не сотни за вечер, как в покере.
            "miner_stats": fresh_daily_stats(),
            "last_seen": time.time(),
            "cats": [fresh_cat() for _ in CATS],
            "cat_roster_version": 3,
        }

    @staticmethod
    def cat_migration(saved_count: int, roster_version: int) -> tuple[int, ...]:
        """Откуда каждому нынешнему котику брать прогресс из старого файла.

        Новый котик не всегда приезжает в конец списка: Сучок встал сразу за
        Кварцем, потому что рядом с ним ему и место на главном экране. Но
        индекс в файле — это позиция, и вставка в середину сдвигает всё, что
        за ней: без этой карты прогресс Фаворита достался бы Сучку, а Блефа —
        Тузу. Поэтому набор котиков и пронумерован ``cat_roster_version``.

        Позиция ``-1`` означает, что котику в старом файле соответствия нет и
        он начинает с нуля.
        """
        if saved_count == 6 and roster_version < 2:
            # Шесть мем-котов: у будущих Крупье, Звонка, Кирпича и Вольта
            # прогресс есть, у остальных его не было вовсе.
            return (5, 1, 3, 4)
        if roster_version < 3 and saved_count >= 9:
            # Девять котиков до Сучка: он вставлен пятым, дальше всё сдвинуто.
            return (0, 1, 2, 3, 4, -1, 5, 6, 7, 8)
        return tuple(range(len(CATS)))

    def load_game(self) -> dict:
        game = self.fresh_game()
        try:
            saved = json.loads(progress_path().read_text(encoding="utf-8"))
            if isinstance(saved, dict) and isinstance(saved.get("cats"), list):
                game["sound_enabled"] = bool(saved.get("sound_enabled", True))
                game["roulette_sound_enabled"] = bool(saved.get("roulette_sound_enabled", True))
                game["coins"] = max(0, int(saved.get("coins", 0)))
                game["purr_charge"] = min(100, max(0, int(saved.get("purr_charge", 0))))
                game["daily_claim_date"] = str(saved.get("daily_claim_date", ""))
                game["daily_streak"] = max(0, int(saved.get("daily_streak", 0)))
                game["achievements"] = [str(item) for item in saved.get("achievements", [])]
                if isinstance(saved.get("decor"), dict):
                    for key in DECOR:
                        game["decor"][key] = max(0, int(saved["decor"].get(key, 0)))
                if isinstance(saved.get("adventure"), dict) and saved["adventure"].get("kind") in ADVENTURES:
                    game["adventure"] = {
                        "kind": str(saved["adventure"]["kind"]),
                        "cat": min(len(CATS) - 1, max(0, int(saved["adventure"].get("cat", 0)))),
                        "ready_at": float(saved["adventure"].get("ready_at", 0)),
                        "reward": max(0, int(saved["adventure"].get("reward", 0))),
                    }
                game["adventures_completed"] = max(0, int(saved.get("adventures_completed", 0)))
                if isinstance(saved.get("miner_stats"), dict):
                    # Разбирать поштучно незачем: OreTally чинит дату, счётчики
                    # и незнакомые ключи на первом же обращении.
                    game["miner_stats"] = dict(saved["miner_stats"])
                game["last_seen"] = float(saved.get("last_seen", time.time()))
                saved_cats = saved["cats"]
                source_indices = self.cat_migration(
                    len(saved_cats), int(saved.get("cat_roster_version", 0)),
                )
                for index, source_index in enumerate(source_indices):
                    if index >= len(CATS) or not 0 <= source_index < len(saved_cats):
                        continue
                    data = saved_cats[source_index]
                    if isinstance(data, dict):
                        for key in fresh_cat():
                            game["cats"][index][key] = max(0, int(data.get(key, 0)))
                game["lifetime_meows"] = max(
                    int(saved.get("lifetime_meows", 0)),
                    sum(cat["meows"] for cat in game["cats"]),
                )
                if saved.get("daily_date") == today_key() and isinstance(saved.get("daily"), dict):
                    game["daily_date"] = today_key()
                    game["daily_level"] = max(1, int(saved.get("daily_level", level_progress(game["lifetime_meows"])[0])))
                    for key in game["daily"]:
                        game["daily"][key] = max(0, int(saved["daily"].get(key, 0)))
                    game["quest_claims"] = [str(item) for item in saved.get("quest_claims", [])]
                else:
                    game["daily_level"] = level_progress(game["lifetime_meows"])[0]
        except (OSError, ValueError, TypeError):
            pass
        return game

    def save_game(self) -> None:
        self.game["sound_enabled"] = self.sound_enabled
        self.game["last_seen"] = time.time()
        try:
            progress_path().write_text(json.dumps(self.game, ensure_ascii=False), encoding="utf-8")
            self.last_disk_save = time.monotonic()
        except OSError:
            pass

    def set_roulette_sound(self, enabled: bool) -> None:
        self.game["roulette_sound_enabled"] = bool(enabled)
        self.save_game()

    def cat_data(self) -> dict:
        return self.game["cats"][self.selected]

    def click_power(self) -> int:
        return cat_click_power(self.cat_data(), self.game["decor"])

    def passive_income(self) -> int:
        return cat_passive_income(self.cat_data(), self.game["decor"])

    def total_passive_income(self) -> int:
        return sum(cat_passive_income(cat, self.game["decor"]) for cat in self.game["cats"])

    def upgrade_price(self, key: str) -> int:
        return cat_upgrade_price(self.cat_data(), key)

    def club_level(self) -> tuple[int, int, int]:
        return level_progress(self.game["lifetime_meows"])

    def boost_active(self) -> bool:
        return time.monotonic() < self.boost_until

    def ensure_daily(self) -> None:
        if self.game.get("daily_date") != today_key():
            self.game["daily_date"] = today_key()
            self.game["daily_level"] = self.club_level()[0]
            self.game["daily"] = {"taps": 0, "earned": 0, "upgrades": 0}
            self.game["quest_claims"] = []

    def add_reward(self, amount: int) -> None:
        """Award currency without feeding rewards back into task progress."""
        amount = max(0, int(amount))
        self.game["coins"] += amount
        self.cat_data()["meows"] += amount
        self.game["lifetime_meows"] += amount

    def add_earnings(self, cat_index: int, amount: int, *, taps: int = 0) -> None:
        amount = max(0, int(amount))
        if not amount and not taps:
            return
        self.ensure_daily()
        self.game["coins"] += amount
        self.game["cats"][cat_index]["meows"] += amount
        self.game["lifetime_meows"] += amount
        self.game["daily"]["earned"] += amount
        self.game["daily"]["taps"] += taps

    def apply_offline_progress(self) -> int:
        elapsed = time.time() - float(self.game.get("last_seen", time.time()))
        rewards, total = offline_income(self.game["cats"], elapsed, decor=self.game["decor"])
        if total:
            self.ensure_daily()
            for index, reward in enumerate(rewards):
                self.game["cats"][index]["meows"] += reward
            self.game["coins"] += total
            self.game["lifetime_meows"] += total
        self.game["last_seen"] = time.time()
        return total

    def apply_window_icon(self) -> None:
        try:
            self.iconbitmap(default=str(resource_path("orange_cat.ico")))
        except tk.TclError:
            # Для запуска исходника остаётся рабочий вариант через PNG.
            self.iconphoto(True, self.photo(2, 64, 64))

    def caption_font(self, size: int) -> tkfont.Font:
        if size not in self.caption_fonts:
            self.caption_fonts[size] = tkfont.Font(family="Segoe UI", size=size, weight="bold")
        return self.caption_fonts[size]

    def category_caption(
        self, icon: str, category: str, count: int | None, columns: int, column_width: float,
    ) -> tuple[str, int]:
        """Самая подробная подпись полосы, которая влезает в её ширину.

        Категория из двух котиков попадает на перенос строки, и её полоса
        сжимается до одной колонки. Подпись сдаёт сначала счёт, потом значок,
        потом кегль — обрезать название нельзя, оно и есть смысл полосы.
        """
        available = columns * column_width - self.CAT_CELL_PADDING
        variants = [f"{icon}  {category.upper()}", category.upper()]
        if count is not None:
            variants.insert(0, f"{icon}  {category.upper()}  ·  {count}")
        for size in (FONT_CAPTION, FONT_CAPTION - 1, FONT_CAPTION - 2):
            for caption in variants:
                if self.caption_font(size).measure(caption) <= available:
                    return caption, size
        return category.upper(), FONT_CAPTION - 2

    def photo(self, index: int, width: int, height: int) -> tk.PhotoImage:
        key = (index, width, height)
        if key not in self.images:
            self.images[key] = rounded_photo(resource_path(*Path(CATS[index][2]).parts), width, height)
        return self.images[key]

    def food_photo(self, food_id: str, width: int, height: int) -> tk.PhotoImage | None:
        """Вернуть иконку еды, когда PNG появится в assets/food."""
        filename = next(filename for item_id, _title, filename in FOOD_CATALOG if item_id == food_id)
        path = resource_path("assets", "food", filename)
        if not path.is_file():
            return None
        key = (food_id, width, height)
        if key not in self.food_images:
            self.food_images[key] = rounded_photo(path, width, height)
        return self.food_images[key]

    def prewarm_images(self, queue: list[tuple[str, object]] | None = None) -> None:
        """Нарезать оставшиеся размеры картинок в фоне, по одной за такт.

        Домашний экран уже распаковал оригиналы котиков, поэтому прогрев
        в основном лишь уменьшает готовые изображения. Разбивка по тактам
        не даёт окну замереть, а в конце оригиналы уходят из памяти.
        """
        if self.closing:
            return
        if queue is None:
            queue = [("cat", index) for index in range(len(CATS))]
            queue += [("food", food_id) for food_id, _title, _filename in FOOD_CATALOG]
        if not queue:
            release_source_images()
            return
        kind, item = queue.pop()
        if kind == "cat":
            for width, height in CAT_IMAGE_SIZES:
                self.photo(int(item), width, height)
        else:
            for width, height in FOOD_IMAGE_SIZES:
                self.food_photo(str(item), width, height)
        self.after(16, lambda rest=queue: self.prewarm_images(rest))

    def hide_content_while_building(self) -> None:
        """Собирать экран на скрытом контейнере, чтобы он не мерцал.

        CTkScrollbar заканчивает отрисовку вызовом update_idletasks, то есть
        заставляет Tk перерисовать окно на каждый добавленный виджет. Пока
        контейнер не показан, эти перерисовки не видны, а сборка экрана
        заодно идёт вдвое быстрее. Возвращаем контейнер через after —
        таймерное событие переживает чужие update_idletasks, тогда как
        after_idle показал бы наполовину собранный экран.
        """
        if self.content_restore_job is not None:
            self.after_cancel(self.content_restore_job)
        self.content.pack_forget()
        self.content_restore_job = self.after(0, self.show_content)

    def show_content(self) -> None:
        self.content_restore_job = None
        self.content.pack(fill="both", expand=True)

    def clear(self) -> None:
        self.cancel_hold()
        self.click_effects.clear()
        self.hide_content_while_building()
        for child in self.content.winfo_children():
            if self.secret_modules.is_managed(child):
                child.pack_forget()
            else:
                child.destroy()

    def header(self, subtitle: str, back=None) -> None:
        bar = ctk.CTkFrame(self.content, fg_color="transparent")
        bar.pack(fill="x", padx=42, pady=(24, 18))
        if back:
            ctk.CTkButton(
                bar, text="←", command=back, width=42, height=42, corner_radius=13,
                fg_color=SURFACE_ALT, hover_color=SURFACE_HOVER,
                font=ctk.CTkFont("Segoe UI", 17, "bold"),
            ).pack(side="left", padx=(0, 12))
        brand = ctk.CTkFrame(bar, fg_color="transparent")
        brand.pack(side="left")
        self.brand_title = ctk.CTkLabel(
            brand, text="КИСИКИСИМЯУМЯУ", cursor="hand2",
            font=ctk.CTkFont("Segoe UI", 20, "bold"), text_color=TEXT,
        )
        self.brand_title.pack(anchor="w")
        self.brand_title.bind("<Button-1>", self.track_recipe_title)
        ctk.CTkLabel(brand, text=subtitle, font=ctk.CTkFont("Segoe UI", 10), text_color=MUTED).pack(anchor="w")

        nav = ctk.CTkFrame(bar, fg_color=SURFACE, corner_radius=14)
        nav.pack(side="right")
        destinations = (
            ("Котики", self.show_home, "home"),
            ("Задания", lambda: self.open_progress("clicker" if self.current_view == "clicker" else "home"), "progress"),
            ("Домик", self.show_club, "club"),
        )
        for label, command, view in destinations:
            active = self.current_view == view
            ctk.CTkButton(
                nav, text=label, command=command, width=88, height=36, corner_radius=11,
                fg_color="#FF8F72" if active else "transparent",
                hover_color="#303C50", text_color="#151A23" if active else TEXT,
                font=ctk.CTkFont("Segoe UI", 10, "bold"),
            ).pack(side="left", padx=3, pady=3)
        self.sound_button = ctk.CTkButton(
            nav, command=self.toggle_sound, width=43, height=36, corner_radius=11,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        )
        self.sound_button.pack(side="left", padx=(3, 4), pady=3)
        self.refresh_sound_button()

    def track_recipe_title(self, _event=None) -> None:
        """Пять нажатий на название открывают временную книгу рецептов."""
        now = time.monotonic()
        if now > self.recipe_title_deadline:
            self.recipe_title_taps = 0
        self.recipe_title_taps += 1
        self.recipe_title_deadline = now + 4.0
        if self.recipe_title_taps >= 5:
            self.recipe_title_taps = 0
            self.recipe_title_deadline = 0.0
            self.after(0, self.show_recipes)

    # Рецептов десять, и списком в одну колонку книга уезжала под экран: до
    # покера приходилось крутить. Три колонки укладывают её в четыре ряда, а
    # полки по категориям котиков режут её до одного-двух рядов.
    RECIPE_COLUMNS = 3
    RECIPE_PADDING = 46
    SHELF_PADDING = 20
    SHELF_GAP = 6

    def show_recipes(self) -> None:
        """Книга рецептов: полки по категориям котиков и сетка карточек."""
        back = self.show_clicker if self.current_view == "clicker" else self.show_home
        self.clear()
        self.current_view = "recipes"
        self.header("скрытая книга кормления", back)

        intro = ctk.CTkFrame(self.content, fg_color=SURFACE, corner_radius=24, border_width=1, border_color="#303B50")
        intro.pack(fill="x", padx=self.RECIPE_PADDING, pady=(0, 14))
        headline = ctk.CTkFrame(intro, fg_color="transparent")
        headline.pack(fill="x", padx=24, pady=(16, 4))
        ctk.CTkLabel(
            headline, text="КНИГА РЕЦЕПТОВ", font=ctk.CTkFont("Segoe UI", 18, "bold"),
            text_color=GOLD,
        ).pack(side="left")
        ctk.CTkLabel(
            headline, text=f"{len(SECRET_RECIPES)} рецептов  ·  по одному на котика",
            font=ctk.CTkFont("Segoe UI", FONT_NOTE, "bold"), text_color=MUTED,
        ).pack(side="right")
        ctk.CTkLabel(
            intro, text="Накорми нужного котика блюдами в указанном порядке.",
            font=ctk.CTkFont("Segoe UI", FONT_BODY), text_color=BODY,
        ).pack(anchor="w", padx=24, pady=(0, 12))

        shelves = ctk.CTkFrame(intro, fg_color="transparent")
        shelves.pack(fill="x", padx=self.SHELF_PADDING, pady=(0, 16))
        self.recipe_shelf_buttons = {}
        # Шесть полок в одну строку не помещаются на узком окне — последняя
        # обрезалась до «эманы». Ширину кнопки берём по подписи и переносим
        # ряд, когда следующая не влезает: полок станет больше, разметка
        # переживёт это сама.
        available = self.MIN_WIDTH - 2 * self.RECIPE_PADDING - 2 * self.SHELF_PADDING
        row = ctk.CTkFrame(shelves, fg_color="transparent")
        row.pack(fill="x")
        used = 0
        for shelf, title in self.recipe_shelves():
            width = self.caption_font(FONT_NOTE).measure(title) + 30
            if used and used + width > available:
                row = ctk.CTkFrame(shelves, fg_color="transparent")
                row.pack(fill="x", pady=(6, 0))
                used = 0
            button = ctk.CTkButton(
                row, text=title, width=width, height=32, corner_radius=11,
                command=lambda chosen=shelf: self.choose_recipe_shelf(chosen),
                font=ctk.CTkFont("Segoe UI", FONT_NOTE, "bold"),
            )
            button.pack(side="left", padx=(0, self.SHELF_GAP))
            self.recipe_shelf_buttons[shelf] = button
            used += width + self.SHELF_GAP

        self.recipe_grid = ctk.CTkFrame(self.content, fg_color="transparent")
        self.recipe_grid.pack(fill="both", expand=True, padx=42, pady=(0, 22))
        for column in range(self.RECIPE_COLUMNS):
            self.recipe_grid.grid_columnconfigure(column, weight=1, uniform="recipe")
        self.refresh_recipe_shelves()

    def recipe_shelves(self) -> list[tuple[str | None, str]]:
        """Полки книги: «Все» и категории котиков, у которых есть рецепт.

        Порядок берётся из ``CAT_CATEGORIES``, чтобы полки стояли так же, как
        карточки на главном экране, а не в порядке словаря.
        """
        with_recipes = {CATS[index][4] for index in SECRET_CAT_INDICES.values()}
        shelves: list[tuple[str | None, str]] = [(None, f"Все  ·  {len(SECRET_RECIPES)}")]
        for category in CAT_CATEGORIES:
            if category not in with_recipes:
                continue
            count = sum(
                1 for index in SECRET_CAT_INDICES.values() if CATS[index][4] == category
            )
            shelves.append((category, f"{category}  ·  {count}"))
        return shelves

    def choose_recipe_shelf(self, shelf: str | None) -> None:
        """Повторный клик по выбранной полке возвращает всю книгу."""
        self.recipe_filter = None if shelf == self.recipe_filter else shelf
        self.refresh_recipe_shelves()

    def refresh_recipe_shelves(self) -> None:
        for shelf, button in self.recipe_shelf_buttons.items():
            active = shelf == self.recipe_filter
            button.configure(
                fg_color=GOLD if active else "#252D3C",
                hover_color="#F7D68C" if active else "#334158",
                text_color="#151A23" if active else TEXT,
            )
        self.render_recipe_cards()

    def render_recipe_cards(self) -> None:
        for child in self.recipe_grid.winfo_children():
            child.destroy()
        shown = [
            recipe for recipe in SECRET_RECIPES
            if self.recipe_filter is None
            or CATS[SECRET_CAT_INDICES[recipe[0]]][4] == self.recipe_filter
        ]
        for position, (secret_id, module_title, ingredients) in enumerate(shown):
            cat_index = SECRET_CAT_INDICES[secret_id]
            cat_name, _description, _filename, cat_color, _category = CATS[cat_index]
            card = ctk.CTkFrame(
                self.recipe_grid, fg_color=SURFACE, corner_radius=21,
                border_width=1, border_color="#303B50",
            )
            card.grid(
                row=position // self.RECIPE_COLUMNS,
                column=position % self.RECIPE_COLUMNS,
                padx=5, pady=(0, 10), sticky="nsew",
            )
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=16, pady=(14, 8))
            ctk.CTkLabel(top, text="", image=self.photo(cat_index, 54, 44)).pack(side="left", padx=(0, 9))
            copy = ctk.CTkFrame(top, fg_color="transparent")
            copy.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(
                copy, text=cat_name, font=ctk.CTkFont("Segoe UI", 17, "bold"),
                text_color=cat_color,
            ).pack(anchor="w")
            ctk.CTkLabel(
                copy, text=module_title,
                font=ctk.CTkFont("Segoe UI", FONT_CAPTION, "bold"), text_color=MUTED,
            ).pack(anchor="w")
            if not self.secret_modules.is_registered(secret_id):
                ctk.CTkLabel(
                    copy, text="модуль в разработке",
                    font=ctk.CTkFont("Segoe UI", FONT_NOTE), text_color="#8A97AB",
                ).pack(anchor="w")
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=(0, 15))
            for index, food_id in enumerate(ingredients):
                if index:
                    ctk.CTkLabel(
                        row, text="→", width=20,
                        font=ctk.CTkFont("Segoe UI", FONT_LEAD, "bold"), text_color=GOLD,
                    ).pack(side="left")
                icon = self.food_photo(food_id, 38, 38)
                ctk.CTkLabel(
                    row, text="" if icon else FOOD_NAMES[food_id], image=icon, height=38,
                    width=48 if icon else 0, corner_radius=11, fg_color="#252D3C",
                    text_color=TEXT, font=ctk.CTkFont("Segoe UI", FONT_NOTE, "bold"),
                ).pack(side="left", padx=2)

    def show_pantry(self) -> None:
        """Кормилка: правильная последовательность открывает пасхалку."""
        self.clear()
        self.current_view = "pantry"
        self.recipe_progress.reset()
        self.recipe_unlock_pending = False
        cat_name = CATS[self.selected][0]
        self.header(f"кормилка · {cat_name}", self.show_clicker)
        intro = ctk.CTkFrame(self.content, fg_color=SURFACE, corner_radius=24, border_width=1, border_color="#303B50")
        intro.pack(fill="x", padx=46, pady=(0, 18))
        ctk.CTkLabel(intro, text="Чем сегодня угостим котика?", font=ctk.CTkFont("Segoe UI", 20, "bold"), text_color=TEXT).pack(anchor="w", padx=24, pady=(17, 2))
        ctk.CTkLabel(
            intro, text="Выбирай блюда в нужном порядке — рецепт откроет секретный модуль.",
            font=ctk.CTkFont("Segoe UI", 11), text_color=MUTED,
        ).pack(anchor="w", padx=24, pady=(0, 17))
        grid = ctk.CTkFrame(self.content, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=46)
        for column in range(5):
            grid.grid_columnconfigure(column, weight=1)
        for index, (food_id, title, _filename) in enumerate(FOOD_CATALOG):
            icon = self.food_photo(food_id, 48, 48)
            tile = ctk.CTkButton(
                grid, text="" if icon else f"{index + 1:02d}\\n{title}", image=icon,
                command=lambda chosen=food_id: self.choose_food_placeholder(chosen),
                height=78, corner_radius=15, fg_color="#252D3C", hover_color="#334158",
                font=ctk.CTkFont("Segoe UI", 11, "bold"),
            )
            tile.grid(row=index // 5, column=index % 5, padx=5, pady=5, sticky="nsew")
        self.pantry_status = ctk.StringVar(value="Давай составим котику вкусное блюдо.")
        ctk.CTkLabel(self.content, textvariable=self.pantry_status, font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=GOLD).pack(pady=(10, 20))

    def choose_food_placeholder(self, food_id: str) -> None:
        if self.recipe_unlock_pending:
            return
        secret_id, sequence = self.recipe_progress.feed(self.selected, food_id)
        recipe_text = "  →  ".join(FOOD_NAMES[item] for item in sequence)
        if secret_id is None:
            if sequence:
                self.pantry_status.set(f"Собираем угощение: {recipe_text}")
            else:
                self.pantry_status.set(f"Ура, ты покормил котика: {FOOD_NAMES[food_id]}!")
            return
        if not self.secret_modules.is_registered(secret_id):
            # Рецепт есть, а экрана нет. Сейчас экраны есть у всех девяти, но
            # так уже жил ORE HUNT, пока логику шахтёра снимали с проекта.
            # Серию RecipeProgress уже сбросил, поэтому кормить можно дальше.
            self.pantry_status.set("Котик доволен, но его модуль ещё в разработке.")
            return
        self.recipe_unlock_pending = True
        self.pantry_status.set("Котик довольно мурчит…")
        self.after(450, lambda unlocked=secret_id: self.open_secret(unlocked))

    def refresh_sound_button(self) -> None:
        if not hasattr(self, "sound_button") or not self.sound_button.winfo_exists():
            return
        if self.sound_enabled:
            self.sound_button.configure(text="🔊", fg_color="#2B374A", hover_color="#36445B")
        else:
            self.sound_button.configure(text="🔇", fg_color="#252D3C", hover_color="#303A4D")

    def toggle_sound(self) -> None:
        self.sound_enabled = not self.sound_enabled
        if not self.sound_enabled:
            self.stop_all_sounds()
        self.save_game()
        self.refresh_sound_button()

    def play_meme_sound(self) -> None:
        """Воспроизводит MP3 тихо, не дожидаясь завершения эффекта."""
        if not self.sound_enabled:
            return
        self.sound_sequence += 1
        alias = f"kiski_sfx_{self.sound_sequence}"
        path = resource_path("sounds", cat_sound(self.selected))
        opened = winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
        if opened != 0:
            return
        self.sound_aliases.add(alias)
        # 130/1000: заметно тише системной громкости, но эффект различим.
        winmm.mciSendStringW(f"setaudio {alias} volume to 130", None, 0, None)
        winmm.mciSendStringW(f"play {alias}", None, 0, None)
        self.after(3500, lambda sound_alias=alias: self.close_sound(sound_alias))

    def close_sound(self, alias: str) -> None:
        if alias in self.sound_aliases:
            winmm.mciSendStringW(f"close {alias}", None, 0, None)
            self.sound_aliases.discard(alias)

    def stop_all_sounds(self) -> None:
        for alias in tuple(self.sound_aliases):
            winmm.mciSendStringW(f"close {alias}", None, 0, None)
        self.sound_aliases.clear()

    def close_app(self) -> None:
        """Полностью завершить приложение при нажатии на крестик окна."""
        if self.closing:
            return
        self.closing = True
        if self.content_restore_job is not None:
            self.after_cancel(self.content_restore_job)
            self.content_restore_job = None
        self.recipe_unlock_pending = True
        self.secret_modules.deactivate_all("Приложение закрыто.")
        self.stop_all_sounds()
        self.save_game()
        self.quit()
        self.destroy()

    def show_home(self) -> None:
        self.clear()
        self.current_view = "home"
        self.header("уютный клуб хвостатых героев")
        home = ctk.CTkScrollableFrame(
            self.content, fg_color="transparent", corner_radius=0,
            scrollbar_button_color="#33435A", scrollbar_button_hover_color="#405570",
        )
        home.pack(fill="both", expand=True)
        intro = ctk.CTkFrame(home, fg_color="#1D2838", corner_radius=22, border_width=1, border_color="#34445C")
        intro.pack(fill="x", padx=42, pady=(0, 13))
        intro.grid_columnconfigure(0, weight=1)
        copy = ctk.CTkFrame(intro, fg_color="transparent")
        copy.grid(row=0, column=0, sticky="w", padx=22, pady=16)
        ctk.CTkLabel(copy, text="Сегодня в кошачьем клубе", font=ctk.CTkFont("Segoe UI", 22, "bold"), text_color=TEXT).pack(anchor="w")
        self.home_balance = ctk.StringVar(value="")
        ctk.CTkLabel(copy, textvariable=self.home_balance, font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=GOLD).pack(anchor="w", pady=(3, 0))
        actions = ctk.CTkFrame(intro, fg_color="transparent")
        actions.grid(row=0, column=1, padx=18, pady=14)
        self.daily_button = ctk.CTkButton(
            actions, command=self.claim_daily_reward, width=225, height=42, corner_radius=13,
            font=ctk.CTkFont("Segoe UI", 10, "bold"),
        )
        self.daily_button.pack()
        self.refresh_home_balance()
        self.refresh_daily_button()
        if self.offline_reward:
            offline = ctk.CTkFrame(home, fg_color="#17333C", corner_radius=14, border_width=1, border_color="#376274")
            offline.pack(fill="x", padx=42, pady=(0, 10))
            ctk.CTkLabel(
                offline, text=f"🌙  Пока тебя не было, котики намурчали {format_number(self.offline_reward)} мяу",
                font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color="#89E0CA",
            ).pack(padx=16, pady=9)

        active = self.game.get("adventure") or {}
        if active and active.get("kind") in ADVENTURES:
            self.home_adventure_text = ctk.StringVar()
            self.home_adventure_button = ctk.CTkButton(
                home, textvariable=self.home_adventure_text, command=self.show_club, height=36, corner_radius=12,
                fg_color="#243448", hover_color="#30445E", font=ctk.CTkFont("Segoe UI", 9, "bold"),
            )
            self.home_adventure_button.pack(fill="x", padx=42, pady=(0, 10))
            self.refresh_home_adventure()

        title_row = ctk.CTkFrame(home, fg_color="transparent")
        title_row.pack(fill="x", padx=42, pady=(1, 7))
        ctk.CTkLabel(title_row, text="КОТИКИ КЛУБА", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=MUTED).pack(side="left")
        ctk.CTkLabel(title_row, text="игровые котики и будущие новички", font=ctk.CTkFont("Segoe UI", 9), text_color="#718097").pack(side="right")
        grid = ctk.CTkFrame(home, fg_color="transparent")
        grid.pack(fill="x", padx=36, pady=(0, 12))
        cat_columns_count = 5
        cat_column_width = (self.MIN_WIDTH - 2 * self.CAT_GRID_PADDING) / cat_columns_count
        for column in range(cat_columns_count):
            grid.grid_columnconfigure(column, weight=1, uniform="club_cat")

        category_styles = {
            "Сонные котики": ("☾", "#222E46", "#9EB4FF"),
            "Строитель": ("▦", "#3A3223", GOLD),
            "Электрик": ("⚡", "#20363A", "#72D7D2"),
            "Добывающие котики": ("◆", "#33361F", "#C9D06B"),
            "Рыбак": ("♆", "#17383B", "#63C7BC"),
            "Лудоманы": ("♠", "#392A42", "#D4A7FF"),
        }
        # Категория из двух котиков попадает на перенос строки: Кварц
        # заканчивает первый ряд, Сучок начинает второй. Счёт стоит только на
        # первой полосе — повторённое «· 2» читалось бы как ошибка.
        counted_categories: set[str] = set()
        for row_start in range(0, len(CATS), cat_columns_count):
            row_cats = CATS[row_start:row_start + cat_columns_count]
            segment_start = 0
            while segment_start < len(row_cats):
                category = row_cats[segment_start][4]
                segment_end = segment_start + 1
                while segment_end < len(row_cats) and row_cats[segment_end][4] == category:
                    segment_end += 1
                icon, background, accent = category_styles[category]
                if category in counted_categories:
                    category_count = None
                else:
                    category_count = sum(1 for cat in CATS if cat[4] == category)
                    counted_categories.add(category)
                caption, caption_size = self.category_caption(
                    icon, category, category_count,
                    segment_end - segment_start, cat_column_width,
                )
                # A single widget avoids the rectangular seam that a transparent
                # child label can leave over a rounded CTkFrame border.
                category_bar = ctk.CTkLabel(
                    grid, text=caption,
                    height=34, fg_color=background, corner_radius=11,
                    font=ctk.CTkFont("Segoe UI", caption_size, "bold"), text_color=accent,
                )
                category_bar.grid(
                    row=(row_start // cat_columns_count) * 2,
                    column=segment_start, columnspan=segment_end - segment_start,
                    padx=6, pady=(0, 7), sticky="ew",
                )
                segment_start = segment_end

        for index, (name, description, _filename, color, _category) in enumerate(CATS):
            card = ctk.CTkFrame(grid, fg_color=SURFACE, corner_radius=19, border_width=1, border_color="#303B50", cursor="hand2")
            card.grid(
                row=(index // cat_columns_count) * 2 + 1,
                column=index % cat_columns_count,
                padx=6, pady=(0, 12), sticky="nsew",
            )
            picture = ctk.CTkLabel(card, text="", image=self.photo(index, 125, 98), cursor="hand2")
            picture.pack(pady=(15, 5))
            ctk.CTkLabel(card, text=name, font=ctk.CTkFont("Segoe UI", 16, "bold"), text_color=TEXT, cursor="hand2").pack()
            ctk.CTkLabel(card, text=description, font=ctk.CTkFont("Segoe UI", 9), text_color=MUTED, cursor="hand2", wraplength=140).pack(padx=8, pady=(2, 10))
            cat_meows = self.game["cats"][index]["meows"]
            cat_auto = cat_passive_income(self.game["cats"][index], self.game["decor"])
            cat_power = cat_click_power(self.game["cats"][index], self.game["decor"])
            cat_level, cat_progress, cat_needed = level_progress(cat_meows)
            ctk.CTkLabel(card, text=f"{format_number(cat_meows)} мяу", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=color, cursor="hand2").pack()
            ctk.CTkLabel(
                card, text=f"+{format_number(cat_power)}/клик  ·  +{format_number(cat_auto)}/сек",
                font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=MUTED, cursor="hand2",
            ).pack(pady=(2, 9))
            ctk.CTkLabel(
                card, text=f"УРОВЕНЬ {cat_level}", font=ctk.CTkFont("Segoe UI", 8, "bold"),
                text_color="#8291A8", cursor="hand2",
            ).pack()
            cat_level_bar = ctk.CTkProgressBar(
                card, height=5, corner_radius=3, fg_color="#303B4D", progress_color=color,
                cursor="hand2",
            )
            cat_level_bar.set(cat_progress / cat_needed)
            cat_level_bar.pack(fill="x", padx=18, pady=(4, 10))
            ctk.CTkLabel(
                card, text="ОТКРЫТЬ КОТИКА  →", height=31, corner_radius=10,
                fg_color="#273449", font=ctk.CTkFont("Segoe UI", 8, "bold"),
                text_color=TEXT, cursor="hand2",
            ).pack(fill="x", padx=14, pady=(0, 14))
            for widget in [card, *card.winfo_children()]:
                widget.bind("<Button-1>", lambda _event, cat=index: self.open_cat(cat))
                widget.bind("<Enter>", lambda _event, target=card: target.configure(fg_color=SURFACE_HOVER, border_color="#536784"))
                widget.bind("<Leave>", lambda _event, target=card: target.configure(fg_color=SURFACE, border_color="#303B50"))

        planned_grid = ctk.CTkFrame(home, fg_color="transparent")
        planned_grid.pack(fill="x", padx=36, pady=(0, 12))
        planned_columns_count = min(cat_columns_count, len(COMING_SOON_CATS))
        planned_column_width = (
            self.MIN_WIDTH - 2 * self.CAT_GRID_PADDING
        ) / planned_columns_count
        for column in range(planned_columns_count):
            planned_grid.grid_columnconfigure(column, weight=1, uniform="planned_cat")

        for row_start in range(0, len(COMING_SOON_CATS), cat_columns_count):
            row_cats = COMING_SOON_CATS[row_start:row_start + cat_columns_count]
            segment_start = 0
            while segment_start < len(row_cats):
                category = row_cats[segment_start][3]
                segment_end = segment_start + 1
                while segment_end < len(row_cats) and row_cats[segment_end][3] == category:
                    segment_end += 1
                icon, background, accent = category_styles[category]
                category_count = sum(1 for cat in COMING_SOON_CATS if cat[3] == category)
                caption, caption_size = self.category_caption(
                    icon, category, category_count,
                    segment_end - segment_start, planned_column_width,
                )
                ctk.CTkLabel(
                    planned_grid, text=caption,
                    height=30, fg_color=background, corner_radius=10,
                    font=ctk.CTkFont("Segoe UI", caption_size, "bold"), text_color=accent,
                ).grid(
                    row=(row_start // cat_columns_count) * 2,
                    column=segment_start, columnspan=segment_end - segment_start,
                    padx=6, pady=(0, 7), sticky="ew",
                )
                segment_start = segment_end

        for index, (name, description, color, _category) in enumerate(COMING_SOON_CATS):
            card = ctk.CTkFrame(
                planned_grid, fg_color="#171F2B", corner_radius=16,
                border_width=1, border_color="#2C3748",
            )
            card.grid(
                row=(index // cat_columns_count) * 2 + 1,
                column=index % cat_columns_count,
                padx=6, pady=(0, 12), sticky="nsew",
            )
            ctk.CTkLabel(
                card, text=name, font=ctk.CTkFont("Segoe UI", 14, "bold"),
                text_color=color,
            ).pack(padx=8, pady=(11, 1))
            ctk.CTkLabel(
                card, text=description, font=ctk.CTkFont("Segoe UI", 8),
                text_color="#7F8B9D", wraplength=135,
            ).pack(padx=8)
            ctk.CTkLabel(
                card, text="В РАЗРАБОТКЕ", height=24, corner_radius=8,
                fg_color="#252D3A", font=ctk.CTkFont("Segoe UI", 8, "bold"),
                text_color="#9AA6B7",
            ).pack(fill="x", padx=10, pady=(8, 10))

        club_level, club_progress, club_needed = self.club_level()
        club_summary = ctk.CTkFrame(
            home, fg_color="#182436", corner_radius=17,
            border_width=1, border_color="#30425A",
        )
        club_summary.pack(fill="x", padx=42, pady=(0, 18))
        summary_copy = ctk.CTkFrame(club_summary, fg_color="transparent")
        summary_copy.pack(side="left", fill="x", expand=True, padx=(18, 24), pady=12)
        ctk.CTkLabel(
            summary_copy, text=f"ПРОГРЕСС КЛУБА  ·  УРОВЕНЬ {club_level}",
            font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=GOLD,
        ).pack(anchor="w")
        club_bar = ctk.CTkProgressBar(
            summary_copy, height=6, corner_radius=3, fg_color="#2D394A", progress_color=GOLD,
        )
        club_bar.set(club_progress / club_needed)
        club_bar.pack(fill="x", pady=(6, 3))
        ctk.CTkLabel(
            summary_copy,
            text=f"Ещё {format_number(club_needed - club_progress)} мяу до нового уровня",
            font=ctk.CTkFont("Segoe UI", 8), text_color=MUTED,
        ).pack(anchor="w")
        ctk.CTkButton(
            club_summary, text="Задания", command=lambda: self.open_progress("home"),
            width=105, height=35, corner_radius=11, fg_color="#33435A", hover_color="#405570",
            font=ctk.CTkFont("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(0, 7), pady=12)
        ctk.CTkButton(
            club_summary, text="Домик", command=self.show_club,
            width=105, height=35, corner_radius=11, fg_color=PINK, hover_color="#FF7D80",
            text_color="#21171A", font=ctk.CTkFont("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(0, 14), pady=12)

    def open_progress(self, return_to: str) -> None:
        self.progress_return = return_to
        self.show_progress()

    def progress_back(self) -> None:
        if self.progress_return == "clicker":
            self.show_clicker()
        else:
            self.show_home()

    def refresh_daily_button(self) -> None:
        if not hasattr(self, "daily_button") or not self.daily_button.winfo_exists():
            return
        claimed = self.game.get("daily_claim_date") == today_key()
        streak = next_daily_streak(
            str(self.game.get("daily_claim_date", "")),
            int(self.game.get("daily_streak", 0)),
            date.today(),
        )
        reward = daily_reward(streak)
        if claimed:
            self.daily_button.configure(
                text=f"✓ Награда дня получена · серия {self.game['daily_streak']}",
                state="disabled", fg_color="#344054", hover_color="#344054",
            )
        else:
            self.daily_button.configure(
                text=f"☀ Забрать {format_number(reward)} · день {streak}",
                state="normal", fg_color="#E47D66", hover_color="#F08F77",
            )

    def claim_daily_reward(self) -> None:
        if self.game.get("daily_claim_date") == today_key():
            return
        streak = next_daily_streak(
            str(self.game.get("daily_claim_date", "")),
            int(self.game.get("daily_streak", 0)),
            date.today(),
        )
        reward = daily_reward(streak)
        self.game["daily_streak"] = streak
        self.game["daily_claim_date"] = today_key()
        self.add_reward(reward)
        self.save_game()
        if self.current_view == "home":
            self.refresh_home_balance()
            self.refresh_daily_button()
        elif self.current_view == "progress":
            self.refresh_progress_view()

    def quest_progress(self, metric: str) -> int:
        self.ensure_daily()
        return int(self.game["daily"].get(metric, 0))

    def claim_quest(self, key: str) -> None:
        quest = next((item for item in daily_quests(int(self.game["daily_level"])) if item["key"] == key), None)
        if quest is None or key in self.game["quest_claims"]:
            return
        if self.quest_progress(str(quest["metric"])) < int(quest["target"]):
            return
        self.game["quest_claims"].append(key)
        self.add_reward(int(quest["reward"]))
        self.save_game()
        self.refresh_progress_view()
        self.refresh_quick_quests()

    def claim_achievement(self, key: str) -> None:
        achievement = next((item for item in ACHIEVEMENTS if item.key == key), None)
        if achievement is None or key in self.game["achievements"]:
            return
        if achievement_metric(self.game, achievement.metric) < achievement.target:
            return
        self.game["achievements"].append(key)
        self.add_reward(achievement.reward)
        self.save_game()
        self.refresh_progress_view()

    def refresh_quick_quests(self) -> None:
        if self.current_view != "clicker" or not hasattr(self, "quick_quest_widgets"):
            return
        for quest in daily_quests(int(self.game["daily_level"])):
            key = str(quest["key"])
            widgets = self.quick_quest_widgets.get(key)
            if widgets is None:
                continue
            label, button = widgets
            value = min(int(quest["target"]), self.quest_progress(str(quest["metric"])))
            ready = value >= int(quest["target"])
            claimed = key in self.game["quest_claims"]
            label.configure(
                text=f"{quest['title']}  ·  {format_number(value)}/{format_number(int(quest['target']))}",
                text_color=MINT if ready else TEXT,
            )
            button.configure(
                text="✓" if claimed else ("Забрать" if ready else f"+{format_number(int(quest['reward']))}"),
                state="normal" if ready and not claimed else "disabled",
                fg_color="#4B8A72" if ready else "#344054",
                hover_color="#5CA587" if ready else "#344054",
            )

    def refresh_progress_view(self) -> None:
        """Update progress widgets in place so claiming never flashes the screen."""
        if self.current_view != "progress" or not hasattr(self, "progress_level_label"):
            return
        if not self.progress_level_label.winfo_exists():
            return

        self.ensure_daily()
        level, current, needed = self.club_level()
        self.progress_level_label.configure(text=f"КЛУБ · УРОВЕНЬ {level}")
        self.progress_lifetime_label.configure(
            text=f"За всё время: {format_number(self.game['lifetime_meows'])} мяу"
        )
        self.progress_level_bar.set(current / needed)
        self.progress_level_remaining.configure(
            text=f"До следующего уровня: {format_number(needed - current)}"
        )
        self.progress_streak_label.configure(
            text=f"Серия входов: {self.game['daily_streak']} дн."
        )

        for quest in daily_quests(int(self.game["daily_level"])):
            key = str(quest["key"])
            widgets = self.quest_widgets.get(key)
            if widgets is None:
                continue
            progress_label, button = widgets
            progress = min(int(quest["target"]), self.quest_progress(str(quest["metric"])))
            claimed = key in self.game["quest_claims"]
            ready = progress >= int(quest["target"])
            progress_label.configure(
                text=f"{format_number(progress)} / {format_number(int(quest['target']))}",
                text_color=MINT if ready else GOLD,
            )
            button.configure(
                text="Получено" if claimed else (
                    f"Забрать +{format_number(int(quest['reward']))}"
                    if ready else f"Награда {format_number(int(quest['reward']))}"
                ),
                state="normal" if ready and not claimed else "disabled",
                fg_color="#4B8A72" if ready else "#3A4354",
                hover_color="#5CA587" if ready else "#3A4354",
            )

        for achievement in ACHIEVEMENTS:
            widgets = self.achievement_widgets.get(achievement.key)
            if widgets is None:
                continue
            title_label, progress_label, button = widgets
            value = min(achievement.target, achievement_metric(self.game, achievement.metric))
            claimed = achievement.key in self.game["achievements"]
            ready = value >= achievement.target
            title_label.configure(
                text=f"{'✓' if claimed else '◆'}  {achievement.title}",
                text_color=MINT if claimed else TEXT,
            )
            progress_label.configure(
                text=f"{format_number(value)} / {format_number(achievement.target)} · +{format_number(achievement.reward)}"
            )
            button.configure(
                text="Готово" if claimed else ("Забрать" if ready else "В пути"),
                state="normal" if ready and not claimed else "disabled",
                fg_color="#4B8A72" if ready else "#3A4354",
                hover_color="#5CA587" if ready else "#3A4354",
            )

    def show_progress(self) -> None:
        self.clear()
        self.current_view = "progress"
        self.ensure_daily()
        self.quest_widgets: dict[str, tuple[ctk.CTkLabel, ctk.CTkButton]] = {}
        self.achievement_widgets: dict[str, tuple[ctk.CTkLabel, ctk.CTkLabel, ctk.CTkButton]] = {}
        self.header("задания, достижения и статистика", self.progress_back)
        scroll = ctk.CTkScrollableFrame(self.content, fg_color="transparent", corner_radius=0)
        scroll.pack(fill="both", expand=True, padx=46, pady=(0, 24))

        level, current, needed = self.club_level()
        hero = ctk.CTkFrame(scroll, fg_color="#1D2838", corner_radius=24, border_width=1, border_color="#34445C")
        hero.pack(fill="x", pady=(0, 14))
        top = ctk.CTkFrame(hero, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(19, 9))
        self.progress_level_label = ctk.CTkLabel(top, text=f"КЛУБ · УРОВЕНЬ {level}", font=ctk.CTkFont("Segoe UI", 19, "bold"), text_color=GOLD)
        self.progress_level_label.pack(side="left")
        self.progress_lifetime_label = ctk.CTkLabel(top, text=f"За всё время: {format_number(self.game['lifetime_meows'])} мяу", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=MUTED)
        self.progress_lifetime_label.pack(side="right")
        self.progress_level_bar = ctk.CTkProgressBar(hero, height=9, corner_radius=7, progress_color="#FF8F72", fg_color="#3A455A")
        self.progress_level_bar.pack(fill="x", padx=24)
        self.progress_level_bar.set(current / needed)
        self.progress_level_remaining = ctk.CTkLabel(hero, text=f"До следующего уровня: {format_number(needed - current)}", font=ctk.CTkFont("Segoe UI", 10), text_color=MUTED)
        self.progress_level_remaining.pack(anchor="w", padx=24, pady=(6, 17))

        daily = ctk.CTkFrame(scroll, fg_color=SURFACE, corner_radius=22, border_width=1, border_color="#303B50")
        daily.pack(fill="x", pady=(0, 14))
        daily_top = ctk.CTkFrame(daily, fg_color="transparent")
        daily_top.pack(fill="x", padx=22, pady=(16, 9))
        ctk.CTkLabel(daily_top, text="СЕГОДНЯШНИЕ ЗАДАНИЯ", font=ctk.CTkFont("Segoe UI", 12, "bold"), text_color=GOLD).pack(side="left")
        self.progress_streak_label = ctk.CTkLabel(daily_top, text=f"Серия входов: {self.game['daily_streak']} дн.", font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=MUTED)
        self.progress_streak_label.pack(side="right")
        quest_row = ctk.CTkFrame(daily, fg_color="transparent")
        quest_row.pack(fill="x", padx=16, pady=(0, 16))
        quests = daily_quests(int(self.game["daily_level"]))
        for column, quest in enumerate(quests):
            quest_row.grid_columnconfigure(column, weight=1)
            progress = min(int(quest["target"]), self.quest_progress(str(quest["metric"])))
            claimed = str(quest["key"]) in self.game["quest_claims"]
            ready = progress >= int(quest["target"])
            card = ctk.CTkFrame(quest_row, fg_color=SURFACE_ALT, corner_radius=15)
            card.grid(row=0, column=column, padx=5, sticky="nsew")
            ctk.CTkLabel(card, text=str(quest["title"]), font=ctk.CTkFont("Segoe UI", 12, "bold"), text_color=TEXT).pack(anchor="w", padx=13, pady=(12, 2))
            ctk.CTkLabel(card, text=str(quest["description"]), font=ctk.CTkFont("Segoe UI", 9), text_color=MUTED).pack(anchor="w", padx=13)
            progress_label = ctk.CTkLabel(card, text=f"{format_number(progress)} / {format_number(int(quest['target']))}", font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=MINT if ready else GOLD)
            progress_label.pack(anchor="w", padx=13, pady=(8, 5))
            button = ctk.CTkButton(
                card, text="Получено" if claimed else (f"Забрать +{format_number(int(quest['reward']))}" if ready else f"Награда {format_number(int(quest['reward']))}"),
                command=lambda item=str(quest["key"]): self.claim_quest(item), height=31, corner_radius=10,
                state="normal" if ready and not claimed else "disabled",
                fg_color="#4B8A72" if ready else "#3A4354", hover_color="#5CA587",
                font=ctk.CTkFont("Segoe UI", 9, "bold"),
            )
            button.pack(fill="x", padx=11, pady=(0, 11))
            self.quest_widgets[str(quest["key"])] = (progress_label, button)

        ctk.CTkLabel(scroll, text="ДОСТИЖЕНИЯ", font=ctk.CTkFont("Segoe UI", 12, "bold"), text_color=GOLD).pack(anchor="w", pady=(5, 7))
        achievement_grid = ctk.CTkFrame(scroll, fg_color="transparent")
        achievement_grid.pack(fill="x")
        for column in range(2):
            achievement_grid.grid_columnconfigure(column, weight=1)
        for index, achievement in enumerate(ACHIEVEMENTS):
            value = min(achievement.target, achievement_metric(self.game, achievement.metric))
            claimed = achievement.key in self.game["achievements"]
            ready = value >= achievement.target
            card = ctk.CTkFrame(achievement_grid, fg_color=SURFACE, corner_radius=17, border_width=1, border_color="#303B50")
            card.grid(row=index // 2, column=index % 2, padx=5, pady=5, sticky="nsew")
            copy = ctk.CTkFrame(card, fg_color="transparent")
            copy.pack(side="left", fill="both", expand=True, padx=(15, 6), pady=12)
            title_label = ctk.CTkLabel(copy, text=f"{'✓' if claimed else '◆'}  {achievement.title}", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=MINT if claimed else TEXT)
            title_label.pack(anchor="w")
            ctk.CTkLabel(copy, text=achievement.description, font=ctk.CTkFont("Segoe UI", 9), text_color=MUTED).pack(anchor="w", pady=(1, 5))
            progress_label = ctk.CTkLabel(copy, text=f"{format_number(value)} / {format_number(achievement.target)} · +{format_number(achievement.reward)}", font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=GOLD)
            progress_label.pack(anchor="w")
            button = ctk.CTkButton(
                card, text="Готово" if claimed else ("Забрать" if ready else "В пути"),
                command=lambda item=achievement.key: self.claim_achievement(item), width=74, height=34,
                state="normal" if ready and not claimed else "disabled", corner_radius=11,
                fg_color="#4B8A72" if ready else "#3A4354", hover_color="#5CA587",
                font=ctk.CTkFont("Segoe UI", 9, "bold"),
            )
            button.pack(side="right", padx=(4, 13), pady=13)
            self.achievement_widgets[achievement.key] = (title_label, progress_label, button)

    def show_club(self) -> None:
        self.clear()
        self.current_view = "club"
        self.adventure_cat = getattr(self, "adventure_cat", self.selected)
        active_adventure = self.game.get("adventure") or {}
        if active_adventure:
            self.adventure_cat = min(len(CATS) - 1, max(0, int(active_adventure.get("cat", 0))))
        self.header("домик, постоянные бонусы и вылазки")

        body = ctk.CTkScrollableFrame(self.content, fg_color="transparent", corner_radius=0)
        body.pack(fill="both", expand=True, padx=42, pady=(0, 22))

        hero = ctk.CTkFrame(body, fg_color="#1D2838", corner_radius=22, border_width=1, border_color="#34445C")
        hero.pack(fill="x", pady=(0, 14))
        copy = ctk.CTkFrame(hero, fg_color="transparent")
        copy.pack(side="left", fill="x", expand=True, padx=22, pady=17)
        ctk.CTkLabel(copy, text="ДОМИК КОШАЧЬЕГО КЛУБА", font=ctk.CTkFont("Segoe UI", 17, "bold"), text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(copy, text="Обустраивай пространство — бонусы работают сразу для всех четырёх котиков.", font=ctk.CTkFont("Segoe UI", 10), text_color=MUTED).pack(anchor="w", pady=(2, 0))
        self.club_bonus_text = ctk.StringVar()
        ctk.CTkLabel(hero, textvariable=self.club_bonus_text, font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=MINT, justify="right").pack(side="right", padx=22)

        columns = ctk.CTkFrame(body, fg_color="transparent")
        columns.pack(fill="both", expand=True)
        columns.grid_columnconfigure(0, weight=1)
        columns.grid_columnconfigure(1, weight=1)

        decor_panel = ctk.CTkFrame(columns, fg_color=SURFACE, corner_radius=22, border_width=1, border_color="#303B50")
        decor_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        ctk.CTkLabel(decor_panel, text="ОБУСТРОЙСТВО", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=GOLD).pack(anchor="w", padx=18, pady=(17, 3))
        ctk.CTkLabel(decor_panel, text="Постоянные улучшения клуба", font=ctk.CTkFont("Segoe UI", 10), text_color=MUTED).pack(anchor="w", padx=18, pady=(0, 12))
        self.decor_buttons: dict[str, ctk.CTkButton] = {}
        for key, item in DECOR.items():
            button = ctk.CTkButton(
                decor_panel, command=lambda decor_key=key: self.buy_decor(decor_key),
                height=62, corner_radius=14, fg_color=SURFACE_ALT, hover_color=SURFACE_HOVER,
                border_width=1, border_color=item.color, anchor="w",
                font=ctk.CTkFont("Segoe UI", 10, "bold"),
            )
            button.pack(fill="x", padx=16, pady=(0, 8))
            self.decor_buttons[key] = button

        adventure_panel = ctk.CTkFrame(columns, fg_color=SURFACE, corner_radius=22, border_width=1, border_color="#303B50")
        adventure_panel.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        ctk.CTkLabel(adventure_panel, text="КОШАЧЬИ ВЫЛАЗКИ", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=GOLD).pack(anchor="w", padx=18, pady=(17, 3))
        ctk.CTkLabel(adventure_panel, text="Отправь котика — награда созреет сама", font=ctk.CTkFont("Segoe UI", 10), text_color=MUTED).pack(anchor="w", padx=18, pady=(0, 10))
        selector = ctk.CTkFrame(adventure_panel, fg_color=SURFACE_ALT, corner_radius=12)
        selector.pack(fill="x", padx=16, pady=(0, 10))
        self.adventure_cat_buttons: list[ctk.CTkButton] = []
        for index, cat in enumerate(CATS):
            button = ctk.CTkButton(
                selector, text=cat[0], command=lambda cat_index=index: self.select_adventure_cat(cat_index),
                height=32, corner_radius=9, fg_color="transparent", hover_color=SURFACE_HOVER,
                font=ctk.CTkFont("Segoe UI", 9, "bold"),
            )
            button.pack(side="left", fill="x", expand=True, padx=2, pady=3)
            self.adventure_cat_buttons.append(button)
        self.adventure_content = ctk.CTkFrame(adventure_panel, fg_color="transparent")
        self.adventure_content.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        self.refresh_club_view()
        self.build_adventure_panel()

    def buy_decor(self, key: str) -> None:
        price = decor_price(self.game["decor"], key)
        if self.game["coins"] < price:
            self.club_bonus_text.set(f"Не хватает {format_number(price - self.game['coins'])} мяу")
            return
        self.game["coins"] -= price
        self.game["decor"][key] += 1
        self.ensure_daily()
        self.game["daily"]["upgrades"] += 1
        self.save_game()
        self.refresh_club_view()
        if not self.game.get("adventure"):
            self.refresh_adventure_options()

    def refresh_club_view(self) -> None:
        if self.current_view != "club" or not hasattr(self, "decor_buttons"):
            return
        decor = self.game["decor"]
        self.club_bonus_text.set(
            f"Клики +{decor['scratcher'] * 8}%   ·   Авто +{decor['lamp'] * 10}%\n"
            f"Офлайн {min(95, 75 + decor['window'] * 5)}%   ·   Мур-режим {purr_duration(decor)} сек"
        )
        for key, button in self.decor_buttons.items():
            item = DECOR[key]
            price = decor_price(decor, key)
            level = decor[key]
            button.configure(
                text=f"{item.title}  ·  {item.description}\n{format_number(price)} мяу   ·   уровень {level}",
                state="normal",
            )
        self.refresh_adventure_selector()

    def refresh_adventure_selector(self) -> None:
        if self.current_view != "club" or not hasattr(self, "adventure_cat_buttons"):
            return
        for index, button in enumerate(self.adventure_cat_buttons):
            active = index == self.adventure_cat
            button.configure(
                fg_color="#FF8F72" if active else "transparent",
                text_color="#151A23" if active else TEXT,
            )

    def select_adventure_cat(self, index: int) -> None:
        if self.game.get("adventure"):
            return
        if index == self.adventure_cat:
            return
        self.adventure_cat = index
        self.refresh_adventure_selector()
        self.refresh_adventure_options()

    def refresh_adventure_options(self) -> None:
        """Refresh reward values without destroying the adventure cards."""
        if self.current_view != "club" or self.game.get("adventure"):
            return
        if not hasattr(self, "adventure_reward_labels"):
            return
        cat = self.game["cats"][self.adventure_cat]
        for key, label in self.adventure_reward_labels.items():
            adventure = ADVENTURES[key]
            reward = adventure_reward(cat, key, self.game["decor"])
            label.configure(
                text=f"{adventure.duration // 60} мин · награда {format_number(reward)}"
            )

    def build_adventure_panel(self) -> None:
        if self.current_view != "club" or not hasattr(self, "adventure_content"):
            return
        for child in self.adventure_content.winfo_children():
            child.destroy()
        self.adventure_reward_labels: dict[str, ctk.CTkLabel] = {}
        active = self.game.get("adventure") or {}
        if active and active.get("kind") in ADVENTURES:
            kind = str(active["kind"])
            adventure = ADVENTURES[kind]
            cat_index = min(len(CATS) - 1, max(0, int(active.get("cat", 0))))
            card = ctk.CTkFrame(self.adventure_content, fg_color="#1D3140", corner_radius=16, border_width=1, border_color="#3C6172")
            card.pack(fill="x", pady=3)
            ctk.CTkLabel(card, text=f"{CATS[cat_index][0]} сейчас: {adventure.title}", font=ctk.CTkFont("Segoe UI", 14, "bold"), text_color=TEXT).pack(anchor="w", padx=16, pady=(16, 3))
            ctk.CTkLabel(card, text=adventure.description, font=ctk.CTkFont("Segoe UI", 9), text_color=MUTED).pack(anchor="w", padx=16)
            self.adventure_timer_text = ctk.StringVar()
            ctk.CTkLabel(card, textvariable=self.adventure_timer_text, font=ctk.CTkFont("Segoe UI", 22, "bold"), text_color=MINT).pack(anchor="w", padx=16, pady=(15, 3))
            ctk.CTkLabel(card, text=f"Награда: {format_number(int(active.get('reward', 0)))} мяу", font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=GOLD).pack(anchor="w", padx=16, pady=(0, 12))
            self.adventure_action_button = ctk.CTkButton(
                card, command=self.claim_adventure, height=39, corner_radius=12,
                fg_color="#4BAF8D", hover_color="#5DC4A0", font=ctk.CTkFont("Segoe UI", 10, "bold"),
            )
            self.adventure_action_button.pack(fill="x", padx=14, pady=(0, 14))
            self.refresh_adventure_timer()
            return

        for key, adventure in ADVENTURES.items():
            reward = adventure_reward(self.game["cats"][self.adventure_cat], key, self.game["decor"])
            minutes = adventure.duration // 60
            card = ctk.CTkFrame(self.adventure_content, fg_color=SURFACE_ALT, corner_radius=14)
            card.pack(fill="x", pady=(0, 7))
            copy = ctk.CTkFrame(card, fg_color="transparent")
            copy.pack(side="left", fill="x", expand=True, padx=13, pady=10)
            ctk.CTkLabel(copy, text=adventure.title, font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=TEXT).pack(anchor="w")
            reward_label = ctk.CTkLabel(copy, text=f"{minutes} мин · награда {format_number(reward)}", font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=GOLD)
            reward_label.pack(anchor="w", pady=(2, 0))
            self.adventure_reward_labels[key] = reward_label
            ctk.CTkButton(
                card, text="Отправить", command=lambda adventure_key=key: self.start_adventure(adventure_key),
                width=88, height=34, corner_radius=10, fg_color="#34455E", hover_color="#415674",
                font=ctk.CTkFont("Segoe UI", 9, "bold"),
            ).pack(side="right", padx=11)

    def start_adventure(self, key: str) -> None:
        if self.game.get("adventure") or key not in ADVENTURES:
            return
        adventure = ADVENTURES[key]
        self.game["adventure"] = {
            "kind": key,
            "cat": self.adventure_cat,
            "ready_at": time.time() + adventure.duration,
            "reward": adventure_reward(self.game["cats"][self.adventure_cat], key, self.game["decor"]),
        }
        self.save_game()
        self.build_adventure_panel()

    def refresh_adventure_timer(self) -> None:
        if self.current_view != "club" or not hasattr(self, "adventure_timer_text"):
            return
        active = self.game.get("adventure") or {}
        if not active:
            return
        remaining = max(0, int(float(active.get("ready_at", 0)) - time.time() + 0.999))
        minutes, seconds = divmod(remaining, 60)
        if remaining:
            self.adventure_timer_text.set(f"{minutes:02d}:{seconds:02d}")
            self.adventure_action_button.configure(text="Вылазка идёт", state="disabled", fg_color="#344B55", hover_color="#344B55")
        else:
            self.adventure_timer_text.set("НАГРАДА ГОТОВА")
            self.adventure_action_button.configure(text="Забрать награду", state="normal", fg_color="#4BAF8D", hover_color="#5DC4A0")

    def claim_adventure(self) -> None:
        active = self.game.get("adventure") or {}
        if not active or time.time() < float(active.get("ready_at", 0)):
            return
        cat_index = min(len(CATS) - 1, max(0, int(active.get("cat", 0))))
        self.add_earnings(cat_index, int(active.get("reward", 0)))
        self.game["adventures_completed"] += 1
        self.game["adventure"] = {}
        self.save_game()
        self.build_adventure_panel()
        self.refresh_club_view()

    def open_cat(self, index: int) -> None:
        self.selected = index
        self.clicks = self.cat_data()["meows"]
        self.reset_combo()
        self.show_clicker()

    def show_clicker(self) -> None:
        self.clear()
        self.current_view = "clicker"
        name, description, _, color, _category = CATS[self.selected]
        self.header(f"личный кликер · {name}", self.show_home)
        body = ctk.CTkFrame(self.content, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=42, pady=(0, 20))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        left = ctk.CTkFrame(body, fg_color=SURFACE, corner_radius=24, border_width=1, border_color="#303B50")
        left.grid(row=0, column=0, sticky="new", padx=(0, 12))
        right = ctk.CTkFrame(body, fg_color=SURFACE, corner_radius=24, width=356, border_width=1, border_color="#303B50")
        right.grid(row=0, column=1, sticky="new")

        cat_top = ctk.CTkFrame(left, fg_color="transparent")
        cat_top.pack(fill="x", padx=24, pady=(18, 3))
        cat_copy = ctk.CTkFrame(cat_top, fg_color="transparent")
        cat_copy.pack(side="left")
        ctk.CTkLabel(cat_copy, text=name, font=ctk.CTkFont("Segoe UI", 26, "bold"), text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(cat_copy, text=description, font=ctk.CTkFont("Segoe UI", 11), text_color=MUTED).pack(anchor="w")
        self.power_text = ctk.CTkLabel(cat_top, text="", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=GOLD, justify="right")
        self.power_text.pack(side="right")
        self.click_area = left
        self.hero = ctk.CTkLabel(left, text="", image=self.photo(self.selected, 320, 255), cursor="hand2")
        self.hero.pack(pady=(0, 0))
        self.hero.bind("<ButtonPress-1>", self.hero_press)
        self.hero.bind("<ButtonRelease-1>", self.hero_release)
        self.hero.bind("<Leave>", self.cancel_hold)
        self.hint = ctk.CTkLabel(left, text="нажимай прямо на котика", font=ctk.CTkFont("Segoe UI", 12, "bold"), text_color=color)
        self.hint.pack(pady=(0, 8))
        combo = ctk.CTkFrame(left, fg_color=SURFACE_ALT, corner_radius=13)
        combo.pack(fill="x", padx=22, pady=(0, 7))
        self.combo_text = ctk.StringVar(value="КОМБО · начинай гладить")
        ctk.CTkLabel(combo, textvariable=self.combo_text, font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=GOLD).pack(anchor="w", padx=13, pady=(8, 4))
        self.combo_bar = ctk.CTkProgressBar(combo, height=5, corner_radius=5, progress_color="#FF987C", fg_color="#3A455A")
        self.combo_bar.pack(fill="x", padx=13, pady=(0, 9))
        self.combo_bar.set(0)
        purr = ctk.CTkFrame(left, fg_color="#1A3540", corner_radius=13, border_width=1, border_color="#376274")
        purr.pack(fill="x", padx=22, pady=(0, 7))
        purr.grid_columnconfigure(0, weight=1)
        self.purr_text = ctk.StringVar()
        ctk.CTkLabel(purr, textvariable=self.purr_text, font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color="#89E0CA").grid(row=0, column=0, sticky="w", padx=13, pady=(8, 4))
        self.purr_button = ctk.CTkButton(
            purr, command=self.start_purr_mode, width=116, height=29, corner_radius=9,
            font=ctk.CTkFont("Segoe UI", 9, "bold"),
        )
        self.purr_button.grid(row=0, column=1, rowspan=2, padx=9, pady=7)
        self.purr_bar = ctk.CTkProgressBar(purr, height=5, corner_radius=5, progress_color="#68D6B4", fg_color="#36515B")
        self.purr_bar.grid(row=1, column=0, sticky="ew", padx=13, pady=(0, 9))
        stats = ctk.CTkFrame(left, fg_color="transparent")
        stats.pack(fill="x", padx=18, pady=(0, 16))
        for column in range(3):
            stats.grid_columnconfigure(column, weight=1)
        self.tap_text = ctk.StringVar()
        self.best_combo_text = ctk.StringVar()
        self.combo_bonus_text = ctk.StringVar()
        stat_items = (
            ("НАЖАТИЙ", self.tap_text, PURPLE),
            ("ЛУЧШИЙ КОМБО", self.best_combo_text, MINT),
            ("БОНУС СЕРИИ", self.combo_bonus_text, GOLD),
        )
        for column, (caption, variable, shade) in enumerate(stat_items):
            card = ctk.CTkFrame(stats, fg_color=SURFACE_ALT, corner_radius=12)
            card.grid(row=0, column=column, padx=4, sticky="ew")
            ctk.CTkLabel(card, text=caption, font=ctk.CTkFont("Segoe UI", 8, "bold"), text_color=MUTED).pack(pady=(8, 1))
            ctk.CTkLabel(card, textvariable=variable, font=ctk.CTkFont("Segoe UI", 15, "bold"), text_color=shade).pack(pady=(0, 8))

        wallet = ctk.CTkFrame(right, fg_color="#1D2838", corner_radius=16)
        wallet.pack(fill="x", padx=16, pady=(16, 12))
        ctk.CTkLabel(wallet, text="БАЛАНС КЛУБА", font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=MUTED).pack(anchor="w", padx=15, pady=(11, 1))
        self.coins_text = ctk.StringVar()
        ctk.CTkLabel(wallet, textvariable=self.coins_text, font=ctk.CTkFont("Segoe UI", 24, "bold"), text_color=GOLD).pack(anchor="w", padx=15)
        self.level_text = ctk.StringVar()
        ctk.CTkLabel(wallet, textvariable=self.level_text, font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=MUTED).pack(anchor="w", padx=15, pady=(1, 4))
        self.level_bar = ctk.CTkProgressBar(wallet, height=6, corner_radius=5, progress_color="#FF8F72", fg_color="#384357")
        self.level_bar.pack(fill="x", padx=15)
        self.score_text = ctk.StringVar()
        ctk.CTkLabel(wallet, textvariable=self.score_text, font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=TEXT).pack(anchor="w", padx=15, pady=(6, 11))

        ctk.CTkLabel(right, text="БЫСТРЫЙ МАГАЗИН", font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=MUTED).pack(anchor="w", padx=18, pady=(0, 7))
        shop = ctk.CTkFrame(right, fg_color="transparent")
        shop.pack(fill="x", padx=12)
        for column in range(2):
            shop.grid_columnconfigure(column, weight=1)
        self.upgrade_buttons: dict[str, ctk.CTkButton] = {}
        for index, key in enumerate(("paw", "laser", "crown", "treat", "basket", "cafe")):
            upgrade = UPGRADES[key]
            button = ctk.CTkButton(
                shop, command=lambda upgrade_key=key: self.buy_upgrade(upgrade_key),
                height=70, corner_radius=13, fg_color=SURFACE_ALT, hover_color=SURFACE_HOVER,
                border_width=1, border_color=upgrade.color, font=ctk.CTkFont("Segoe UI", 9, "bold"),
            )
            button.grid(row=index // 2, column=index % 2, padx=4, pady=4, sticky="nsew")
            self.upgrade_buttons[key] = button

        utilities = ctk.CTkFrame(right, fg_color="transparent")
        utilities.pack(fill="x", padx=16, pady=(11, 0))
        self.gift_button = ctk.CTkButton(
            utilities, command=self.claim_lucky_yarn, height=42, corner_radius=12,
            fg_color="#6957A0", hover_color="#7B68B8", font=ctk.CTkFont("Segoe UI", 9, "bold"),
        )
        self.gift_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(
            utilities, text="🍽  Кормилка", command=self.show_pantry,
            height=42, corner_radius=12, fg_color="#315A57", hover_color="#3B6D68",
            font=ctk.CTkFont("Segoe UI", 9, "bold"),
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        ctk.CTkButton(
            right, text="🏠  Улучшения Домика и вылазки", command=self.show_club,
            height=38, corner_radius=12, fg_color="#27364A", hover_color="#33465F",
            font=ctk.CTkFont("Segoe UI", 9, "bold"),
        ).pack(fill="x", padx=16, pady=(8, 14))

        quick = ctk.CTkFrame(body, fg_color=SURFACE, corner_radius=18, border_width=1, border_color="#303B50")
        quick.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(11, 0))
        quick_top = ctk.CTkFrame(quick, fg_color="transparent")
        quick_top.pack(fill="x", padx=16, pady=(11, 6))
        ctk.CTkLabel(quick_top, text="СЕГОДНЯ В КЛУБЕ", font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=GOLD).pack(side="left")
        ctk.CTkButton(
            quick_top, text="Все задания →", command=lambda: self.open_progress("clicker"),
            width=105, height=27, corner_radius=9, fg_color="transparent", hover_color=SURFACE_HOVER,
            font=ctk.CTkFont("Segoe UI", 8, "bold"),
        ).pack(side="right")
        quick_row = ctk.CTkFrame(quick, fg_color="transparent")
        quick_row.pack(fill="x", padx=12, pady=(0, 11))
        self.quick_quest_widgets: dict[str, tuple[ctk.CTkLabel, ctk.CTkButton]] = {}
        for column, quest in enumerate(daily_quests(int(self.game["daily_level"]))):
            quick_row.grid_columnconfigure(column, weight=1)
            card = ctk.CTkFrame(quick_row, fg_color=SURFACE_ALT, corner_radius=11)
            card.grid(row=0, column=column, padx=4, sticky="ew")
            label = ctk.CTkLabel(card, text="", font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=TEXT)
            label.pack(side="left", padx=11, pady=9)
            button = ctk.CTkButton(
                card, text="", command=lambda quest_key=str(quest["key"]): self.claim_quest(quest_key),
                width=75, height=28, corner_radius=9, font=ctk.CTkFont("Segoe UI", 8, "bold"),
            )
            button.pack(side="right", padx=8, pady=6)
            self.quick_quest_widgets[str(quest["key"])] = (label, button)
        self.refresh_score()
        self.refresh_shop()
        self.refresh_gift_button()
        self.refresh_quick_quests()

    def refresh_score(self) -> None:
        data = self.cat_data()
        self.clicks = data["meows"]
        if hasattr(self, "score_text") and self.score_text is not None:
            self.score_text.set(f"{format_number(self.clicks)} мяу")
            self.coins_text.set(f"{format_number(self.game['coins'])}  🐾")
            level, progress, needed = level_progress(self.clicks)
            self.level_text.set(f"Котик ур. {level} · ещё {format_number(needed - progress)}")
            self.level_bar.set(progress / needed)
            self.power_text.configure(text=f"Сила клика: +{format_number(self.click_power())}  ·  Авто: +{format_number(self.passive_income())}/сек")
            self.tap_text.set(str(data.get("taps", 0)))
            self.best_combo_text.set(f"×{data.get('best_combo', 0)}")
            self.update_combo_display()
            self.refresh_purr_mode()
            self.refresh_quick_quests()

    def combo_bonus(self) -> int:
        return min(4, self.combo_count // 5)

    def update_combo_display(self) -> None:
        if not hasattr(self, "combo_text") or not self.combo_text:
            return
        bonus = self.combo_bonus()
        if self.combo_count:
            self.combo_text.set(f"КОМБО ×{self.combo_count}  ·  бонус к клику +{bonus}")
            self.combo_bar.set((self.combo_count % 5) / 5)
        else:
            self.combo_text.set("КОМБО · кликай быстрее, чтобы собрать серию")
            self.combo_bar.set(0)
        self.combo_bonus_text.set(f"+{bonus}")

    def expire_combo(self) -> None:
        self.combo_reset_job = None
        self.combo_count = 0
        self.combo_deadline = 0.0
        if self.current_view == "clicker":
            self.update_combo_display()

    def reset_combo(self) -> None:
        if self.combo_reset_job is not None:
            try:
                self.after_cancel(self.combo_reset_job)
            except tk.TclError:
                pass
        self.combo_reset_job = None
        self.combo_count = 0
        self.combo_deadline = 0.0
        if self.current_view == "clicker" and hasattr(self, "combo_text"):
            self.update_combo_display()

    def register_click(self) -> int:
        now = time.monotonic()
        self.combo_count = self.combo_count + 1 if now <= self.combo_deadline else 1
        self.combo_deadline = now + 1.25
        if self.combo_reset_job is not None:
            try:
                self.after_cancel(self.combo_reset_job)
            except tk.TclError:
                pass
        self.combo_reset_job = self.after(1300, self.expire_combo)
        data = self.cat_data()
        data["taps"] = data.get("taps", 0) + 1
        data["best_combo"] = max(data.get("best_combo", 0), self.combo_count)
        return self.click_power() + self.combo_bonus()

    def refresh_gift_button(self) -> None:
        if not hasattr(self, "gift_button") or not self.gift_button.winfo_exists():
            return
        remaining = max(0, int(self.gift_ready_at - time.monotonic() + 0.999))
        if remaining == 0:
            self.gift_button.configure(
                text="🎁  Забрать клубок", state="normal",
                fg_color="#8C6AC1", hover_color="#A47BD9",
            )
        else:
            minutes, seconds = divmod(remaining, 60)
            self.gift_button.configure(
                text=f"Новый клубок через {minutes:02d}:{seconds:02d}", state="disabled",
                fg_color="#3B4050", hover_color="#3B4050",
            )

    def refresh_purr_mode(self) -> None:
        if not hasattr(self, "purr_button") or not self.purr_button.winfo_exists():
            return
        charge = min(100, int(self.game.get("purr_charge", 0)))
        remaining = max(0, int(self.boost_until - time.monotonic() + 0.999))
        self.purr_bar.set(charge / 100)
        if remaining:
            self.purr_text.set(f"МУР-РЕЖИМ · двойной доход ещё {remaining} сек")
            self.purr_button.configure(text="×2 АКТИВЕН", state="disabled", fg_color="#4B8A72", hover_color="#4B8A72")
        elif charge >= 100:
            self.purr_text.set(f"МУР-РЕЖИМ заряжен · {purr_duration(self.game['decor'])} сек двойного дохода")
            self.purr_button.configure(text="ВКЛЮЧИТЬ ×2", state="normal", fg_color="#4B8A72", hover_color="#5CA587")
        else:
            self.purr_text.set(f"МУР-РЕЖИМ · гладь котика, заряд {charge}%")
            self.purr_button.configure(text=f"ЗАРЯД {charge}%", state="disabled", fg_color="#40555A", hover_color="#40555A")

    def start_purr_mode(self) -> None:
        if self.boost_active() or int(self.game.get("purr_charge", 0)) < 100:
            return
        self.game["purr_charge"] = 0
        duration = purr_duration(self.game["decor"])
        self.boost_until = time.monotonic() + duration
        self.save_game()
        self.hint.configure(text=f"мур-режим включён на {duration} сек: весь доход ×2!", text_color=MINT)
        self.refresh_purr_mode()

    def claim_lucky_yarn(self) -> None:
        if time.monotonic() < self.gift_ready_at:
            return
        reward = random.randint(max(5, self.click_power() * 5), max(12, self.click_power() * 12))
        self.add_earnings(self.selected, reward)
        self.gift_ready_at = time.monotonic() + 45
        self.save_game()
        self.hint.configure(text=f"клубок удачи: +{reward} мяу!", text_color=PURPLE)
        self.refresh_score()
        self.refresh_shop()
        self.refresh_gift_button()

    def refresh_home_balance(self) -> None:
        if hasattr(self, "home_balance"):
            level = self.club_level()[0]
            self.home_balance.set(
                f"Уровень клуба {level}  ·  {format_number(self.game['coins'])} мяукоинов  ·  +{format_number(self.total_passive_income())}/сек"
            )

    def refresh_home_adventure(self) -> None:
        if self.current_view != "home" or not hasattr(self, "home_adventure_text"):
            return
        active = self.game.get("adventure") or {}
        if not active or active.get("kind") not in ADVENTURES:
            return
        adventure = ADVENTURES[str(active["kind"])]
        cat_index = min(len(CATS) - 1, max(0, int(active.get("cat", 0))))
        remaining = max(0, int(float(active.get("ready_at", 0)) - time.time() + 0.999))
        minutes, seconds = divmod(remaining, 60)
        status = "награда готова" if not remaining else f"{minutes:02d}:{seconds:02d}"
        self.home_adventure_text.set(f"🐾  {CATS[cat_index][0]}: {adventure.title}  ·  {status}")

    def refresh_shop(self) -> None:
        if not hasattr(self, "upgrade_buttons"):
            return
        for key, button in self.upgrade_buttons.items():
            upgrade = UPGRADES[key]
            price = self.upgrade_price(key)
            owned = self.cat_data().get(key, 0)
            can_buy = self.game["coins"] >= price
            button.configure(
                text=f"{upgrade.title}\n{upgrade.description}\n{format_number(price)} · ур. {owned}",
                border_width=1,
                border_color=upgrade.color if can_buy else "#344054",
            )

    def buy_upgrade(self, key: str) -> None:
        price = self.upgrade_price(key)
        if self.game["coins"] < price:
            self.hint.configure(text=f"Нужно ещё {price - self.game['coins']} мяу", text_color=PINK)
            return
        self.game["coins"] -= price
        self.cat_data()[key] += 1
        self.ensure_daily()
        self.game["daily"]["upgrades"] += 1
        self.save_game()
        self.hint.configure(text="улучшение куплено!", text_color=MINT)
        self.refresh_score()
        self.refresh_shop()

    def passive_tick(self) -> None:
        boost = 2 if self.boost_active() else 1
        total = 0
        for index, cat in enumerate(self.game["cats"]):
            income = cat_passive_income(cat, self.game["decor"]) * boost
            if income:
                self.add_earnings(index, income)
                total += income
        if total and time.monotonic() - self.last_disk_save >= 5:
            self.save_game()
        if self.current_view == "clicker":
            self.refresh_score()
            self.refresh_shop()
            self.refresh_gift_button()
        elif self.current_view == "home":
            self.refresh_home_balance()
            self.refresh_home_adventure()
        elif self.current_view == "club":
            self.refresh_adventure_timer()
        self.after(1000, self.passive_tick)

    def hero_press(self, _event) -> None:
        # Пасхалки теперь открываются только через рецепты в кормилке.
        return

    def hero_release(self, _event) -> None:
        # Короткий клик — основа активного заработка. Крит и мур-режим
        # добавляют редкие пики, не меняя базовую отзывчивость кликера.
        self.cancel_hold()
        amount = self.register_click()
        critical = random.random() < 0.08
        if critical:
            amount *= 3
        if self.boost_active():
            amount *= 2
        self.game["purr_charge"] = min(100, int(self.game.get("purr_charge", 0)) + 2)
        self.add_earnings(self.selected, amount, taps=1)
        self.save_game()
        self.refresh_score()
        self.refresh_shop()
        self.hint.configure(text=f"КРИТИЧЕСКОЕ МЯУ +{amount}!" if critical else f"мяу +{amount}!", text_color=GOLD if critical else CATS[self.selected][3])
        self.spawn_click_effect(amount, critical)
        self.play_meme_sound()

    def spawn_click_effect(self, amount: int, critical: bool = False) -> None:
        """Лёгкий всплывающий эффект вместо резкой анимации."""
        symbols = (f"✦  +{amount}", f"♡  +{amount}", f"✧  +{amount}", f"мяу!  +{amount}") if not critical else (f"★ КРИТ +{amount}",)
        effect = ctk.CTkLabel(
            self.click_area,
            text=random.choice(symbols),
            font=ctk.CTkFont("Segoe UI", 16, "bold"),
            text_color=GOLD if critical else CATS[self.selected][3],
        )
        self.click_effects.append(effect)
        relx = random.uniform(0.31, 0.66)
        effect.place(relx=relx, rely=0.50, anchor="center")

        def float_up(step: int = 0) -> None:
            if not effect.winfo_exists():
                return
            if step >= 13:
                effect.destroy()
                if effect in self.click_effects:
                    self.click_effects.remove(effect)
                return
            effect.place_configure(rely=0.50 - step * 0.017)
            effect.after(38, lambda: float_up(step + 1))

        float_up()

    def cancel_hold(self, _event=None) -> None:
        # Раньше здесь было удержание Grumpy Cat; оно заменено рецептами.
        return

    def open_secret(self, secret_id: str) -> None:
        """Единая точка входа для правил из easter_eggs.py.

        Чтобы добавить ещё один секрет, зарегистрируй его правило в
        ``easter_eggs.py`` и фабрику экрана выше в ``__init__``.
        """
        stop_messages = {
            "roulette": "Остановлено: открыт модуль казино.",
            "phone": "Остановлено: открыт модуль телефона.",
            "builder": "Остановлено: открыт модуль строителя.",
            "volt": "Остановлено: открыт модуль электрика.",
            "miner": "Остановлено: открыт модуль шахтёра.",
            "lumberjack": "Остановлено: открыт модуль лесоруба.",
            "race_bettor": "Остановлено: открыт модуль ставок на скачки.",
            "slot_spinner": "Остановлено: открыт модуль слотов.",
            "blackjack": "Остановлено: открыт модуль блэкджека.",
            "poker": "Остановлено: открыт модуль покера.",
        }
        self.cancel_hold()
        self.secret_modules.deactivate_others(secret_id, stop_messages[secret_id])
        self.clear()
        self.current_view = secret_id
        module = self.secret_modules.get(secret_id)
        module.pack(fill="both", expand=True)
        module.activate()
