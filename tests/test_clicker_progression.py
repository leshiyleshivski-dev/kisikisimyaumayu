from __future__ import annotations

import sys
import ast
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kisiki.core import (  # noqa: E402
    CATS,
    PLACEHOLDER_SOUND,
    SOUND_FILES,
    cat_sound,
)
from kisiki.clicker import (  # noqa: E402
    adventure_reward,
    achievement_metric,
    click_power,
    daily_quests,
    decor_price,
    format_number,
    fresh_cat,
    level_progress,
    next_daily_streak,
    offline_income,
    passive_income,
    upgrade_price,
)


class ClickerProgressionTests(unittest.TestCase):
    def test_claim_handlers_refresh_in_place_without_rebuilding_screen(self) -> None:
        tree = ast.parse((ROOT / "kisiki" / "app.py").read_text(encoding="utf-8"))
        handlers = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"claim_daily_reward", "claim_quest", "claim_achievement"}
        }

        for name, handler in handlers.items():
            called_methods = {
                call.func.attr
                for call in ast.walk(handler)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            }
            self.assertNotIn("show_progress", called_methods, name)
            if name != "claim_daily_reward":
                self.assertIn("refresh_progress_view", called_methods, name)

    def test_switching_adventure_cat_does_not_rebuild_cards(self) -> None:
        tree = ast.parse((ROOT / "kisiki" / "app.py").read_text(encoding="utf-8"))
        handler = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "select_adventure_cat"
        )
        called_methods = {
            call.func.attr
            for call in ast.walk(handler)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        }

        self.assertNotIn("build_adventure_panel", called_methods)
        self.assertIn("refresh_adventure_selector", called_methods)
        self.assertIn("refresh_adventure_options", called_methods)

    def test_click_and_passive_upgrades_stack(self) -> None:
        cat = fresh_cat()
        cat.update({"paw": 2, "laser": 1, "crown": 1, "treat": 2, "basket": 1, "cafe": 1})

        self.assertEqual(click_power(cat), 8)
        self.assertEqual(passive_income(cat), 33)

    def test_club_decor_applies_global_bonuses(self) -> None:
        cat = fresh_cat()
        cat.update({"paw": 9, "treat": 10})
        decor = {"scratcher": 2, "lamp": 2, "window": 2, "radio": 0}

        self.assertEqual(click_power(cat, decor), 11)
        self.assertEqual(passive_income(cat, decor), 12)
        self.assertGreater(decor_price({"scratcher": 1}, "scratcher"), 1_500)

    def test_longer_adventures_pay_more(self) -> None:
        cat = fresh_cat()
        cat.update({"paw": 2, "treat": 1})

        self.assertGreater(adventure_reward(cat, "roofs"), adventure_reward(cat, "yard"))
        self.assertGreater(adventure_reward(cat, "night"), adventure_reward(cat, "roofs"))

    def test_upgrade_price_grows_from_catalog_base(self) -> None:
        cat = fresh_cat()
        first = upgrade_price(cat, "paw")
        cat["paw"] = 3

        self.assertEqual(first, 25)
        self.assertGreater(upgrade_price(cat, "paw"), first)

    def test_levels_have_increasing_thresholds(self) -> None:
        self.assertEqual(level_progress(0), (1, 0, 100))
        level, progress, needed = level_progress(100)

        self.assertEqual(level, 2)
        self.assertEqual(progress, 0)
        self.assertGreater(needed, 100)

    def test_offline_income_is_reduced_and_capped_at_eight_hours(self) -> None:
        cat = fresh_cat()
        cat["treat"] = 4

        short_rewards, short_total = offline_income([cat], 10)
        long_rewards, long_total = offline_income([cat], 24 * 3600)

        self.assertEqual(short_rewards, [30])
        self.assertEqual(short_total, 30)
        self.assertEqual(long_rewards, [4 * 8 * 3600 * 3 // 4])
        self.assertEqual(long_total, long_rewards[0])

        decorated_rewards, _ = offline_income([cat], 10, decor={"window": 2})
        self.assertEqual(decorated_rewards, [34])

    def test_daily_streak_continues_resets_and_does_not_double_claim(self) -> None:
        current = date(2026, 8, 21)

        self.assertEqual(next_daily_streak("2026-08-20", 4, current), 5)
        self.assertEqual(next_daily_streak("2026-08-18", 4, current), 1)
        self.assertEqual(next_daily_streak("2026-08-21", 4, current), 4)

    def test_daily_quests_scale_but_keep_three_clear_goals(self) -> None:
        low = daily_quests(1)
        high = daily_quests(12)

        self.assertEqual(len(low), 3)
        self.assertGreater(high[1]["target"], low[1]["target"])

    def test_achievement_metrics_cover_the_whole_club(self) -> None:
        cats = [fresh_cat() for _ in range(4)]
        for cat in cats:
            cat.update({"meows": 1_000, "taps": 20, "treat": 1})
        game = {"cats": cats, "lifetime_meows": 4_000}

        self.assertEqual(achievement_metric(game, "taps"), 80)
        self.assertEqual(achievement_metric(game, "all_cats"), 1_000)
        self.assertEqual(achievement_metric(game, "passive"), 4)

        game.update({"adventures_completed": 7, "decor": {"lamp": 2, "window": 1}})
        self.assertEqual(achievement_metric(game, "adventures"), 7)
        self.assertEqual(achievement_metric(game, "decor"), 3)

    def test_large_numbers_are_compact_and_readable(self) -> None:
        self.assertEqual(format_number(999), "999")
        self.assertEqual(format_number(12_500), "12,5 тыс.")
        self.assertEqual(format_number(2_000_000), "2 млн")


class CatSoundTests(unittest.TestCase):
    """Звук не должен ронять игру, когда котиков больше, чем звуков."""

    def test_each_listed_sound_belongs_to_its_cat(self) -> None:
        for index, filename in enumerate(SOUND_FILES):
            self.assertEqual(cat_sound(index), filename)

    def test_cats_without_a_sound_get_the_placeholder(self) -> None:
        # Ровно тот случай, ради которого функция и появилась: котиков в CATS
        # становится больше, чем записей в SOUND_FILES.
        for index in range(len(SOUND_FILES), len(SOUND_FILES) + 6):
            self.assertEqual(cat_sound(index), PLACEHOLDER_SOUND)

    def test_every_current_cat_resolves_to_a_file_on_disk(self) -> None:
        for index in range(len(CATS)):
            self.assertTrue(
                (ROOT / "sounds" / cat_sound(index)).is_file(),
                f"нет файла звука для котика {index}: {cat_sound(index)}",
            )

    def test_placeholder_file_exists(self) -> None:
        self.assertTrue((ROOT / "sounds" / PLACEHOLDER_SOUND).is_file())


if __name__ == "__main__":
    unittest.main()
