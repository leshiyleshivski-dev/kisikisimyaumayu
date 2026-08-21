"""Majestic quarry helper for the 2560x1440 mining mini-game."""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import customtkinter as ctk
import mss
import numpy as np

from ..core import (
    APP_BG, MINT, MUTED, SURFACE_ALT, TEXT, VK_E, VK_F9, VK_F11,
    activate_window, client_bounds, cursor_position, find_game_window,
    resource_path, rounded_photo, send_left_click, user32, window_title,
)
from ..clicker import today_key
from .miner_vision import (
    ORE_NAMES, ORE_TYPES, find_ore_targets, is_supported_miner_resolution,
    miner_overlay_visible, mining_progress_visible, read_ore_notification,
)
from .ui import connection_panel, hotkey_bar, module_header, panel


class MinerModule(ctk.CTkFrame):
    """Mine one rock per F9 press and keep confirmed quarry statistics."""

    SCAN_INTERVAL_MS = 55
    STRIKE_INTERVAL_RANGE_SECONDS = (0.95, 1.25)
    STRIKE_HOLD_RANGE_SECONDS = (0.055, 0.12)
    MAX_STRIKING_SECONDS = 38.0
    MAX_STRIKES = 30
    ACTIVATION_TIMEOUT_SECONDS = 5.0
    TARGET_STALL_SECONDS = 6.0
    TARGET_SETTLE_RANGE_SECONDS = (0.20, 0.34)
    TARGET_RETRY_SECONDS = 0.50
    MAX_ATTEMPTS_PER_TARGET = 3
    RESULT_TIMEOUT_SECONDS = 7.0
    MAX_TARGET_ATTEMPTS = 45

    def __init__(
        self,
        parent: ctk.CTkFrame,
        on_back: Callable[[], None],
        *,
        daily_stats: dict | None = None,
        on_stats_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, fg_color=APP_BG, corner_radius=0)
        self.on_back = on_back
        self.daily_stats = daily_stats if isinstance(daily_stats, dict) else {}
        self.on_stats_change = on_stats_change
        self.session_counts = {key: 0 for key, _title in ORE_TYPES}
        self.session_unknown = 0
        self.session_total = 0

        self.active = False
        self.running = False
        self.phase = "idle"
        self.game_window: int | None = None
        self.previous_window: int | None = None
        self.saved_cursor: tuple[int, int] | None = None
        self.strike_count = 0
        self.target_attempts = 0
        self.target_history: list[tuple[int, int, int, float]] = []
        self.overlay_missing_frames = 0
        self.phase_started_at = 0.0
        self.next_action_at = 0.0
        self.last_capture_at = 0.0
        self.last_target_at = 0.0
        self.last_daily_check = 0.0
        self.keys = {key: False for key in (VK_E, VK_F9, VK_F11)}

        self.process = ctk.StringVar(value="GTA5.exe")
        self.connection = ctk.StringVar(value="Ищу GTA5.exe…")
        self.resolution = ctk.StringVar(value="Разрешение ещё не проверено")
        self.status = ctk.StringVar(value="Включи автодетект один раз, затем начинай каждый камень клавишей E.")
        self.stage = ctk.StringVar(value="АВТОДЕТЕКТ ВЫКЛЮЧЕН")
        self.session_total_text = ctk.StringVar(value="0")
        self.daily_total_text = ctk.StringVar(value="0")
        self.ore_stat_text = {
            key: ctk.StringVar(value="сеанс 0  ·  день 0") for key, _title in ORE_TYPES
        }
        self.unknown_text = ctk.StringVar(value="Нераспознано: 0 / 0")
        self.ore_images: dict[str, object] = {}

        self.ensure_daily_stats(notify=False)
        self.build_ui()
        self.refresh_stats_view()
        self.after(40, self.poll_hotkeys)
        self.after(self.SCAN_INTERVAL_MS, self.scan_tick)
        self.after(0, self.refresh_connection)

    def build_ui(self) -> None:
        accent = "#E2B85B"
        module_header(
            self, code="ORE", title="Ore Hunt",
            subtitle="автодобыча и журнал руды карьерщика",
            accent=accent, on_back=self.on_back,
        )
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=42, pady=(0, 24))
        self.indicator = connection_panel(
            body, process=self.process, connection=self.connection,
            accent=accent, on_check=self.refresh_connection, trailing=self.resolution,
        )

        workspace = ctk.CTkFrame(body, fg_color="transparent")
        workspace.pack(fill="both", expand=True)
        workspace.grid_columnconfigure(0, weight=5, uniform="miner")
        workspace.grid_columnconfigure(1, weight=7, uniform="miner")
        workspace.grid_rowconfigure(0, weight=1)

        live = panel(workspace, "LIVE  /  ДОБЫЧА", accent)
        live.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        resolution_row = ctk.CTkFrame(live, fg_color="#24291F", corner_radius=13)
        resolution_row.pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkLabel(
            resolution_row, text="2K  2560×1440", height=26, corner_radius=8,
            fg_color="#3E5135", text_color="#AEE69B",
            font=ctk.CTkFont("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(10, 6), pady=9)
        ctk.CTkLabel(
            resolution_row, text="FULL HD  1920×1080", height=26,
            corner_radius=8, fg_color="#3E5135", text_color="#AEE69B",
            font=ctk.CTkFont("Segoe UI", 8, "bold"),
        ).pack(side="left", padx=(0, 8), pady=9)

        automation = ctk.CTkFrame(live, fg_color=SURFACE_ALT, corner_radius=14)
        automation.pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkLabel(
            automation, text="АВТОМАТИКА", font=ctk.CTkFont("Segoe UI", 8, "bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=12, pady=(10, 5))
        ctk.CTkLabel(
            automation,
            text="F9 включается один раз · E ставит камень в ожидание\nУдары — только после синей полосы · затем точечный сбор руды",
            font=ctk.CTkFont("Segoe UI", 8, "bold"), text_color=accent,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 10))

        monitor = ctk.CTkFrame(live, fg_color="#181F2C", corner_radius=15)
        monitor.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        ctk.CTkLabel(
            monitor, textvariable=self.stage, font=ctk.CTkFont("Segoe UI", 16, "bold"),
            text_color=accent,
        ).pack(padx=14, pady=(18, 5))
        ctk.CTkLabel(
            monitor, textvariable=self.status, font=ctk.CTkFont("Segoe UI", 9),
            text_color=MUTED, justify="center", wraplength=315,
        ).pack(fill="x", padx=18, pady=(0, 14))
        self.main_button = ctk.CTkButton(
            live, text="Включить автодетект  ·  F9", command=self.toggle,
            height=48, corner_radius=14, fg_color=accent, hover_color="#EDC66E",
            text_color="#17130B", font=ctk.CTkFont("Segoe UI", 12, "bold"),
        )
        self.main_button.pack(fill="x", padx=14, pady=(0, 14))

        stats = panel(
            workspace, "СТАТИСТИКА РУДЫ  /  MAJESTIC", accent,
            title_font_size=11,
        )
        stats.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        totals = ctk.CTkFrame(stats, fg_color="#24291F", corner_radius=14)
        totals.pack(fill="x", padx=14, pady=(0, 9))
        for column in range(2):
            totals.grid_columnconfigure(column, weight=1)
        self._total_card(totals, 0, "ЗА СЕАНС", self.session_total_text, accent)
        self._total_card(totals, 1, "ЗА СЕГОДНЯ", self.daily_total_text, "#AEE69B")

        ore_grid = ctk.CTkFrame(stats, fg_color="transparent")
        ore_grid.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        for column in range(3):
            ore_grid.grid_columnconfigure(column, weight=1, uniform="ore")
        for index, (ore_key, title) in enumerate(ORE_TYPES):
            card = ctk.CTkFrame(ore_grid, fg_color=SURFACE_ALT, corner_radius=11)
            card.grid(row=index // 3, column=index % 3, padx=4, pady=4, sticky="nsew")
            card.grid_columnconfigure(1, weight=1)
            icon_path = resource_path("assets", "ores", f"{index + 1:02d}_{ore_key}.png")
            try:
                self.ore_images[ore_key] = rounded_photo(icon_path, 40, 40)
            except Exception:
                self.ore_images[ore_key] = None
            if self.ore_images[ore_key] is not None:
                ctk.CTkLabel(
                    card, text="", image=self.ore_images[ore_key], width=34, height=34,
                ).grid(row=0, column=0, rowspan=2, padx=(8, 5), pady=8)
            ctk.CTkLabel(
                card, text=title.upper(), font=ctk.CTkFont("Segoe UI", 10, "bold"),
                text_color=TEXT,
            ).grid(row=0, column=1, sticky="sw", padx=(0, 6), pady=(9, 0))
            ctk.CTkLabel(
                card, textvariable=self.ore_stat_text[ore_key],
                font=ctk.CTkFont("Segoe UI", 9), text_color=MUTED,
            ).grid(row=1, column=1, sticky="nw", padx=(0, 6), pady=(0, 8))
        ctk.CTkLabel(
            stats, textvariable=self.unknown_text,
            font=ctk.CTkFont("Segoe UI", 10), text_color=MUTED,
        ).pack(anchor="w", padx=18, pady=(2, 3))
        ctk.CTkLabel(
            stats, text="Дневной счётчик автоматически обнуляется в 00:00.",
            font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=accent,
        ).pack(anchor="w", padx=18, pady=(0, 12))
        hotkey_bar(
            body,
            (("F9", "автодетект вкл / выкл"), ("E", "следующий камень"), ("F11", "аварийный стоп")),
            accent,
        )

    @staticmethod
    def _total_card(parent: ctk.CTkFrame, column: int, title: str, value: ctk.StringVar, color: str) -> None:
        card = ctk.CTkFrame(parent, fg_color="transparent")
        card.grid(row=0, column=column, padx=16, pady=11, sticky="ew")
        ctk.CTkLabel(
            card, text=title, font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=MUTED,
        ).pack()
        ctk.CTkLabel(
            card, textvariable=value, font=ctk.CTkFont("Segoe UI", 22, "bold"),
            text_color=color,
        ).pack()

    def ensure_daily_stats(self, *, notify: bool = True) -> bool:
        changed = self.daily_stats.get("date") != today_key()
        ores = self.daily_stats.get("ores")
        if changed or not isinstance(ores, dict):
            self.daily_stats.clear()
            self.daily_stats.update({
                "date": today_key(),
                "total": 0,
                "unknown": 0,
                "ores": {key: 0 for key, _title in ORE_TYPES},
            })
            if notify and self.on_stats_change is not None:
                self.on_stats_change()
        else:
            for key, _title in ORE_TYPES:
                ores[key] = max(0, int(ores.get(key, 0)))
            self.daily_stats["total"] = max(0, int(self.daily_stats.get("total", 0)))
            self.daily_stats["unknown"] = max(0, int(self.daily_stats.get("unknown", 0)))
        return changed

    def refresh_stats_view(self) -> None:
        self.ensure_daily_stats()
        daily_ores = self.daily_stats["ores"]
        self.session_total_text.set(str(self.session_total))
        self.daily_total_text.set(str(self.daily_stats["total"]))
        for key, _title in ORE_TYPES:
            self.ore_stat_text[key].set(
                f"сеанс {self.session_counts[key]}  ·  день {daily_ores[key]}"
            )
        self.unknown_text.set(
            f"Нераспознано: сеанс {self.session_unknown}  ·  день {self.daily_stats['unknown']}"
        )

    def record_result(self, ore_key: str | None) -> str:
        self.ensure_daily_stats()
        self.session_total += 1
        self.daily_stats["total"] += 1
        if ore_key in ORE_NAMES:
            self.session_counts[ore_key] += 1
            self.daily_stats["ores"][ore_key] += 1
            result = f"{ORE_NAMES[ore_key]} руда записана в статистику."
        else:
            self.session_unknown += 1
            self.daily_stats["unknown"] += 1
            result = "Руда добыта, но её вид не распознан. Общий счётчик обновлён."
        self.refresh_stats_view()
        if self.on_stats_change is not None:
            self.on_stats_change()
        return result

    def refresh_connection(self) -> None:
        if not self.winfo_exists():
            return
        self.game_window = find_game_window(self.process.get())
        if self.game_window:
            title = window_title(self.game_window)
            self.connection.set(f"Подключено · {title[:22]}" if title else "Подключено")
            self.indicator.configure(text_color=MINT)
            bounds = client_bounds(self.game_window)
            if bounds and is_supported_miner_resolution(bounds[2], bounds[3]):
                self.resolution.set(f"{bounds[2]}×{bounds[3]} · готово")
            elif bounds:
                self.resolution.set(f"{bounds[2]}×{bounds[3]} · не поддерживается")
            else:
                self.resolution.set("Размер окна не прочитан")
        else:
            self.connection.set("Не подключено")
            self.resolution.set("Разрешение ещё не проверено")
            self.indicator.configure(text_color="#F05A67")
        self.after(2000, self.refresh_connection)

    def capture_game_image(self) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            return None
        bounds = client_bounds(self.game_window)
        if not bounds:
            return None
        left, top, width, height = bounds
        try:
            with mss.mss() as screen:
                image = np.asarray(screen.grab({
                    "left": left, "top": top, "width": width, "height": height,
                }))[:, :, :3]
        except Exception:
            return None
        return image, bounds

    def toggle(self) -> None:
        if self.running:
            self.stop("Автодетект выключен через F9.")
            return
        self.start_cycle()

    def start_cycle(self) -> None:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            self.status.set(f"Процесс «{self.process.get()}» не найден.")
            return
        bounds = client_bounds(self.game_window)
        if not bounds:
            self.status.set("Не получилось прочитать размер игрового окна.")
            return
        if not is_supported_miner_resolution(bounds[2], bounds[3]):
            self.status.set(
                f"Поддерживаются 2560×1440 и 1920×1080. Обнаружено {bounds[2]}×{bounds[3]}."
            )
            self.stage.set("РАЗРЕШЕНИЕ НЕ ПОДДЕРЖИВАЕТСЯ")
            return
        previous = user32.GetForegroundWindow()
        if not activate_window(self.game_window):
            self.status.set("Не получилось вывести GTA на передний план.")
            return
        self.previous_window = previous if previous and previous != self.game_window else None
        self.saved_cursor = None
        self.running = True
        self.phase = "watching"
        self.phase_started_at = time.monotonic()
        self.stage.set("АВТОДЕТЕКТ ВКЛЮЧЁН")
        self.status.set("Подойди к камню и нажми E. Удары начнутся автоматически и продолжатся до появления стола.")
        self.main_button.configure(
            text="Выключить автодетект  ·  F9", fg_color="#E95E69",
            hover_color="#C94C57", text_color=TEXT,
        )

    def _arm_cycle(self) -> None:
        if not self.running or self.phase != "watching":
            return
        if not self._foreground_is_game():
            self.status.set("Нажатие E замечено, но GTA не активна — клики не отправлены.")
            return
        self.saved_cursor = cursor_position()
        self.strike_count = 0
        self.target_attempts = 0
        self.target_history.clear()
        self.overlay_missing_frames = 0
        self.phase = "arming"
        self.phase_started_at = time.monotonic()
        self.next_action_at = self.phase_started_at + random.uniform(0.28, 0.45)
        self.last_target_at = self.phase_started_at
        self.stage.set("ПРОВЕРЯЮ ЭКРАН")
        self.status.set("E замечена. Жду справа полосу «Добыча руды»; до неё мышь не нажимаю.")

    def _enter_collecting(self, now: float, message: str) -> None:
        if self.saved_cursor is None:
            self.saved_cursor = cursor_position()
        self.target_attempts = 0
        self.target_history.clear()
        self.overlay_missing_frames = 0
        self.phase = "collecting"
        self.phase_started_at = now
        self.last_target_at = now
        self.next_action_at = now
        self.stage.set("СОБИРАЮ РУДУ")
        self.status.set(message)

    def _foreground_is_game(self) -> bool:
        return bool(self.game_window and user32.GetForegroundWindow() == self.game_window)

    def _strike_step(self, now: float) -> None:
        if now < self.next_action_at:
            return
        if not self._foreground_is_game():
            self.stop("Добыча остановлена: GTA потеряла фокус до завершения ударов.")
            return
        captured = self.capture_game_image()
        if captured is None:
            self.stop("Потеряно окно GTA во время ударов по камню.")
            return
        if miner_overlay_visible(captured[0]):
            self._enter_collecting(now, "Стол появился — прекращаю удары и собираю включения.")
            return
        if now - self.phase_started_at >= self.MAX_STRIKING_SECONDS or self.strike_count >= self.MAX_STRIKES:
            self.stop("Стол с рудой не появился после 30 ударов. Автодетект остановлен без дальнейших кликов.")
            return
        hold_seconds = random.uniform(*self.STRIKE_HOLD_RANGE_SECONDS)
        if not send_left_click(hold_seconds):
            self.stop("Windows не принял левый клик. Добыча остановлена.")
            return
        self.strike_count += 1
        self.stage.set(f"УДАР {self.strike_count} · ЖДУ СТОЛ")
        delay = random.uniform(*self.STRIKE_INTERVAL_RANGE_SECONDS)
        self.status.set(f"Разбиваю камень. Следующий удар примерно через {delay:.1f} с.")
        self.next_action_at = now + delay

    def _arming_step(self, now: float) -> None:
        """Wait for Majestic to acknowledge E before sending any mouse input."""
        if now < self.next_action_at:
            return
        if not self._foreground_is_game():
            self.phase = "watching"
            self.stage.set("АВТОДЕТЕКТ ВКЛЮЧЁН")
            self.status.set("GTA потеряла фокус. Удары не отправлены; жду новую E.")
            return
        captured = self.capture_game_image()
        if captured is None:
            self.stop("Не удалось получить кадр GTA перед началом добычи.")
            return
        image, bounds = captured
        if not is_supported_miner_resolution(bounds[2], bounds[3]):
            self.stop("Размер окна GTA изменился. Нужны 2560×1440 или 1920×1080.")
            return
        if miner_overlay_visible(image):
            self._enter_collecting(now, "Стол уже открыт — собираю включения без лишних ударов.")
            return
        if mining_progress_visible(image):
            self.phase = "striking"
            self.phase_started_at = now
            self.next_action_at = now
            self.stage.set("ДОБЫЧА ПОДТВЕРЖДЕНА")
            self.status.set("Полоса «Добыча руды» появилась. Начинаю бить камень.")
            return
        if now - self.phase_started_at >= self.ACTIVATION_TIMEOUT_SECONDS:
            self.phase = "watching"
            self.phase_started_at = now
            self.saved_cursor = None
            self.stage.set("АВТОДЕТЕКТ ВКЛЮЧЁН")
            self.status.set("Полоса «Добыча руды» не появилась — в воздух не бью. Подойди к камню и нажми E ещё раз.")
            return
        self.next_action_at = now + 0.10
        self.stage.set("ЖДУ ПОЛОСУ ДОБЫЧИ")
        self.status.set("E замечена. Жду подтверждение игры справа; мышь пока не нажимаю.")

    def _watching_step(self, now: float) -> None:
        if now - self.last_capture_at < 0.16:
            return
        self.last_capture_at = now
        captured = self.capture_game_image()
        if captured is None:
            self.stop("Потеряно окно GTA во время автодетекта.")
            return
        image, bounds = captured
        if not is_supported_miner_resolution(bounds[2], bounds[3]):
            self.stop("Размер окна GTA изменился. Нужны 2560×1440 или 1920×1080.")
            return
        if miner_overlay_visible(image):
            if self._foreground_is_game():
                self._enter_collecting(now, "Стол найден автодетектом. Ищу включения на камнях.")
            else:
                self.status.set("Стол с рудой найден. Верни фокус в GTA, чтобы начать сбор.")

    def _target_record(self, point: tuple[int, int]) -> tuple[int, tuple[int, int, int, float]] | None:
        x, y = point
        for index, record in enumerate(self.target_history):
            old_x, old_y, _attempts, _last_attempt = record
            if (x - old_x) ** 2 + (y - old_y) ** 2 <= 32 ** 2:
                return index, record
        return None

    def _target_ready(self, point: tuple[int, int], now: float) -> bool:
        matched = self._target_record(point)
        if matched is None:
            return True
        _index, (_x, _y, attempts, last_attempt) = matched
        return attempts < self.MAX_ATTEMPTS_PER_TARGET and now - last_attempt >= self.TARGET_RETRY_SECONDS

    def _remember_target_attempt(self, point: tuple[int, int], now: float) -> int:
        matched = self._target_record(point)
        if matched is None:
            self.target_history.append((point[0], point[1], 1, now))
            return 1
        index, (old_x, old_y, attempts, _last_attempt) = matched
        attempts += 1
        self.target_history[index] = (old_x, old_y, attempts, now)
        return attempts

    def _collecting_step(self, now: float) -> None:
        if now < self.next_action_at:
            return
        if now - self.last_capture_at < 0.11:
            return
        self.last_capture_at = now
        captured = self.capture_game_image()
        if captured is None:
            self.stop("Потеряно окно GTA во время сбора руды.")
            return
        image, bounds = captured
        if not miner_overlay_visible(image):
            self.overlay_missing_frames += 1
            if self.overlay_missing_frames >= 2:
                self.phase = "waiting_result"
                self.phase_started_at = now
                self.stage.set("ПРОВЕРЯЮ РЕЗУЛЬТАТ")
                self.status.set("Мини-игра закрылась. Жду уведомление о полученной руде.")
                self.restore_cursor()
            return
        self.overlay_missing_frames = 0
        if not self._foreground_is_game():
            self.stop("Добыча остановлена: GTA потеряла фокус во время сбора.")
            return
        detected = find_ore_targets(image)
        available = [target for target in detected if self._target_ready((target[0], target[1]), now)]
        if available:
            target_x, target_y, _score, kind = available[0]
            screen_x, screen_y = bounds[0] + target_x, bounds[1] + target_y
            if not user32.SetCursorPos(screen_x, screen_y) or not send_left_click(random.uniform(0.045, 0.085)):
                self.stop("Windows не принял клик по включению руды.")
                return
            clicked_at = time.monotonic()
            target_attempt = self._remember_target_attempt((target_x, target_y), clicked_at)
            self.target_attempts += 1
            self.last_target_at = clicked_at
            self.next_action_at = clicked_at + random.uniform(*self.TARGET_SETTLE_RANGE_SECONDS)
            label = "цветное" if kind == "color" else "контрастное"
            self.stage.set(f"КЛИК ПО РУДЕ · {self.target_attempts}")
            self.status.set(
                f"Найдено {label} включение. Проверяю, исчезло ли оно после клика"
                + (f" (попытка {target_attempt})." if target_attempt > 1 else ".")
            )
            if self.target_attempts >= self.MAX_TARGET_ATTEMPTS:
                self.stop("Достигнут безопасный лимит кликов. Мини-игра осталась открыта.")
            return
        if detected:
            if now - self.last_target_at >= self.TARGET_STALL_SECONDS:
                self.stop(
                    "Вкрапления не исчезли после повторных кликов. Ничего не считаю завершённым — оставил игру для ручной проверки."
                )
            else:
                self.status.set("Вкрапления ещё видны. Жду реакцию игры или готовлю повторный клик.")
        elif now - self.last_target_at >= self.TARGET_STALL_SECONDS:
            self.stop("Стол всё ещё открыт, но руда не распознана. Ничего не считаю завершённым — оставил игру для ручной проверки.")

    def _waiting_result_step(self, now: float) -> None:
        if now - self.last_capture_at < 0.12:
            return
        self.last_capture_at = now
        captured = self.capture_game_image()
        if captured is None:
            self.stop("Потеряно окно GTA до подтверждения результата.")
            return
        image, _bounds = captured
        toast_present, ore_key, _score = read_ore_notification(image)
        if toast_present:
            message = self.record_result(ore_key)
            self.finish(message)
            return
        if now - self.phase_started_at >= self.RESULT_TIMEOUT_SECONDS:
            self.finish("Мини-игра завершилась, но уведомление о руде не найдено — статистика не изменена.")

    def restore_cursor(self) -> None:
        if self.saved_cursor is not None:
            user32.SetCursorPos(*self.saved_cursor)
            self.saved_cursor = None

    def restore_previous_window(self) -> None:
        previous, self.previous_window = self.previous_window, None
        if previous and user32.IsWindow(previous):
            activate_window(previous)

    def finish(self, message: str) -> None:
        self.restore_cursor()
        if not self.running:
            return
        self.phase = "watching"
        self.phase_started_at = time.monotonic()
        self.stage.set("АВТОДЕТЕКТ ВКЛЮЧЁН")
        self.status.set(f"{message}\nЖду следующий камень — подойди к нему и нажми E.")
        self.main_button.configure(
            text="Выключить автодетект  ·  F9", fg_color="#E95E69",
            hover_color="#C94C57", text_color=TEXT,
        )

    def stop(self, message: str) -> None:
        self.running = False
        self.phase = "idle"
        self.restore_cursor()
        self.restore_previous_window()
        self.stage.set("ОСТАНОВЛЕНО")
        self.status.set(message)
        self.main_button.configure(
            text="Включить автодетект  ·  F9", fg_color="#E2B85B",
            hover_color="#EDC66E", text_color="#17130B",
        )

    def activate(self) -> None:
        self.active = True
        self.ensure_daily_stats()
        self.refresh_stats_view()
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def deactivate(self, message: str) -> None:
        self.stop(message)
        self.active = False
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def scan_tick(self) -> None:
        now = time.monotonic()
        if now - self.last_daily_check >= 1.0:
            self.last_daily_check = now
            if self.ensure_daily_stats():
                self.refresh_stats_view()
        if self.winfo_exists() and self.active and self.running:
            if self.phase == "watching":
                self._watching_step(now)
            elif self.phase == "arming":
                self._arming_step(now)
            elif self.phase == "striking":
                self._strike_step(now)
            elif self.phase == "collecting":
                self._collecting_step(now)
            elif self.phase == "waiting_result":
                self._waiting_result_step(now)
        if self.winfo_exists():
            self.after(self.SCAN_INTERVAL_MS, self.scan_tick)

    def poll_hotkeys(self) -> None:
        actions = {VK_F9: self.toggle, VK_F11: lambda: self.stop("Экстренно остановлено через F11.")}
        for key, action in actions.items():
            down = bool(user32.GetAsyncKeyState(key) & 0x8000)
            if self.active and down and not self.keys[key]:
                action()
            self.keys[key] = down
        e_down = bool(user32.GetAsyncKeyState(VK_E) & 0x8000)
        if (
            self.active and self.running and self.phase == "watching"
            and e_down and not self.keys[VK_E]
        ):
            self._arm_cycle()
        self.keys[VK_E] = e_down
        if self.winfo_exists():
            self.after(40, self.poll_hotkeys)
