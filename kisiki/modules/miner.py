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
    miner_overlay_visible, mining_progress_fill, mining_progress_visible,
    read_ore_notification,
)
from .ui import connection_panel, hotkey_bar, module_header, panel


class MinerModule(ctk.CTkFrame):
    """Mine one rock per F9 press and keep confirmed quarry statistics."""

    SCAN_INTERVAL_MS = 55
    # How the swing cycle actually works, measured frame by frame on the
    # recordings of 22.08 (see tests/fixtures/miner/README.md):
    #
    #   credited step -> the game ignores the mouse for about 0.75 s
    #                 -> a click after that starts the next swing
    #                 -> the step is credited ~0.30 s later
    #
    # So the floor is ~1.05 s per hit, and hitting it is purely a question of
    # placing one click just after the deaf window ends. Clicking earlier is
    # not merely wasted: the click is dropped, and the module then sits out its
    # whole fallback interval, which is how the swing rate fell to 2.0 s.
    STRIKE_READY_DELAY_RANGE_SECONDS = (0.78, 0.88)
    # A click that still landed too early is dropped silently - the only sign
    # is that no step arrives. Retry after a gap slightly longer than the
    # ~0.30 s it takes a good click to be credited, so a swing that did land
    # reschedules from its own step and the retry never fires: one click per
    # hit in the normal case, and a 0.45 s recovery when a click was missed.
    STRIKE_RETRY_RANGE_SECONDS = (0.42, 0.52)
    # Used only before the first step has been credited, when there is nothing
    # to lock onto yet.
    STRIKE_INTERVAL_RANGE_SECONDS = (0.45, 0.60)
    STRIKE_HOLD_RANGE_SECONDS = (0.055, 0.12)
    # A swing moves the bar by 4-10% depending on the rock.
    PROGRESS_STEP = 0.03
    # Safety nets only - a rock really ends when the bar disappears. They are
    # counted in credited hits now: while this limit counted clicks, a rock
    # needing 15 hits burned through 24 "strikes" in half a minute and the
    # module walked away from a stone it was still working (a 1.5 s freeze at
    # 32.9 s of the 08-12-31 recording). Rocks take 10-16 hits; the bar cannot
    # step by less than about 4%, so 25 is the most a stone can need.
    MAX_STRIKING_SECONDS = 42.0
    MAX_STRIKES = 30
    # Clicks are not hits: an early one is dropped. Bound them separately so a
    # rock that never credits cannot turn into an endless click stream.
    MAX_STRIKE_CLICKS = 70
    STRIKE_OBSERVE_INTERVAL_SECONDS = 0.08
    PROGRESS_MISSING_FRAMES = 3
    ACTIVATION_TIMEOUT_SECONDS = 5.0
    TARGET_STALL_SECONDS = 6.0
    TARGET_SETTLE_RANGE_SECONDS = (0.20, 0.34)
    TARGET_RETRY_SECONDS = 0.50
    MAX_ATTEMPTS_PER_TARGET = 3
    # Three clicks half a second apart are all inside one short stretch of
    # time, and the game does refuse a sprite for a while: on the 10-43-16
    # table the module clicked a gold inclusion dead centre three times with
    # no effect, gave up, and the player collected that very sprite by hand
    # seventeen seconds later. So an exhausted target is not a lost one - it
    # goes back into the queue after a wait, instead of the table standing
    # open and idle for the rest of its life.
    TARGET_COOLDOWN_SECONDS = 2.5
    MAX_TARGET_ROUNDS = 4
    RESULT_TIMEOUT_SECONDS = 7.0
    MAX_TARGET_ATTEMPTS = 45
    # Phases from which E may start the next rock. The result check is
    # included on purpose: it runs for several seconds after a rock is
    # worked out, and the player is usually at the next stone by then.
    ARMABLE_PHASES = ("watching", "waiting_result")

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
        self.strike_clicks = 0
        self.target_attempts = 0
        self.target_rounds = 0
        self.target_history: list[tuple[int, int, int, float]] = []
        self.overlay_missing_frames = 0
        self.progress_missing_frames = 0
        self.progress_confirmed = False
        self.progress_fill: float | None = None
        self.result_allows_table = False
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
            text="F9 включается один раз · E ставит камень в ожидание\nУдары идут, пока горит синяя полоса · затем точечный сбор руды",
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
        if not self.running or self.phase not in self.ARMABLE_PHASES:
            return
        if not self._foreground_is_game():
            self.status.set("Нажатие E замечено, но GTA не активна — клики не отправлены.")
            return
        self.saved_cursor = cursor_position()
        self.strike_count = 0
        self.strike_clicks = 0
        self.target_attempts = 0
        self.target_rounds = 0
        self.target_history.clear()
        self.overlay_missing_frames = 0
        self.progress_missing_frames = 0
        self.progress_confirmed = False
        self.progress_fill = None
        self.result_allows_table = False
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
        self.result_allows_table = False
        self.phase = "collecting"
        self.phase_started_at = now
        self.last_target_at = now
        self.next_action_at = now
        self.stage.set("СОБИРАЮ РУДУ")
        self.status.set(message)

    def _enter_result_wait(self, now: float, message: str, *, allow_table: bool = False) -> None:
        """Stop every click and watch for the ore notification.

        ``allow_table`` is set only when the strike phase ended by itself: the
        sorting table can open a moment after the rock is worked out. Coming
        back from an already finished table must not re-enter collecting,
        otherwise the same inclusions would be clicked a second time.
        """
        self.restore_cursor()
        self.phase = "waiting_result"
        self.phase_started_at = now
        self.result_allows_table = allow_table
        self.overlay_missing_frames = 0
        self.stage.set("ПРОВЕРЯЮ РЕЗУЛЬТАТ")
        self.status.set(message)

    def _pause_for_manual_review(self, now: float, message: str) -> None:
        """Keep auto-detect enabled while the player finishes this table."""
        self.restore_cursor()
        if not self.running:
            return
        self.phase = "manual_review"
        self.phase_started_at = now
        self.overlay_missing_frames = 0
        self.stage.set("НУЖНА РУЧНАЯ ПРОВЕРКА")
        self.status.set(
            f"{message}\nАвтодетект остаётся включён. Закончи этот стол вручную — "
            "после его закрытия программа продолжит со следующим камнем."
        )
        self.main_button.configure(
            text="Выключить автодетект  ·  F9", fg_color="#E95E69",
            hover_color="#C94C57", text_color=TEXT,
        )

    def _foreground_is_game(self) -> bool:
        return bool(self.game_window and user32.GetForegroundWindow() == self.game_window)

    def _strike_step(self, now: float) -> None:
        if not self._foreground_is_game():
            self.stop("Добыча остановлена: GTA потеряла фокус до завершения ударов.")
            return
        # The screen is inspected between strikes and not only right before the
        # next one. A rock is finished in ten to fifteen swings, so a check
        # tied to the click schedule noticed the end a whole strike too late.
        if now - self.last_capture_at >= self.STRIKE_OBSERVE_INTERVAL_SECONDS:
            self.last_capture_at = now
            captured = self.capture_game_image()
            if captured is None:
                self.stop("Потеряно окно GTA во время ударов по камню.")
                return
            image = captured[0]
            if miner_overlay_visible(image):
                self._enter_collecting(now, "Стол появился — прекращаю удары и собираю включения.")
                return
            # The blue «Добыча руды» bar is Majestic's own "this rock is still
            # being mined" flag: it disappears as soon as the rock is worked
            # out. Ending the strike phase on its loss is what keeps the
            # pickaxe from swinging at an already empty spot.
            fill = mining_progress_fill(image)
            if fill is not None:
                self.progress_confirmed = True
                self.progress_missing_frames = 0
                # A step on the bar is the one honest signal that a swing
                # landed, so it is what the counter reports and what the whole
                # phase is timed from. Counting clicks instead showed two hits
                # where the character had made one.
                if (
                    self.progress_fill is not None
                    and fill - self.progress_fill >= self.PROGRESS_STEP
                ):
                    self.strike_count += 1
                    self.next_action_at = now + random.uniform(
                        *self.STRIKE_READY_DELAY_RANGE_SECONDS
                    )
                    self.stage.set(f"УДАР {self.strike_count} · ДОБЫЧА ИДЁТ")
                    self.status.set(
                        f"Разбиваю камень. Полоса добычи: {fill * 100:.0f}%."
                    )
                self.progress_fill = fill
            elif self.progress_confirmed:
                self.progress_missing_frames += 1
                if self.progress_missing_frames >= self.PROGRESS_MISSING_FRAMES:
                    self._enter_result_wait(
                        now,
                        f"Полоса «Добыча руды» пропала после {self.strike_count} "
                        "ударов — камень отработан. Клики прекращены.",
                        allow_table=True,
                    )
                    return
        if now < self.next_action_at:
            return
        if (
            now - self.phase_started_at >= self.MAX_STRIKING_SECONDS
            or self.strike_count >= self.MAX_STRIKES
            or self.strike_clicks >= self.MAX_STRIKE_CLICKS
        ):
            self.finish(
                f"Полоса «Добыча руды» не пропала за {self.strike_count} ударов. "
                "Клики прекращены, но автодетект остаётся включён."
            )
            return
        hold_seconds = random.uniform(*self.STRIKE_HOLD_RANGE_SECONDS)
        if not send_left_click(hold_seconds):
            self.stop("Windows не принял левый клик. Добыча остановлена.")
            return
        self.strike_clicks += 1
        # This click either starts a swing or was still inside the deaf window
        # and was dropped. Only a step on the bar tells the two apart, so retry
        # soon and let the step reschedule properly when it arrives.
        self.next_action_at = now + random.uniform(
            *(self.STRIKE_RETRY_RANGE_SECONDS if self.strike_count
              else self.STRIKE_INTERVAL_RANGE_SECONDS)
        )

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
            self.last_capture_at = now
            self.progress_confirmed = True
            self.progress_missing_frames = 0
            self.progress_fill = mining_progress_fill(image)
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

    def _retry_round(self, now: float) -> bool:
        """Forgive the attempt counts once more, so a stubborn sprite is retried.

        Returns ``False`` once the rounds are used up, and the caller falls
        through to the manual-review pause it always had.
        """
        if self.target_rounds >= self.MAX_TARGET_ROUNDS:
            return False
        self.target_rounds += 1
        self.target_history.clear()
        self.last_target_at = now
        return True

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
                self._enter_result_wait(
                    now, "Мини-игра закрылась. Жду уведомление о полученной руде.",
                )
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
                self._pause_for_manual_review(
                    clicked_at,
                    "Достигнут безопасный лимит кликов. Мини-игра осталась открыта.",
                )
            return
        if detected:
            if now - self.last_target_at >= self.TARGET_COOLDOWN_SECONDS and self._retry_round(now):
                self.status.set(
                    f"Вкрапления не поддались — захожу на них ещё раз "
                    f"(круг {self.target_rounds})."
                )
            elif now - self.last_target_at >= self.TARGET_STALL_SECONDS:
                self._pause_for_manual_review(
                    now,
                    "Вкрапления не исчезли после повторных кликов. Ничего не считаю завершённым — оставил игру для ручной проверки."
                )
            else:
                self.status.set("Вкрапления ещё видны. Жду реакцию игры или готовлю повторный клик.")
        elif now - self.last_target_at >= self.TARGET_STALL_SECONDS:
            self._pause_for_manual_review(
                now,
                "Стол всё ещё открыт, но руда не распознана. Ничего не считаю завершённым — оставил игру для ручной проверки.",
            )

    def _manual_review_step(self, now: float) -> None:
        """Observe without clicking until the manually completed table closes."""
        if now - self.last_capture_at < 0.16:
            return
        self.last_capture_at = now
        captured = self.capture_game_image()
        if captured is None:
            self.stop("Потеряно окно GTA во время ручной проверки.")
            return
        image, _bounds = captured
        if miner_overlay_visible(image):
            self.overlay_missing_frames = 0
            return
        self.overlay_missing_frames += 1
        if self.overlay_missing_frames >= 2:
            self._enter_result_wait(
                now,
                "Стол закрыт вручную. Жду уведомление о руде; автодетект остаётся включён.",
            )

    def _waiting_result_step(self, now: float) -> None:
        if now - self.last_capture_at < 0.12:
            return
        self.last_capture_at = now
        captured = self.capture_game_image()
        if captured is None:
            self.stop("Потеряно окно GTA до подтверждения результата.")
            return
        image, _bounds = captured
        if self.result_allows_table and miner_overlay_visible(image):
            self._enter_collecting(now, "Стол открылся после ударов. Ищу включения на камнях.")
            return
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
            elif self.phase == "manual_review":
                self._manual_review_step(now)
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
            self.active and self.running and self.phase in self.ARMABLE_PHASES
            and e_down and not self.keys[VK_E]
        ):
            self._arm_cycle()
        self.keys[VK_E] = e_down
        if self.winfo_exists():
            self.after(40, self.poll_hotkeys)
