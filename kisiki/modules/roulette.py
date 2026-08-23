"""Roulette automation screen."""

from __future__ import annotations

import random
import threading
import time
import winsound

import cv2
import customtkinter as ctk
import mss
import numpy as np

from ..core import (
    APP_BG, BET_DELAY_SECONDS, BLACK_DIAMOND_RATIO,
    GAME_CURSOR_DISTANCE_MULTIPLIER, GAME_RECONNECT_SECONDS,
    GAME_ROUND_SECONDS, GOLD, MINT, MISSING_TIMER_SHUTDOWN_SECONDS,
    MUTED, RED_DIAMOND_RATIO, SURFACE, SURFACE_ALT, TEXT,
    TIMER_ACTIVATE_SETTLE_SECONDS, TIMER_SCAN_INTERVAL_SECONDS,
    TIME_READOUT_RATIO, VK_F4, VK_F9, VK_F11, activate_window,
    client_bounds, confine_cursor_to_client, cursor_position,
    find_game_window, move_relative_and_bet, resource_path, user32,
    window_title, winmm,
)
from .ui import connection_panel, hotkey_bar, module_header, panel, step_list

class RouletteModule(ctk.CTkFrame):
    """Встроенный, а не отдельный, экран V6."""

    def __init__(self, parent: ctk.CTkFrame, on_back, *, alert_sound_enabled: bool = True, on_alert_sound_change=None) -> None:
        super().__init__(parent, fg_color=APP_BG, corner_radius=0)
        self.on_back = on_back
        self.on_alert_sound_change = on_alert_sound_change
        self.alert_sound_enabled = alert_sound_enabled
        self.game_cursor_at: str | None = None
        self.game_window: int | None = None
        self.running = False
        self.active = False
        self.next_run: float | None = None
        self.last_bet_started: float | None = None
        self.last_bet_completed: float | None = None
        self.f9_started_at: float | None = None
        self.cycle_started_at: float | None = None
        self.cycle_count = 0
        self.last_trigger_lag = 0.0
        self.last_first_click_delay: float | None = None
        self.last_cycle_duration: float | None = None
        self.last_wait_after_black: float | None = None
        self.waiting_for_betting_window = False
        self.waiting_previous_window: int | None = None
        self.scan_ready_at = 0.0
        self.last_timer_scan_at = 0.0
        self.missing_timer_since: float | None = None
        self.last_timer_observation = "ожидание"
        self.next_delay_seconds: int | None = None
        self.warning_alerted = False
        self.keys = {key: False for key in (VK_F4, VK_F9, VK_F11)}
        self.process = ctk.StringVar(value="GTA5.exe")
        self.connection = ctk.StringVar(value="Ищу GTA5.exe…")
        self.point_status = ctk.StringVar(value="○ Точка не захвачена")
        self.status = ctk.StringVar(value="Ожидание подключения к игре")
        self.timer = ctk.StringVar(value="Таймер не запущен")
        self.build_ui()
        self.after(40, self.poll_hotkeys)
        self.after(100, self.tick)
        self.after(0, self.refresh_connection)

    def build_ui(self) -> None:
        accent = "#FF8F91"
        actions = module_header(
            self, code="RLT", title="Roulette Control",
            subtitle="умная проверка окна ставок и безопасные циклы",
            accent=accent, on_back=self.back,
        )
        self.alert_sound_button = ctk.CTkButton(
            actions, command=self.toggle_alert_sound, width=134, height=38,
            corner_radius=12, font=ctk.CTkFont("Segoe UI", 9, "bold"),
        )
        self.alert_sound_button.pack(side="right", padx=(0, 8))
        self.refresh_alert_sound_button()

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=42, pady=(0, 24))
        self.indicator = connection_panel(
            body, process=self.process, connection=self.connection,
            accent=accent, on_check=self.refresh_connection,
        )

        workspace = ctk.CTkFrame(body, fg_color="transparent")
        workspace.pack(fill="both", expand=True)
        workspace.grid_columnconfigure(0, weight=6, uniform="roulette")
        workspace.grid_columnconfigure(1, weight=5, uniform="roulette")
        workspace.grid_rowconfigure(0, weight=1)

        live = panel(workspace, "LIVE  /  ЦИКЛ СТАВОК", accent)
        live.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        wheel = ctk.CTkFrame(live, fg_color="#2A2029", corner_radius=17)
        wheel.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkLabel(
            wheel, text="◆   00:34 — 00:10   ◆", font=ctk.CTkFont("Segoe UI", 20, "bold"),
            text_color=accent,
        ).pack(pady=(16, 3))
        self.point_badge = ctk.CTkLabel(
            wheel, textvariable=self.point_status, height=26, corner_radius=8,
            fg_color="#352936", font=ctk.CTkFont("Segoe UI", 9, "bold"),
            text_color="#D8A3AE",
        )
        self.point_badge.pack(pady=(0, 15))
        ctk.CTkLabel(
            live, textvariable=self.timer, font=ctk.CTkFont("Segoe UI", 21, "bold"),
            text_color=TEXT,
        ).pack(padx=18, pady=(7, 4))
        ctk.CTkLabel(
            live, textvariable=self.status, font=ctk.CTkFont("Segoe UI", 10),
            text_color=MUTED, wraplength=410, justify="center",
        ).pack(fill="x", padx=24, pady=(0, 16))
        self.main_button = ctk.CTkButton(
            live, text="Запустить  ·  F9", command=self.toggle,
            height=50, corner_radius=14, font=ctk.CTkFont("Segoe UI", 13, "bold"),
            fg_color=accent, hover_color="#FFA3A5", text_color="#21171A",
        )
        self.main_button.pack(fill="x", side="bottom", padx=16, pady=16)

        guide = panel(workspace, "СЦЕНАРИЙ ЗАПУСКА", accent)
        guide.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        step_list(guide, (
            ("1", "Подготовь фишку", "Наведи игровую фишку на красный ромб и нажми F4."),
            ("2", "Запусти модуль", "Нажми F9 — GTA откроется только для проверки."),
            ("3", "Дальше автоматически", "Ставка проходит в безопасном окне, затем вернётся прежнее окно."),
        ), accent)
        warning = ctk.CTkFrame(guide, fg_color="#30291F", corner_radius=12)
        warning.pack(fill="x", padx=14, pady=(2, 14))
        ctk.CTkLabel(
            warning,
            text="🔔 Сигнал за 10 сек.  ·  Интервал 7–9 мин.\n⚠ Потеря строки «ВРЕМЯ» остановит приложение.",
            font=ctk.CTkFont("Segoe UI", 8, "bold"), text_color=GOLD,
            justify="left", wraplength=320,
        ).pack(anchor="w", padx=12, pady=10)
        hotkey_bar(body, (("F4", "точка ставки"), ("F9", "старт / стоп"), ("F11", "аварийный стоп")), accent)

    def back(self) -> None:
        self.on_back()

    def refresh_alert_sound_button(self) -> None:
        if self.alert_sound_enabled:
            self.alert_sound_button.configure(
                text="🔔  Сигнал: вкл", fg_color="#34445C", hover_color="#405570",
            )
        else:
            self.alert_sound_button.configure(
                text="🔕  Сигнал: выкл", fg_color="#252E3D", hover_color="#313C4F",
            )

    def toggle_alert_sound(self) -> None:
        self.alert_sound_enabled = not self.alert_sound_enabled
        self.refresh_alert_sound_button()
        if self.on_alert_sound_change is not None:
            self.on_alert_sound_change(self.alert_sound_enabled)

    def play_alert_sound(self, *, urgent: bool = False) -> None:
        if not self.alert_sound_enabled:
            return
        pattern = ((620, 240), (460, 300), (620, 240))
        wave_path = resource_path("sounds", "grumpy_fail.wav")
        ringtone_path = resource_path("sounds", "cat-iphone-ringtone.mp3")

        def play_pattern() -> None:
            if not urgent:
                alias = f"kiski_secret_alert_{time.time_ns()}"
                opened = winmm.mciSendStringW(
                    f'open "{ringtone_path}" type mpegvideo alias {alias}', None, 0, None,
                )
                if opened == 0:
                    try:
                        winmm.mciSendStringW(f"set {alias} time format milliseconds", None, 0, None)
                        winmm.mciSendStringW(f"setaudio {alias} volume to 1000", None, 0, None)
                        winmm.mciSendStringW(f"play {alias} from 0 to 3000 wait", None, 0, None)
                        return
                    finally:
                        winmm.mciSendStringW(f"close {alias}", None, 0, None)
            try:
                winsound.PlaySound(str(wave_path), winsound.SND_FILENAME)
            except RuntimeError:
                pass
            try:
                for index, (frequency, duration) in enumerate(pattern):
                    winsound.Beep(frequency, duration)
                    if index + 1 < len(pattern):
                        time.sleep(0.08)
            except RuntimeError:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

        threading.Thread(target=play_pattern, daemon=True, name="kiski-secret-alert").start()

    def refresh_connection(self) -> None:
        if not self.winfo_exists():
            return
        self.game_window = find_game_window(self.process.get())
        if self.game_window:
            title = window_title(self.game_window)
            self.connection.set(f"Подключено · {title[:22]}" if title else "Подключено")
            self.indicator.configure(text_color=MINT)
            if not self.running:
                self.status.set("GTA найден. Окно может оставаться на другом мониторе.")
        else:
            self.connection.set("Не подключено")
            self.indicator.configure(text_color="#F05A67")
            if not self.running:
                self.status.set(f"Процесс «{self.process.get()}» не найден. Проверь название выше.")
        self.after(2000, self.refresh_connection)

    def table_points(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if not self.game_window:
            return None
        bounds = client_bounds(self.game_window)
        if not bounds:
            return None
        left, top, width, height = bounds
        red = (left + round(width * RED_DIAMOND_RATIO[0]), top + round(height * RED_DIAMOND_RATIO[1]))
        black = (left + round(width * BLACK_DIAMOND_RATIO[0]), top + round(height * BLACK_DIAMOND_RATIO[1]))
        return red, black

    def sync_game_cursor(self) -> None:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            self.point_status.set("○ Точка не захвачена")
            self.point_badge.configure(text_color="#D5A29B")
            self.status.set("Сначала запусти GTA.")
            return
        self.game_cursor_at = "red"
        self.point_status.set("● Точка захвачена")
        self.point_badge.configure(text_color=MINT)
        self.status.set("Красная точка готова. Теперь можно запускать модуль через F9.")

    @staticmethod
    def format_elapsed(seconds: float) -> str:
        seconds = max(0.0, seconds)
        minutes, remainder = divmod(int(seconds), 60)
        tenths = int((seconds % 1) * 10)
        return f"{minutes:02d}:{remainder:02d}.{tenths}"

    def capture_game_timer(self) -> np.ndarray | None:
        """Capture and binarize the GTA timer readout in the bottom-right HUD."""
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            return None
        bounds = client_bounds(self.game_window)
        if not bounds:
            return None
        left, top, width, height = bounds
        ratio_x, ratio_y, ratio_width, ratio_height = TIME_READOUT_RATIO
        region = {
            "left": left + round(width * ratio_x),
            "top": top + round(height * ratio_y),
            "width": max(1, round(width * ratio_width)),
            "height": max(1, round(height * ratio_height)),
        }
        try:
            with mss.mss() as screen:
                image = np.asarray(screen.grab(region))[:, :, :3]
        except Exception:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def glyph_has_hole(glyph: np.ndarray) -> bool:
        _contours, hierarchy = cv2.findContours(glyph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        return bool(hierarchy is not None and any(item[3] >= 0 for item in hierarchy[0]))

    def betting_window_state(self) -> bool | None:
        """Return True for 00:10..00:34, False for 00:00..00:09, or None if unreadable.

        The timer never exceeds 00:34. Its tens digit therefore has a closed
        centre only for the unsafe 00:0x range. The first two zeroes are used
        as anchors so unrelated bright HUD elements are not accepted as time.
        """
        binary = self.capture_game_timer()
        if binary is None:
            return None
        height, width = binary.shape
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary)
        characters: list[tuple[int, int, int, int]] = []
        for index in range(1, count):
            x, y, char_width, char_height, area = map(int, stats[index])
            if (
                y >= height * 0.40
                and char_height >= height * 0.14
                and char_height <= height * 0.40
                and char_width >= width * 0.025
                and area >= 10
            ):
                characters.append((x, y, char_width, char_height))
        characters.sort(key=lambda item: item[0])
        for start in range(max(0, len(characters) - 3)):
            group = characters[start:start + 4]
            holes = []
            for x, y, char_width, char_height in group:
                glyph = binary[y:y + char_height, x:x + char_width]
                holes.append(self.glyph_has_hole(glyph))
            if holes[0] and holes[1]:
                return not holes[2]
        return None

    def restore_waiting_window(self) -> None:
        previous = self.waiting_previous_window
        self.waiting_previous_window = None
        if previous and previous != self.game_window and user32.IsWindow(previous):
            activate_window(previous)

    def shutdown_missing_timer(self) -> None:
        self.running = False
        self.next_run = None
        self.waiting_for_betting_window = False
        self.missing_timer_since = None
        self.restore_waiting_window()
        self.timer.set("Строка «ВРЕМЯ» не найдена")
        self.status.set("Игрок не за столом. Приложение полностью закрывается.")
        self.play_alert_sound(urgent=True)
        top_level = self.winfo_toplevel()
        self.after(350, top_level.destroy)

    def begin_betting_window_wait(self) -> None:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            self.next_run = time.monotonic() + GAME_RECONNECT_SECONDS
            self.status.set("GTA не найдена. Повторю подключение через 5 секунд.")
            return
        previous = user32.GetForegroundWindow()
        if not activate_window(self.game_window):
            self.next_run = time.monotonic() + GAME_RECONNECT_SECONDS
            self.status.set("GTA не получила фокус. Повторю через 5 секунд.")
            return
        self.waiting_previous_window = previous if previous and previous != self.game_window else None
        self.waiting_for_betting_window = True
        self.scan_ready_at = time.monotonic() + TIMER_ACTIVATE_SETTLE_SECONDS
        self.last_timer_scan_at = 0.0
        self.missing_timer_since = None
        self.last_timer_observation = "GTA открыта, читаю таймер"
        self.timer.set("GTA открыта · жду ВРЕМЯ 00:34–00:10")
        self.status.set("Проверяю реальный таймер GTA. До безопасного окна игра останется открытой.")

    def toggle(self) -> None:
        if self.running:
            self.stop("Таймер остановлен.")
            return
        if self.game_cursor_at not in {"red", "black"}:
            self.status.set("Сначала наведи игровую фишку на КРАСНЫЙ ромб и нажми F4.")
            return
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            self.status.set("GTA не подключена.")
            return
        started = time.monotonic()
        self.running, self.next_run = True, started
        self.f9_started_at = started
        self.cycle_started_at = None
        self.cycle_count = 0
        self.last_trigger_lag = 0.0
        self.last_first_click_delay = None
        self.last_cycle_duration = None
        self.last_wait_after_black = None
        self.waiting_for_betting_window = False
        self.waiting_previous_window = None
        self.missing_timer_since = None
        self.next_delay_seconds = None
        self.last_timer_observation = "первая проверка"
        self.warning_alerted = False
        self.main_button.configure(text="Остановить  ·  F9", fg_color="#E95E69", hover_color="#C94C57", text_color=TEXT)
        self.status.set("F9 принят. Открываю GTA и проверяю реальный таймер перед первой ставкой.")
        self.after(0, self.tick)

    def stop(self, message: str) -> None:
        self.running, self.next_run = False, None
        self.waiting_for_betting_window = False
        self.missing_timer_since = None
        self.restore_waiting_window()
        self.main_button.configure(text="Запустить  ·  F9", fg_color="#FF8F91", hover_color="#FFA3A5", text_color="#21171A")
        self.timer.set("Таймер не запущен")
        self.status.set(message)

    def activate(self) -> None:
        """Разрешить горячие клавиши только открытому секретному модулю."""
        self.active = True
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def deactivate(self, message: str) -> None:
        """Остановить модуль при переходе в другую скрытую часть приложения."""
        self.stop(message)
        self.active = False
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def execute_bets(self, return_previous: bool = True) -> bool:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            self.status.set("GTA не найдена: клик отменён.")
            return False
        previous_window = user32.GetForegroundWindow()
        previous_cursor = cursor_position()
        cursor_confined = False
        cursor_relocated = False
        if not activate_window(self.game_window):
            self.status.set("GTA не получила фокус — ввод не отправлен.")
            return False
        try:
            points = self.table_points()
            if not points or self.game_cursor_at not in {"red", "black"}:
                self.status.set("Сначала синхронизируй игровую фишку на красном через F4.")
                return False
            red, black = points
            bounds = client_bounds(self.game_window)
            if not bounds:
                self.status.set("Не удалось определить границы GTA — ввод отменён.")
                return False
            cursor_confined = confine_cursor_to_client(bounds)
            left, top, width, height = bounds
            cursor_relocated = bool(user32.SetCursorPos(left + width // 2, top + height // 2))
            if not cursor_relocated:
                self.status.set("Не удалось перенести системный курсор внутрь GTA — ввод отменён.")
                return False
            time.sleep(1.25)
            step_x = round((black[0] - red[0]) * GAME_CURSOR_DISTANCE_MULTIPLIER)
            step_y = round((black[1] - red[1]) * GAME_CURSOR_DISTANCE_MULTIPLIER)
            if not (step_x or step_y):
                self.status.set("Красный и чёрный ромбы совпали в настройке — клик отменён.")
                return False
            red_delta = (0, 0) if self.game_cursor_at == "red" else (-step_x, -step_y)
            red_sent = move_relative_and_bet(*red_delta, settle_seconds=0.55, hold_seconds=0.18)
            if not red_sent:
                self.status.set("Windows не приняла ввод для красной ставки; чёрная пропущена.")
                return False
            self.game_cursor_at, self.last_bet_started = "red", time.monotonic()
            if self.cycle_started_at is not None:
                self.last_first_click_delay = self.last_bet_started - self.cycle_started_at
            # Не начинаем движение сразу после красного: это помогает GTA
            # завершить анимацию фишки перед переходом к чёрному ромбу.
            time.sleep(0.72)
            black_sent = move_relative_and_bet(step_x, step_y, settle_seconds=0.92, hold_seconds=0.22)
            if black_sent:
                self.game_cursor_at = "black"
                self.last_bet_completed = time.monotonic()
                if self.cycle_started_at is not None:
                    self.last_cycle_duration = self.last_bet_completed - self.cycle_started_at
            if not black_sent:
                self.status.set("Windows не приняла ввод для чёрной ставки.")
                return False
            self.status.set(f"Ставки отправлены: красный {red[0]}×{red[1]}, чёрный {black[0]}×{black[1]}.")
            return True
        finally:
            time.sleep(0.35)
            if cursor_confined:
                user32.ClipCursor(None)
            if return_previous and previous_window and previous_window != self.game_window and user32.IsWindow(previous_window):
                activate_window(previous_window)
            user32.SetCursorPos(*previous_cursor)

    def place_bets(self) -> None:
        triggered_at = time.monotonic()
        self.cycle_started_at = triggered_at
        self.cycle_count += 1
        self.last_first_click_delay = None
        self.last_cycle_duration = None
        self.last_wait_after_black = None
        self.warning_alerted = False
        self.waiting_for_betting_window = False
        self.missing_timer_since = None
        previous = self.waiting_previous_window
        self.waiting_previous_window = None
        try:
            if not self.execute_bets(return_previous=False):
                self.next_run = time.monotonic() + GAME_ROUND_SECONDS
                self.status.set("Ставка не прошла. Через 56 секунд снова проверю реальный таймер.")
                return
        except Exception as error:
            self.status.set(f"Ошибка автоматической ставки: {type(error).__name__}: {error}")
            self.next_run = time.monotonic() + GAME_ROUND_SECONDS
            return
        finally:
            if previous and previous != self.game_window and user32.IsWindow(previous):
                activate_window(previous)
        delay = random.randint(*BET_DELAY_SECONDS)
        completed_at = self.last_bet_completed or time.monotonic()
        self.next_delay_seconds = delay
        self.last_wait_after_black = float(delay)
        self.next_run = completed_at + delay
        minutes, seconds = divmod(delay, 60)
        self.last_timer_observation = "безопасное окно найдено"
        self.status.set(
            f"Ставки сделаны, прежнее окно возвращено. Новый случайный таймер: {minutes:02d}:{seconds:02d}."
        )

    def tick(self) -> None:
        if self.winfo_exists() and self.running and self.next_run is not None:
            now = time.monotonic()
            remaining = self.next_run - now
            if remaining <= 0:
                if not self.waiting_for_betting_window:
                    self.begin_betting_window_wait()
                elif not self.game_window or not user32.IsWindow(self.game_window):
                    self.last_timer_observation = "GTA потеряна"
                    self.shutdown_missing_timer()
                elif user32.GetForegroundWindow() != self.game_window:
                    if activate_window(self.game_window):
                        self.scan_ready_at = now + TIMER_ACTIVATE_SETTLE_SECONDS
                        self.missing_timer_since = None
                    self.timer.set("Возвращаю GTA для чтения таймера…")
                elif now >= self.scan_ready_at and now - self.last_timer_scan_at >= TIMER_SCAN_INTERVAL_SECONDS:
                    self.last_timer_scan_at = now
                    state = self.betting_window_state()
                    if state is True:
                        self.missing_timer_since = None
                        self.last_timer_observation = "00:34–00:10"
                        self.timer.set("Безопасное окно найдено · ставлю")
                        self.place_bets()
                    elif state is False:
                        self.missing_timer_since = None
                        self.last_timer_observation = "00:09–00:00"
                        self.timer.set("Поздно для ставки · жду следующий круг")
                    else:
                        if self.missing_timer_since is None:
                            self.missing_timer_since = now
                        missing_for = now - self.missing_timer_since
                        remaining_check = max(0.0, MISSING_TIMER_SHUTDOWN_SECONDS - missing_for)
                        self.last_timer_observation = "строка ВРЕМЯ не найдена"
                        self.timer.set(f"ВРЕМЯ не найдено · закрытие через {remaining_check:.1f} с")
                        if missing_for >= MISSING_TIMER_SHUTDOWN_SECONDS:
                            self.shutdown_missing_timer()
            else:
                if remaining <= 10 and not self.warning_alerted:
                    self.warning_alerted = True
                    self.play_alert_sound()
                    self.status.set("🔔 Через 10 секунд открою GTA и начну проверять реальный таймер.")
                minutes, seconds = divmod(int(remaining + 0.999), 60)
                self.timer.set(f"Проверка GTA через {minutes:02d}:{seconds:02d}")
        if self.winfo_exists():
            self.after(100, self.tick)

    def poll_hotkeys(self) -> None:
        actions = {VK_F4: self.sync_game_cursor, VK_F9: self.toggle, VK_F11: lambda: self.stop("Экстренно остановлено.")}
        for key, action in actions.items():
            down = bool(user32.GetAsyncKeyState(key) & 0x8000)
            if self.active and down and not self.keys[key]:
                action()
            self.keys[key] = down
        if self.winfo_exists():
            self.after(40, self.poll_hotkeys)
