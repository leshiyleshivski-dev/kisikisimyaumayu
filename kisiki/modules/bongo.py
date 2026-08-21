"""Two-beat phone mini-game screen."""

from __future__ import annotations

import random
import time

import customtkinter as ctk

from ..core import (
    APP_BG, GOLD, MINT, MUTED, SURFACE, SURFACE_ALT, TEXT,
    VK_F9, VK_F11, VK_UP, activate_window, find_game_window,
    send_key_tap, user32, window_title,
)
from .ui import connection_panel, hotkey_bar, module_header, panel, step_list

class BongoModule(ctk.CTkFrame):
    SECOND_TAP_SECONDS = 3
    CYCLE_DELAY_SECONDS = (7 * 60, 9 * 60)
    RECONNECT_SECONDS = 5

    def __init__(self, parent: ctk.CTkFrame, on_back) -> None:
        super().__init__(parent, fg_color=APP_BG, corner_radius=0)
        self.on_back = on_back
        self.running = False
        self.active = False
        self.phase = "idle"
        self.next_action: float | None = None
        self.game_window: int | None = None
        self.previous_window: int | None = None
        self.cycle_count = 0
        self.keys = {key: False for key in (VK_F9, VK_F11)}
        self.process = ctk.StringVar(value="GTA5.exe")
        self.connection = ctk.StringVar(value="Ищу GTA5.exe…")
        self.status = ctk.StringVar(value="Готов к ритму Бонго.")
        self.timer = ctk.StringVar(value="Таймер не запущен")
        self.build_ui()
        self.after(40, self.poll_hotkeys)
        self.after(100, self.tick)
        self.after(0, self.refresh_connection)

    def build_ui(self) -> None:
        accent = "#FF9D76"
        module_header(
            self, code="BG", title="Bongo Beat",
            subtitle="ритм из двух нажатий с автоматическим повтором",
            accent=accent, on_back=self.back,
        )
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=42, pady=(0, 24))
        self.indicator = connection_panel(
            body, process=self.process, connection=self.connection,
            accent=accent, on_check=self.refresh_connection,
        )

        workspace = ctk.CTkFrame(body, fg_color="transparent")
        workspace.pack(fill="both", expand=True)
        workspace.grid_columnconfigure(0, weight=6, uniform="bongo")
        workspace.grid_columnconfigure(1, weight=5, uniform="bongo")
        workspace.grid_rowconfigure(0, weight=1)

        live = panel(workspace, "LIVE  /  СОСТОЯНИЕ РИТМА", accent)
        live.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        beat = ctk.CTkFrame(live, fg_color="#251F2B", corner_radius=17)
        beat.pack(fill="x", padx=16, pady=(0, 14))
        ctk.CTkLabel(
            beat, text="↑    ·  3 СЕК  ·    ↑", font=ctk.CTkFont("Segoe UI", 24, "bold"),
            text_color=accent,
        ).pack(pady=(18, 4))
        ctk.CTkLabel(
            beat, text="пара ударов повторяется каждые 7–9 минут",
            font=ctk.CTkFont("Segoe UI", 9), text_color=MUTED,
        ).pack(pady=(0, 17))
        ctk.CTkLabel(
            live, textvariable=self.timer, font=ctk.CTkFont("Segoe UI", 21, "bold"),
            text_color=TEXT,
        ).pack(padx=18, pady=(5, 4))
        ctk.CTkLabel(
            live, textvariable=self.status, font=ctk.CTkFont("Segoe UI", 10),
            text_color=MUTED, wraplength=410, justify="center",
        ).pack(fill="x", padx=24, pady=(0, 16))
        self.main_button = ctk.CTkButton(
            live, text="Запустить ритм  ·  F9", command=self.toggle,
            height=50, corner_radius=14, font=ctk.CTkFont("Segoe UI", 13, "bold"),
            fg_color=accent, hover_color="#FFB08D", text_color="#17121A",
        )
        self.main_button.pack(fill="x", side="bottom", padx=16, pady=16)

        guide = panel(workspace, "СЦЕНАРИЙ ЗАПУСКА", accent)
        guide.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        steps = (
            ("1", "Запусти игру", "Окно процесса можно выбрать в поле выше."),
            ("2", "Нажми F9", "Модуль выведет игру на передний план и нажмёт ↑."),
            ("3", "Второй удар", "Через 3 секунды отправит ещё одно ↑, вернёт прежнее окно и запустит таймер."),
        )
        step_list(guide, steps, accent)
        ctk.CTkLabel(
            guide, text="После пары окно вернётся автоматически.",
            font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=GOLD,
        ).pack(anchor="w", padx=18, pady=(4, 14))
        hotkey_bar(body, (("F9", "старт / стоп"), ("F11", "экстренная остановка")), accent)

    def back(self) -> None:
        self.on_back()

    def refresh_connection(self) -> None:
        if not self.winfo_exists():
            return
        self.game_window = find_game_window(self.process.get())
        if self.game_window:
            title = window_title(self.game_window)
            self.connection.set(f"Подключено · {title[:28]}" if title else "Подключено")
            self.indicator.configure(text_color=MINT)
        else:
            self.connection.set("Не подключено")
            self.indicator.configure(text_color="#F05A67")
        self.after(2000, self.refresh_connection)

    def restore_previous_window(self) -> None:
        previous = self.previous_window
        self.previous_window = None
        if previous and previous != self.game_window and user32.IsWindow(previous):
            activate_window(previous)

    def open_game_for_beat(self) -> bool:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            self.status.set(f"Процесс «{self.process.get()}» не найден. Повторю через 5 секунд.")
            self.next_action = time.monotonic() + self.RECONNECT_SECONDS
            self.phase = "retry"
            return False
        previous = user32.GetForegroundWindow()
        if not activate_window(self.game_window):
            self.status.set("Не получилось вывести игру на передний план. Повторю через 5 секунд.")
            self.next_action = time.monotonic() + self.RECONNECT_SECONDS
            self.phase = "retry"
            return False
        self.previous_window = previous if previous and previous != self.game_window else None
        return True

    def start_first_tap(self) -> None:
        if not self.open_game_for_beat():
            return
        if not send_key_tap(VK_UP):
            self.status.set("Windows не принял первое нажатие ↑.")
            self.restore_previous_window()
            self.stop("Ритм остановлен: первое нажатие не отправлено.")
            return
        self.phase = "second_tap"
        self.next_action = time.monotonic() + self.SECOND_TAP_SECONDS
        self.timer.set("Второй удар через 00:03")
        self.status.set("Первое ↑ отправлено. Игра останется открытой ещё 3 секунды.")

    def send_second_tap(self) -> None:
        if not self.game_window or not user32.IsWindow(self.game_window) or not activate_window(self.game_window):
            self.restore_previous_window()
            self.status.set("Игра закрыта или потеряла фокус — второй ↑ отменён.")
            self.next_action = time.monotonic() + self.RECONNECT_SECONDS
            self.phase = "retry"
            return
        if not send_key_tap(VK_UP):
            self.restore_previous_window()
            self.stop("Ритм остановлен: второе нажатие не отправлено.")
            return
        self.cycle_count += 1
        self.restore_previous_window()
        delay = random.randint(*self.CYCLE_DELAY_SECONDS)
        self.phase = "waiting"
        self.next_action = time.monotonic() + delay
        minutes, seconds = divmod(delay, 60)
        self.status.set(f"Пара ↑ готова · цикл {self.cycle_count}. Прежнее окно возвращено.")
        self.timer.set(f"Следующий ритм через {minutes:02d}:{seconds:02d}")

    def toggle(self) -> None:
        if self.running:
            self.stop("Таймер остановлен.")
            return
        self.running = True
        self.phase = "first_tap"
        self.next_action = time.monotonic()
        self.cycle_count = 0
        self.main_button.configure(text="Остановить ритм  ·  F9", fg_color="#E95E69", hover_color="#C94C57", text_color=TEXT)
        self.status.set("F9 принят. Открываю игру для первого ↑.")
        self.after(0, self.tick)

    def stop(self, message: str) -> None:
        self.running, self.phase, self.next_action = False, "idle", None
        self.restore_previous_window()
        self.main_button.configure(text="Запустить ритм  ·  F9", fg_color="#FF9D76", hover_color="#FFB08D", text_color="#17121A")
        self.timer.set("Таймер не запущен")
        self.status.set(message)

    def activate(self) -> None:
        """Разрешить F9/F11 только пока открыт экран Бонго."""
        self.active = True
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def deactivate(self, message: str) -> None:
        """Остановить таймер и перестать слушать горячие клавиши."""
        self.stop(message)
        self.active = False
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def tick(self) -> None:
        if self.winfo_exists() and self.running and self.next_action is not None:
            now = time.monotonic()
            remaining = self.next_action - now
            if remaining <= 0:
                if self.phase in {"first_tap", "retry", "waiting"}:
                    self.start_first_tap()
                elif self.phase == "second_tap":
                    self.send_second_tap()
            elif self.phase == "second_tap":
                self.timer.set(f"Второй удар через 00:{max(0, int(remaining + 0.999)):02d}")
            elif self.phase in {"waiting", "retry"}:
                minutes, seconds = divmod(max(0, int(remaining + 0.999)), 60)
                prefix = "Повторное подключение" if self.phase == "retry" else "Следующий ритм"
                self.timer.set(f"{prefix} через {minutes:02d}:{seconds:02d}")
        if self.winfo_exists():
            self.after(100, self.tick)

    def poll_hotkeys(self) -> None:
        actions = {VK_F9: self.toggle, VK_F11: lambda: self.stop("Экстренно остановлено.")}
        for key, action in actions.items():
            down = bool(user32.GetAsyncKeyState(key) & 0x8000)
            if self.active and down and not self.keys[key]:
                action()
            self.keys[key] = down
        if self.winfo_exists():
            self.after(40, self.poll_hotkeys)
