"""Buff timing mini-game screen."""

from __future__ import annotations

import time

import cv2
import customtkinter as ctk
import mss
import numpy as np

from ..core import (
    APP_BG, MINT, MUTED, SURFACE, SURFACE_ALT, TEXT, TIMING_BAR_RATIO,
    VK_F9, VK_F11, VK_SPACE, activate_window, client_bounds,
    find_game_window, send_key_tap, user32, window_title,
)
from .ui import connection_panel, hotkey_bar, module_header, panel, step_list

class BuffTimingModule(ctk.CTkFrame):
    SCAN_INTERVAL_MS = 8
    ZONE_CHANGE_PIXELS = 8
    BAR_MISSING_RESET_SECONDS = 0.45
    MIN_MARKER_OVERLAP_PIXELS = 4
    SPACE_COOLDOWN_SECONDS = 0.5

    def __init__(self, parent: ctk.CTkFrame, on_back) -> None:
        super().__init__(parent, fg_color=APP_BG, corner_radius=0)
        self.on_back = on_back
        self.active = False
        self.running = False
        self.game_window: int | None = None
        self.keys = {key: False for key in (VK_F9, VK_F11)}
        self.process = ctk.StringVar(value="GTA5.exe")
        self.connection = ctk.StringVar(value="Ищу GTA5.exe…")
        self.status = ctk.StringVar(value="Готов к ловле зелёной зоны.")
        self.target = ctk.StringVar(value="Жду запуск мини-игры")
        self.hits = 0
        self.zone: tuple[int, int] | None = None
        self.armed = True
        self.last_space_at = 0.0
        self.last_bar_seen_at = 0.0
        self.build_ui()
        self.after(40, self.poll_hotkeys)
        self.after(0, self.refresh_connection)
        self.after(self.SCAN_INTERVAL_MS, self.scan_tick)

    def build_ui_legacy(self) -> None:
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=46, pady=(36, 30))
        ctk.CTkButton(
            top, text="←  К котикам", command=self.on_back, width=126, height=38,
            corner_radius=12, fg_color="#26314E", hover_color="#344263",
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        ).pack(side="left")
        title = ctk.CTkFrame(top, fg_color="transparent")
        title.pack(side="right")
        ctk.CTkLabel(title, text="BUFF TIMING", font=ctk.CTkFont("Segoe UI", 24, "bold"), text_color=TEXT).pack(anchor="e")
        ctk.CTkLabel(title, text="ловит ритм строительной шкалы", font=ctk.CTkFont("Segoe UI", 11), text_color=MUTED).pack(anchor="e")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=46, pady=(0, 26))
        connection = ctk.CTkFrame(body, corner_radius=22, fg_color=SURFACE)
        connection.pack(fill="x", pady=(0, 16))
        connection.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(connection, text="ПОДКЛЮЧЕНИЕ К ИГРЕ", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color="#9EE0BE").grid(row=0, column=0, columnspan=2, padx=26, pady=(20, 8), sticky="w")
        self.indicator = ctk.CTkLabel(connection, text="●", font=ctk.CTkFont(size=18), text_color="#F05A67")
        self.indicator.grid(row=1, column=0, padx=(26, 10), pady=(0, 21))
        ctk.CTkEntry(connection, textvariable=self.process, height=42, border_width=0, corner_radius=13, fg_color="#33453D", font=ctk.CTkFont("Segoe UI", 14)).grid(row=1, column=1, padx=(0, 12), pady=(0, 21), sticky="ew")
        ctk.CTkButton(connection, text="Проверить", command=self.refresh_connection, width=122, height=42, corner_radius=13, fg_color="#4BB889", hover_color="#62C99B").grid(row=1, column=2, padx=(0, 26), pady=(0, 21))
        ctk.CTkLabel(connection, textvariable=self.connection, font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=MUTED).grid(row=2, column=0, columnspan=3, padx=26, pady=(0, 17), sticky="w")

        guide = ctk.CTkFrame(body, corner_radius=22, fg_color=SURFACE)
        guide.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(guide, text="КАК РАБОТАЕТ", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color="#9EE0BE").pack(anchor="w", padx=26, pady=(20, 10))
        steps = (
            ("1", "Запусти мини-игру", "В GTA должна быть видна шкала в правом нижнем углу."),
            ("2", "Нажми F9", "Модуль откроет игру и начнёт отслеживать зелёный сектор."),
            ("3", "Удар по бегунку", "Space нажимается один раз, когда розовый бегунок заходит в зелёный сектор."),
        )
        for number, heading, description in steps:
            row = ctk.CTkFrame(guide, fg_color=SURFACE_ALT, corner_radius=14)
            row.pack(fill="x", padx=22, pady=(0, 8))
            ctk.CTkLabel(row, text=number, width=30, height=30, corner_radius=15, fg_color="#4BB889", text_color=TEXT, font=ctk.CTkFont("Segoe UI", 13, "bold")).pack(side="left", padx=(12, 12), pady=10)
            copy = ctk.CTkFrame(row, fg_color="transparent")
            copy.pack(side="left", fill="x", expand=True, pady=8)
            ctk.CTkLabel(copy, text=heading, font=ctk.CTkFont("Segoe UI", 12, "bold"), text_color=TEXT).pack(anchor="w")
            ctk.CTkLabel(copy, text=description, font=ctk.CTkFont("Segoe UI", 10), text_color=MUTED).pack(anchor="w")

        self.main_button = ctk.CTkButton(body, text="Запустить ловлю  ·  F9", command=self.toggle, height=56, corner_radius=16, font=ctk.CTkFont("Segoe UI", 15, "bold"), fg_color="#4BB889", hover_color="#62C99B")
        self.main_button.pack(fill="x", pady=(0, 14))
        monitor = ctk.CTkFrame(body, corner_radius=18, fg_color=SURFACE_ALT, border_width=1, border_color="#3E5B4D")
        monitor.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(monitor, text="СОСТОЯНИЕ", font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color="#9EE0BE").pack(pady=(14, 4))
        ctk.CTkLabel(monitor, textvariable=self.target, font=ctk.CTkFont("Segoe UI", 20, "bold"), text_color=TEXT).pack(pady=(0, 5))
        ctk.CTkLabel(monitor, textvariable=self.status, font=ctk.CTkFont("Segoe UI", 11), text_color=MUTED, wraplength=650, justify="center").pack(padx=22, pady=(0, 14))
        ctk.CTkLabel(body, text="F9 — запуск / остановка     ·     F11 — экстренная остановка", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color="#9EE0BE").pack(pady=(0, 16))

    def build_ui(self) -> None:
        accent = "#68D6B4"
        module_header(
            self, code="BT", title="Buff Timing",
            subtitle="точный удар по зелёной зоне строительной шкалы",
            accent=accent, on_back=self.on_back,
        )
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=42, pady=(0, 24))
        self.indicator = connection_panel(
            body, process=self.process, connection=self.connection,
            accent=accent, on_check=self.refresh_connection,
        )

        workspace = ctk.CTkFrame(body, fg_color="transparent")
        workspace.pack(fill="both", expand=True)
        workspace.grid_columnconfigure(0, weight=6, uniform="buff")
        workspace.grid_columnconfigure(1, weight=5, uniform="buff")
        workspace.grid_rowconfigure(0, weight=1)

        live = panel(workspace, "LIVE  /  ЗАХВАТ ШКАЛЫ", accent)
        live.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        timing = ctk.CTkFrame(live, fg_color="#172E2C", corner_radius=17)
        timing.pack(fill="x", padx=16, pady=(0, 14))
        scale = ctk.CTkFrame(timing, height=24, fg_color="#293748", corner_radius=8)
        scale.pack(fill="x", padx=22, pady=(24, 8))
        scale.pack_propagate(False)
        ctk.CTkLabel(scale, text="", width=96, height=24, corner_radius=8, fg_color=accent).place(relx=0.56, rely=0.5, anchor="center")
        ctk.CTkLabel(scale, text="◆", width=22, font=ctk.CTkFont("Segoe UI Symbol", 15, "bold"), text_color="#FF8F91").place(relx=0.43, rely=0.5, anchor="center")
        ctk.CTkLabel(
            timing, text="РОЗОВЫЙ БЕГУНОК  →  ЗЕЛЁНАЯ ЗОНА",
            font=ctk.CTkFont("Segoe UI", 8, "bold"), text_color=accent,
        ).pack(pady=(0, 18))
        ctk.CTkLabel(
            live, textvariable=self.target, font=ctk.CTkFont("Segoe UI", 20, "bold"),
            text_color=TEXT,
        ).pack(padx=18, pady=(5, 4))
        ctk.CTkLabel(
            live, textvariable=self.status, font=ctk.CTkFont("Segoe UI", 10),
            text_color=MUTED, wraplength=410, justify="center",
        ).pack(fill="x", padx=24, pady=(0, 16))
        self.main_button = ctk.CTkButton(
            live, text="Запустить ловлю  ·  F9", command=self.toggle,
            height=50, corner_radius=14, font=ctk.CTkFont("Segoe UI", 13, "bold"),
            fg_color=accent, hover_color="#7BE2C2", text_color="#111B19",
        )
        self.main_button.pack(fill="x", side="bottom", padx=16, pady=16)

        guide = panel(workspace, "СЦЕНАРИЙ ЗАПУСКА", accent)
        guide.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        step_list(guide, (
            ("1", "Запусти мини-игру", "В GTA должна быть видна шкала в правом нижнем углу."),
            ("2", "Нажми F9", "Модуль откроет игру и начнёт отслеживать зелёный сектор."),
            ("3", "Дождись попадания", "Space отправится, когда бегунок подтвердится внутри зоны."),
        ), accent)
        ctk.CTkLabel(
            guide, text="Одно подтверждённое пересечение — одно нажатие Space.",
            font=ctk.CTkFont("Segoe UI", 8, "bold"), text_color=accent,
            wraplength=330, justify="left",
        ).pack(anchor="w", padx=18, pady=(4, 14))
        hotkey_bar(body, (("F9", "старт / стоп"), ("F11", "экстренная остановка")), accent)

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

    @staticmethod
    def color_box(mask: np.ndarray, *, min_width: int, min_height: int, min_area: int) -> tuple[int, int, int, int] | None:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        candidates = [tuple(map(int, row)) for row in stats[1:] if row[2] >= min_width and row[3] >= min_height and row[4] >= min_area]
        if not candidates:
            return None
        x, y, width, height, _area = max(candidates, key=lambda row: row[4])
        return x, y, width, height

    def read_timing_bar(self) -> tuple[tuple[int, int], tuple[int, int] | None] | None:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            return None
        bounds = client_bounds(self.game_window)
        if not bounds:
            return None
        left, top, width, height = bounds
        ratio_x, ratio_y, ratio_width, ratio_height = TIMING_BAR_RATIO
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
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array((35, 50, 55), dtype=np.uint8), np.array((95, 255, 255), dtype=np.uint8))
        green_box = self.color_box(green, min_width=max(8, region["width"] // 35), min_height=max(10, region["height"] // 5), min_area=80)
        if not green_box:
            return None
        pink = cv2.inRange(hsv, np.array((160, 80, 80), dtype=np.uint8), np.array((179, 255, 255), dtype=np.uint8))
        pink_box = self.color_box(pink, min_width=max(5, region["width"] // 80), min_height=max(10, region["height"] // 5), min_area=60)
        green_x, _green_y, green_width, _green_height = green_box
        zone = (region["left"] + green_x, region["left"] + green_x + green_width)
        if not pink_box:
            return zone, None
        pink_x, _pink_y, pink_width, _pink_height = pink_box
        return zone, (region["left"] + pink_x, region["left"] + pink_x + pink_width)

    def reset_round(self) -> None:
        self.zone = None
        self.armed = True
        self.last_space_at = 0.0
        self.target.set("Ищу зелёную зону…")

    def zone_changed(self, zone: tuple[int, int]) -> bool:
        if self.zone is None:
            return True
        old_start, old_end = self.zone
        start, end = zone
        return abs(start - old_start) >= self.ZONE_CHANGE_PIXELS or abs(end - old_end) >= self.ZONE_CHANGE_PIXELS

    def handle_timing_state(self, zone: tuple[int, int], marker: tuple[int, int] | None) -> None:
        now = time.monotonic()
        self.last_bar_seen_at = now
        if self.zone_changed(zone):
            self.zone = zone
            self.target.set("Зелёная зона найдена")
        if marker is None:
            self.status.set("Зелёная зона найдена. Ищу розовый бегунок.")
            return
        zone_start, zone_end = zone
        marker_start, marker_end = marker
        overlap = min(zone_end, marker_end) - max(zone_start, marker_start)
        if overlap < self.MIN_MARKER_OVERLAP_PIXELS:
            self.armed = True
            self.status.set("Бегунок вне зелёной зоны — жду входа.")
            return
        if not self.armed:
            self.status.set("Удар уже отправлен — жду, пока бегунок выйдет из зоны.")
            return
        cooldown_left = self.SPACE_COOLDOWN_SECONDS - (now - self.last_space_at)
        if cooldown_left > 0:
            self.status.set(f"Пауза после удара: {cooldown_left:.1f} сек.")
            return
        if send_key_tap(VK_SPACE):
            self.hits += 1
            self.last_space_at = now
            self.target.set(f"Space отправлен · попаданий: {self.hits}")
            self.status.set("Жду выхода бегунка и следующего попадания.")
            self.armed = False
        else:
            self.status.set("Windows не принял Space. Продолжаю отслеживание.")

    def toggle(self) -> None:
        if self.running:
            self.stop("Ловля остановлена.")
            return
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            self.status.set(f"Процесс «{self.process.get()}» не найден.")
            return
        previous = user32.GetForegroundWindow()
        if not activate_window(self.game_window):
            self.status.set("Не получилось вывести игру на передний план.")
            return
        self.previous_window = previous if previous and previous != self.game_window else None
        self.running = True
        self.hits = 0
        self.reset_round()
        self.main_button.configure(text="Остановить ловлю  ·  F9", fg_color="#E95E69", hover_color="#C94C57", text_color=TEXT)
        self.status.set("F9 принят. Игра открыта, ищу шкалу в правом нижнем углу.")

    def stop(self, message: str) -> None:
        self.running = False
        self.reset_round()
        self.restore_previous_window()
        self.main_button.configure(text="Запустить ловлю  ·  F9", fg_color="#68D6B4", hover_color="#7BE2C2", text_color="#111B19")
        self.status.set(message)

    def activate(self) -> None:
        self.active = True
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def deactivate(self, message: str) -> None:
        self.stop(message)
        self.active = False
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def scan_tick(self) -> None:
        if self.winfo_exists() and self.active and self.running:
            state = self.read_timing_bar()
            if state is None:
                if self.last_bar_seen_at and time.monotonic() - self.last_bar_seen_at >= self.BAR_MISSING_RESET_SECONDS:
                    self.reset_round()
                self.status.set("Шкала не видна — жду запуск мини-игры.")
            else:
                self.handle_timing_state(*state)
        if self.winfo_exists():
            self.after(self.SCAN_INTERVAL_MS, self.scan_tick)

    def poll_hotkeys(self) -> None:
        actions = {VK_F9: self.toggle, VK_F11: lambda: self.stop("Экстренно остановлено.")}
        for key, action in actions.items():
            down = bool(user32.GetAsyncKeyState(key) & 0x8000)
            if self.active and down and not self.keys[key]:
                action()
            self.keys[key] = down
        if self.winfo_exists():
            self.after(40, self.poll_hotkeys)
