"""Volt screen coordinating the three independent electrical solvers."""

from __future__ import annotations

import threading
import tkinter as tk

import customtkinter as ctk
import mss
import numpy as np

from ...core import (
    APP_BG, MINT, MUTED, SURFACE, SURFACE_ALT, TEXT, VK_F9, VK_F11,
    VK_SPACE, activate_window, client_bounds, find_game_window,
    send_key_tap, user32, window_title,
)
from ..ui import connection_panel, hotkey_bar, module_header, panel
from .current_grid import CurrentGridMixin
from .maze import MazeMixin
from .tumblers import TumblerMixin


class ElectricianModule(TumblerMixin, MazeMixin, CurrentGridMixin, ctk.CTkFrame):
    def __init__(self, parent: ctk.CTkFrame, on_back) -> None:
        super().__init__(parent, fg_color=APP_BG, corner_radius=0)
        self.on_back = on_back
        self.active = False
        self.running = False
        self.game_window: int | None = None
        self.keys = {key: False for key in (VK_F9, VK_F11)}
        self.process = ctk.StringVar(value="GTA5.exe")
        self.connection = ctk.StringVar(value="Ищу GTA5.exe…")
        self.target = ctk.StringVar(value="Автодетект выключен")
        self.status = ctk.StringVar(value="Нажми F9 один раз перед началом работы.")
        self.action_job: str | None = None
        self.actions: list[int] = []
        self.action_kind = ""
        self.detection_generation = 0
        self.current_recovery_round = 0
        self.current_verify_misses = 0
        # Перемещение по лабиринту требует больше времени, чем навигация по
        # статичной сетке проводов: GTA сначала должна отрисовать новый узел.
        self.action_interval = 135
        self.tumbler_positions: list[tuple[int, int]] = []
        self.tumbler_effects: list[tuple[float, float]] = []
        self.tumbler_baseline = (0.0, 0.0)
        self.tumbler_targets = ((0.0, 0.0), (0.0, 0.0))
        self.tumbler_initial_states: list[bool] = []
        self.tumbler_previous = (0.0, 0.0)
        self.tumbler_index = 0
        self.tumbler_phase = ""
        self.tumbler_solution: list[int] = []
        self.tumbler_apply: list[int] = []
        self.tumbler_click_retries = 0
        self.tumbler_correction_round = 0
        self.tumbler_candidate_signature: tuple[int, ...] | None = None
        self.tumbler_candidate_hits = 0
        self.tumbler_pending_values = (0.0, 0.0)
        self.tumbler_settle_retries = 0
        self.build_ui()
        self.after(40, self.poll_hotkeys)
        self.after(0, self.refresh_connection)

    def build_ui(self) -> None:
        accent = "#72D7D2"
        module_header(
            self, code="V", title="Volt Grid",
            subtitle="единый автодетектор электрических мини-игр",
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
        workspace.grid_columnconfigure(0, weight=7, uniform="volt")
        workspace.grid_columnconfigure(1, weight=4, uniform="volt")
        workspace.grid_rowconfigure(0, weight=1)

        catalog = panel(workspace, "ПОДДЕРЖИВАЕМЫЕ ЭКРАНЫ  /  3 ИЗ 4", accent)
        catalog.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        cards = ctk.CTkFrame(catalog, fg_color="transparent")
        cards.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        for column in range(2):
            cards.grid_columnconfigure(column, weight=1, uniform="volt-card")
        for row in range(2):
            cards.grid_rowconfigure(row, weight=1, uniform="volt-card")
        games = (
            ("⚡", "Проведение тока", "ГОТОВО", "строит схему и вращает плитки"),
            ("⌁", "Контроль напряжения", "СКОРО", "фиксация стрелки в секторе"),
            ("▣", "Тумблеры", "ГОТОВО", "измеряет влияние переключателей"),
            ("◎", "Электро-лабиринт", "ГОТОВО", "находит маршрут от A до Б"),
        )
        for index, (icon, title, state, description) in enumerate(games):
            ready = state == "ГОТОВО"
            card = ctk.CTkFrame(
                cards, fg_color=SURFACE_ALT if ready else "#1C2330", corner_radius=15,
                border_width=1, border_color="#34565B" if ready else "#2B3444",
            )
            card.grid(row=index // 2, column=index % 2, padx=5, pady=5, sticky="nsew")
            ctk.CTkLabel(
                card, text=icon, width=36, height=36, corner_radius=11,
                fg_color="#234247" if ready else "#29303C",
                font=ctk.CTkFont("Segoe UI Symbol", 17, "bold"),
                text_color=accent if ready else "#69778B",
            ).pack(anchor="w", padx=13, pady=(13, 8))
            ctk.CTkLabel(
                card, text=title, font=ctk.CTkFont("Segoe UI", 11, "bold"),
                text_color=TEXT if ready else MUTED,
            ).pack(anchor="w", padx=13)
            ctk.CTkLabel(
                card, text=description, font=ctk.CTkFont("Segoe UI", 8),
                text_color=MUTED, wraplength=205, justify="left",
            ).pack(anchor="w", padx=13, pady=(2, 8))
            ctk.CTkLabel(
                card, text=state, height=22, corner_radius=7,
                fg_color="#29473F" if ready else "#2A303B",
                font=ctk.CTkFont("Segoe UI", 7, "bold"),
                text_color="#83E0C1" if ready else "#778397",
            ).pack(anchor="w", padx=13, pady=(0, 12))
        ctk.CTkLabel(
            catalog,
            text="Один запуск F9 включает постоянное наблюдение: тип экрана определяется автоматически.",
            font=ctk.CTkFont("Segoe UI", 8, "bold"), text_color=accent,
            wraplength=550, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 15))

        live = panel(workspace, "LIVE  /  АВТООПРЕДЕЛЕНИЕ", accent)
        live.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        scanner = ctk.CTkFrame(live, fg_color="#172C32", corner_radius=18)
        scanner.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkLabel(
            scanner, text="⌁", width=72, height=72, corner_radius=36,
            fg_color="#203D43", font=ctk.CTkFont("Segoe UI Symbol", 32, "bold"),
            text_color=accent,
        ).pack(pady=(20, 8))
        ctk.CTkLabel(
            scanner, text="СКАНЕР ЭКРАНА", font=ctk.CTkFont("Segoe UI", 8, "bold"),
            text_color=accent,
        ).pack(pady=(0, 17))
        ctk.CTkLabel(
            live, textvariable=self.target, font=ctk.CTkFont("Segoe UI", 16, "bold"),
            text_color=TEXT, wraplength=300, justify="center",
        ).pack(padx=18, pady=(2, 5))
        ctk.CTkLabel(
            live, textvariable=self.status, font=ctk.CTkFont("Segoe UI", 9),
            text_color=MUTED, wraplength=300, justify="center",
        ).pack(fill="x", padx=18, pady=(0, 15))
        self.main_button = ctk.CTkButton(
            live, text="Включить автодетект  ·  F9", command=self.toggle,
            height=50, corner_radius=14, font=ctk.CTkFont("Segoe UI", 12, "bold"),
            fg_color=accent, hover_color="#86E4DF", text_color="#102022",
        )
        self.main_button.pack(fill="x", side="bottom", padx=16, pady=16)
        hotkey_bar(body, (("F9", "автодетект"), ("F11", "экстренная остановка")), accent)

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

    def capture_game_image(self) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        """Снять клиентскую область GTA без привязки к масштабу окна."""
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            return None
        bounds = client_bounds(self.game_window)
        if not bounds:
            return None
        left, top, width, height = bounds
        try:
            with mss.mss() as screen:
                image = np.asarray(screen.grab({"left": left, "top": top, "width": width, "height": height}))[:, :, :3]
        except Exception:
            return None
        return image, bounds

    def detect_electric_minigame(self) -> None:
        self.action_job = None
        if not self.running or not self.active:
            return
        captured = self.capture_game_image()
        if captured is None:
            self.reset_tumbler_candidate()
            self.target.set("Автодетект включён · жду GTA")
            self.status.set(f"Процесс «{self.process.get()}» пока не найден. Продолжаю наблюдение.")
            self.schedule_detection(900)
            return
        if not self.game_window or user32.GetForegroundWindow() != self.game_window:
            self.reset_tumbler_candidate()
            self.target.set("Автодетект на паузе · GTA не активно")
            self.status.set("Вернись в GTA — наблюдение продолжится автоматически.")
            self.schedule_detection(700)
            return
        image, _bounds = captured

        # Зелёную плату можно узнать за миллисекунды. Тяжёлое восстановление
        # графа выполняется после этого в фоне, поэтому интерфейс сразу
        # сообщает правильный тип игры и не зависает на несколько секунд.
        if self.is_maze_screen(image):
            self.reset_tumbler_candidate()
            self.action_kind = "maze_detection"
            self.target.set("Электро-лабиринт распознан")
            self.status.set("Строю маршрут от зелёной точки A до красной Б…")
            generation = self.detection_generation

            def solve_maze() -> None:
                route = self.analyze_maze_route(image)
                self.after(0, lambda: self.finish_maze_detection(route, generation))

            threading.Thread(target=solve_maze, daemon=True, name="kiski-maze-detector").start()
            return
        tumbler_layout = self.detect_tumbler_layout(captured)
        if tumbler_layout is not None:
            tumblers, initial_states = tumbler_layout
            signature = self.tumbler_layout_signature(tumblers, _bounds)
            # Цветной решётки недостаточно: на том же кадре обязательно должны
            # присутствовать два настоящих прибора с красными целевыми зонами.
            gauges = self.analyze_tumbler_gauges(image)
            if signature is None or gauges is None:
                self.reset_tumbler_candidate()
                self.recognize_and_solve(captured)
                return
            if signature == self.tumbler_candidate_signature:
                self.tumbler_candidate_hits += 1
            else:
                self.tumbler_candidate_signature = signature
                self.tumbler_candidate_hits = 1
            if self.tumbler_candidate_hits < 2:
                self.target.set("Проверяю экран тумблеров…")
                self.status.set("Жду второй совпадающий кадр перед первым кликом.")
                self.schedule_detection(240)
                return
            self.reset_tumbler_candidate()
            self.tumbler_positions = tumblers
            self.tumbler_initial_states = initial_states
            self.tumbler_effects = []
            self.tumbler_index = 0
            self.tumbler_phase = "normalize"
            self.tumbler_click_retries = 0
            self.tumbler_correction_round = 0
            _initial_values, self.tumbler_targets = gauges
            self.target.set(f"Тумблеры найдены · {len(tumblers)} шт.")
            self.status.set("Сначала выключаю тумблеры, затем измеряю их по одному.")
            self.run_tumbler_step()
            return
        self.reset_tumbler_candidate()
        self.recognize_and_solve(captured)

    def finish_maze_detection(self, route: list[int] | None, generation: int) -> None:
        if generation != self.detection_generation or not self.running or not self.active:
            return
        if self.action_kind != "maze_detection":
            return
        if route is None:
            self.resume_monitoring(
                "Лабиринт распознан, но маршрут пока не читается. Повторю автоматически.",
                1000,
            )
            return
        if not route:
            self.resume_monitoring("Точка A уже в точке Б. Продолжаю автодетект.", 1200)
            return
        self.actions = route
        self.action_kind = "maze"
        self.action_interval = 300
        self.target.set(f"Электро-лабиринт · путь {len(route)} шагов")
        self.status.set("Маршрут от A до Б найден. Отправляю W/A/S/D.")
        self.run_next_action()

    def schedule_detection(self, delay: int = 450) -> None:
        if not self.running or not self.active:
            return
        self.action_job = self.after(delay, self.detect_electric_minigame)

    def reset_tumbler_candidate(self) -> None:
        self.tumbler_candidate_signature = None
        self.tumbler_candidate_hits = 0

    def resume_monitoring(self, message: str, delay: int = 900) -> None:
        """Завершить один вызов, не выключая постоянный автодетект."""
        if not self.running or not self.active:
            return
        self.actions.clear()
        if self.action_job is not None:
            try:
                self.after_cancel(self.action_job)
            except tk.TclError:
                pass
        self.action_job = None
        self.action_kind = ""
        self.reset_tumbler_candidate()
        self.target.set("Автодетект включён · жду мини-игру")
        self.status.set(message)
        self.schedule_detection(delay)

    def toggle(self) -> None:
        if self.running:
            self.stop("Автодетект остановлен.")
            return
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            self.status.set(f"Процесс «{self.process.get()}» не найден.")
            return
        if not activate_window(self.game_window):
            self.status.set("Не удалось вывести GTA на передний план.")
            return
        self.running = True
        self.detection_generation += 1
        self.main_button.configure(text="Остановить автодетект  ·  F9", fg_color="#E95E69", hover_color="#C94C57", text_color=TEXT)
        self.target.set("Автодетект включён · жду мини-игру")
        self.status.set("Можно ехать на вызов: Вольт сам распознает появившуюся мини-игру.")
        self.schedule_detection(250)

    def run_next_action(self) -> None:
        self.action_job = None
        if not self.running or not self.active:
            return
        if not self.actions:
            if self.action_kind == "current":
                self.target.set("Проверяю обе цепи…")
                self.status.set("Жду реакцию мини-игры и сверяю итоговую схему.")
                self.action_job = self.after(420, self.verify_current_completion)
                return
            self.resume_monitoring("Лабиринт пройден. Жду следующий вызов.", 1400)
            return
        if self.action_kind in ("current", "maze"):
            current_window = user32.GetForegroundWindow()
            if not self.game_window or not find_game_window(self.process.get()) or current_window != self.game_window:
                self.stop("Ввод остановлен: окно GTA больше не активно.")
                return
        key = self.actions.pop(0)
        if not send_key_tap(key):
            self.stop("Windows не принял клавишу. Решение остановлено.")
            return
        # Лабиринт обновляет позицию между командами. 38 мс были быстрее
        # игрового интерфейса: приложение считало шаг отправленным, а GTA
        # фактически пропускала его. Держим безопасный темп.
        delay = 155 if key == VK_SPACE else self.action_interval
        self.action_job = self.after(delay, self.run_next_action)

    def stop(self, message: str) -> None:
        self.running = False
        self.detection_generation += 1
        self.actions.clear()
        if self.action_job is not None:
            try:
                self.after_cancel(self.action_job)
            except tk.TclError:
                pass
        self.action_job = None
        self.action_kind = ""
        self.reset_tumbler_candidate()
        self.main_button.configure(text="Включить автодетект  ·  F9", fg_color="#72D7D2", hover_color="#86E4DF", text_color="#102022")
        self.target.set("Автодетект выключен")
        self.status.set(message)

    def activate(self) -> None:
        self.active = True
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}
        self.target.set("Автодетект выключен")
        self.status.set("Нажми F9 один раз перед началом работы.")

    def deactivate(self, message: str) -> None:
        self.stop(message)
        self.active = False
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def poll_hotkeys(self) -> None:
        actions = {VK_F9: self.toggle, VK_F11: lambda: self.stop("Экстренно остановлено.")}
        for key, action in actions.items():
            down = bool(user32.GetAsyncKeyState(key) & 0x8000)
            if self.active and down and not self.keys[key]:
                action()
            self.keys[key] = down
        if self.winfo_exists():
            self.after(40, self.poll_hotkeys)
