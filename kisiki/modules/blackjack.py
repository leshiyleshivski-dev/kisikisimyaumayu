"""Automatic blackjack rounds until a requested number of wins."""

from __future__ import annotations

import random
import time

import customtkinter as ctk
import mss
import numpy as np

from ..core import (
    APP_BG, GOLD, MINT, MUTED, SURFACE_ALT, TEXT,
    VK_F9, VK_F11, VK_RETURN, VK_SPACE,
    activate_window, client_bounds, find_game_window, send_key_tap,
    user32, window_title,
)
from .blackjack_vision import (
    HandReader, classify_round_toast, decide_move, hand_totals, table_phase,
)
from .ui import connection_panel, hotkey_bar, module_header, panel, step_list


MOVE_KEYS = {"hit": VK_RETURN, "stand": VK_SPACE}
MOVE_TITLES = {"hit": "Ещё", "stand": "Достаточно"}


def parse_win_target(value: str, maximum: int = 999) -> int | None:
    """Return a positive number of wins or ``None`` for unsafe input."""
    try:
        target = int(value.strip())
    except (AttributeError, ValueError):
        return None
    return target if 1 <= target <= maximum else None


class BlackjackModule(ctk.CTkFrame):
    """Bet, play the hand by the book and stop at the requested wins."""

    NEXT_ROUND_DELAY_SECONDS = (1.4, 2.1)
    # Раздача, ход дилера и уведомление укладываются секунд в двадцать пять.
    # Это только предохранитель: любой распознанный результат принимается
    # сразу, поэтому обычный круг столько не ждёт.
    ROUND_TIMEOUT_SECONDS = 60.0
    BET_ACCEPT_TIMEOUT_SECONDS = 8.0
    MOVE_ACCEPT_TIMEOUT_SECONDS = 12.0
    TOAST_CLEAR_TIMEOUT_SECONDS = 15.0
    FOCUS_SETTLE_SECONDS = 0.30
    RETRY_SECONDS = 1.0
    # Рука читается по кадрам, и один кадр может испортить рукой дилера или
    # бабочкой. Ход делаем только после трёх одинаковых чтений подряд.
    STABLE_READS = 3
    SCAN_INTERVAL_MS = 120
    MAX_TARGET_WINS = 999

    def __init__(self, parent: ctk.CTkFrame, on_back) -> None:
        super().__init__(parent, fg_color=APP_BG, corner_radius=0)
        self.on_back = on_back
        self.active = False
        self.running = False
        self.stage = "idle"
        self.deadline: float | None = None
        self.next_action: float | None = None
        self.game_window: int | None = None
        self.previous_window: int | None = None
        self.target_wins = 15
        self.wins = 0
        self.rounds = 0
        self.no_wins = 0
        self.hits = 0
        self.pending_move: str | None = None
        self.hand_reader = HandReader()
        self.last_reading: tuple[int, int] | None = None
        self.stable_count = 0
        self.first_bet = True
        self.keys = {key: False for key in (VK_F9, VK_F11)}

        self.process = ctk.StringVar(value="GTA5.exe")
        self.connection = ctk.StringVar(value="Ищу GTA5.exe…")
        self.target_input = ctk.StringVar(value="15")
        self.progress = ctk.StringVar(value="0 / 15 побед")
        self.stats = ctk.StringVar(value="Раздач 0  ·  без выигрыша 0  ·  добрано карт 0")
        self.hand = ctk.StringVar(value="Рука не прочитана")
        self.status = ctk.StringVar(
            value="Сядь за стол и сам выставь размер ставки в GTA. Модуль её не меняет."
        )
        self.timer = ctk.StringVar(value="Автоигра не запущена")

        self.build_ui()
        self.after(40, self.poll_hotkeys)
        self.after(100, self.tick)
        self.after(self.SCAN_INTERVAL_MS, self.scan_tick)
        self.after(0, self.refresh_connection)

    # ------------------------------------------------------------------ UI

    def build_ui(self) -> None:
        accent = "#68D6B4"
        module_header(
            self, code="21", title="Blackjack Run",
            subtitle="не меняет ставку и играет раздачи до заданного числа побед",
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
        workspace.grid_columnconfigure(0, weight=6, uniform="blackjack")
        workspace.grid_columnconfigure(1, weight=5, uniform="blackjack")
        workspace.grid_rowconfigure(0, weight=1)

        live = panel(workspace, "LIVE  /  ЦЕЛЬ ПО ПОБЕДАМ", accent)
        live.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        settings = ctk.CTkFrame(live, fg_color="#233A34", corner_radius=17)
        settings.pack(fill="x", padx=16, pady=(0, 13))
        settings.grid_columnconfigure(0, weight=1)

        target_box = ctk.CTkFrame(settings, fg_color="transparent")
        target_box.grid(row=0, column=0, padx=14, pady=11, sticky="ew")
        ctk.CTkLabel(
            target_box, text="ОСТАНОВИТЬСЯ ПОСЛЕ", font=ctk.CTkFont("Segoe UI", 9, "bold"),
            text_color=accent,
        ).pack(anchor="w")
        self.target_entry = ctk.CTkEntry(
            target_box, textvariable=self.target_input, height=36, justify="center",
            corner_radius=10, border_color="#3F7A68", fg_color="#171B22",
            font=ctk.CTkFont("Segoe UI", 17, "bold"),
        )
        self.target_entry.pack(fill="x", pady=(4, 0))

        ctk.CTkLabel(
            live, textvariable=self.progress, font=ctk.CTkFont("Segoe UI", 25, "bold"),
            text_color=TEXT,
        ).pack(padx=18, pady=(3, 2))
        ctk.CTkLabel(
            live, textvariable=self.stats, font=ctk.CTkFont("Segoe UI", 10, "bold"),
            text_color=accent,
        ).pack(padx=18, pady=(0, 5))
        ctk.CTkLabel(
            live, textvariable=self.hand, font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color=TEXT,
        ).pack(padx=18, pady=(0, 5))
        ctk.CTkLabel(
            live, textvariable=self.timer, font=ctk.CTkFont("Segoe UI", 12, "bold"),
            text_color=MUTED,
        ).pack(fill="x", padx=20, pady=(2, 4))
        ctk.CTkLabel(
            live, textvariable=self.status, font=ctk.CTkFont("Segoe UI", 9),
            text_color=MUTED, wraplength=410, justify="center",
        ).pack(fill="x", padx=24, pady=(0, 14))
        self.main_button = ctk.CTkButton(
            live, text="Запустить раздачи  ·  F9", command=self.toggle,
            height=50, corner_radius=14, font=ctk.CTkFont("Segoe UI", 13, "bold"),
            fg_color=accent, hover_color="#84E7C8", text_color="#10211C",
        )
        self.main_button.pack(fill="x", padx=16, pady=(0, 16))

        guide = panel(workspace, "КАК ЗАПУСТИТЬ", accent)
        guide.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        step_list(guide, (
            ("1", "Сядь за стол", "Оставь на экране «РАЗМЕР СТАВКИ» и нижнюю строку подсказок."),
            ("2", "Выставь ставку в GTA", "Размер выбирается стрелками вручную — модуль их не нажимает."),
            ("3", "Укажи победы", "Цель 15 означает: проигрыши не останавливают цикл до пятнадцатой победы."),
            ("4", "Нажми F9", "Модуль сам сделает ставку, доберёт карты по базовой стратегии и посчитает результат."),
        ), accent)
        note = ctk.CTkFrame(guide, fg_color=SURFACE_ALT, corner_radius=12)
        note.pack(fill="x", padx=14, pady=(1, 14))
        ctk.CTkLabel(
            note,
            text=(
                "Дабл модуль не берёт: ставка остаётся ровно такой, какую выставил ты. "
                "Если стол или счёт руки не распознаны, цикл остановится и не нажмёт вслепую."
            ),
            font=ctk.CTkFont("Segoe UI", 8, "bold"), text_color=GOLD,
            wraplength=315, justify="left",
        ).pack(anchor="w", padx=12, pady=10)
        hotkey_bar(body, (("F9", "старт / стоп"), ("F11", "аварийная остановка")), accent)

    # ------------------------------------------------------------- GTA glue

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

    def capture_client(self) -> np.ndarray | None:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            return None
        bounds = client_bounds(self.game_window)
        if not bounds:
            return None
        left, top, width, height = bounds
        try:
            with mss.mss() as screen:
                return np.asarray(screen.grab({
                    "left": left, "top": top, "width": width, "height": height,
                }))[:, :, :3]
        except Exception:
            return None

    def focus_game(self) -> bool:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            return False
        previous = user32.GetForegroundWindow()
        if self.previous_window is None and previous and previous != self.game_window:
            self.previous_window = previous
        return activate_window(self.game_window)

    def restore_previous_window(self) -> None:
        previous = self.previous_window
        self.previous_window = None
        if previous and previous != self.game_window and user32.IsWindow(previous):
            activate_window(previous)

    def press(self, virtual_key: int) -> bool:
        """Перевести фокус в GTA и отправить одно нажатие."""
        if not self.focus_game():
            return False
        # На первом нажатии перед игрой стоит наше окно: даём GTA принять
        # фокус, иначе клавиша уходит в пустоту.
        if self.first_bet:
            time.sleep(self.FOCUS_SETTLE_SECONDS)
            self.first_bet = False
        return send_key_tap(virtual_key)

    # ------------------------------------------------------------ scoreboard

    def update_score(self) -> None:
        self.progress.set(f"{self.wins} / {self.target_wins} побед")
        self.stats.set(
            f"Раздач {self.rounds}  ·  без выигрыша {self.no_wins}  ·  добрано карт {self.hits}"
        )

    def complete_round(self, result: str | None) -> None:
        self.rounds += 1
        if result == "win":
            self.wins += 1
        else:
            self.no_wins += 1
        self.update_score()
        if self.wins >= self.target_wins:
            self.finish(f"Цель выполнена: {self.wins} побед за {self.rounds} раздач.")
            return
        self.stage = "settle"
        self.deadline = time.monotonic() + self.TOAST_CLEAR_TIMEOUT_SECONDS
        self.next_action = None
        if result == "win":
            self.timer.set("Победа засчитана · жду, пока уведомление уйдёт")
        elif result == "loss":
            self.timer.set("Раздача проиграна · жду, пока уведомление уйдёт")
        else:
            self.timer.set("Раздача закрыта без уведомления · готовлю следующую ставку")

    # ------------------------------------------------------------ lifecycle

    def toggle(self) -> None:
        if self.running:
            self.stop("Автоигра остановлена вручную.")
            return
        target = parse_win_target(self.target_input.get(), self.MAX_TARGET_WINS)
        if target is None:
            self.status.set(f"Введи целое число побед от 1 до {self.MAX_TARGET_WINS}.")
            self.target_entry.focus_set()
            return
        if not find_game_window(self.process.get()):
            self.status.set(f"Процесс «{self.process.get()}» не найден.")
            return
        self.target_wins = target
        self.wins = self.rounds = self.no_wins = self.hits = 0
        self.pending_move = None
        self.hand_reader.reset()
        self.last_reading = None
        self.stable_count = 0
        self.first_bet = True
        self.running = True
        self.stage = "await_bet"
        self.deadline = time.monotonic() + self.ROUND_TIMEOUT_SECONDS
        self.next_action = None
        self.target_entry.configure(state="disabled")
        self.main_button.configure(
            text="Остановить раздачи  ·  F9", fg_color="#E95E69",
            hover_color="#C94C57", text_color=TEXT,
        )
        self.update_score()
        self.hand.set("Рука не прочитана")
        self.timer.set("Жду экран ставки…")
        self.status.set("F9 принят. Ручной размер ставки останется без изменений.")

    def stop(self, message: str) -> None:
        self.running = False
        self.stage = "idle"
        self.deadline = self.next_action = None
        self.restore_previous_window()
        self.target_entry.configure(state="normal")
        self.main_button.configure(
            text="Запустить раздачи  ·  F9", fg_color="#68D6B4",
            hover_color="#84E7C8", text_color="#10211C",
        )
        self.timer.set("Автоигра не запущена")
        self.status.set(message)

    def finish(self, message: str) -> None:
        self.stop(message)
        self.timer.set("Цель по победам выполнена")

    def activate(self) -> None:
        self.active = True
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def deactivate(self, message: str) -> None:
        self.stop(message)
        self.active = False
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    # ------------------------------------------------------------ main loop

    def stable_totals(self, frame: np.ndarray | None) -> tuple[int, int] | None:
        """Отдать счёт руки, подтверждённый несколькими кадрами подряд."""
        reading = hand_totals(frame)
        if reading is None or reading != self.last_reading:
            self.last_reading = reading
            self.stable_count = 1 if reading is not None else 0
            return None
        self.stable_count += 1
        return reading if self.stable_count >= self.STABLE_READS else None

    def place_bet(self) -> None:
        if not self.press(VK_RETURN):
            self.next_action = time.monotonic() + self.RETRY_SECONDS
            self.status.set("Не удалось нажать «Сделать ставку». Повторю через секунду.")
            return
        self.hand_reader.reset()
        self.last_reading = None
        self.stable_count = 0
        self.stage = "bet_sent"
        self.deadline = time.monotonic() + self.BET_ACCEPT_TIMEOUT_SECONDS
        self.next_action = None
        self.hand.set("Рука не прочитана")
        self.timer.set(f"Раздача №{self.rounds + 1} · ставка сделана")
        self.status.set("Жду раздачу карт.")

    def play_move(self, player: int, dealer: int, *, soft: bool) -> None:
        move = decide_move(player, dealer, soft=soft)
        if not self.press(MOVE_KEYS[move]):
            self.next_action = time.monotonic() + self.RETRY_SECONDS
            self.status.set(f"Не удалось нажать «{MOVE_TITLES[move]}». Повторю через секунду.")
            return
        if move == "hit":
            self.hits += 1
            self.hand_reader.draw()
            self.update_score()
        self.pending_move = move
        self.last_reading = None
        self.stable_count = 0
        self.stage = "move_sent"
        self.deadline = time.monotonic() + self.MOVE_ACCEPT_TIMEOUT_SECONDS
        self.next_action = None
        self.status.set(
            f"У вас {'мягкие ' if soft else ''}{player}, у дилера {dealer}"
            f" — нажал «{MOVE_TITLES[move]}»."
        )

    def scan_tick(self) -> None:
        if self.winfo_exists() and self.running and self.next_action is None:
            self.advance(self.capture_client())
        if self.winfo_exists():
            self.after(self.SCAN_INTERVAL_MS, self.scan_tick)

    def advance(self, frame: np.ndarray | None) -> None:
        phase = table_phase(frame)
        toast = classify_round_toast(frame)
        if self.stage == "await_bet":
            if toast is not None:
                # Старое уведомление ещё висит: подождём, иначе засчитаем
                # его как результат новой раздачи.
                return
            if phase == "bet":
                self.place_bet()
            return
        if self.stage == "settle":
            if toast is None:
                delay = random.uniform(*self.NEXT_ROUND_DELAY_SECONDS)
                self.stage = "await_bet"
                self.deadline = time.monotonic() + self.ROUND_TIMEOUT_SECONDS
                self.next_action = time.monotonic() + delay
                self.timer.set(f"Следующая ставка через {delay:.1f} с")
            return
        if self.stage == "bet_sent":
            if phase is not None and phase != "bet":
                self.stage = "play"
                self.deadline = time.monotonic() + self.ROUND_TIMEOUT_SECONDS
                self.timer.set(f"Раздача №{self.rounds + 1} · читаю руку")
            return
        if self.stage == "move_sent":
            if phase is not None and phase != "action":
                self.stage = "play"
                self.deadline = time.monotonic() + self.ROUND_TIMEOUT_SECONDS
                self.timer.set(
                    f"Раздача №{self.rounds + 1} · "
                    + ("жду новую карту" if self.pending_move == "hit" else "жду ход дилера")
                )
            return
        if self.stage == "play":
            if toast is not None:
                self.complete_round(toast)
                return
            if phase == "bet":
                # Раздача закрылась без уведомления — например ничья.
                self.complete_round(None)
                return
            totals = self.stable_totals(frame)
            if totals is None:
                return
            player, dealer = totals
            soft = self.hand_reader.update(player, settled=phase == "action")
            self.hand.set(
                f"У вас {'мягкие ' if soft else ''}{player}  ·  у дилера {dealer}"
            )
            if phase == "action":
                self.play_move(player, dealer, soft=soft)

    def tick(self) -> None:
        if self.winfo_exists() and self.running:
            now = time.monotonic()
            if self.next_action is not None and now >= self.next_action:
                self.next_action = None
            if self.deadline is not None and now >= self.deadline:
                self.stop(self.timeout_message())
        if self.winfo_exists():
            self.after(100, self.tick)

    def timeout_message(self) -> str:
        messages = {
            "await_bet": "Экран ставки так и не появился. Остановлено без лишних нажатий.",
            "bet_sent": "Ставка не принята столом. Остановлено без лишних нажатий.",
            "play": "Результат раздачи не найден. Остановлено без лишних нажатий.",
            "move_sent": "Стол не принял ход. Остановлено без лишних нажатий.",
            "settle": "Уведомление не исчезло. Остановлено без лишних нажатий.",
        }
        return messages.get(self.stage, "Остановлено по тайм-ауту.")

    def poll_hotkeys(self) -> None:
        actions = {
            VK_F9: self.toggle,
            VK_F11: lambda: self.stop("Экстренно остановлено."),
        }
        for key, action in actions.items():
            down = bool(user32.GetAsyncKeyState(key) & 0x8000)
            if self.active and down and not self.keys[key]:
                action()
            self.keys[key] = down
        if self.winfo_exists():
            self.after(40, self.poll_hotkeys)
