"""Automatic slot spins until a requested number of wins."""

from __future__ import annotations

import random
import time

import cv2
import customtkinter as ctk
import mss
import numpy as np

from ..core import (
    APP_BG, BODY, FONT_BODY, FONT_CAPTION, FONT_NOTE, GOLD, MINT, SURFACE_ALT, TEXT,
    VK_F, VK_F9, VK_F11,
    activate_window, client_bounds, find_game_window, send_key_tap,
    user32, window_title,
)
from .ui import connection_panel, hotkey_bar, module_header, panel, step_list


SLOT_BET_PANEL_RATIO = (0.006, 0.015, 0.175, 0.052)
SLOT_CONTROLS_RATIO = (0.555, 0.935, 0.270, 0.064)
SLOT_TOAST_ACCENT_RATIO = (0.595, 0.895, 0.018, 0.095)


def _ratio_crop(image: np.ndarray, ratio: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, region_width, region_height = ratio
    return image[
        max(0, round(height * y)):min(height, round(height * (y + region_height))),
        max(0, round(width * x)):min(width, round(width * (x + region_width))),
    ]


def parse_win_target(value: str, maximum: int = 999) -> int | None:
    """Return a positive number of wins or ``None`` for unsafe input."""
    try:
        target = int(value.strip())
    except (AttributeError, ValueError):
        return None
    return target if 1 <= target <= maximum else None


def slot_interface_visible(frame: np.ndarray | None) -> bool:
    """Confirm the stake panel and the bottom slot-control bar before sending keys."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return False
    bet_panel = _ratio_crop(frame[:, :, :3], SLOT_BET_PANEL_RATIO)
    controls = _ratio_crop(frame[:, :, :3], SLOT_CONTROLS_RATIO)
    if bet_panel.size == 0 or controls.size == 0:
        return False
    bet_gray = cv2.cvtColor(bet_panel, cv2.COLOR_BGR2GRAY)
    controls_gray = cv2.cvtColor(controls, cv2.COLOR_BGR2GRAY)
    return (
        float(np.mean(bet_gray < 85)) >= 0.42
        and float(np.mean(bet_gray > 145)) >= 0.025
        and float(np.mean(controls_gray < 85)) >= 0.30
        and float(np.mean(controls_gray > 170)) >= 0.025
    )


def can_start_slot_spin(frame: np.ndarray | None, *, first_spin: bool) -> bool:
    """The first F9 must start a spin even if the initial UI capture is obscured."""
    return first_spin or slot_interface_visible(frame)


def _strongest_vertical_accent(mask: np.ndarray) -> int:
    height, width = mask.shape
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    minimum_height = max(10, round(height * 0.24))
    minimum_area = max(18, round(width * height * 0.012))
    return max((
        int(area)
        for _x, _y, component_width, component_height, area in stats[1:count]
        if component_width >= 2
        and component_height >= minimum_height
        and area >= minimum_area
    ), default=0)


def classify_slot_result(frame: np.ndarray | None) -> str | None:
    """Return ``win`` or ``loss`` for the slot result toast shown in the video."""
    if frame is None or not slot_interface_visible(frame):
        return None
    accent = _ratio_crop(frame[:, :, :3], SLOT_TOAST_ACCENT_RATIO)
    if accent.size == 0:
        return None
    hsv = cv2.cvtColor(accent, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv, np.array((35, 70, 90), dtype=np.uint8),
        np.array((95, 255, 255), dtype=np.uint8),
    )
    red_low = cv2.inRange(
        hsv, np.array((0, 70, 90), dtype=np.uint8),
        np.array((15, 255, 255), dtype=np.uint8),
    )
    red_high = cv2.inRange(
        hsv, np.array((155, 70, 90), dtype=np.uint8),
        np.array((179, 255, 255), dtype=np.uint8),
    )
    green_score = _strongest_vertical_accent(green)
    red_score = _strongest_vertical_accent(cv2.bitwise_or(red_low, red_high))
    if green_score > red_score and green_score:
        return "win"
    if red_score > green_score and red_score:
        return "loss"
    return None


def slot_win_visible(frame: np.ndarray | None) -> bool:
    """Recognise the green ``Вы выиграли … фишек!`` notification from Majestic."""
    return classify_slot_result(frame) == "win"


class SlotSpinnerModule(ctk.CTkFrame):
    """Press F with slight timing variation and stop at the requested wins."""

    NEXT_SPIN_DELAY_SECONDS = (1.7, 2.4)
    # Different slot machines keep their reels moving for different lengths.
    # This is only a safety ceiling: a red/green result is accepted immediately,
    # so a normal spin never waits out all twenty seconds.
    SPIN_RESULT_TIMEOUT_SECONDS = 20.0
    FIRST_FOCUS_SETTLE_SECONDS = 0.30
    SPIN_RETRY_SECONDS = 1.0
    SCAN_INTERVAL_MS = 90
    MAX_TARGET_WINS = 999

    def __init__(self, parent: ctk.CTkFrame, on_back) -> None:
        super().__init__(parent, fg_color=APP_BG, corner_radius=0)
        self.on_back = on_back
        self.active = False
        self.running = False
        self.phase = "idle"
        self.next_action: float | None = None
        self.deadline: float | None = None
        self.game_window: int | None = None
        self.previous_window: int | None = None
        self.target_wins = 4
        self.wins = 0
        self.spins = 0
        self.no_wins = 0
        self.round_won = False
        self.round_result: str | None = None
        self.first_spin = True
        self.keys = {key: False for key in (VK_F9, VK_F11)}

        self.process = ctk.StringVar(value="GTA5.exe")
        self.connection = ctk.StringVar(value="Ищу GTA5.exe…")
        self.target_input = ctk.StringVar(value="4")
        self.progress = ctk.StringVar(value="0 / 4 побед")
        self.stats = ctk.StringVar(value="Вращений 0  ·  без выигрыша 0")
        self.status = ctk.StringVar(
            value="Сядь за слот и сам выставь ставку в GTA. Модуль её не меняет."
        )
        self.timer = ctk.StringVar(value="Автовращение не запущено")

        self.build_ui()
        self.after(40, self.poll_hotkeys)
        self.after(100, self.tick)
        self.after(self.SCAN_INTERVAL_MS, self.scan_tick)
        self.after(0, self.refresh_connection)

    def build_ui(self) -> None:
        accent = "#D4A7FF"
        module_header(
            self, code="777", title="Slot Spinner",
            subtitle="не меняет ставку и крутит слот до заданного числа побед",
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
        workspace.grid_columnconfigure(0, weight=6, uniform="slots")
        workspace.grid_columnconfigure(1, weight=5, uniform="slots")
        workspace.grid_rowconfigure(0, weight=1)

        live = panel(workspace, "LIVE  /  ЦЕЛЬ ПО ПОБЕДАМ", accent)
        live.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        settings = ctk.CTkFrame(live, fg_color="#30263A", corner_radius=17)
        settings.pack(fill="x", padx=16, pady=(0, 13))
        settings.grid_columnconfigure(0, weight=1)

        target_box = ctk.CTkFrame(settings, fg_color="transparent")
        target_box.grid(row=0, column=0, padx=14, pady=11, sticky="ew")
        ctk.CTkLabel(
            target_box, text="ОСТАНОВИТЬСЯ ПОСЛЕ",
            font=ctk.CTkFont("Segoe UI", FONT_CAPTION, "bold"), text_color=accent,
        ).pack(anchor="w")
        self.target_entry = ctk.CTkEntry(
            target_box, textvariable=self.target_input, height=36, justify="center",
            corner_radius=10, border_color="#705682", fg_color="#171B22",
            font=ctk.CTkFont("Segoe UI", 17, "bold"),
        )
        self.target_entry.pack(fill="x", pady=(4, 0))

        ctk.CTkLabel(
            live, textvariable=self.progress, font=ctk.CTkFont("Segoe UI", 25, "bold"),
            text_color=TEXT,
        ).pack(padx=18, pady=(3, 2))
        ctk.CTkLabel(
            live, textvariable=self.stats,
            font=ctk.CTkFont("Segoe UI", FONT_NOTE, "bold"), text_color=accent,
        ).pack(padx=18, pady=(0, 7))
        ctk.CTkLabel(
            live, textvariable=self.timer, font=ctk.CTkFont("Segoe UI", 14, "bold"),
            text_color=TEXT,
        ).pack(fill="x", padx=20, pady=(5, 4))
        ctk.CTkLabel(
            live, textvariable=self.status, font=ctk.CTkFont("Segoe UI", FONT_BODY),
            text_color=BODY, wraplength=410, justify="center",
        ).pack(fill="x", padx=24, pady=(0, 14))
        self.main_button = ctk.CTkButton(
            live, text="Запустить вращения  ·  F9", command=self.toggle,
            height=50, corner_radius=14, font=ctk.CTkFont("Segoe UI", 13, "bold"),
            fg_color=accent, hover_color="#E2C2FF", text_color="#201526",
        )
        self.main_button.pack(fill="x", padx=16, pady=(0, 16))

        guide = panel(workspace, "КАК ЗАПУСТИТЬ", accent)
        guide.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        step_list(guide, (
            ("1", "Сядь за слот", "Оставь на экране строку «Пуск F» и данные текущей ставки."),
            ("2", "Выставь ставку в GTA", "Выбери нужный размер вручную — модуль не нажимает стрелки."),
            ("3", "Укажи победы", "Цель 4 означает: проигрыши не останавливают цикл до четвёртой победы."),
            ("4", "Нажми F9", "Между вращениями будет небольшой случайный интервал; победа читается автоматически."),
        ), accent)
        note = ctk.CTkFrame(guide, fg_color=SURFACE_ALT, corner_radius=12)
        note.pack(fill="x", padx=14, pady=(1, 14))
        ctk.CTkLabel(
            note,
            text="Если интерфейс слота пропал, модуль остановится и не станет нажимать F вслепую.",
            font=ctk.CTkFont("Segoe UI", FONT_BODY, "bold"), text_color=GOLD,
            wraplength=365, justify="left",
        ).pack(anchor="w", padx=12, pady=10)
        hotkey_bar(body, (("F9", "старт / стоп"), ("F11", "аварийная остановка")), accent)

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

    def update_score(self) -> None:
        self.progress.set(f"{self.wins} / {self.target_wins} побед")
        self.stats.set(f"Вращений {self.spins}  ·  без выигрыша {self.no_wins}")

    def start_spin(self) -> None:
        # On the first F9 the helper window is normally in front of GTA.  Focus
        # the game before taking a screenshot; otherwise the capture contains
        # our own UI and the slot check falsely fails.
        if not self.focus_game():
            self.phase = "ready"
            self.next_action = time.monotonic() + self.SPIN_RETRY_SECONDS
            self.status.set("Не удалось открыть GTA. Повторю первое нажатие через секунду.")
            return
        time.sleep(self.FIRST_FOCUS_SETTLE_SECONDS)
        frame = self.capture_client()
        if not can_start_slot_spin(frame, first_spin=self.first_spin):
            self.stop("Интерфейс слота пропал. Остановлено без лишнего нажатия.")
            return
        previous_result_visible = classify_slot_result(frame) is not None
        if not send_key_tap(VK_F):
            self.phase = "ready"
            self.next_action = time.monotonic() + self.SPIN_RETRY_SECONDS
            self.status.set("Не удалось запустить вращение. Повторю через секунду.")
            return
        self.first_spin = False
        now = time.monotonic()
        self.round_won = False
        self.round_result = None
        # The previous toast is allowed to remain on screen when F is pressed.
        # Ignore it until the notification slot becomes clean once; otherwise
        # it could be counted as the result of this new spin.
        self.phase = "spin_wait_stale" if previous_result_visible else "spin_wait"
        self.next_action = now + self.SPIN_RESULT_TIMEOUT_SECONDS
        self.deadline = self.next_action
        self.timer.set(f"Вращение №{self.spins + 1} · проверяю результат")
        self.status.set("Ищу зелёное уведомление о выигрыше.")

    def complete_spin(self) -> None:
        self.spins += 1
        if self.round_result == "win":
            self.wins += 1
        else:
            self.no_wins += 1
        self.update_score()
        if self.wins >= self.target_wins:
            self.finish(f"Цель выполнена: {self.wins} побед за {self.spins} вращений.")
            return
        delay = random.uniform(*self.NEXT_SPIN_DELAY_SECONDS)
        self.phase = "ready"
        self.next_action = time.monotonic() + delay
        self.deadline = None
        if self.round_result == "win":
            self.timer.set(f"Победа засчитана · следующий спин через {delay:.1f} с")
        else:
            self.timer.set(f"Без выигрыша · следующий спин через {delay:.1f} с")
        self.status.set("Уведомление может оставаться на экране — новый F нажмётся примерно через 2 секунды.")

    def toggle(self) -> None:
        if self.running:
            self.stop("Автовращение остановлено вручную.")
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
        self.wins = self.spins = self.no_wins = 0
        self.round_won = False
        self.round_result = None
        self.first_spin = True
        self.running = True
        self.phase = "ready"
        self.next_action = time.monotonic()
        self.deadline = None
        self.target_entry.configure(state="disabled")
        self.main_button.configure(
            text="Остановить вращения  ·  F9", fg_color="#E95E69",
            hover_color="#C94C57", text_color=TEXT,
        )
        self.update_score()
        self.timer.set("Проверяю слот…")
        self.status.set("F9 принят. Ручная ставка останется без изменений.")
        self.after(0, self.tick)

    def stop(self, message: str) -> None:
        self.running = False
        self.phase = "idle"
        self.next_action = self.deadline = None
        self.restore_previous_window()
        self.target_entry.configure(state="normal")
        self.main_button.configure(
            text="Запустить вращения  ·  F9", fg_color="#D4A7FF",
            hover_color="#E2C2FF", text_color="#201526",
        )
        self.timer.set("Автовращение не запущено")
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

    def tick(self) -> None:
        if self.winfo_exists() and self.running and self.next_action is not None:
            now = time.monotonic()
            remaining = self.next_action - now
            if remaining <= 0:
                if self.phase == "ready":
                    self.start_spin()
                elif self.phase in {"spin_wait", "spin_wait_stale"}:
                    self.stop("Результат вращения не найден. Остановлено без лишнего нажатия F.")
                elif self.phase == "result_found":
                    self.complete_spin()
            elif self.phase in {"spin_wait", "spin_wait_stale"}:
                self.timer.set(
                    f"Вращение №{self.spins + 1} · результат через {max(0.0, remaining):.1f} с"
                )
            elif self.phase == "ready":
                self.timer.set(f"Следующее вращение через {max(0.0, remaining):.1f} с")
        if self.winfo_exists():
            self.after(100, self.tick)

    def scan_tick(self) -> None:
        if self.winfo_exists() and self.running and self.phase == "spin_wait_stale":
            if classify_slot_result(self.capture_client()) is None:
                self.phase = "spin_wait"
                self.status.set("Старое уведомление исчезло. Жду результат текущего вращения.")
        if self.winfo_exists() and self.running and self.phase == "spin_wait":
            result = classify_slot_result(self.capture_client())
            if result is not None:
                self.round_result = result
                self.round_won = result == "win"
                self.phase = "result_found"
                self.next_action = time.monotonic() + 0.15
                self.status.set(
                    "Зелёное уведомление найдено — победа будет засчитана один раз."
                    if self.round_won else
                    "Красное уведомление найдено — вращение учтено без победы."
                )
        if self.winfo_exists():
            self.after(self.SCAN_INTERVAL_MS, self.scan_tick)

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
