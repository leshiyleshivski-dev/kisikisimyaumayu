"""Кисикисимяумяу — кликер с секретным модулем RLT Control.

Запускать можно прямо этим файлом или собранным .exe из папки dist.
"""

from __future__ import annotations

import ctypes
import json
import os
import random
import sys
import threading
import time
import tkinter as tk
import winsound
from ctypes import wintypes
from pathlib import Path

import cv2
import customtkinter as ctk
import mss
import numpy as np


# ---------------------------------------------------------------------------
# Общие данные кликера

APP_BG = "#21191C"
SURFACE = "#35272B"
SURFACE_HOVER = "#423035"
SURFACE_ALT = "#2C2024"
TEXT = "#FFF6EE"
MUTED = "#D8BEB1"
PURPLE = "#CC9AF2"
PINK = "#F4A6A5"
GOLD = "#FFD58A"
MINT = "#96DBB4"
HOLD_MS = 1250
BONGO_SECRET_TAPS = 10
BONGO_SECRET_TAP_WINDOW_SECONDS = 4.0
BUFF_SECRET_TAPS_PER_BEAT = 3
BUFF_SECRET_TAP_SPACING_SECONDS = 0.55
BUFF_SECRET_PAUSE_MIN_SECONDS = 0.70
BUFF_SECRET_PAUSE_MAX_SECONDS = 1.80

CATS = (
    ("Смайлик", "лукавый и очень уверенный", "01_smug_cat.png", "#7A6CF6"),
    ("Бонго", "отбивает ритм для удачи", "02_bongo_cat.png", "#FF9F68"),
    ("Апельсин", "заряжает настроение", "03_orange_meme_cat.png", "#F7B84C"),
    ("Бафф", "силён, но любит обнимашки", "04_buff_cat.png", "#4CC6A5"),
    ("Клавишник", "печатает только мяу", "05_keyboard_cat.png", "#5D9EFA"),
    ("Grumpy Cat", "недовольный хранитель секрета", "06_grumpy_cat.png", "#D37B78"),
)

SOUND_FILES = (
    "smug_laugh.mp3", "bongo_drums.mp3", "orange_boing.mp3",
    "buff_impact.mp3", "keyboard_type.mp3", "grumpy_trombone.mp3",
)
UPGRADES = {
    "paw": ("Лапка", "+1 к клику", 25, "#D88470"),
    "treat": ("Лакомство", "+1 мяу / сек", 90, "#CFA05C"),
    "laser": ("Лазер", "+3 к клику", 260, "#AF86CA"),
}


def resource_path(*parts: str) -> Path:
    """Путь работает одинаково для исходника и собранного приложения."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root.joinpath(*parts)


def rounded_photo(path: Path, max_width: int, max_height: int) -> tk.PhotoImage:
    """Tk умеет показывать PNG без дополнительных библиотек; масштабируем кратно."""
    image = tk.PhotoImage(file=str(path))
    width, height = image.width(), image.height()
    factor = max(1, (max(width / max_width, height / max_height) + 0.999).__int__())
    return image.subsample(factor, factor)


def progress_path() -> Path:
    data_root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "KisikiSimyaumyau"
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root / "progress.json"


# ---------------------------------------------------------------------------
# Оригинальная рабочая логика V6. Она перенесена в CTkFrame, а не запускается
# как отдельное приложение/окно.

user32 = ctypes.windll.user32
winmm = ctypes.windll.winmm
kernel32 = ctypes.windll.kernel32
kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = (
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.LPWSTR,
    ctypes.POINTER(wintypes.DWORD),
)
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL

VK_SPACE, VK_UP, VK_F4, VK_F9, VK_F11 = 0x20, 0x26, 0x73, 0x78, 0x7A
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_MOVE, MOUSEEVENTF_MOVE_NOCOALESCE = 0x0001, 0x2000
KEYEVENTF_KEYUP = 0x0002
SW_RESTORE = 9
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
RED_DIAMOND_RATIO = (1540 / 2560, 980 / 1440)
BLACK_DIAMOND_RATIO = (1706 / 2560, 980 / 1440)
GAME_CURSOR_DISTANCE_MULTIPLIER = 1.5
GAME_ROUND_SECONDS = 56
BET_DELAY_SECONDS = (7 * 60, 9 * 60)
SAFE_TIMER_MIN_SECONDS = 10
SAFE_TIMER_MAX_SECONDS = 34
TIME_READOUT_RATIO = (0.948, 0.892, 0.048, 0.055)
TIMER_SCAN_INTERVAL_SECONDS = 0.25
TIMER_ACTIVATE_SETTLE_SECONDS = 0.8
GAME_RECONNECT_SECONDS = 5
MISSING_TIMER_SHUTDOWN_SECONDS = 3.0
TIMING_BAR_RATIO = (0.80, 0.84, 0.19, 0.12)


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouse_data", wintypes.DWORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD), ("extra_info", ctypes.c_size_t)]


class KeyboardInput(ctypes.Structure):
    _fields_ = [("virtual_key", wintypes.WORD), ("scan_code", wintypes.WORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD), ("extra_info", ctypes.c_size_t)]


class InputUnion(ctypes.Union):
    _fields_ = [("mouse", MouseInput), ("keyboard", KeyboardInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("data", InputUnion)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.ClipCursor.argtypes = (ctypes.POINTER(wintypes.RECT),)
user32.ClipCursor.restype = wintypes.BOOL
user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
user32.SetCursorPos.restype = wintypes.BOOL


def make_dpi_aware() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        user32.SetProcessDPIAware()


def cursor_position() -> tuple[int, int]:
    point = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def confine_cursor_to_client(bounds: tuple[int, int, int, int]) -> bool:
    left, top, width, height = bounds
    clip_rect = wintypes.RECT(left, top, left + width, top + height)
    return bool(user32.ClipCursor(ctypes.byref(clip_rect)))


def process_for_window(hwnd: int) -> str:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        path = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
            return os.path.basename(path.value)
    finally:
        kernel32.CloseHandle(handle)
    return ""


def window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if not length:
        return ""
    text = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, text, length + 1)
    return text.value


def find_game_window(process_name: str) -> int | None:
    wanted = process_name.strip().casefold()
    if not wanted:
        return None
    found: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def inspect(hwnd: int, _lparam: int) -> bool:
        if user32.IsWindowVisible(hwnd) and process_for_window(hwnd).casefold() == wanted:
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(inspect, 0)
    return found[0] if found else None


def client_bounds(hwnd: int) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    top_left, bottom_right = wintypes.POINT(0, 0), wintypes.POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)) or not user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
        return None
    width, height = bottom_right.x - top_left.x, bottom_right.y - top_left.y
    return (top_left.x, top_left.y, width, height) if width > 0 and height > 0 else None


def activate_window(hwnd: int) -> bool:
    foreground = user32.GetForegroundWindow()
    foreground_thread = user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
    target_thread = user32.GetWindowThreadProcessId(hwnd, None)
    attached = False
    if foreground_thread and target_thread and foreground_thread != target_thread:
        attached = bool(user32.AttachThreadInput(foreground_thread, target_thread, True))
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    user32.BringWindowToTop(hwnd)
    user32.SetActiveWindow(hwnd)
    user32.SetForegroundWindow(hwnd)
    if attached:
        user32.AttachThreadInput(foreground_thread, target_thread, False)
    for _ in range(5):
        if user32.GetForegroundWindow() == hwnd:
            return True
        time.sleep(0.05)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    return False


def send_left_click(hold_seconds: float) -> bool:
    down, up = Input(), Input()
    down.type, down.data.mouse.flags = 0, MOUSEEVENTF_LEFTDOWN
    sent_down = user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(Input)) == 1
    time.sleep(hold_seconds)
    up.type, up.data.mouse.flags = 0, MOUSEEVENTF_LEFTUP
    return sent_down and user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(Input)) == 1


def send_key_tap(virtual_key: int) -> bool:
    """Отправить короткое нажатие клавиши активному окну через Windows input."""
    down, up = Input(), Input()
    down.type, down.data.keyboard.virtual_key = 1, virtual_key
    sent_down = user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(Input)) == 1
    up.type, up.data.keyboard.virtual_key, up.data.keyboard.flags = 1, virtual_key, KEYEVENTF_KEYUP
    return sent_down and user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(Input)) == 1


def send_relative_move(dx: int, dy: int) -> bool:
    distance = max(abs(dx), abs(dy))
    steps, sent_x, sent_y = max(1, (distance + 7) // 8), 0, 0
    for step in range(1, steps + 1):
        target_x, target_y = round(dx * step / steps), round(dy * step / steps)
        step_x, step_y = target_x - sent_x, target_y - sent_y
        if step_x or step_y:
            move = Input()
            move.type, move.data.mouse.dx, move.data.mouse.dy = 0, step_x, step_y
            move.data.mouse.flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_MOVE_NOCOALESCE
            if user32.SendInput(1, ctypes.byref(move), ctypes.sizeof(Input)) != 1:
                return False
        sent_x, sent_y = target_x, target_y
        if step < steps:
            time.sleep(0.008)
    return True


def move_relative_and_bet(dx: int, dy: int, *, settle_seconds: float, hold_seconds: float) -> bool:
    """Дойти до ромба, дождаться его обработки GTA и сделать один клик.

    После относительного перемещения GTA обновляет свой курсор не в тот же
    кадр. На втором (чёрном) ромбе прежняя пауза 0.3 с иногда приводила к
    тому, что нажатие терялось. Здесь задержка задаётся отдельно для каждого
    этапа, но второй клик всё так же ровно один.
    """
    if (dx or dy) and not send_relative_move(dx, dy):
        return False
    time.sleep(settle_seconds)
    return send_left_click(hold_seconds)


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
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=46, pady=(36, 30))
        ctk.CTkButton(top, text="←  К котикам", command=self.back, width=126, height=38,
                      corner_radius=12, fg_color="#26314E", hover_color="#344263",
                      font=ctk.CTkFont("Segoe UI", 12, "bold")).pack(side="left")
        self.alert_sound_button = ctk.CTkButton(
            top, command=self.toggle_alert_sound, width=142, height=38, corner_radius=12,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
        )
        self.alert_sound_button.pack(side="left", padx=(12, 0))
        self.refresh_alert_sound_button()
        title = ctk.CTkFrame(top, fg_color="transparent")
        title.pack(side="right")
        ctk.CTkLabel(title, text="RLT CONTROL", font=ctk.CTkFont("Segoe UI", 24, "bold"), text_color=TEXT).pack(anchor="e")
        ctk.CTkLabel(title, text="умная проверка окна ставок", font=ctk.CTkFont("Segoe UI", 11), text_color=MUTED).pack(anchor="e")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=46, pady=(0, 26))
        connection = ctk.CTkFrame(body, corner_radius=22, fg_color=SURFACE)
        connection.pack(fill="x", pady=(0, 16))
        connection.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(connection, text="ПОДКЛЮЧЕНИЕ К GTA", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color="#BFA9A2").grid(row=0, column=0, columnspan=2, padx=26, pady=(20, 8), sticky="w")
        self.point_badge = ctk.CTkLabel(connection, textvariable=self.point_status, font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color="#D5A29B")
        self.point_badge.grid(row=0, column=2, padx=(8, 26), pady=(20, 8), sticky="e")
        self.indicator = ctk.CTkLabel(connection, text="●", font=ctk.CTkFont(size=18), text_color="#F05A67")
        self.indicator.grid(row=1, column=0, padx=(26, 10), pady=(0, 21))
        ctk.CTkEntry(connection, textvariable=self.process, height=42, border_width=0, corner_radius=13, fg_color="#4A353B", font=ctk.CTkFont("Segoe UI", 14)).grid(row=1, column=1, padx=(0, 12), pady=(0, 21), sticky="ew")
        ctk.CTkButton(connection, text="Проверить", command=self.refresh_connection, width=122, height=42, corner_radius=13, fg_color="#C9715D", hover_color="#D8826B").grid(row=1, column=2, padx=(0, 26), pady=(0, 21))

        guide = ctk.CTkFrame(body, corner_radius=22, fg_color=SURFACE)
        guide.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(guide, text="КАК ЗАПУСТИТЬ", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color="#BFA9A2").pack(anchor="w", padx=26, pady=(20, 10))
        steps = (
            ("1", "Подготовь фишку", "Наведи игровую фишку на красный ромб и нажми F4."),
            ("2", "Запусти модуль", "Нажми F9 — GTA откроется, когда придёт время проверки."),
            ("3", "Дальше автоматически", "Ставка только при 00:34–00:10, затем возврат прежнего окна."),
        )
        for number, heading, description in steps:
            row = ctk.CTkFrame(guide, fg_color=SURFACE_ALT, corner_radius=14)
            row.pack(fill="x", padx=22, pady=(0, 8))
            ctk.CTkLabel(row, text=number, width=30, height=30, corner_radius=15, fg_color="#C9715D",
                         text_color=TEXT, font=ctk.CTkFont("Segoe UI", 13, "bold")).pack(side="left", padx=(12, 12), pady=10)
            copy = ctk.CTkFrame(row, fg_color="transparent")
            copy.pack(side="left", fill="x", expand=True, pady=8)
            ctk.CTkLabel(copy, text=heading, font=ctk.CTkFont("Segoe UI", 12, "bold"), text_color=TEXT).pack(anchor="w")
            ctk.CTkLabel(copy, text=description, font=ctk.CTkFont("Segoe UI", 10), text_color=MUTED).pack(anchor="w")
        ctk.CTkLabel(
            guide,
            text=(
                "🔔 За 10 секунд прозвучит отдельный сигнал  ·  случайный интервал: 07:00–09:00\n"
                "⚠ Если строка «ВРЕМЯ» пропала, приложение полностью закроется."
            ),
            font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=GOLD,
            justify="left",
        ).pack(anchor="w", padx=26, pady=(5, 16))

        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.pack(fill="x", pady=(0, 14))
        self.main_button = ctk.CTkButton(actions, text="Запустить  ·  F9", command=self.toggle, height=56, corner_radius=16, font=ctk.CTkFont("Segoe UI", 15, "bold"), fg_color="#C9715D", hover_color="#D8826B")
        self.main_button.pack(fill="x")
        monitor = ctk.CTkFrame(body, corner_radius=18, fg_color=SURFACE_ALT, border_width=1, border_color="#503A40")
        monitor.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(monitor, text="СОСТОЯНИЕ", font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color="#BFA9A2").pack(pady=(14, 4))
        ctk.CTkLabel(monitor, textvariable=self.timer, font=ctk.CTkFont("Segoe UI", 20, "bold"), text_color=TEXT).pack(pady=(0, 5))
        ctk.CTkLabel(monitor, textvariable=self.status, font=ctk.CTkFont("Segoe UI", 11), text_color=MUTED, wraplength=650, justify="center").pack(padx=22, pady=(0, 14))
        ctk.CTkLabel(body, text="F9 — запуск / остановка     ·     F11 — экстренная остановка", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color="#EEB28E").pack(pady=(0, 16))

    def back(self) -> None:
        self.on_back()

    def refresh_alert_sound_button(self) -> None:
        if self.alert_sound_enabled:
            self.alert_sound_button.configure(
                text="🔔  Сигнал: вкл", fg_color="#A66A45", hover_color="#BC7B50",
            )
        else:
            self.alert_sound_button.configure(
                text="🔕  Сигнал: выкл", fg_color="#58474B", hover_color="#6B555A",
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
        self.main_button.configure(text="Остановить  ·  F9", fg_color="#D34D5C", hover_color="#B93D4A")
        self.status.set("F9 принят. Открываю GTA и проверяю реальный таймер перед первой ставкой.")
        self.after(0, self.tick)

    def stop(self, message: str) -> None:
        self.running, self.next_run = False, None
        self.waiting_for_betting_window = False
        self.missing_timer_since = None
        self.restore_waiting_window()
        self.main_button.configure(text="Запустить  ·  F9", fg_color="#C9715D", hover_color="#D8826B")
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


# ---------------------------------------------------------------------------
# Секретный модуль Бонго. Он повторяет аккуратный жизненный цикл RLT Control:
# открывает игру только на время действия, возвращает предыдущее окно и оставляет
# между циклами обычный таймер, чтобы не мешать работе за компьютером.


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
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=46, pady=(36, 30))
        ctk.CTkButton(
            top, text="←  К котикам", command=self.back, width=126, height=38,
            corner_radius=12, fg_color="#26314E", hover_color="#344263",
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        ).pack(side="left")
        title = ctk.CTkFrame(top, fg_color="transparent")
        title.pack(side="right")
        ctk.CTkLabel(title, text="BONGO BEAT", font=ctk.CTkFont("Segoe UI", 24, "bold"), text_color=TEXT).pack(anchor="e")
        ctk.CTkLabel(title, text="секретный ритм двух нажатий ↑", font=ctk.CTkFont("Segoe UI", 11), text_color=MUTED).pack(anchor="e")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=46, pady=(0, 26))

        connection = ctk.CTkFrame(body, corner_radius=22, fg_color=SURFACE)
        connection.pack(fill="x", pady=(0, 16))
        connection.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(connection, text="ПОДКЛЮЧЕНИЕ К ИГРЕ", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color="#FFB58A").grid(row=0, column=0, columnspan=2, padx=26, pady=(20, 8), sticky="w")
        self.indicator = ctk.CTkLabel(connection, text="●", font=ctk.CTkFont(size=18), text_color="#F05A67")
        self.indicator.grid(row=1, column=0, padx=(26, 10), pady=(0, 21))
        ctk.CTkEntry(connection, textvariable=self.process, height=42, border_width=0, corner_radius=13, fg_color="#4A353B", font=ctk.CTkFont("Segoe UI", 14)).grid(row=1, column=1, padx=(0, 12), pady=(0, 21), sticky="ew")
        ctk.CTkButton(connection, text="Проверить", command=self.refresh_connection, width=122, height=42, corner_radius=13, fg_color="#E1864B", hover_color="#EF9A61").grid(row=1, column=2, padx=(0, 26), pady=(0, 21))
        ctk.CTkLabel(connection, textvariable=self.connection, font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=MUTED).grid(row=2, column=0, columnspan=3, padx=26, pady=(0, 17), sticky="w")

        guide = ctk.CTkFrame(body, corner_radius=22, fg_color=SURFACE)
        guide.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(guide, text="КАК РАБОТАЕТ РИТМ", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color="#FFB58A").pack(anchor="w", padx=26, pady=(20, 10))
        steps = (
            ("1", "Запусти игру", "Окно процесса можно выбрать в поле выше."),
            ("2", "Нажми F9", "Модуль выведет игру на передний план и нажмёт ↑."),
            ("3", "Второй удар", "Через 3 секунды отправит ещё одно ↑, вернёт прежнее окно и запустит таймер."),
        )
        for number, heading, description in steps:
            row = ctk.CTkFrame(guide, fg_color=SURFACE_ALT, corner_radius=14)
            row.pack(fill="x", padx=22, pady=(0, 8))
            ctk.CTkLabel(row, text=number, width=30, height=30, corner_radius=15, fg_color="#E1864B", text_color=TEXT, font=ctk.CTkFont("Segoe UI", 13, "bold")).pack(side="left", padx=(12, 12), pady=10)
            copy = ctk.CTkFrame(row, fg_color="transparent")
            copy.pack(side="left", fill="x", expand=True, pady=8)
            ctk.CTkLabel(copy, text=heading, font=ctk.CTkFont("Segoe UI", 12, "bold"), text_color=TEXT).pack(anchor="w")
            ctk.CTkLabel(copy, text=description, font=ctk.CTkFont("Segoe UI", 10), text_color=MUTED).pack(anchor="w")
        ctk.CTkLabel(guide, text="После каждой пары модуль ждёт случайные 7–9 минут и повторяет ритм.", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=GOLD).pack(anchor="w", padx=26, pady=(5, 16))

        self.main_button = ctk.CTkButton(body, text="Запустить ритм  ·  F9", command=self.toggle, height=56, corner_radius=16, font=ctk.CTkFont("Segoe UI", 15, "bold"), fg_color="#E1864B", hover_color="#EF9A61")
        self.main_button.pack(fill="x", pady=(0, 14))
        monitor = ctk.CTkFrame(body, corner_radius=18, fg_color=SURFACE_ALT, border_width=1, border_color="#5B4036")
        monitor.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(monitor, text="СОСТОЯНИЕ", font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color="#FFB58A").pack(pady=(14, 4))
        ctk.CTkLabel(monitor, textvariable=self.timer, font=ctk.CTkFont("Segoe UI", 20, "bold"), text_color=TEXT).pack(pady=(0, 5))
        ctk.CTkLabel(monitor, textvariable=self.status, font=ctk.CTkFont("Segoe UI", 11), text_color=MUTED, wraplength=650, justify="center").pack(padx=22, pady=(0, 14))
        ctk.CTkLabel(body, text="F9 — запуск / остановка     ·     F11 — экстренная остановка", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color="#FFBA86").pack(pady=(0, 16))

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
        self.main_button.configure(text="Остановить ритм  ·  F9", fg_color="#D34D5C", hover_color="#B93D4A")
        self.status.set("F9 принят. Открываю игру для первого ↑.")
        self.after(0, self.tick)

    def stop(self, message: str) -> None:
        self.running, self.phase, self.next_action = False, "idle", None
        self.restore_previous_window()
        self.main_button.configure(text="Запустить ритм  ·  F9", fg_color="#E1864B", hover_color="#EF9A61")
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


# ---------------------------------------------------------------------------
# Секретный модуль Баффа. Следит за мини-игрой строительства в GTA: зелёная
# зона — нужный сектор, розовая метка — движущийся индикатор.


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
        self.previous_window: int | None = None
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

    def build_ui(self) -> None:
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
        self.main_button.configure(text="Остановить ловлю  ·  F9", fg_color="#D34D5C", hover_color="#B93D4A")
        self.status.set("F9 принят. Игра открыта, ищу шкалу в правом нижнем углу.")

    def stop(self, message: str) -> None:
        self.running = False
        self.reset_round()
        self.restore_previous_window()
        self.main_button.configure(text="Запустить ловлю  ·  F9", fg_color="#4BB889", hover_color="#62C99B")
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


# ---------------------------------------------------------------------------
# Основное приложение-кликер

class KisikiApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        make_dpi_aware()
        ctk.set_appearance_mode("dark")
        self.title("Кисикисимяумяу")
        self.geometry("920x860")
        self.minsize(880, 820)
        self.configure(fg_color=APP_BG)
        self.selected = 0
        self.current_view = "home"
        self.game = self.load_game()
        self.sound_enabled = bool(self.game["sound_enabled"])
        self.clicks = self.cat_data()["meows"]
        self.hold_job: str | None = None
        self.hold_tick: str | None = None
        self.hold_started = 0.0
        self.bongo_secret_taps = 0
        self.bongo_secret_deadline = 0.0
        self.buff_secret_stage = 0
        self.buff_secret_taps = 0
        self.buff_secret_last_tap = 0.0
        self.combo_count = 0
        self.combo_deadline = 0.0
        self.combo_reset_job: str | None = None
        self.gift_ready_at = time.monotonic()
        self.images: dict[tuple[int, int, int], tk.PhotoImage] = {}
        self.click_effects: list[ctk.CTkLabel] = []
        self.sound_aliases: set[str] = set()
        self.sound_sequence = 0
        self.content = ctk.CTkFrame(self, fg_color=APP_BG, corner_radius=0)
        self.content.pack(fill="both", expand=True)
        self.roulette_module: RouletteModule | None = None
        self.bongo_module: BongoModule | None = None
        self.buff_module: BuffTimingModule | None = None
        # CTk иногда возвращает свою стандартную иконку позднее при старте.
        # Поэтому устанавливаем cat-иконку после полной инициализации окна.
        self.after(120, self.apply_window_icon)
        self.after(1000, self.passive_tick)
        self.show_home()

    @staticmethod
    def fresh_game() -> dict:
        return {
            "sound_enabled": True,
            "roulette_sound_enabled": True,
            "coins": 0,
            "cats": [
                {"meows": 0, "paw": 0, "treat": 0, "laser": 0, "taps": 0, "best_combo": 0}
                for _ in CATS
            ],
        }

    def load_game(self) -> dict:
        game = self.fresh_game()
        try:
            saved = json.loads(progress_path().read_text(encoding="utf-8"))
            if isinstance(saved, dict) and isinstance(saved.get("cats"), list) and len(saved["cats"]) == len(CATS):
                game["sound_enabled"] = bool(saved.get("sound_enabled", True))
                game["roulette_sound_enabled"] = bool(saved.get("roulette_sound_enabled", True))
                game["coins"] = max(0, int(saved.get("coins", 0)))
                for index, data in enumerate(saved["cats"]):
                    if isinstance(data, dict):
                        for key in ("meows", "paw", "treat", "laser", "taps", "best_combo"):
                            game["cats"][index][key] = max(0, int(data.get(key, 0)))
        except (OSError, ValueError, TypeError):
            pass
        return game

    def save_game(self) -> None:
        self.game["sound_enabled"] = self.sound_enabled
        try:
            progress_path().write_text(json.dumps(self.game, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def set_roulette_sound(self, enabled: bool) -> None:
        self.game["roulette_sound_enabled"] = bool(enabled)
        self.save_game()

    def cat_data(self) -> dict:
        return self.game["cats"][self.selected]

    def click_power(self) -> int:
        data = self.cat_data()
        return 1 + data["paw"] + data["laser"] * 3

    def passive_income(self) -> int:
        return self.cat_data()["treat"]

    def upgrade_price(self, key: str) -> int:
        return int(UPGRADES[key][2] * (1.65 ** self.cat_data()[key]))

    def apply_window_icon(self) -> None:
        try:
            self.iconbitmap(default=str(resource_path("orange_cat.ico")))
        except tk.TclError:
            # Для запуска исходника остаётся рабочий вариант через PNG.
            self.iconphoto(True, self.photo(2, 64, 64))

    def photo(self, index: int, width: int, height: int) -> tk.PhotoImage:
        key = (index, width, height)
        if key not in self.images:
            self.images[key] = rounded_photo(resource_path("kiski kartinki", CATS[index][2]), width, height)
        return self.images[key]

    def clear(self) -> None:
        self.cancel_hold()
        self.click_effects.clear()
        for child in self.content.winfo_children():
            if child is self.roulette_module or child is self.bongo_module or child is self.buff_module:
                child.pack_forget()
            else:
                child.destroy()

    def header(self, subtitle: str, back=None) -> None:
        bar = ctk.CTkFrame(self.content, fg_color="transparent")
        bar.pack(fill="x", padx=46, pady=(34, 24))
        if back:
            ctk.CTkButton(bar, text="←  Все котики", command=back, width=128, height=38, corner_radius=12, fg_color="#26314E", hover_color="#344263", font=ctk.CTkFont("Segoe UI", 12, "bold")).pack(side="left")
        else:
            self.sound_button = ctk.CTkButton(
                bar, command=self.toggle_sound, width=142, height=38, corner_radius=12,
                font=ctk.CTkFont("Segoe UI", 12, "bold"),
            )
            self.sound_button.pack(side="right")
            self.refresh_sound_button()
        brand = ctk.CTkFrame(bar, fg_color="transparent")
        brand.pack(side="right" if back else "left")
        ctk.CTkLabel(brand, text="КИСИКИСИМЯУМЯУ", font=ctk.CTkFont("Segoe UI", 23, "bold"), text_color=TEXT).pack(anchor="e" if back else "w")
        ctk.CTkLabel(brand, text=subtitle, font=ctk.CTkFont("Segoe UI", 11), text_color=MUTED).pack(anchor="e" if back else "w")

    def refresh_sound_button(self) -> None:
        if not hasattr(self, "sound_button") or not self.sound_button.winfo_exists():
            return
        if self.sound_enabled:
            self.sound_button.configure(text="🔊  Звук: вкл", fg_color="#C9715D", hover_color="#D8826B")
        else:
            self.sound_button.configure(text="🔇  Звук: выкл", fg_color="#5B4548", hover_color="#71565A")

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
        path = resource_path("sounds", SOUND_FILES[self.selected])
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

    def show_home(self) -> None:
        self.clear()
        self.current_view = "home"
        self.header("маленький клуб больших мяу")
        intro = ctk.CTkFrame(self.content, fg_color=SURFACE, corner_radius=24, border_width=1, border_color="#5B4141")
        intro.pack(fill="x", padx=46, pady=(0, 24))
        ctk.CTkLabel(intro, text="Кого сегодня будем гладить?", font=ctk.CTkFont("Segoe UI", 25, "bold"), text_color=TEXT).pack(anchor="w", padx=24, pady=(20, 2))
        self.home_balance = ctk.StringVar(value="")
        ctk.CTkLabel(intro, textvariable=self.home_balance, font=ctk.CTkFont("Segoe UI", 13, "bold"), text_color=GOLD).pack(anchor="w", padx=24, pady=(0, 20))
        self.refresh_home_balance()
        grid = ctk.CTkFrame(self.content, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=46, pady=(0, 10))
        for column in range(3):
            grid.grid_columnconfigure(column, weight=1)
        for index, (name, description, _, color) in enumerate(CATS):
            card = ctk.CTkFrame(grid, fg_color=SURFACE, corner_radius=21, border_width=1, border_color="#574040", cursor="hand2")
            card.grid(row=index // 3, column=index % 3, padx=10, pady=10, sticky="nsew")
            picture = ctk.CTkLabel(card, text="", image=self.photo(index, 150, 120), cursor="hand2")
            picture.pack(pady=(14, 4))
            ctk.CTkLabel(card, text=name, font=ctk.CTkFont("Segoe UI", 15, "bold"), text_color=TEXT, cursor="hand2").pack()
            ctk.CTkLabel(card, text=description, font=ctk.CTkFont("Segoe UI", 10), text_color=MUTED, cursor="hand2").pack(pady=(2, 14))
            cat_meows = self.game["cats"][index]["meows"]
            ctk.CTkLabel(card, text=f"{cat_meows} мяу", font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=CATS[index][3], cursor="hand2").pack(pady=(0, 13))
            widgets = [card, picture, *card.winfo_children()[1:]]
            for widget in widgets:
                widget.bind("<Button-1>", lambda _event, cat=index: self.open_cat(cat))
                widget.bind("<Enter>", lambda _event, target=card: target.configure(fg_color=SURFACE_HOVER, border_color="#B57570"))
                widget.bind("<Leave>", lambda _event, target=card: target.configure(fg_color=SURFACE, border_color="#574040"))
        ctk.CTkLabel(self.content, text="мяу — это тоже язык любви", font=ctk.CTkFont("Segoe UI", 11), text_color="#B18F83").pack(pady=(10, 28))

    def open_cat(self, index: int) -> None:
        self.selected = index
        self.clicks = self.cat_data()["meows"]
        self.reset_combo()
        self.show_clicker()

    def show_clicker(self) -> None:
        self.clear()
        self.current_view = "clicker"
        name, description, _, color = CATS[self.selected]
        self.header(f"личный кликер · {name}", self.show_home)
        body = ctk.CTkFrame(self.content, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=54, pady=(0, 22))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        left = ctk.CTkFrame(body, fg_color=SURFACE, corner_radius=27, border_width=1, border_color="#5B4141")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        right = ctk.CTkFrame(body, fg_color=SURFACE_ALT, corner_radius=27, width=255, border_width=1, border_color="#523A3B")
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)
        ctk.CTkLabel(left, text=name, font=ctk.CTkFont("Segoe UI", 30, "bold"), text_color=TEXT).pack(pady=(24, 0))
        ctk.CTkLabel(left, text=description, font=ctk.CTkFont("Segoe UI", 13), text_color=MUTED).pack(pady=(3, 9))
        self.power_text = ctk.CTkLabel(left, text="", font=ctk.CTkFont("Segoe UI", 12, "bold"), text_color=GOLD)
        self.power_text.pack(pady=(0, 8))
        self.click_area = left
        self.hero = ctk.CTkLabel(left, text="", image=self.photo(self.selected, 350, 285), cursor="hand2")
        self.hero.pack(pady=(0, 2))
        self.hero.bind("<ButtonPress-1>", self.hero_press)
        self.hero.bind("<ButtonRelease-1>", self.hero_release)
        self.hero.bind("<Leave>", self.cancel_hold)
        combo = ctk.CTkFrame(left, fg_color=SURFACE_ALT, corner_radius=14)
        combo.pack(fill="x", padx=28, pady=(2, 8))
        self.combo_text = ctk.StringVar(value="КОМБО · начинай гладить")
        ctk.CTkLabel(combo, textvariable=self.combo_text, font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=GOLD).pack(anchor="w", padx=14, pady=(10, 5))
        self.combo_bar = ctk.CTkProgressBar(combo, height=6, corner_radius=5, progress_color="#F0A56D", fg_color="#594247")
        self.combo_bar.pack(fill="x", padx=14, pady=(0, 11))
        self.combo_bar.set(0)
        self.hint = ctk.CTkLabel(left, text="нажимай прямо на котика", font=ctk.CTkFont("Segoe UI", 13, "bold"), text_color=color)
        self.hint.pack(pady=(1, 10))
        stats = ctk.CTkFrame(left, fg_color="transparent")
        stats.pack(fill="x", padx=24, pady=(0, 20))
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
        ctk.CTkLabel(right, text="МЯУКОИНЫ", font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color="#C5AFA4").pack(anchor="w", padx=24, pady=(25, 5))
        self.coins_text = ctk.StringVar()
        ctk.CTkLabel(right, textvariable=self.coins_text, font=ctk.CTkFont("Segoe UI", 29, "bold"), text_color=GOLD).pack(anchor="w", padx=24)
        self.level_text = ctk.StringVar()
        ctk.CTkLabel(right, textvariable=self.level_text, font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=MUTED).pack(anchor="w", padx=24, pady=(0, 5))
        self.level_bar = ctk.CTkProgressBar(right, height=7, corner_radius=6, progress_color="#D88470", fg_color="#5A4040")
        self.level_bar.pack(fill="x", padx=24)
        ctk.CTkFrame(right, height=1, fg_color="#5A4040").pack(fill="x", padx=24, pady=18)
        ctk.CTkLabel(right, text="СЧЁТ ЭТОГО КОТИКА", font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color="#C5AFA4").pack(anchor="w", padx=24)
        self.score_text = ctk.StringVar()
        ctk.CTkLabel(right, textvariable=self.score_text, font=ctk.CTkFont("Segoe UI", 23, "bold"), text_color=TEXT, justify="left").pack(anchor="w", padx=24, pady=(3, 0))
        ctk.CTkFrame(right, height=1, fg_color="#5A4040").pack(fill="x", padx=24, pady=16)
        ctk.CTkLabel(right, text="УЛУЧШЕНИЯ", font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color="#C5AFA4").pack(anchor="w", padx=24, pady=(0, 7))
        self.upgrade_buttons: dict[str, ctk.CTkButton] = {}
        for key in ("paw", "treat", "laser"):
            title, description, _, shade = UPGRADES[key]
            button = ctk.CTkButton(right, command=lambda upgrade=key: self.buy_upgrade(upgrade), height=48, corner_radius=12, fg_color=shade, hover_color="#B56D5F", font=ctk.CTkFont("Segoe UI", 11, "bold"), anchor="w")
            button.pack(fill="x", padx=24, pady=(0, 7))
            self.upgrade_buttons[key] = button
        ctk.CTkFrame(right, height=1, fg_color="#5A4040").pack(fill="x", padx=24, pady=(10, 12))
        ctk.CTkLabel(right, text="КЛУБОК УДАЧИ", font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color="#C5AFA4").pack(anchor="w", padx=24, pady=(0, 7))
        self.gift_button = ctk.CTkButton(
            right, command=self.claim_lucky_yarn, height=46, corner_radius=12,
            fg_color="#8C6AC1", hover_color="#A47BD9", font=ctk.CTkFont("Segoe UI", 11, "bold"),
        )
        self.gift_button.pack(fill="x", padx=24)
        self.refresh_score()
        self.refresh_shop()
        self.refresh_gift_button()
        ctk.CTkLabel(self.content, text="мяу-мяу ♡", font=ctk.CTkFont("Segoe UI", 11), text_color="#B18F83").pack(pady=(0, 20))

    def refresh_score(self) -> None:
        data = self.cat_data()
        self.clicks = data["meows"]
        if hasattr(self, "score_text") and self.score_text is not None:
            self.score_text.set(f"{self.clicks} мяу")
            self.coins_text.set(f"{self.game['coins']}  🐾")
            level = 1 + self.clicks // 100
            self.level_text.set(f"Уровень {level} · до следующего: {100 - self.clicks % 100}")
            self.level_bar.set((self.clicks % 100) / 100)
            self.power_text.configure(text=f"Сила клика: +{self.click_power()}  ·  Авто: +{self.passive_income()}/сек")
            self.tap_text.set(str(data.get("taps", 0)))
            self.best_combo_text.set(f"×{data.get('best_combo', 0)}")
            self.update_combo_display()

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
                fg_color="#594A60", hover_color="#594A60",
            )

    def claim_lucky_yarn(self) -> None:
        if time.monotonic() < self.gift_ready_at:
            return
        reward = random.randint(max(5, self.click_power() * 5), max(12, self.click_power() * 12))
        self.game["coins"] += reward
        self.cat_data()["meows"] += reward
        self.gift_ready_at = time.monotonic() + 45
        self.save_game()
        self.hint.configure(text=f"клубок удачи: +{reward} мяу!", text_color=PURPLE)
        self.refresh_score()
        self.refresh_shop()
        self.refresh_gift_button()

    def refresh_home_balance(self) -> None:
        if hasattr(self, "home_balance"):
            self.home_balance.set(f"Общий запас: {self.game['coins']} мяукоинов  ·  улучши любимого котика")

    def refresh_shop(self) -> None:
        if not hasattr(self, "upgrade_buttons"):
            return
        for key, button in self.upgrade_buttons.items():
            title, description, _, _ = UPGRADES[key]
            price = self.upgrade_price(key)
            owned = self.cat_data()[key]
            button.configure(text=f"{title} · {description}\n{price} мяу  ·  ур. {owned}")

    def buy_upgrade(self, key: str) -> None:
        price = self.upgrade_price(key)
        if self.game["coins"] < price:
            self.hint.configure(text=f"Нужно ещё {price - self.game['coins']} мяу", text_color=PINK)
            return
        self.game["coins"] -= price
        self.cat_data()[key] += 1
        self.save_game()
        self.hint.configure(text="улучшение куплено!", text_color=MINT)
        self.refresh_score()
        self.refresh_shop()

    def passive_tick(self) -> None:
        income = self.passive_income()
        if income:
            self.game["coins"] += income
            self.cat_data()["meows"] += income
            self.save_game()
            if self.current_view == "clicker":
                self.refresh_score()
                self.refresh_shop()
                self.refresh_gift_button()
        self.after(1000, self.passive_tick)

    def hero_press(self, _event) -> None:
        self.cancel_hold()
        # Пасхалка остаётся невидимой: удержание не показывает индикатор.
        if self.selected == 5:
            self.hold_job = self.after(HOLD_MS, self.open_roulette)

    def hero_release(self, _event) -> None:
        # Обычный короткий клик засчитывается для всех шести котиков.
        self.cancel_hold()
        amount = self.register_click()
        self.game["coins"] += amount
        self.cat_data()["meows"] += amount
        self.save_game()
        self.refresh_score()
        self.refresh_shop()
        self.hint.configure(text=f"мяу +{amount}!", text_color=CATS[self.selected][3])
        self.spawn_click_effect(amount)
        self.play_meme_sound()
        self.track_bongo_secret()
        self.track_buff_secret()

    def track_bongo_secret(self) -> None:
        """Открыть пасхалку Бонго после десяти быстрых обычных кликов."""
        if self.selected != 1:
            self.bongo_secret_taps = 0
            self.bongo_secret_deadline = 0.0
            return
        now = time.monotonic()
        if now > self.bongo_secret_deadline:
            self.bongo_secret_taps = 0
        self.bongo_secret_taps += 1
        self.bongo_secret_deadline = now + BONGO_SECRET_TAP_WINDOW_SECONDS
        if self.bongo_secret_taps >= BONGO_SECRET_TAPS:
            self.bongo_secret_taps = 0
            self.bongo_secret_deadline = 0.0
            self.after(0, self.open_bongo)

    def reset_buff_secret(self) -> None:
        self.buff_secret_stage = 0
        self.buff_secret_taps = 0
        self.buff_secret_last_tap = 0.0

    def track_buff_secret(self) -> None:
        """Три быстрых клика, пауза, ещё три — код силы Баффа."""
        if self.selected != 3:
            self.reset_buff_secret()
            return
        now = time.monotonic()
        if self.buff_secret_stage == 0:
            if self.buff_secret_taps and now - self.buff_secret_last_tap > BUFF_SECRET_TAP_SPACING_SECONDS:
                self.buff_secret_taps = 0
            self.buff_secret_taps += 1
            self.buff_secret_last_tap = now
            if self.buff_secret_taps >= BUFF_SECRET_TAPS_PER_BEAT:
                self.buff_secret_stage = 1
            return
        if self.buff_secret_stage == 1:
            pause = now - self.buff_secret_last_tap
            if BUFF_SECRET_PAUSE_MIN_SECONDS <= pause <= BUFF_SECRET_PAUSE_MAX_SECONDS:
                self.buff_secret_stage = 2
                self.buff_secret_taps = 1
                self.buff_secret_last_tap = now
                return
            self.reset_buff_secret()
            self.buff_secret_taps = 1
            self.buff_secret_last_tap = now
            return
        if now - self.buff_secret_last_tap > BUFF_SECRET_TAP_SPACING_SECONDS:
            self.reset_buff_secret()
            self.buff_secret_taps = 1
            self.buff_secret_last_tap = now
            return
        self.buff_secret_taps += 1
        self.buff_secret_last_tap = now
        if self.buff_secret_taps >= BUFF_SECRET_TAPS_PER_BEAT:
            self.reset_buff_secret()
            self.after(0, self.open_buff)

    def spawn_click_effect(self, amount: int) -> None:
        """Лёгкий всплывающий эффект вместо резкой анимации."""
        symbols = (f"✦  +{amount}", f"♡  +{amount}", f"✧  +{amount}", f"мяу!  +{amount}")
        effect = ctk.CTkLabel(
            self.click_area,
            text=random.choice(symbols),
            font=ctk.CTkFont("Segoe UI", 16, "bold"),
            text_color=CATS[self.selected][3],
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
        if self.hold_job is not None:
            self.after_cancel(self.hold_job)
        if self.hold_tick is not None:
            self.after_cancel(self.hold_tick)
        self.hold_job = self.hold_tick = None

    def open_roulette(self) -> None:
        self.cancel_hold()
        if self.bongo_module is not None and self.bongo_module.winfo_exists():
            self.bongo_module.deactivate("Остановлено: открыт модуль казино.")
        if self.buff_module is not None and self.buff_module.winfo_exists():
            self.buff_module.deactivate("Остановлено: открыт модуль казино.")
        self.clear()
        self.current_view = "roulette"
        if self.roulette_module is None or not self.roulette_module.winfo_exists():
            self.roulette_module = RouletteModule(
                self.content,
                self.show_clicker,
                alert_sound_enabled=bool(self.game.get("roulette_sound_enabled", True)),
                on_alert_sound_change=self.set_roulette_sound,
            )
        self.roulette_module.pack(fill="both", expand=True)
        self.roulette_module.activate()

    def open_bongo(self) -> None:
        self.cancel_hold()
        if self.roulette_module is not None and self.roulette_module.winfo_exists():
            self.roulette_module.deactivate("Остановлено: открыт модуль телефона.")
        if self.buff_module is not None and self.buff_module.winfo_exists():
            self.buff_module.deactivate("Остановлено: открыт модуль телефона.")
        self.clear()
        self.current_view = "bongo"
        if self.bongo_module is None or not self.bongo_module.winfo_exists():
            self.bongo_module = BongoModule(self.content, self.show_clicker)
        self.bongo_module.pack(fill="both", expand=True)
        self.bongo_module.activate()

    def open_buff(self) -> None:
        self.cancel_hold()
        if self.roulette_module is not None and self.roulette_module.winfo_exists():
            self.roulette_module.deactivate("Остановлено: открыт модуль Баффа.")
        if self.bongo_module is not None and self.bongo_module.winfo_exists():
            self.bongo_module.deactivate("Остановлено: открыт модуль Баффа.")
        self.clear()
        self.current_view = "buff"
        if self.buff_module is None or not self.buff_module.winfo_exists():
            self.buff_module = BuffTimingModule(self.content, self.show_clicker)
        self.buff_module.pack(fill="both", expand=True)
        self.buff_module.activate()

if __name__ == "__main__":
    KisikiApp().mainloop()
