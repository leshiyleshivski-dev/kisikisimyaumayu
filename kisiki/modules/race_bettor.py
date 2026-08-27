"""Automatic Inside Track betting until a requested number of wins."""

from __future__ import annotations

import random
import time

import cv2
import customtkinter as ctk
import mss
import numpy as np

from ..core import (
    APP_BG, GOLD, MINT, MUTED, SURFACE_ALT, TEXT, VK_F, VK_F9, VK_F11,
    activate_window, client_bounds, cursor_position, find_game_window,
    glide_cursor_to, resource_path, send_key_tap, send_left_click, user32,
    window_title,
)
from .ui import connection_panel, hotkey_bar, module_header, panel, step_list


# The narrow coloured line on the right edge of Majestic's bottom-centre
# notification. It is stable between the 2560x1440 video and the supplied
# 2048x1152 screenshot, unlike the text whose size and anti-aliasing vary.
RACE_TOAST_ACCENT_RATIO = (0.595, 0.895, 0.018, 0.095)
ODDS_X_RATIO = (0.101, 0.157)
ODDS_FIRST_Y_RATIO = 0.388
ODDS_ROW_STEP_RATIO = 0.0965
ODDS_HALF_HEIGHT_RATIO = 0.016
HORSE_CLICK_X_RATIO = 0.160
HORSE_FIRST_Y_RATIO = 0.372
HORSE_ROW_STEP_RATIO = 0.0965
BET_BUTTON_RATIO = (0.369, 0.405)
BET_DIALOG_RATIO = (0.305, 0.075, 0.247, 0.275)
_ODDS_TEMPLATE_SIGNATURE: np.ndarray | None = None


def _vertical_accent_score(mask: np.ndarray) -> int:
    """Return the area of the strongest notification-like vertical segment."""
    height, width = mask.shape
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    minimum_height = max(10, round(height * 0.24))
    minimum_area = max(20, round(width * height * 0.012))
    return max(
        (
            int(area)
            for _x, _y, component_width, component_height, area in stats[1:]
            if component_width >= 2
            and component_height >= minimum_height
            and area >= minimum_area
        ),
        default=0,
    )


def classify_race_toast(image: np.ndarray | None) -> str | None:
    """Classify Majestic's result accent as ``green`` or ``red``.

    Green is used both for an accepted bet and a win. The module distinguishes
    those two messages by its state: immediately after the third F it confirms
    the bet, while after the race it increments the win counter.
    """
    if image is None or image.size == 0 or image.ndim != 3:
        return None
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv, np.array((35, 75, 90), dtype=np.uint8),
        np.array((95, 255, 255), dtype=np.uint8),
    )
    red_low = cv2.inRange(
        hsv, np.array((0, 75, 90), dtype=np.uint8),
        np.array((15, 255, 255), dtype=np.uint8),
    )
    red_high = cv2.inRange(
        hsv, np.array((155, 75, 90), dtype=np.uint8),
        np.array((179, 255, 255), dtype=np.uint8),
    )
    green_score = _vertical_accent_score(green)
    red_score = _vertical_accent_score(cv2.bitwise_or(red_low, red_high))
    if green_score > red_score and green_score:
        return "green"
    if red_score > green_score and red_score:
        return "red"
    return None


def parse_win_target(value: str, maximum: int = 999) -> int | None:
    """Return a validated positive target or ``None`` for unsafe input."""
    try:
        target = int(value.strip())
    except (AttributeError, ValueError):
        return None
    return target if 1 <= target <= maximum else None


def odds_signature(image: np.ndarray | None) -> tuple[np.ndarray, float] | None:
    """Normalize a rendered odds label so it can be compared across resolutions."""
    if image is None or image.size == 0 or image.ndim != 3:
        return None
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    background = float(np.median(gray))
    threshold = max(52, round(background + 28))
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary)
    clean = np.zeros_like(binary)
    minimum_area = max(2, round(binary.size * 0.0005))
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area:
            clean[labels == index] = 255
    points = cv2.findNonZero(clean)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    if width < 6 or height < 6:
        return None
    glyphs = clean[y:y + height, x:x + width]
    aspect = width / height
    normalized_height = 24
    normalized_width = max(1, round(width * normalized_height / height))
    glyphs = cv2.resize(
        glyphs, (normalized_width, normalized_height), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.zeros((32, 80), dtype=np.uint8)
    canvas[4:28, :min(canvas.shape[1], normalized_width)] = glyphs[:, :canvas.shape[1]]
    return canvas, aspect


def two_to_one_template() -> np.ndarray | None:
    """Load the small 2/1 reference captured from the real Majestic interface."""
    global _ODDS_TEMPLATE_SIGNATURE
    if _ODDS_TEMPLATE_SIGNATURE is None:
        template = cv2.imread(str(resource_path("assets", "vision", "race_odds_2_1.png")))
        normalized = odds_signature(template)
        if normalized is not None:
            _ODDS_TEMPLATE_SIGNATURE = normalized[0]
    return _ODDS_TEMPLATE_SIGNATURE


def signature_similarity(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union else 0.0


def find_two_to_one_horse(
    frame: np.ndarray | None, template: np.ndarray | None = None,
) -> tuple[int, int, int, float] | None:
    """Find the row whose odds read 2/1 and return its safe click point."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None
    template = template if template is not None else two_to_one_template()
    if template is None:
        return None
    height, width = frame.shape[:2]
    x1 = round(width * ODDS_X_RATIO[0])
    x2 = round(width * ODDS_X_RATIO[1])
    half_height = max(8, round(height * ODDS_HALF_HEIGHT_RATIO))
    candidates: list[tuple[float, int]] = []
    for row in range(6):
        centre_y = round(height * (ODDS_FIRST_Y_RATIO + row * ODDS_ROW_STEP_RATIO))
        crop = frame[max(0, centre_y - half_height):min(height, centre_y + half_height), x1:x2]
        normalized = odds_signature(crop)
        if normalized is None:
            candidates.append((0.0, row))
            continue
        signature, aspect = normalized
        score = signature_similarity(signature, template) if 1.18 <= aspect <= 1.62 else 0.0
        candidates.append((score, row))
    candidates.sort(reverse=True)
    best_score, best_row = candidates[0]
    second_score = candidates[1][0]
    if best_score < 0.62 or best_score - second_score < 0.08:
        return None
    click_x = round(width * HORSE_CLICK_X_RATIO)
    click_y = round(height * (HORSE_FIRST_Y_RATIO + best_row * HORSE_ROW_STEP_RATIO))
    return best_row, click_x, click_y, best_score


def bet_dialog_visible(frame: np.ndarray | None) -> bool:
    """Confirm that the dark bet card is open before clicking its button."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return False
    height, width = frame.shape[:2]
    x, y, region_width, region_height = BET_DIALOG_RATIO
    crop = frame[
        round(height * y):round(height * (y + region_height)),
        round(width * x):round(width * (x + region_width)),
    ]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
    dark_fraction = float(np.mean(gray < 70))
    bright_fraction = float(np.mean(gray > 100))
    return dark_fraction >= 0.82 and bright_fraction >= 0.03 and float(gray.std()) >= 20.0


def race_result_board_visible(frame: np.ndarray | None) -> bool:
    """Require the large purple finish board before accepting any toast colour."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return False
    height, width = frame.shape[:2]
    roi = frame[
        round(height * 0.02):round(height * 0.35),
        round(width * 0.18):round(width * 0.72),
    ]
    hsv = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2HSV)
    purple = cv2.inRange(
        hsv, np.array((125, 70, 45), dtype=np.uint8),
        np.array((175, 255, 255), dtype=np.uint8),
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(purple)
    minimum_width = max(80, round(roi.shape[1] * 0.25))
    minimum_height = max(45, round(roi.shape[0] * 0.20))
    minimum_area = max(2500, round(roi.shape[0] * roi.shape[1] * 0.035))
    return any(
        component_width >= minimum_width
        and component_height >= minimum_height
        and area >= minimum_area
        for _x, _y, component_width, component_height, area in stats[1:count]
    )


def classify_race_result(frame: np.ndarray | None) -> str | None:
    """Return a result only in a verified race UI state.

    Majestic may leave the bet dialog over the finish board. Either the large
    purple board or that persistent dialog is therefore a valid race anchor.
    """
    if frame is None or not (race_result_board_visible(frame) or bet_dialog_visible(frame)):
        return None
    height, width = frame.shape[:2]
    x, y, region_width, region_height = RACE_TOAST_ACCENT_RATIO
    toast = frame[
        round(height * y):round(height * (y + region_height)),
        round(width * x):round(width * (x + region_width)),
    ]
    kind = classify_race_toast(toast)
    if kind == "green":
        return "win"
    if kind == "red":
        return "loss"
    return None


class RaceBettorModule(ctk.CTkFrame):
    """Mouse-bet on the 2/1 horse and stop at the requested wins."""

    BET_WINDOW_SECONDS = 28.0
    BET_REMAINING_SECONDS = (15.0, 27.0)
    LIST_DETECT_TIMEOUT_SECONDS = 4.0
    DIALOG_SETTLE_SECONDS = 0.75
    BET_CONFIRM_TIMEOUT_SECONDS = 5.0
    BET_RETRY_SECONDS = 3.0
    RESULT_SCAN_AFTER_SECONDS = 30.0
    RESULT_TIMEOUT_SECONDS = 85.0
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
        self.saved_cursor: tuple[int, int] | None = None
        self.bet_confirmed_at: float | None = None
        self.toast_kind: str | None = None
        self.horse_row: int | None = None
        self.horse_score = 0.0
        self.scheduled_remaining = 0.0
        self.target_wins = 4
        self.wins = 0
        self.losses = 0
        self.bets = 0
        self.attempts = 0
        self.keys = {key: False for key in (VK_F9, VK_F11)}

        self.process = ctk.StringVar(value="GTA5.exe")
        self.connection = ctk.StringVar(value="Ищу GTA5.exe…")
        self.target_input = ctk.StringVar(value="4")
        self.progress = ctk.StringVar(value="0 / 4 побед")
        self.stats = ctk.StringVar(value="Ставок 0  ·  проигрышей 0")
        self.status = ctk.StringVar(value="Сядь за терминал скачек. Лошадь 2/1 и кнопки модуль найдёт сам.")
        self.timer = ctk.StringVar(value="Автоставки не запущены")

        self.build_ui()
        self.after(40, self.poll_hotkeys)
        self.after(100, self.tick)
        self.after(self.SCAN_INTERVAL_MS, self.scan_tick)
        self.after(0, self.refresh_connection)

    def build_ui(self) -> None:
        accent = "#F2C66D"
        module_header(
            self, code="RACE", title="Race Bettor",
            subtitle="мышью находит коэффициент 2/1 и ставит до заданного числа побед",
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
        workspace.grid_columnconfigure(0, weight=6, uniform="race")
        workspace.grid_columnconfigure(1, weight=5, uniform="race")
        workspace.grid_rowconfigure(0, weight=1)

        live = panel(workspace, "LIVE  /  ЦЕЛЬ ПО ПОБЕДАМ", accent)
        live.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        goal = ctk.CTkFrame(live, fg_color="#302A1C", corner_radius=17)
        goal.pack(fill="x", padx=16, pady=(0, 13))
        ctk.CTkLabel(
            goal, text="ОСТАНОВИТЬСЯ ПОСЛЕ", font=ctk.CTkFont("Segoe UI", 9, "bold"),
            text_color=GOLD,
        ).pack(side="left", padx=(18, 10), pady=18)
        self.target_entry = ctk.CTkEntry(
            goal, textvariable=self.target_input, width=72, height=36,
            justify="center", corner_radius=10, border_color="#6F5A2B",
            fg_color="#171B22", font=ctk.CTkFont("Segoe UI", 17, "bold"),
        )
        self.target_entry.pack(side="left", pady=11)
        ctk.CTkLabel(
            goal, text="ПОБЕД", font=ctk.CTkFont("Segoe UI", 9, "bold"),
            text_color=GOLD,
        ).pack(side="left", padx=10, pady=18)
        ctk.CTkLabel(
            live, textvariable=self.progress, font=ctk.CTkFont("Segoe UI", 25, "bold"),
            text_color=TEXT,
        ).pack(padx=18, pady=(3, 2))
        ctk.CTkLabel(
            live, textvariable=self.stats, font=ctk.CTkFont("Segoe UI", 10, "bold"),
            text_color=accent,
        ).pack(padx=18, pady=(0, 7))
        ctk.CTkLabel(
            live, textvariable=self.timer, font=ctk.CTkFont("Segoe UI", 14, "bold"),
            text_color=TEXT,
        ).pack(fill="x", padx=20, pady=(5, 4))
        ctk.CTkLabel(
            live, textvariable=self.status, font=ctk.CTkFont("Segoe UI", 9),
            text_color=MUTED, wraplength=410, justify="center",
        ).pack(fill="x", padx=24, pady=(0, 14))
        self.main_button = ctk.CTkButton(
            live, text="Запустить автоставки  ·  F9", command=self.toggle,
            height=50, corner_radius=14, font=ctk.CTkFont("Segoe UI", 13, "bold"),
            fg_color=accent, hover_color="#FFD981", text_color="#201B10",
        )
        self.main_button.pack(fill="x", side="bottom", padx=16, pady=16)

        guide = panel(workspace, "КАК РАБОТАЕТ ЦИКЛ", accent)
        guide.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        step_list(guide, (
            ("1", "Подготовь терминал", "Сядь за скачки. Размер ставки оставь нужным — обычно 10 фишек."),
            ("2", "Укажи число побед", "Например, 4 — цикл завершится только после четвёртой победы."),
            ("3", "Нажми F9", "F откроет список; модуль увидит строку 2/1 и плавно нажмёт на неё мышью."),
            ("4", "Случайный момент", "Ставка мышью подтверждается при случайном остатке примерно 27–15 секунд."),
        ), accent)
        note = ctk.CTkFrame(guide, fg_color=SURFACE_ALT, corner_radius=12)
        note.pack(fill="x", padx=14, pady=(1, 14))
        ctk.CTkLabel(
            note,
            text="Всегда выбирается найденная строка 2/1. Размер ставки не изменяется.",
            font=ctk.CTkFont("Segoe UI", 8, "bold"), text_color=GOLD,
            wraplength=315, justify="left",
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

    def update_score(self) -> None:
        self.progress.set(f"{self.wins} / {self.target_wins} побед")
        self.stats.set(f"Ставок {self.bets}  ·  проигрышей {self.losses}")

    def smooth_move_to(self, target_x: int, target_y: int) -> bool:
        """Плавный подвод курсора живёт в ``core``: им пользуется и шахтёр."""
        return glide_cursor_to(target_x, target_y)

    def restore_previous_window(self, *, restore_cursor: bool = False) -> None:
        previous = self.previous_window
        self.previous_window = None
        if previous and previous != self.game_window and user32.IsWindow(previous):
            activate_window(previous)
        if restore_cursor and self.saved_cursor is not None:
            self.smooth_move_to(*self.saved_cursor)
            self.saved_cursor = None

    def focus_game(self, *, save_cursor: bool = False) -> bool:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            return False
        previous = user32.GetForegroundWindow()
        if save_cursor and self.saved_cursor is None:
            self.saved_cursor = cursor_position()
        if not activate_window(self.game_window):
            return False
        if self.previous_window is None and previous and previous != self.game_window:
            self.previous_window = previous
        return True

    def click_client_point(
        self, client_x: int, client_y: int, *, jitter_x: int, jitter_y: int,
    ) -> bool:
        if not self.game_window:
            return False
        bounds = client_bounds(self.game_window)
        if not bounds:
            return False
        left, top, width, height = bounds
        safe_x = min(width - 2, max(1, client_x + random.randint(-jitter_x, jitter_x)))
        safe_y = min(height - 2, max(1, client_y + random.randint(-jitter_y, jitter_y)))
        if not self.smooth_move_to(left + safe_x, top + safe_y):
            return False
        time.sleep(random.uniform(0.08, 0.15))
        return send_left_click(random.uniform(0.10, 0.16))

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

    def capture_toast(self) -> np.ndarray | None:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            return None
        bounds = client_bounds(self.game_window)
        if not bounds:
            return None
        left, top, width, height = bounds
        ratio_x, ratio_y, ratio_width, ratio_height = RACE_TOAST_ACCENT_RATIO
        region = {
            "left": left + round(width * ratio_x),
            "top": top + round(height * ratio_y),
            "width": max(2, round(width * ratio_width)),
            "height": max(2, round(height * ratio_height)),
        }
        try:
            with mss.mss() as screen:
                return np.asarray(screen.grab(region))[:, :, :3]
        except Exception:
            return None

    def prepare_bet(self) -> None:
        if not self.focus_game(save_cursor=True):
            self.phase = "retry"
            self.next_action = time.monotonic() + self.BET_RETRY_SECONDS
            self.status.set("GTA не найдена или не получила фокус. Повторю подключение.")
            return
        self.attempts += 1
        self.horse_row = None
        self.horse_score = 0.0
        if bet_dialog_visible(self.capture_client()):
            self.schedule_button_click("Открытая карточка 2/1 уже найдена — повторно гонщика не выбираю.")
            return
        if not send_key_tap(VK_F):
            self.restore_previous_window(restore_cursor=True)
            self.stop("Windows не приняла нажатие F. Автоставки остановлены.")
            return
        now = time.monotonic()
        self.phase = "list_wait"
        self.next_action = now + self.LIST_DETECT_TIMEOUT_SECONDS
        self.deadline = self.next_action
        self.timer.set("Ищу в списке коэффициент 2/1…")
        self.status.set("F открыл список. Ставка мышью будет только после уверенного распознавания 2/1.")

    def plan_horse_click(self, detection: tuple[int, int, int, float]) -> None:
        row, _click_x, _click_y, score = detection
        self.horse_row = row
        self.horse_score = score
        self.phase = "horse_click"
        self.next_action = time.monotonic()
        self.deadline = None
        self.timer.set("2/1 найден · открываю постоянную карточку ставки")
        self.status.set(f"Строка №{row + 1} распознана ({score:.0%}). Сейчас выберу её один раз.")

    def click_detected_horse(self) -> None:
        if not self.focus_game():
            self.schedule_bet_retry("Не удалось вернуть GTA для клика по лошади 2/1.")
            return
        detection = find_two_to_one_horse(self.capture_client())
        if detection is None:
            self.schedule_bet_retry("К моменту клика список или строка 2/1 исчезли. Жду новое окно ставок.")
            return
        row, click_x, click_y, score = detection
        self.horse_row, self.horse_score = row, score
        bounds = client_bounds(self.game_window) if self.game_window else None
        if not bounds:
            self.schedule_bet_retry("Не удалось определить размер окна GTA перед кликом.")
            return
        _left, _top, width, height = bounds
        if not self.click_client_point(
            click_x, click_y,
            jitter_x=max(5, round(width * 0.010)),
            jitter_y=max(3, round(height * 0.006)),
        ):
            self.schedule_bet_retry("Windows не приняла плавный клик по строке 2/1.")
            return
        self.phase = "dialog_wait"
        self.next_action = time.monotonic() + self.DIALOG_SETTLE_SECONDS
        self.deadline = self.next_action + 2.0
        self.timer.set("Лошадь 2/1 выбрана · открываю ставку")
        self.status.set(f"Плавный клик по строке №{row + 1} выполнен. Жду карточку подтверждения.")

    def schedule_button_click(self, message: str) -> None:
        """Pick a fresh human-like moment inside the 28-second betting window."""
        self.scheduled_remaining = random.uniform(*self.BET_REMAINING_SECONDS)
        delay = max(0.2, self.BET_WINDOW_SECONDS - self.scheduled_remaining)
        self.phase = "button_delay"
        self.next_action = time.monotonic() + delay
        self.deadline = None
        self.restore_previous_window(restore_cursor=True)
        self.timer.set(f"Ставка при остатке ≈ {self.scheduled_remaining:.1f} с")
        self.status.set(f"{message} Случайная пауза перед кнопкой: {delay:.1f} с.")

    def click_bet_button(self) -> None:
        if not self.focus_game(save_cursor=True):
            self.schedule_bet_retry("GTA потеряла фокус перед кнопкой «Сделать ставку».")
            return
        if not bet_dialog_visible(self.capture_client()):
            self.schedule_bet_retry("Постоянная карточка ставки исчезла. Открою список заново.")
            return
        # The previous win/loss toast can still be visible when the shortest
        # random delay is selected. Never let it masquerade as confirmation of
        # the new bet: wait for a clean notification slot before clicking.
        if classify_race_toast(self.capture_toast()) is not None:
            self.phase = "button_delay"
            self.next_action = time.monotonic() + 0.20
            self.status.set("Жду, пока исчезнет уведомление прошлого заезда; кнопку ещё не нажимаю.")
            return
        bounds = client_bounds(self.game_window)
        if not bounds:
            self.schedule_bet_retry("Не удалось определить кнопку «Сделать ставку».")
            return
        _left, _top, width, height = bounds
        button_x = round(width * BET_BUTTON_RATIO[0])
        button_y = round(height * BET_BUTTON_RATIO[1])
        if not self.click_client_point(
            button_x, button_y,
            jitter_x=max(8, round(width * 0.014)),
            jitter_y=max(3, round(height * 0.005)),
        ):
            self.schedule_bet_retry("Windows не приняла клик «Сделать ставку».")
            return
        now = time.monotonic()
        self.phase = "bet_confirm"
        self.next_action = now + self.BET_CONFIRM_TIMEOUT_SECONDS
        self.deadline = self.next_action
        self.timer.set("Проверяю подтверждение ставки…")
        self.status.set("Кнопка «Сделать ставку» нажата мышью. Жду зелёное уведомление.")

    def confirm_bet(self) -> None:
        now = time.monotonic()
        self.bets += 1
        self.bet_confirmed_at = now
        self.phase = "outcome_wait"
        self.next_action = now + self.RESULT_SCAN_AFTER_SECONDS
        self.deadline = now + self.RESULT_TIMEOUT_SECONDS
        self.restore_previous_window(restore_cursor=True)
        self.update_score()
        self.timer.set("Заезд идёт · результат проверю автоматически")
        self.status.set("Ставка принята. Перед финишем GTA ненадолго откроется для чтения результата.")

    def schedule_bet_retry(self, message: str) -> None:
        self.restore_previous_window(restore_cursor=True)
        self.phase = "retry"
        self.next_action = time.monotonic() + self.BET_RETRY_SECONDS
        self.deadline = None
        self.timer.set("Новый поиск окна ставок через 00:03")
        self.status.set(message)

    def begin_outcome_scan(self) -> None:
        if not self.focus_game():
            self.next_action = time.monotonic() + self.BET_RETRY_SECONDS
            self.status.set("Не могу открыть GTA для результата. Повторю через 3 секунды.")
            return
        self.phase = "outcome_scan"
        self.next_action = self.deadline
        self.timer.set("GTA открыта · жду уведомление о результате")
        self.status.set("Распознаю зелёную победу или красный проигрыш внизу экрана.")

    def accept_outcome(self, won: bool) -> None:
        if won:
            self.wins += 1
        else:
            self.losses += 1
        self.update_score()
        self.restore_previous_window()
        if self.wins >= self.target_wins:
            self.finish(f"Цель выполнена: {self.wins} побед из {self.bets} ставок.")
            return
        result = "Победа засчитана" if won else "Проигрыш засчитан"
        self.phase = "retry"
        self.next_action = time.monotonic() + 0.25
        self.deadline = None
        self.timer.set("Проверяю, осталась ли карточка ставки открытой…")
        self.status.set(f"{result}. Если карточка на месте, повторного выбора 2/1 не будет.")

    def parse_target(self) -> int | None:
        return parse_win_target(self.target_input.get(), self.MAX_TARGET_WINS)

    def toggle(self) -> None:
        if self.running:
            self.stop("Автоставки остановлены вручную.")
            return
        target = self.parse_target()
        if target is None:
            self.status.set(f"Введи целое число побед от 1 до {self.MAX_TARGET_WINS}.")
            self.target_entry.focus_set()
            return
        if not find_game_window(self.process.get()):
            self.status.set(f"Процесс «{self.process.get()}» не найден.")
            return
        self.target_wins = target
        self.wins = self.losses = self.bets = self.attempts = 0
        self.toast_kind = None
        self.bet_confirmed_at = None
        self.horse_row = None
        self.horse_score = 0.0
        self.scheduled_remaining = 0.0
        self.saved_cursor = None
        self.running = True
        self.phase = "prepare_bet"
        self.next_action = time.monotonic()
        self.deadline = None
        self.target_entry.configure(state="disabled")
        self.main_button.configure(
            text="Остановить автоставки  ·  F9", fg_color="#E95E69",
            hover_color="#C94C57", text_color=TEXT,
        )
        self.update_score()
        self.timer.set("Подготовка первой ставки…")
        self.status.set("F9 принят. Открываю GTA.")
        self.after(0, self.tick)

    def stop(self, message: str) -> None:
        self.running = False
        self.phase = "idle"
        self.next_action = self.deadline = None
        self.restore_previous_window(restore_cursor=True)
        self.target_entry.configure(state="normal")
        self.main_button.configure(
            text="Запустить автоставки  ·  F9", fg_color="#F2C66D",
            hover_color="#FFD981", text_color="#201B10",
        )
        self.timer.set("Автоставки не запущены")
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
                if self.phase in {"prepare_bet", "retry"}:
                    self.prepare_bet()
                elif self.phase == "list_wait":
                    self.schedule_bet_retry("Список с коэффициентом 2/1 не найден. Проверю следующее окно ставок.")
                elif self.phase == "horse_click":
                    self.click_detected_horse()
                elif self.phase == "dialog_wait":
                    if bet_dialog_visible(self.capture_client()):
                        self.schedule_button_click("Карточка 2/1 закреплена и останется открытой для следующих кругов.")
                    elif self.deadline is not None and now >= self.deadline:
                        self.schedule_bet_retry("Карточка ставки не открылась. Кнопку вслепую не нажимаю.")
                    else:
                        self.next_action = now + 0.18
                elif self.phase == "button_delay":
                    self.click_bet_button()
                elif self.phase == "bet_confirm":
                    self.schedule_bet_retry("Зелёное подтверждение ставки не найдено. Жду новое окно, не кликаю вслепую.")
                elif self.phase == "outcome_wait":
                    self.begin_outcome_scan()
                elif self.phase == "outcome_scan":
                    self.stop("Результат заезда не найден за 85 секунд. Остановлено без лишней ставки.")
            elif self.phase == "retry":
                seconds = max(0, int(remaining + 0.999))
                self.timer.set(f"Новый поиск через 00:{seconds:02d}")
            elif self.phase == "button_delay":
                seconds = max(0.0, remaining)
                self.timer.set(
                    f"Карточка открыта · кнопка через {seconds:.1f} с "
                    f"(остаток ≈ {self.scheduled_remaining:.1f} с)"
                )
            elif self.phase == "outcome_wait" and self.bet_confirmed_at is not None:
                seconds = max(0, int(remaining + 0.999))
                self.timer.set(f"Заезд идёт · проверка результата через {seconds} с")
        if self.winfo_exists():
            self.after(100, self.tick)

    def scan_tick(self) -> None:
        if self.winfo_exists() and self.running and self.phase == "list_wait":
            detection = find_two_to_one_horse(self.capture_client())
            if detection is not None:
                self.plan_horse_click(detection)
        elif self.winfo_exists() and self.running and self.phase == "bet_confirm":
            self.toast_kind = classify_race_toast(self.capture_toast())
            if self.toast_kind == "green":
                self.confirm_bet()
        elif self.winfo_exists() and self.running and self.phase == "outcome_scan":
            result = classify_race_result(self.capture_client())
            if result is not None:
                self.accept_outcome(result == "win")
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
