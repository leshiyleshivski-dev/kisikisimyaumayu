"""Shared theme, resources and Windows input primitives.

This module has no application-screen dependencies and can be reused by every
mini-game controller without creating circular imports.
"""

from __future__ import annotations

import ctypes
import os
import random
import sys
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

APP_BG = "#111722"
SURFACE = "#1A2230"
SURFACE_HOVER = "#2B374A"
SURFACE_ALT = "#222C3C"
TEXT = "#F5F7FB"
MUTED = "#98A5B8"
PURPLE = "#A994ED"
PINK = "#FF8F91"
GOLD = "#F2C66D"
MINT = "#68D6B4"

# Текст секретных модулей стоял на 8-9 пунктах, а абзацы красились в MUTED —
# цвет подписи, а не текста. На тёмной панели такая строка сливается с фоном, и
# описание приходилось вычитывать. Размеры и цвет абзаца названы здесь, чтобы
# экран не подбирал их на глаз, а десять модулей выглядели одинаково.
BODY = "#C8D3E4"
FONT_CAPTION = 11  # прописные заголовки панелей, полос и бейджей
FONT_NOTE = 12     # короткая строка состояния под заголовком
FONT_BODY = 13     # абзац описания
FONT_LEAD = 15     # первая строка карточки, крупнее абзаца

CATS = (
    ("Крупье", "сонно следит за рулеткой", "assets/cats/01_casino_cat.png", "#D37B78", "Сонные котики"),
    ("Звонок", "сонно держит телефонный ритм", "assets/cats/02_mobile_cat.png", "#4E9BFF", "Сонные котики"),
    ("Кирпич", "ловит идеальный момент", "assets/cats/04_builder_cat.png", "#E3A642", "Строитель"),
    ("Вольт", "пока просто следит за проводами", "assets/cats/03_electrician_cat.png", "#F1C94A", "Электрик"),
    ("Кварц", "выбивает самоцветы из упрямых камней", "assets/cats/05_miner_cat.png", "#E2B85B", "Добывающие котики"),
    ("Сучок", "снимает с бревна всё лишнее", "assets/cats/12_lumberjack_cat.png", "#9BD36F", "Добывающие котики"),
    ("Фаворит", "угадывает победителя по усам", "assets/cats/07_race_bettor_cat.png", "#F2C66D", "Лудоманы"),
    ("Семёрка", "слушает звон барабанов", "assets/cats/08_slot_cat.png", "#D4A7FF", "Лудоманы"),
    ("Туз", "знает, когда хватит карт", "assets/cats/10_blackjack_cat.png", "#68D6B4", "Лудоманы"),
    ("Блеф", "считает шансы и не моргает", "assets/cats/11_poker_cat.png", "#75A7FF", "Лудоманы"),
)

COMING_SOON_CATS = (
    ("Поплавок", "ждёт большого клёва", "#63C7BC", "Рыбак"),
    ("Фишка", "копит фишки на удачу", "#FF8F91", "Лудоманы"),
)

# Шахтёр и лесоруб раньше стояли отдельными категориями по одному котику:
# полоса над карточкой повторяла её же название. Работа у них одна — помощник
# бьёт по цели и подбирает добытое, — поэтому они собраны в одну категорию.
CAT_CATEGORIES = (
    "Сонные котики", "Строитель", "Электрик", "Добывающие котики", "Рыбак",
    "Лудоманы",
)

# Звук берётся по индексу котика из CATS. Список намеренно может быть короче
# CATS: у новых котиков своего звука ещё нет, и им играет заглушка. Раньше
# здесь стоял прямой SOUND_FILES[index], который уронил бы игру на седьмом
# котике. Свой звук добавляется просто — вписать файл на нужную позицию.
SOUND_FILES = (
    "smug_laugh.mp3", "bongo_drums.mp3", "orange_boing.mp3",
    "buff_impact.mp3", "keyboard_type.mp3",
)
PLACEHOLDER_SOUND = "grumpy_trombone.mp3"


def cat_sound(index: int) -> str:
    """Файл звука для котика; без своего звука отдаёт заглушку."""
    if 0 <= index < len(SOUND_FILES):
        return SOUND_FILES[index]
    return PLACEHOLDER_SOUND

def resource_path(*parts: str) -> Path:
    """Путь работает одинаково для исходника и собранного приложения."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return root.joinpath(*parts)


# Исходные PNG котиков весят больше мегабайта (1254x1254), и распаковка
# каждого стоит ~300 мс. Раньше один и тот же файл читался заново на каждый
# новый размер, поэтому экран рецептов подвисал на полторы секунды. Теперь
# распакованный оригинал живёт в кеше, а нарезка размеров стоит миллисекунды.
_SOURCE_IMAGES: dict[str, tk.PhotoImage] = {}


def source_photo(path: Path) -> tk.PhotoImage:
    """Распакованный оригинал; повторные обращения берутся из кеша."""
    key = str(path)
    image = _SOURCE_IMAGES.get(key)
    if image is None:
        image = tk.PhotoImage(file=key)
        _SOURCE_IMAGES[key] = image
    return image


def release_source_images() -> None:
    """Отпустить оригиналы, когда все нужные размеры уже нарезаны.

    Готовые уменьшенные копии не зависят от оригинала, поэтому после прогрева
    приложение освобождает десятки мегабайт распакованных пикселей.
    """
    _SOURCE_IMAGES.clear()


def rounded_photo(path: Path, max_width: int, max_height: int) -> tk.PhotoImage:
    """Tk умеет показывать PNG без дополнительных библиотек; масштабируем кратно."""
    image = source_photo(path)
    width, height = image.width(), image.height()
    factor = max(1, (max(width / max_width, height / max_height) + 0.999).__int__())
    return image.subsample(factor, factor)


def data_path(name: str) -> Path:
    """Файл в папке приложения: прогресс кликера, журнал раздач и что дальше.

    Складывать всё в один файл нельзя: прогресс читается на каждом запуске, а
    покерный журнал за вечер набирает сотни записей — им незачем мешать друг
    другу.
    """
    data_root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "KisikiSimyaumyau"
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root / name


def progress_path() -> Path:
    return data_path("progress.json")


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
VK_RETURN = 0x0D
VK_LEFT, VK_RIGHT = 0x25, 0x27
VK_F = 0x46
VK_E = 0x45
VK_W, VK_A, VK_S, VK_D = 0x57, 0x41, 0x53, 0x44
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
TIME_READOUT_RATIO = (0.948, 0.892, 0.048, 0.055)
TIMER_SCAN_INTERVAL_SECONDS = 0.25
TIMER_ACTIVATE_SETTLE_SECONDS = 0.8
GAME_RECONNECT_SECONDS = 5
MISSING_TIMER_SHUTDOWN_SECONDS = 3.0
TIMING_BAR_RATIO = (0.80, 0.84, 0.19, 0.12)
CURRENT_GRID_ROWS, CURRENT_GRID_COLUMNS = 8, 8
# Область сетки «Проведение тока» внутри клиентской области GTA (16:9).
# Коэффициенты получены по игровому интерфейсу, а не по размеру рабочего стола.
CURRENT_GRID_RATIO = (0.338, 0.213, 0.340, 0.575)
PORT_UP, PORT_RIGHT, PORT_DOWN, PORT_LEFT = 1, 2, 4, 8
CURRENT_DIRECTIONS = (
    (-1, 0, PORT_UP, PORT_DOWN, VK_W),
    (0, 1, PORT_RIGHT, PORT_LEFT, VK_D),
    (1, 0, PORT_DOWN, PORT_UP, VK_S),
    (0, -1, PORT_LEFT, PORT_RIGHT, VK_A),
)


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
    """Отправить нажатие с достаточным удержанием для интерфейса GTA."""
    down, up = Input(), Input()
    down.type, down.data.keyboard.virtual_key = 1, virtual_key
    sent_down = user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(Input)) == 1
    # GTA иногда пропускает события down/up, отправленные в один тик Windows.
    # Небольшое удержание стабильно регистрируется и не мешает мини-играм.
    time.sleep(0.024)
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


def glide_cursor_to(
    target_x: int,
    target_y: int,
    *,
    duration_range: tuple[float, float] = (0.38, 0.62),
    bend_pixels: float = 14.0,
) -> bool:
    """Довести настоящий курсор до точки живой дугой, а не прыжком.

    ``SetCursorPos`` ставит курсор в точку одним кадром: между двумя крупинками
    руды он телепортируется, и со стороны это ровно то, чем является. Здесь
    путь идёт по квадратичной кривой со случайным изгибом, а скорость по нему
    размазана «сглаженным шагом» — без рывка на старте и стука в конце.

    Длительность складывается из случайной базы и добавки за расстояние:
    короткий переход между соседними вкраплениями не должен стоить столько же,
    сколько проход через весь экран. Ею и настраивается характер под модуль —
    ставки на скачки никуда не спешат, а стол сортировки живёт полторы секунды.
    """
    start_x, start_y = cursor_position()
    dx, dy = target_x - start_x, target_y - start_y
    distance = max(1.0, (dx * dx + dy * dy) ** 0.5)
    duration = random.uniform(*duration_range) + min(0.18, distance / 9000)
    steps = max(12, min(72, round(duration / 0.010)))
    bend = random.uniform(-bend_pixels, bend_pixels)
    control_x = (start_x + target_x) / 2 - dy / distance * bend
    control_y = (start_y + target_y) / 2 + dx / distance * bend
    for step in range(1, steps + 1):
        raw = step / steps
        eased = raw * raw * (3.0 - 2.0 * raw)
        inverse = 1.0 - eased
        x = inverse * inverse * start_x + 2 * inverse * eased * control_x + eased * eased * target_x
        y = inverse * inverse * start_y + 2 * inverse * eased * control_y + eased * eased * target_y
        if not user32.SetCursorPos(round(x), round(y)):
            return False
        if step < steps:
            time.sleep(duration / steps)
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
