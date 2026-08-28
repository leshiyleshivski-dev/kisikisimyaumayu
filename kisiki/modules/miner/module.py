"""Экран ORE HUNT: сам отбивает камень и сам собирает руду со стола.

Помощник не ходит по карьеру и не выбирает камни — до камня доходит человек и
сам жмёт `E`, как жал бы всегда. Дальше начинается то, ради чего модуль и
написан: десять ударов в ритме, который принимает игра, и сбор крупинок со
стола сортировки, живущего полторы секунды.

**Ритм задаёт игра, а не таймер.** Каждый засчитанный удар двигает полосу
«Добыча руды» на десятую часть, и следующий клик ставится от этой ступеньки, а
не от прошлого клика. По записи от 26.08 живой игрок отбивает камень
ступеньками через 1,02-1,17 секунды — это пол, ниже которого игра удары просто
не засчитывает. Клик, отправленный раньше, она молча выбрасывает.

**И ни одно ожидание здесь не постоянное.** Каждая пауза берётся случайной из
своего диапазона: и задержка до удара, и удержание кнопки, и пауза после
клика по руде, и время, за которое курсор доезжает до крупинки. Ровный
интервал — это подпись робота, и меняется она не косметикой, а тем, что
константы заменены на границы.

Клики уходят только когда GTA впереди: фокус помощник себе не забирает и у
человека не отнимает.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import customtkinter as ctk
import mss
import numpy as np

from ...core import (
    APP_BG, BODY, FONT_BODY, FONT_CAPTION, FONT_NOTE, MINT, MUTED, SURFACE_ALT,
    TEXT, VK_F9, VK_F11, client_bounds, cursor_position, find_game_window,
    glide_cursor_to, process_for_window, resource_path, rounded_photo,
    send_left_click, user32,
)
from ..ui import connection_panel, hotkey_bar, module_header, panel, step_list
from .stats import OreTally
from .vision import (
    ORE_NAMES, ORE_TYPES, OreTarget, find_ore_targets, mining_bar_fill,
    read_ore_toast, sorting_table_visible, supported_resolution,
)

ACCENT = "#E2B85B"


class MinerModule(ctk.CTkFrame):
    """Отбить камень и собрать с него руду, пока человек ходит по карьеру."""

    # Тик опроса. Он же — шаг, которым квантуется всё остальное: ступеньку
    # видно не раньше следующего тика, и клик уходит тоже на тике. При 55 мс
    # это добавляло к каждому удару около 0,11 с сверх запланированного и
    # выносило темп за 1,20. Чтение полосы стоит около 4 мс, так что тик можно
    # держать частым.
    SCAN_INTERVAL_MS = 30

    # --- ритм ударов -------------------------------------------------------
    # Отсчёт идёт от засчитанной ступеньки. Игра после неё какое-то время мышь
    # не принимает, поэтому клик ставится ближе к концу этого окна: сумма
    # задержки и пути «клик -> ступенька» (около 0,3 с) даёт удар за 1,08-1,18
    # секунды.
    #
    # Границы выбраны по 220 промежуткам между ступеньками из записи 09:50:
    # медиана 1,10 с, девяностый процентиль 1,20 с, и только 2% промежутков
    # длиннее 1,20. Верхняя граница держит сумму под этим потолком, нижняя
    # поднята над глухим окном — клик, отправленный в него, игра молча
    # выбрасывает, и удар обходится в лишние полсекунды на повторе.
    STRIKE_READY_RANGE = (0.78, 0.88)
    # Глухое окно — свойство игры и сервера, а не наше: замерить его по записи
    # нельзя, чужие клики на ней не видны. Поэтому оно не угадывается, а
    # подбирается на ходу. Клик, после которого ступенька не пришла, был
    # отправлен слишком рано — пол задержки поднимается. Удар, взятый с первого
    # клика, отпускает его обратно понемногу. За несколько камней пол сам
    # встаёт чуть выше настоящего глухого окна.
    #
    # Живой прогон показал, зачем это нужно: на выставленных вручную границах
    # треть кликов уходила в пустоту, и каждый такой удар стоил 1,45-1,50 с
    # вместо 1,15 — хуже, чем играет человек.
    STRIKE_READY_SPREAD = 0.10
    # Поднимать решительно, отпускать редко. Отпускать по чуть-чуть на каждом
    # чистом ударе не годится: за камень пол сползает ниже глухого окна, на
    # следующем камне снова ловит пропуск, и получается пила — в живом прогоне
    # это давало каждый десятый клик в пустоту при равновесии в четыре
    # процента. Поэтому пол отпускается только после длинной серии чистых
    # ударов, и то на один шаг.
    STRIKE_READY_FLOOR_STEP = 0.05
    STRIKE_READY_FLOOR_EASE = 0.01
    STRIKE_CLEAN_STREAK_TO_EASE = 25
    # Потолок пола: даже на самой медленной игре удар не должен уходить за
    # 1,25 с вместе с путём «клик -> ступенька».
    STRIKE_READY_FLOOR_MAX = 0.95
    # Как часто переспрашивать, вернулась ли игра вперёд.
    REFOCUS_RECHECK_SECONDS = 0.06
    # Путь «принятый клик -> засчитанная ступенька». Стартовое значение взято
    # с запасом: дальше модуль мерит его сам по каждому чистому удару. Числу
    # 0,30 верить не стоило — на живой игре вышло около 0,17, и повтор после
    # пропущенного клика из-за этого был вдвое длиннее нужного.
    CLICK_TO_STEP_SECONDS = 0.30
    CLICK_TO_STEP_SMOOTHING = 0.2
    STRIKE_PACE_CEILING_SECONDS = 1.20
    # Ступенька не пришла — значит клик выбросили. Повтор нарочно длиннее пути
    # «клик -> ступенька»: иначе удавшийся взмах получит лишний клик до того,
    # как его собственная ступенька перепланирует следующий удар.
    # Повтор считается от измеренного пути «клик -> ступенька»: он обязан быть
    # длиннее его плюс окна опроса, иначе удавшийся взмах получит лишний клик
    # до того, как его собственная ступенька перепланирует следующий удар.
    # Всё, что сверх этого, — чистая потеря на каждом пропущенном клике.
    STRIKE_RETRY_MARGIN = 0.04
    STRIKE_RETRY_SPREAD = 0.10
    STRIKE_RETRY_MINIMUM = 0.20
    # Пока первой ступеньки не было, цепляться не за что.
    STRIKE_COLD_RANGE = (0.45, 0.62)
    CLICK_HOLD_RANGE = (0.045, 0.11)
    # Ступенька полосы — 4-10% в зависимости от камня.
    PROGRESS_STEP = 0.03
    STRIKE_WATCH_SECONDS = 0.04
    # Камень кончился, когда полоса пропала: это собственный флаг игры, а не
    # наша догадка. Три пропуска подряд — примерно четверть секунды.
    BAR_MISSING_FRAMES = 3
    # Страховки, а не рабочие пределы. Считаются в засчитанных ударах: клики
    # ударами не являются, ранний игра выбрасывает, и лимит по кликам увёл бы
    # помощника с недобитого камня.
    MAX_STRIKES = 30
    MAX_STRIKE_CLICKS = 70
    MAX_STRIKING_SECONDS = 45.0

    # --- сбор со стола -----------------------------------------------------
    # Курсор доезжает до крупинки за это время плюс добавка за расстояние.
    # Диапазон узкий: стол живёт полторы-две секунды, и полсекунды на переезд,
    # как у ставок на скачки, съели бы половину вкраплений.
    GLIDE_RANGE = (0.13, 0.24)
    GLIDE_BEND_PIXELS = 9.0
    TARGET_SETTLE_RANGE = (0.16, 0.30)
    MAX_ATTEMPTS_PER_TARGET = 3
    # Отработанная цель не потеряна: игра иногда не отдаёт крупинку, по которой
    # кликнули точно, а через пару секунд отдаёт её же. Поэтому история попыток
    # прощается и заход повторяется, вместо того чтобы стол стоял открытым.
    TARGET_COOLDOWN_SECONDS = 1.6
    MAX_TARGET_ROUNDS = 3
    MAX_TARGET_CLICKS = 45
    TARGET_STALL_SECONDS = 5.0
    # Крупинка со стола сама не уходит: забирает её только наш клик. Значит
    # цель, пропавшая из виду до того, как по ней кликнули, не собрана — её
    # потеряли из виду, и стоит сходить по последнему адресу. Мигание одного
    # кадра адресом не считается: настоящая крупинка видна подряд.
    SIGHTING_MEMORY_SECONDS = 12.0
    SIGHTING_MIN_FRAMES = 2
    TABLE_MISSING_FRAMES = 2
    RESULT_TIMEOUT_SECONDS = 7.0

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
        self.on_stats_change = on_stats_change
        self.tally = OreTally(daily_stats)

        # `active` — экран сверху и горячие клавиши живут; `watching` — идёт
        # ли работа по F9. Путать их нельзя: спрятанный за другим модулем
        # экран обязан молчать на F9, иначе начнёт кликать из-под чужого.
        self.active = False
        self.watching = False
        self.phase = "idle"
        self.game_window: int | None = None
        self.saved_cursor: tuple[int, int] | None = None
        self.phase_started_at = 0.0
        self.next_action_at = 0.0
        self.last_watch_at = 0.0
        self.strikes = 0
        self.strike_clicks = 0
        self.clicks_since_step = 0
        # Удар, у которого промежуток испорчен паузой по фокусу, в ритм не идёт.
        self.rhythm_paused = False
        self.paused_strikes = 0
        # Пол живёт дольше камня: глухое окно принадлежит игре, а не камню.
        self.ready_floor = self.STRIKE_READY_RANGE[0]
        self.clean_streak = 0
        self.click_to_step = self.CLICK_TO_STEP_SECONDS
        self.last_click_at = 0.0
        self.last_step_at = 0.0
        self.step_gaps: list[float] = []
        self.progress_fill: float | None = None
        self.bar_missing = 0
        self.table_missing = 0
        self.target_clicks = 0
        self.target_rounds = 0
        self.last_target_at = 0.0
        self.attempts: list[tuple[int, int, int, float]] = []
        # [x, y, скор, вид, сколько кадров подряд видели, когда видели в последний раз]
        self.sightings: list[list] = []
        self.keys = {key: False for key in (VK_F9, VK_F11)}

        self.process = ctk.StringVar(value="GTA5.exe")
        self.connection = ctk.StringVar(value="Ищу GTA5.exe…")
        self.stage = ctk.StringVar(value="СМОТРЕНИЕ ВЫКЛЮЧЕНО")
        self.status = ctk.StringVar(
            value="Нажми F9 и иди к камню. Дальше только подходишь и жмёшь E.",
        )
        self.rhythm = ctk.StringVar(value="Ритм ударов ещё не замерен")
        self.session_total_text = ctk.StringVar(value="0")
        self.daily_total_text = ctk.StringVar(value="0")
        self.unknown_text = ctk.StringVar(value="Нераспознано: сеанс 0 · день 0")
        self.leaders_text = ctk.StringVar(value="За сеанс пока ничего не добыто")
        self.ore_text = {
            key: ctk.StringVar(value="0  ·  0") for key, _title in ORE_TYPES
        }
        self.ore_images: dict[str, object] = {}

        self.build_ui()
        self.refresh_stats_view()
        self.after(40, self.poll_hotkeys)
        self.after(self.SCAN_INTERVAL_MS, self.scan_tick)
        self.after(0, self.refresh_connection)

    # ----------------------------------------------------------------- вид

    def build_ui(self) -> None:
        module_header(
            self, code="ORE", title="ORE HUNT",
            subtitle="Кот «Кварц» · сам бьёт камень и сам собирает руду",
            accent=ACCENT, on_back=self.on_back,
        )
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=42, pady=(0, 28))
        body.grid_columnconfigure(0, weight=5, uniform="ore")
        body.grid_columnconfigure(1, weight=6, uniform="ore")
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.indicator = connection_panel(
            left, process=self.process, connection=self.connection,
            accent=ACCENT, on_check=self.refresh_connection,
        )

        run = panel(left, "ЧТО СЕЙЧАС ДЕЛАЕТ", ACCENT)
        run.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(
            run, textvariable=self.stage, font=ctk.CTkFont("Segoe UI", 17, "bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=18, pady=(0, 4))
        ctk.CTkLabel(
            run, textvariable=self.status, font=ctk.CTkFont("Segoe UI", FONT_BODY),
            text_color=BODY, wraplength=380, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 7))
        ctk.CTkLabel(
            run, textvariable=self.rhythm,
            font=ctk.CTkFont("Segoe UI", FONT_NOTE, "bold"), text_color=ACCENT,
        ).pack(anchor="w", padx=18, pady=(0, 16))

        steps = panel(left, "КАК РАБОТАТЬ", ACCENT)
        steps.pack(fill="both", expand=True)
        step_list(steps, (
            ("1", "Нажми F9", "Помощник начинает смотреть на экран. Фокус он себе не забирает — переключись в GTA сам."),
            ("2", "Подойди к камню и нажми E", "Как обычно. Полоса «Добыча руды» — знак, что игра приняла камень; по ней помощник и начинает бить."),
            ("3", "Дальше не трогай мышь", "Удары идут в ритме игры, стол сортировки собирается сам. Между камнями просто иди к следующему."),
        ), ACCENT)
        hotkey_bar(left, (("F9", "смотреть / пауза"), ("F11", "стоп")), ACCENT)

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        self.build_stats(right)

    def build_stats(self, parent: ctk.CTkFrame) -> None:
        totals = ctk.CTkFrame(parent, fg_color=SURFACE_ALT, corner_radius=18)
        totals.pack(fill="x", pady=(0, 12))
        totals.grid_columnconfigure((0, 1), weight=1)
        self._total_card(totals, 0, "ЗА СЕАНС", self.session_total_text, MINT)
        self._total_card(totals, 1, "ЗА ДЕНЬ", self.daily_total_text, ACCENT)

        card = panel(parent, "СОБРАНО РУДЫ  ·  СЕАНС  ·  ДЕНЬ", ACCENT)
        card.pack(fill="both", expand=True)
        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=14, pady=(0, 6))
        grid.grid_columnconfigure((0, 1, 2), weight=1, uniform="ore")
        grid.grid_rowconfigure((0, 1, 2), weight=1, uniform="ore")
        for index, (key, title) in enumerate(ORE_TYPES):
            self._ore_card(grid, index, key, title)
        ctk.CTkLabel(
            card, textvariable=self.leaders_text,
            font=ctk.CTkFont("Segoe UI", FONT_BODY, "bold"), text_color=MINT,
        ).pack(anchor="w", padx=18, pady=(6, 0))
        ctk.CTkLabel(
            card, textvariable=self.unknown_text,
            font=ctk.CTkFont("Segoe UI", FONT_NOTE), text_color=BODY,
        ).pack(anchor="w", padx=18, pady=(3, 14))

    @staticmethod
    def _total_card(
        parent: ctk.CTkFrame, column: int, title: str,
        value: ctk.StringVar, colour: str,
    ) -> None:
        card = ctk.CTkFrame(parent, fg_color="transparent")
        card.grid(row=0, column=column, padx=16, pady=11, sticky="ew")
        ctk.CTkLabel(
            card, text=title, font=ctk.CTkFont("Segoe UI", FONT_CAPTION, "bold"),
            text_color=MUTED,
        ).pack()
        ctk.CTkLabel(
            card, textvariable=value, font=ctk.CTkFont("Segoe UI", 24, "bold"),
            text_color=colour,
        ).pack()

    def _ore_card(
        self, parent: ctk.CTkFrame, index: int, key: str, title: str,
    ) -> None:
        row = ctk.CTkFrame(parent, fg_color=SURFACE_ALT, corner_radius=13)
        row.grid(row=index // 3, column=index % 3, padx=4, pady=4, sticky="nsew")
        icon = self.ore_icon(index, key)
        ctk.CTkLabel(
            row, text="" if icon else title[:2], image=icon, width=48, height=48,
            font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=ACCENT,
        ).pack(side="left", padx=(11, 9))
        copy = ctk.CTkFrame(row, fg_color="transparent")
        copy.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(
            copy, text=title, font=ctk.CTkFont("Segoe UI", FONT_NOTE, "bold"),
            text_color=TEXT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            copy, textvariable=self.ore_text[key],
            font=ctk.CTkFont("Segoe UI", FONT_BODY, "bold"), text_color=BODY,
        ).pack(anchor="w")

    def ore_icon(self, index: int, key: str):
        """Иконка руды из ``assets/ores``; без файла карточка живёт подписью."""
        if key not in self.ore_images:
            path = resource_path("assets", "ores", f"{index + 1:02d}_{key}.png")
            try:
                self.ore_images[key] = rounded_photo(path, 48, 48)
            except Exception:
                self.ore_images[key] = None
        return self.ore_images[key]

    def refresh_stats_view(self) -> None:
        self.tally.refresh_day()
        self.session_total_text.set(str(self.tally.session_total))
        self.daily_total_text.set(str(self.tally.daily_total))
        for key, _title in ORE_TYPES:
            self.ore_text[key].set(
                f"{self.tally.session_count(key)}  ·  {self.tally.daily_count(key)}"
            )
        self.unknown_text.set(
            f"Нераспознано: сеанс {self.tally.session_unknown}"
            f"  ·  день {self.tally.daily_unknown}"
        )
        leaders = self.tally.leaders()
        self.leaders_text.set(
            "За сеанс пока ничего не добыто" if not leaders else
            "За сеанс чаще всего: " + ", ".join(
                f"{ORE_NAMES[key].lower()} {count}" for key, count in leaders
            )
        )

    def record_ore(self, ore_key: str | None) -> str:
        message = self.tally.record(ore_key)
        self.refresh_stats_view()
        if self.on_stats_change is not None:
            self.on_stats_change()
        return message

    # ------------------------------------------------------------- связь

    def refresh_connection(self) -> None:
        if not self.winfo_exists():
            return
        self.game_window = find_game_window(self.process.get())
        if self.game_window:
            self.indicator.configure(text_color=MINT)
            bounds = client_bounds(self.game_window)
            if bounds:
                _left, _top, width, height = bounds
                # Разрешение здесь нужнее заголовка окна: вся геометрия разбора
                # снята на 2560x1440, и на непроверенном экране это видно сразу.
                self.connection.set(
                    f"Подключено · {width}×{height}"
                    + ("" if supported_resolution(width, height) else " — не проверено")
                )
            else:
                self.connection.set("Подключено")
        else:
            self.connection.set("Не подключено")
            self.indicator.configure(text_color="#F05A67")
        self.after(2000, self.refresh_connection)

    def capture_client(self) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        self.game_window = find_game_window(self.process.get())
        if not self.game_window:
            return None
        bounds = client_bounds(self.game_window)
        if not bounds:
            return None
        left, top, width, height = bounds
        try:
            with mss.mss() as screen:
                frame = np.asarray(screen.grab({
                    "left": left, "top": top, "width": width, "height": height,
                }))[:, :, :3]
        except Exception:
            return None
        return frame, bounds

    def game_is_foreground(self) -> bool:
        """Клик уходит в игру только когда игра впереди.

        Сравнивается процесс, а не хэндл. ``find_game_window`` отдаёт первое
        видимое окно процесса в Z-порядке, а у игры их несколько, и какое из
        них вернётся — меняется. Сравнение хэндлов из-за этого то и дело давало
        ложное «игра не впереди»: помощник переставал бить посреди камня и
        добирал задержку четвертьсекундными кусками.

        Фокус помощник при этом не забирает: человек в это время играет, и
        перетягивать окно на себя ради клика — верный способ отправить удар в
        чужое окно.
        """
        foreground = user32.GetForegroundWindow()
        if not foreground:
            return False
        wanted = self.process.get().strip().casefold()
        return bool(wanted) and process_for_window(foreground).casefold() == wanted

    # -------------------------------------------------------- жизненный цикл

    def enter(self, phase: str, message: str, stage: str) -> None:
        self.phase = phase
        self.phase_started_at = time.monotonic()
        self.status.set(message)
        self.stage.set(stage)

    def toggle(self) -> None:
        if self.watching:
            self.pause("Пауза. F9 — продолжить смотреть.")
            return
        self.watching = True
        self.enter("watching", "Смотрю на экран. Подойди к камню и нажми E.", "ЖДУ КАМЕНЬ")

    def pause(self, message: str) -> None:
        self.watching = False
        self.enter("idle", message, "ПАУЗА")

    def stop(self, message: str) -> None:
        self.watching = False
        self.enter("idle", message, "ОСТАНОВЛЕНО")

    def activate(self) -> None:
        self.active = True
        # Клавиша, зажатая в момент переключения экранов, не должна прочитаться
        # свежим нажатием: запоминаем её состояние как уже известное.
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    def deactivate(self, message: str) -> None:
        self.stop(message)
        self.active = False
        self.keys = {key: bool(user32.GetAsyncKeyState(key) & 0x8000) for key in self.keys}

    # ------------------------------------------------------------- удары

    def begin_striking(self, now: float) -> None:
        self.strikes = 0
        self.strike_clicks = 0
        self.clicks_since_step = 0
        self.rhythm_paused = False
        self.paused_strikes = 0
        self.step_gaps = []
        self.last_step_at = 0.0
        self.progress_fill = None
        self.bar_missing = 0
        self.next_action_at = now + random.uniform(*self.STRIKE_COLD_RANGE)
        self.enter(
            "striking", "Камень принят игрой — бью в её ритме.", "УДАР 0",
        )

    def strike_step(self, now: float) -> None:
        if now - self.last_watch_at < self.STRIKE_WATCH_SECONDS:
            return
        self.last_watch_at = now
        captured = self.capture_client()
        if captured is None:
            self.stop("Окно GTA потерялось во время добычи.")
            return
        frame, _bounds = captured
        fill = mining_bar_fill(frame)
        if fill is None:
            self.bar_missing += 1
            if self.bar_missing >= self.BAR_MISSING_FRAMES:
                self.enter(
                    "awaiting_table",
                    "Камень отбит. Жду стол сортировки.",
                    "КАМЕНЬ ОТБИТ",
                )
            return
        self.bar_missing = 0
        if self.progress_fill is not None and fill - self.progress_fill >= self.PROGRESS_STEP:
            self.credit_strike(now)
        self.progress_fill = fill

        if self.strikes >= self.MAX_STRIKES or self.strike_clicks >= self.MAX_STRIKE_CLICKS:
            self.stop("Камень не поддаётся — остановился, чтобы не бить вслепую.")
            return
        if now - self.phase_started_at >= self.MAX_STRIKING_SECONDS:
            self.stop("Добыча затянулась дольше сорока пяти секунд — остановился.")
            return
        if now >= self.next_action_at:
            self.send_strike(now)

    def retry_window(self) -> tuple[float, float]:
        """Через сколько повторить клик, если ступенька так и не пришла."""
        floor = max(
            self.STRIKE_RETRY_MINIMUM,
            self.click_to_step + self.STRIKE_WATCH_SECONDS + self.STRIKE_RETRY_MARGIN,
        )
        return floor, floor + self.STRIKE_RETRY_SPREAD

    def credit_strike(self, now: float) -> None:
        """Ступенька полосы — единственный честный признак удара.

        Клик ударом не является: ранний игра молча выбрасывает. Пока счётчик
        считал отправленные клики, в окне стояла цифра вдвое больше правды.
        """
        self.strikes += 1
        if self.last_step_at and self.rhythm_paused:
            # Промежуток, в котором помощник намеренно не бил (GTA ушла из
            # фокуса), — это не его ритм, а пауза. Считать её ударом значит
            # мерить не то: в окне появлялись 1,4-1,5 с ровно тогда, когда
            # человек переключался на помощник посмотреть на эти самые цифры.
            self.paused_strikes += 1
        elif self.last_step_at:
            self.step_gaps.append(now - self.last_step_at)
            average = sum(self.step_gaps) / len(self.step_gaps)
            self.rhythm.set(
                f"Удар за {self.step_gaps[-1]:.2f} с  ·  в среднем {average:.2f} с"
                f"  ·  кликов на удар {self.strike_clicks / max(1, self.strikes):.2f}"
                f"  ·  пауза от {self.ready_floor:.2f} с"
                f"  ·  взмах {self.click_to_step:.2f} с"
                + (f"  ·  вне фокуса {self.paused_strikes}" if self.paused_strikes else "")
            )
        self.rhythm_paused = False
        self.last_step_at = now
        self.stage.set(f"УДАР {self.strikes}")
        # Отсчёт следующего удара — от ступеньки, а не от клика. Сама ступенька
        # случилась не в этот миг, а между прошлым и этим взглядом на экран,
        # поэтому отсчёт ведётся от середины окна: иначе каждый удар получает
        # полшага опроса сверху, и за камень набегает лишняя десятая секунды.
        if self.clicks_since_step == 1 and self.last_click_at:
            # Клик был ровно один — значит ступенька принадлежит ему, и
            # расстояние между ними и есть искомый путь.
            measured = now - self.last_click_at
            if 0.05 <= measured <= 0.60:
                self.click_to_step += self.CLICK_TO_STEP_SMOOTHING * (
                    measured - self.click_to_step
                )
        if self.clicks_since_step == 1:
            self.clean_streak += 1
            if self.clean_streak >= self.STRIKE_CLEAN_STREAK_TO_EASE:
                # Длинная серия без единого промаха — можно осторожно
                # поторопиться: вдруг пол забрался выше, чем нужно игре.
                self.clean_streak = 0
                self.ready_floor = max(
                    self.STRIKE_READY_RANGE[0],
                    self.ready_floor - self.STRIKE_READY_FLOOR_EASE,
                )
        self.clicks_since_step = 0
        stepped_at = now - self.STRIKE_WATCH_SECONDS / 2
        self.next_action_at = stepped_at + random.uniform(
            self.ready_floor, self.ready_floor + self.STRIKE_READY_SPREAD,
        )

    def send_strike(self, now: float) -> None:
        if not self.game_is_foreground():
            self.status.set("Жду, пока GTA снова окажется впереди — вслепую не кликаю.")
            # Проверяться часто: как только окно вернулось, бить надо сразу.
            # Четверть секунды здесь означала четверть секунды к удару просто
            # за то, что человек глянул в помощник.
            self.next_action_at = now + self.REFOCUS_RECHECK_SECONDS
            # Этот промежуток ритмом уже не будет: мы в нём нарочно не били.
            self.rhythm_paused = True
            return
        if self.clicks_since_step and self.strikes:
            # Прошлый клик ступеньки не дал — значит попал в глухое окно.
            # Условие про `strikes` важно: пока первой ступеньки на камне не
            # было, клики идут вслепую по холодному интервалу и попадают в
            # пустоту просто потому, что цепляться не за что. Считать их
            # уликой против пола — значит гнать его вверх на каждом камне;
            # в прогоне он так добегал до потолка и на быстрой игре тоже.
            self.clean_streak = 0
            self.ready_floor = min(
                self.STRIKE_READY_FLOOR_MAX,
                self.ready_floor + self.STRIKE_READY_FLOOR_STEP,
            )
        if not send_left_click(random.uniform(*self.CLICK_HOLD_RANGE)):
            self.stop("Windows не принял удар киркой.")
            return
        self.strike_clicks += 1
        self.clicks_since_step += 1
        self.last_click_at = now
        # Ступенька, если удар засчитан, придёт сама и перепланирует следующий.
        # Это окно — только на случай, когда клик выбросили.
        self.next_action_at = now + random.uniform(*self.retry_window())

    # -------------------------------------------------------------- сбор

    def begin_collecting(self, now: float) -> None:
        self.target_clicks = 0
        self.target_rounds = 0
        self.attempts = []
        self.sightings = []
        self.table_missing = 0
        self.last_target_at = now
        self.next_action_at = now
        if self.saved_cursor is None:
            self.saved_cursor = cursor_position()
        self.enter(
            "collecting", "Стол открыт. Ищу вкрапления на камнях.", "СБОР РУДЫ",
        )

    def target_ready(self, point: tuple[int, int], now: float) -> bool:
        record = self.attempt_record(point)
        if record is None:
            return True
        _index, (_x, _y, tries, last) = record
        if tries >= self.MAX_ATTEMPTS_PER_TARGET:
            return False
        return now - last >= 0.45

    def attempt_record(
        self, point: tuple[int, int],
    ) -> tuple[int, tuple[int, int, int, float]] | None:
        for index, record in enumerate(self.attempts):
            x, y, _tries, _last = record
            if (x - point[0]) ** 2 + (y - point[1]) ** 2 <= 34 ** 2:
                return index, record
        return None

    def remember_attempt(self, point: tuple[int, int], now: float) -> int:
        record = self.attempt_record(point)
        if record is None:
            self.attempts.append((point[0], point[1], 1, now))
            return 1
        index, (x, y, tries, _last) = record
        self.attempts[index] = (x, y, tries + 1, now)
        return tries + 1

    def retry_round(self, now: float) -> bool:
        if self.target_rounds >= self.MAX_TARGET_ROUNDS:
            return False
        self.target_rounds += 1
        self.attempts = []
        self.last_target_at = now
        return True

    def remember_sightings(self, detected: list, now: float) -> None:
        """Запомнить, где на этом столе вообще видели руду."""
        for target in detected:
            record = None
            for sighting in self.sightings:
                if (sighting[0] - target.x) ** 2 + (sighting[1] - target.y) ** 2 <= 34 ** 2:
                    record = sighting
                    break
            if record is None:
                self.sightings.append([target.x, target.y, target.score, target.kind, 1, now])
                continue
            record[0], record[1] = target.x, target.y
            record[2] = max(record[2], target.score)
            record[4] = record[4] + 1 if now - record[5] <= 1.0 else 1
            record[5] = now

    def lost_sighting(self, detected: list, now: float):
        """Крупинка, которую видели своими глазами и не успели взять.

        Со стола руду забирает только наш клик, поэтому цель, пропавшая из
        виду раньше первого клика по ней, не собрана: её потеряло зрение. На
        столе 5:06 записи «тест 3» так и вышло — красная крупинка стояла на
        камне все пятнадцать секунд, а помощник перестал её видеть после
        седьмого клика и простоял рядом до самого закрытия стола.

        Заход по памяти стоит дешевле, чем несобранная руда: промах по голому
        камню не делает ничего, а стол всё равно стоит открытым. Заход при
        этом один: цель, по которой уже кликали, из памяти выбывает. Пропала
        после клика — значит собрана, и гоняться за ней не за чем.
        """
        best = None
        for x, y, score, kind, frames, last in self.sightings:
            if frames < self.SIGHTING_MIN_FRAMES:
                continue
            if now - last > self.SIGHTING_MEMORY_SECONDS:
                continue
            if any((x - t.x) ** 2 + (y - t.y) ** 2 <= 34 ** 2 for t in detected):
                continue
            if self.attempt_record((x, y)) is not None:
                continue
            if best is None or score > best[2]:
                best = (x, y, score, kind)
        if best is None:
            return None
        return OreTarget(best[0], best[1], best[2], best[3])

    def collect_step(self, now: float) -> None:
        if now < self.next_action_at:
            return
        captured = self.capture_client()
        if captured is None:
            self.stop("Окно GTA потерялось во время сбора руды.")
            return
        frame, bounds = captured
        if not sorting_table_visible(frame):
            self.table_missing += 1
            if self.table_missing >= self.TABLE_MISSING_FRAMES:
                self.restore_cursor()
                self.enter(
                    "awaiting_result",
                    "Стол закрылся. Жду уведомление о полученной руде.",
                    "ЖДУ РУДУ",
                )
            return
        self.table_missing = 0
        if not self.game_is_foreground():
            self.status.set("GTA ушла из фокуса — по столу не кликаю.")
            self.next_action_at = now + self.REFOCUS_RECHECK_SECONDS
            return

        # Где стоит стрелка, знает система, а не кадр: искать её зрением значит
        # каждый раз заново находить то, что и так известно. Точка переводится в
        # координаты клиентской области — в тех же живёт и сам кадр.
        pointer_x, pointer_y = cursor_position()
        detected = find_ore_targets(
            frame, cursor=(pointer_x - bounds[0], pointer_y - bounds[1]),
        )
        self.remember_sightings(detected, now)
        available = [
            target for target in detected
            if self.target_ready((target.x, target.y), now)
        ]
        if available:
            self.click_target(available[0], bounds, now)
            return
        if detected and now - self.last_target_at >= self.TARGET_COOLDOWN_SECONDS:
            if self.retry_round(now):
                self.status.set(
                    f"Вкрапления не поддались — захожу ещё раз (круг {self.target_rounds})."
                )
                return
        if not detected:
            # Пустой камень — не заклинивший стол, и уходить с него рано.
            # Крупинки прилетают на камень и после того, как ковёр открылся, а
            # закрывается стол сам. Раньше пустой кадр отсчитывал ту же
            # выдержку, что и не поддающаяся руда, и стол бросали живым.
            lost = self.lost_sighting(detected, now)
            if lost is not None:
                self.click_target(lost, bounds, now, remembered=True)
                return
            self.status.set("На камнях сейчас пусто. Смотрю, пока виден ковёр.")
            return
        if now - self.last_target_at >= self.TARGET_STALL_SECONDS:
            self.restore_cursor()
            self.enter(
                "manual",
                "Руда не поддаётся. Оставил стол тебе — ничего не считаю добытым.",
                "РУЧНАЯ ПРОВЕРКА",
            )

    def click_target(self, target, bounds, now: float, *, remembered: bool = False) -> None:
        screen_x, screen_y = bounds[0] + target.x, bounds[1] + target.y
        moved = glide_cursor_to(
            screen_x, screen_y,
            duration_range=self.GLIDE_RANGE,
            bend_pixels=self.GLIDE_BEND_PIXELS,
        )
        if not moved or not send_left_click(random.uniform(*self.CLICK_HOLD_RANGE)):
            self.stop("Windows не принял клик по вкраплению.")
            return
        clicked_at = time.monotonic()
        attempt = self.remember_attempt((target.x, target.y), clicked_at)
        self.target_clicks += 1
        self.last_target_at = clicked_at
        self.next_action_at = clicked_at + random.uniform(*self.TARGET_SETTLE_RANGE)
        self.stage.set(f"СБОР РУДЫ · {self.target_clicks}")
        self.status.set(
            ("Крупинка пропала из виду — пробую по последнему месту" if remembered
             else f"Нажал на {target.kind} вкрапление")
            + (f" (попытка {attempt})." if attempt > 1 else ".")
        )
        if self.target_clicks >= self.MAX_TARGET_CLICKS:
            self.restore_cursor()
            self.enter(
                "manual",
                "Достигнут безопасный предел кликов по столу. Дальше руками.",
                "РУЧНАЯ ПРОВЕРКА",
            )

    def restore_cursor(self) -> None:
        if self.saved_cursor is not None:
            glide_cursor_to(
                *self.saved_cursor, duration_range=self.GLIDE_RANGE,
                bend_pixels=self.GLIDE_BEND_PIXELS,
            )
            self.saved_cursor = None

    # ------------------------------------------------------------ результат

    def result_step(self, now: float) -> None:
        if now - self.last_watch_at < 0.12:
            return
        self.last_watch_at = now
        captured = self.capture_client()
        if captured is None:
            self.stop("Окно GTA потерялось до подтверждения результата.")
            return
        frame, _bounds = captured
        reading = read_ore_toast(frame)
        if reading.visible:
            message = self.record_ore(reading.ore)
            self.enter("watching", message, "ЖДУ КАМЕНЬ")
            return
        if now - self.phase_started_at >= self.RESULT_TIMEOUT_SECONDS:
            self.enter(
                "watching",
                "Уведомления о руде не было — статистику не трогаю.",
                "ЖДУ КАМЕНЬ",
            )

    # ------------------------------------------------------------- цикл

    def scan_tick(self) -> None:
        if not self.winfo_exists():
            return
        try:
            if self.watching:
                self.advance(time.monotonic())
        finally:
            if self.winfo_exists():
                self.after(self.SCAN_INTERVAL_MS, self.scan_tick)

    def advance(self, now: float) -> None:
        if self.phase in ("watching", "awaiting_table"):
            self.watch_step(now)
        elif self.phase == "striking":
            self.strike_step(now)
        elif self.phase == "collecting":
            self.collect_step(now)
        elif self.phase == "awaiting_result":
            self.result_step(now)
        elif self.phase == "manual":
            self.manual_step(now)

    def watch_step(self, now: float) -> None:
        if now - self.last_watch_at < 0.16:
            return
        self.last_watch_at = now
        captured = self.capture_client()
        if captured is None:
            return
        frame, _bounds = captured
        if sorting_table_visible(frame):
            self.begin_collecting(now)
            return
        if mining_bar_fill(frame) is not None:
            self.begin_striking(now)

    def manual_step(self, now: float) -> None:
        """Смотреть, не кликая, пока человек не закроет стол сам."""
        if now - self.last_watch_at < 0.2:
            return
        self.last_watch_at = now
        captured = self.capture_client()
        if captured is None:
            return
        frame, _bounds = captured
        if sorting_table_visible(frame):
            self.table_missing = 0
            return
        self.table_missing += 1
        if self.table_missing >= self.TABLE_MISSING_FRAMES:
            self.enter(
                "awaiting_result",
                "Стол закрыт вручную. Жду уведомление о руде.",
                "ЖДУ РУДУ",
            )

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
