"""Экран TIMBER CUT: сам рубит дерево и сам очищает бревно от веток.

Помощник не ходит по лесу и не выбирает деревья — до дерева доходит человек и
сам берётся за него, как брался бы всегда. Дальше начинается то, ради чего
модуль и написан: десять ударов в ритме, который принимает игра, и обрубка
веток на столе, живущем три секунды.

**Вступает помощник по полосе, как Кварц в карьере.** Полоса «Рубка дерева»
зажигается, когда игра дерево приняла, и с этого мига удары идут в зачёт. Первую
секунду полоса стоит на нуле и клики выбрасывает — эта пара взмахов и есть вся
плата за то, что человеку не надо бить самому.

**Ритм задаёт игра, а не таймер.** Каждый засчитанный удар двигает полосу
«Рубка дерева» на десятую часть, и следующий клик ставится от этой ступеньки,
а не от прошлого клика. По записи от 27.08 живой игрок валит дерево ступеньками
через 1,00-1,20 секунды при медиане 1,10 — это пол, ниже которого игра удары
просто не засчитывает. Клик, отправленный раньше, она молча выбрасывает.

Полоса рубки — тот же виджет игры, что и «Добыча руды» у шахтёра, в том же
углу и с тем же шагом, поэтому и ритм здесь считается тем же способом. Числа,
однако, свои: они замерены по записи лесопилки, а не по записи карьера.

**И ни одно ожидание здесь не постоянное.** Каждая пауза берётся случайной из
своего диапазона: и задержка до удара, и удержание кнопки, и пауза после клика
по ветке, и время, за которое курсор доезжает до сука. Ровный интервал — это
подпись робота, и меняется она не косметикой, а тем, что константы заменены на
границы.

Счёта добытого здесь нет намеренно. Игра за очистку бревна ничего не называет:
уведомления вроде «Вы собрали … руда!» на лесопилке не бывает, а считать
собственные клики значит выдавать за добычу свои попытки. Что происходит
сейчас, видно в статусе; что накопилось за вечер, знает сама игра.

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
    APP_BG, BODY, FONT_BODY, FONT_NOTE, MINT, TEXT, VK_F9, VK_F11, client_bounds,
    cursor_position, find_game_window, glide_cursor_to, process_for_window,
    send_left_click, user32,
)
from ..ui import (
    connection_panel, hotkey_bar, module_header, note_card, panel, step_list,
)
from .ml_vision import find_branch_targets_ml
from .vision import (
    BranchTarget, chopping_bar_fill, find_branch_targets as find_classical_branch_targets,
    log_table_visible, supported_resolution,
)

ACCENT = "#9BD36F"


def find_branch_targets(frame):
    """Use learned masks, with the old silhouette detector as a safe fallback."""
    detected = find_branch_targets_ml(frame)
    if detected is not None:
        return detected
    return find_classical_branch_targets(frame)


class LumberjackModule(ctk.CTkFrame):
    """Свалить дерево и обрубить с бревна ветки, пока человек ходит по лесу."""

    # Тик опроса. Он же — шаг, которым квантуется всё остальное: ступеньку
    # видно не раньше следующего тика, и клик уходит тоже на тике. Чтение
    # полосы стоит 0,1 мс, так что тик можно держать частым.
    SCAN_INTERVAL_MS = 30

    # --- ритм ударов -------------------------------------------------------
    # Отсчёт идёт от засчитанной ступеньки. Игра после неё какое-то время мышь
    # не принимает, поэтому клик ставится ближе к концу этого окна: сумма
    # задержки и пути «клик -> ступенька» даёт удар за 1,05-1,15 секунды.
    #
    # Границы выбраны по 44 ступенькам пяти деревьев записи: медиана
    # промежутка 1,10 с, разброс 1,00-1,20, и ни одного промежутка длиннее
    # 1,20. Верхняя граница держит сумму под этим потолком, нижняя поднята над
    # глухим окном — клик, отправленный в него, игра молча выбрасывает.
    CHOP_READY_RANGE = (0.78, 0.88)
    # Глухое окно — свойство игры и сервера, а не наше: замерить его по записи
    # нельзя, чужие клики на ней не видны. Поэтому оно не угадывается, а
    # подбирается на ходу. Клик, после которого ступенька не пришла, был
    # отправлен слишком рано — пол задержки поднимается. Удар, взятый с первого
    # клика, отпускает его обратно понемногу.
    CHOP_READY_SPREAD = 0.10
    # Поднимать решительно, отпускать редко. Отпускать по чуть-чуть на каждом
    # чистом ударе не годится: за дерево пол сползает ниже глухого окна, на
    # следующем снова ловит пропуск, и получается пила.
    CHOP_READY_FLOOR_STEP = 0.05
    CHOP_READY_FLOOR_EASE = 0.01
    CHOP_CLEAN_STREAK_TO_EASE = 25
    # Потолок пола: даже на самой медленной игре удар не должен уходить за
    # 1,25 с вместе с путём «клик -> ступенька».
    CHOP_READY_FLOOR_MAX = 0.95
    # Как часто переспрашивать, вернулась ли игра вперёд.
    REFOCUS_RECHECK_SECONDS = 0.06
    # Путь «принятый клик -> засчитанная ступенька». Стартовое значение взято
    # с запасом: дальше модуль мерит его сам по каждому чистому удару.
    CLICK_TO_STEP_SECONDS = 0.25
    CLICK_TO_STEP_SMOOTHING = 0.2
    # Ступенька не пришла — значит клик выбросили. Повтор считается от
    # измеренного пути «клик -> ступенька»: он обязан быть длиннее его плюс
    # окна опроса, иначе удавшийся взмах получит лишний клик до того, как его
    # собственная ступенька перепланирует следующий удар.
    CHOP_RETRY_MARGIN = 0.04
    CHOP_RETRY_SPREAD = 0.10
    CHOP_RETRY_MINIMUM = 0.20
    CLICK_HOLD_RANGE = (0.045, 0.11)
    # Пока первой ступеньки не было, цепляться не за что: полоса зажигается
    # пустой (на дневной записи она стоит на нуле 1,3 с), и первые клики игра
    # выбрасывает. Ждать ради этого человеческого удара помощник не будет —
    # у Кварца в карьере тот же виджет и то же правило: видно полосу — бей.
    # Выброшенные клики пол задержки не поднимают: `send_chop` трогает его
    # только после первой засчитанной ступеньки.
    CHOP_COLD_RANGE = (0.45, 0.62)
    # Как часто смотреть на экран, пока дерева ещё нет.
    WATCH_INTERVAL = 0.16
    # Ступенька полосы — 8-11% на записи. Порог взят заметно ниже: он ловит
    # ступеньку, а не меряет её.
    PROGRESS_STEP = 0.03
    CHOP_WATCH_SECONDS = 0.04
    # Дерево кончилось, когда полоса пропала: это собственный флаг игры, а не
    # наша догадка. Три пропуска подряд — примерно десятая доля секунды.
    BAR_MISSING_FRAMES = 3
    # Страховки, а не рабочие пределы. Считаются в засчитанных ударах: клики
    # ударами не являются, ранний игра выбрасывает, и лимит по кликам увёл бы
    # помощника с недорубленного дерева. На записи дерево берут за десять
    # ударов и семь-девять секунд.
    MAX_CHOPS = 30
    MAX_CHOP_CLICKS = 70
    MAX_CHOPPING_SECONDS = 45.0

    # --- обрубка веток -----------------------------------------------------
    # Курсор доезжает до ветки за это время плюс добавка за расстояние.
    # Диапазон короткий, и это замерено, а не выбрано на глаз: человек на
    # записи снимает двенадцать веток за две секунды, то есть игра принимает
    # по ветке каждые 0,15-0,2 с и никакого глухого окна между ними не держит.
    # Вместе с удержанием кнопки и паузой после клика выходит 4-5 кликов в
    # секунду — столько же, сколько у человека, и вчетверо больше прежнего.
    GLIDE_RANGE = (0.05, 0.10)
    GLIDE_BEND_PIXELS = 7.0
    TARGET_SETTLE_RANGE = (0.05, 0.12)
    MAX_ATTEMPTS_PER_TARGET = 3
    # Отработанная цель не потеряна: игра иногда не снимает ветку, по которой
    # кликнули точно. Поэтому история попыток прощается и заход повторяется,
    # вместо того чтобы стол стоял открытым.
    TARGET_COOLDOWN_SECONDS = 1.4
    MAX_TARGET_ROUNDS = 3
    # Предел оставляет место трём проходам по всем найденным маскам, но не
    # позволяет зациклиться на ошибочной цели.
    MAX_TARGET_CLICKS = 36
    TARGET_STALL_SECONDS = 4.0
    # Две небольшие сегментационные модели на тёплом запуске разбирают кадр
    # примерно за 0,15 с, поэтому новый взгляд планируется с той же частотой.
    BRANCH_LOOK_SECONDS = 0.15
    # Ветку со стола снимает только наш клик. Значит цель, пропавшая из виду до
    # того, как по ней кликнули, не срублена — её потеряло зрение, и стоит
    # сходить по последнему адресу. Мигание одного кадра адресом не считается.
    SIGHTING_MEMORY_SECONDS = 12.0
    SIGHTING_MIN_FRAMES = 2
    TARGET_SAME_DISTANCE = 34
    TABLE_MISSING_FRAMES = 2

    def __init__(self, parent: ctk.CTkFrame, on_back: Callable[[], None]) -> None:
        super().__init__(parent, fg_color=APP_BG, corner_radius=0)
        self.on_back = on_back

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
        self.chops = 0
        self.chop_clicks = 0
        self.clicks_since_step = 0
        # Удар, у которого промежуток испорчен паузой по фокусу, в ритм не идёт.
        self.rhythm_paused = False
        self.paused_chops = 0
        # Пол живёт дольше дерева: глухое окно принадлежит игре, а не дереву.
        self.ready_floor = self.CHOP_READY_RANGE[0]
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
        self.last_look_at = 0.0
        # [x, y, скор, вид, сколько кадров подряд видели, когда видели в последний раз]
        self.sightings: list[list] = []
        self.trees = 0
        self.branches = 0
        self.keys = {key: False for key in (VK_F9, VK_F11)}

        self.process = ctk.StringVar(value="GTA5.exe")
        self.connection = ctk.StringVar(value="Ищу GTA5.exe…")
        self.stage = ctk.StringVar(value="СМОТРЕНИЕ ВЫКЛЮЧЕНО")
        self.status = ctk.StringVar(
            value="Нажми F9 и иди к дереву. Дальше только подходишь и начинаешь рубить.",
        )
        self.rhythm = ctk.StringVar(value="Ритм ударов ещё не замерен")
        self.table_text = ctk.StringVar(value="Стол ещё не открывался")
        self.session_text = ctk.StringVar(value="За сеанс: деревьев 0 · веток 0")

        self.build_ui()
        self.after(40, self.poll_hotkeys)
        self.after(self.SCAN_INTERVAL_MS, self.scan_tick)
        self.after(0, self.refresh_connection)

    # ------------------------------------------------------------------ вид

    def build_ui(self) -> None:
        module_header(
            self, code="LOG", title="TIMBER CUT",
            subtitle="Кот «Сучок» · сам рубит дерево и сам обрубает ветки",
            accent=ACCENT, on_back=self.on_back,
        )
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=42, pady=(0, 28))
        body.grid_columnconfigure(0, weight=5, uniform="log")
        body.grid_columnconfigure(1, weight=6, uniform="log")
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
        ).pack(anchor="w", padx=18, pady=(0, 5))
        ctk.CTkLabel(
            run, textvariable=self.session_text,
            font=ctk.CTkFont("Segoe UI", FONT_NOTE), text_color=BODY,
        ).pack(anchor="w", padx=18, pady=(0, 16))

        steps = panel(left, "КАК РАБОТАТЬ", ACCENT)
        steps.pack(fill="both", expand=True)
        step_list(steps, (
            ("1", "Нажми F9",
             "Помощник начинает смотреть на экран. Фокус он себе не забирает — "
             "переключись в GTA сам."),
            ("2", "Подойди к дереву и возьмись за него",
             "Как обычно. Полоса «Рубка дерева» — знак, что игра приняла "
             "дерево; по ней помощник и начинает бить, дальше руки не нужны."),
            ("3", "Дальше не трогай мышь",
             "Удары идут в ритме игры, ветки со стола снимаются сами. Между "
             "деревьями просто иди к следующему."),
        ), ACCENT)
        hotkey_bar(left, (("F9", "смотреть / пауза"), ("F11", "стоп")), ACCENT)

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        table = panel(right, "ЧТО НА СТОЛЕ", ACCENT)
        table.pack(fill="both", expand=True)
        ctk.CTkLabel(
            table, textvariable=self.table_text,
            font=ctk.CTkFont("Segoe UI", 18, "bold"), text_color=MINT,
            wraplength=380, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 14))
        for title, description in (
            ("Сучки распознаёт обученная модель",
             "Она выделяет отдельной маской каждый сучок, в том числе лежащие "
             "вдоль коры и сросшиеся друг с другом. Помощник нажимает в самое "
             "толстое место маски, поэтому точка остаётся внутри сучка даже у "
             "развилки."),
            ("По бревну вслепую больше не тыкает",
             "Если модель сейчас не нашла сучок, помощник пересматривает стол. "
             "Клик отправляется только по найденной маске или по месту сучка, "
             "который уверенно видели на предыдущих кадрах."),
            ("Стол ищется по ковру, а не по зелени",
             "Ковёр стола — искусственный газон, и насыщенность у него вдвое "
             "выше, чем у любой живой листвы. По «просто зелёному» помощник "
             "днём считал столом весь лес и щёлкал по веткам ивы над головой. "
             "Теперь зелень леса порог не берёт ни при каком свете."),
            ("Счёта добытого здесь нет",
             "Игра за очистку бревна ничего не называет: уведомления вроде «Вы "
             "собрали … руда!» на лесопилке не бывает. Считать вместо неё свои "
             "же клики значит выдавать попытки за добычу, поэтому в строке "
             "сеанса стоят только деревья и заходы по веткам."),
        ):
            note_card(table, title, description, wraplength=455)

    def refresh_session_view(self) -> None:
        self.session_text.set(
            f"За сеанс: деревьев {self.trees} · веток {self.branches}"
        )

    # -------------------------------------------------------------- связь

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

        Сравнивается процесс, а не хэндл: ``find_game_window`` отдаёт первое
        видимое окно процесса в Z-порядке, а у игры их несколько, и какое из
        них вернётся — меняется.

        Фокус помощник при этом не забирает: человек в это время играет, и
        перетягивать окно на себя ради клика — верный способ отправить удар в
        чужое окно.
        """
        foreground = user32.GetForegroundWindow()
        if not foreground:
            return False
        wanted = self.process.get().strip().casefold()
        return bool(wanted) and process_for_window(foreground).casefold() == wanted

    # ------------------------------------------------------- жизненный цикл

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
        self.enter(
            "watching", "Смотрю на экран. Подойди к дереву и начни рубить.",
            "ЖДУ ДЕРЕВО",
        )

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

    # -------------------------------------------------------------- рубка

    def begin_chopping(self, now: float) -> None:
        self.chops = 0
        self.chop_clicks = 0
        self.clicks_since_step = 0
        self.rhythm_paused = False
        self.paused_chops = 0
        self.step_gaps = []
        self.last_step_at = 0.0
        self.progress_fill = None
        self.bar_missing = 0
        self.next_action_at = now + random.uniform(*self.CHOP_COLD_RANGE)
        self.enter("chopping", "Дерево принято игрой — бью в её ритме.", "УДАР 0")

    def chop_step(self, now: float) -> None:
        if now - self.last_watch_at < self.CHOP_WATCH_SECONDS:
            return
        self.last_watch_at = now
        captured = self.capture_client()
        if captured is None:
            self.stop("Окно GTA потерялось во время рубки.")
            return
        frame, _bounds = captured
        fill = chopping_bar_fill(frame)
        if fill is None:
            self.bar_missing += 1
            if self.bar_missing >= self.BAR_MISSING_FRAMES:
                self.trees += 1
                self.refresh_session_view()
                self.enter(
                    "awaiting_table", "Дерево свалено. Жду стол с бревном.",
                    "ДЕРЕВО СВАЛЕНО",
                )
            return
        self.bar_missing = 0
        if self.progress_fill is not None and fill - self.progress_fill >= self.PROGRESS_STEP:
            self.credit_chop(now)
        self.progress_fill = fill

        if self.chops >= self.MAX_CHOPS or self.chop_clicks >= self.MAX_CHOP_CLICKS:
            self.stop("Дерево не поддаётся — остановился, чтобы не бить вслепую.")
            return
        if now - self.phase_started_at >= self.MAX_CHOPPING_SECONDS:
            self.stop("Рубка затянулась дольше сорока пяти секунд — остановился.")
            return
        if now >= self.next_action_at:
            self.send_chop(now)

    def retry_window(self) -> tuple[float, float]:
        """Через сколько повторить клик, если ступенька так и не пришла."""
        floor = max(
            self.CHOP_RETRY_MINIMUM,
            self.click_to_step + self.CHOP_WATCH_SECONDS + self.CHOP_RETRY_MARGIN,
        )
        return floor, floor + self.CHOP_RETRY_SPREAD

    def credit_chop(self, now: float) -> None:
        """Ступенька полосы — единственный честный признак удара.

        Клик ударом не является: ранний игра молча выбрасывает. Считал бы
        счётчик отправленные клики — в окне стояла бы цифра вдвое больше правды.
        """
        self.chops += 1
        if self.last_step_at and self.rhythm_paused:
            # Промежуток, в котором помощник намеренно не бил (GTA ушла из
            # фокуса), — это не его ритм, а пауза. Считать её ударом значит
            # мерить не то.
            self.paused_chops += 1
        elif self.last_step_at:
            self.step_gaps.append(now - self.last_step_at)
            average = sum(self.step_gaps) / len(self.step_gaps)
            self.rhythm.set(
                f"Удар за {self.step_gaps[-1]:.2f} с  ·  в среднем {average:.2f} с"
                f"  ·  кликов на удар {self.chop_clicks / max(1, self.chops):.2f}"
                f"  ·  пауза от {self.ready_floor:.2f} с"
                f"  ·  взмах {self.click_to_step:.2f} с"
                + (f"  ·  вне фокуса {self.paused_chops}" if self.paused_chops else "")
            )
        self.rhythm_paused = False
        self.last_step_at = now
        self.stage.set(f"УДАР {self.chops}")
        # Отсчёт следующего удара — от ступеньки, а не от клика. Сама ступенька
        # случилась не в этот миг, а между прошлым и этим взглядом на экран,
        # поэтому отсчёт ведётся от середины окна: иначе каждый удар получает
        # полшага опроса сверху.
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
            if self.clean_streak >= self.CHOP_CLEAN_STREAK_TO_EASE:
                # Длинная серия без единого промаха — можно осторожно
                # поторопиться: вдруг пол забрался выше, чем нужно игре.
                self.clean_streak = 0
                self.ready_floor = max(
                    self.CHOP_READY_RANGE[0],
                    self.ready_floor - self.CHOP_READY_FLOOR_EASE,
                )
        self.clicks_since_step = 0
        stepped_at = now - self.CHOP_WATCH_SECONDS / 2
        self.next_action_at = stepped_at + random.uniform(
            self.ready_floor, self.ready_floor + self.CHOP_READY_SPREAD,
        )

    def send_chop(self, now: float) -> None:
        if not self.game_is_foreground():
            self.status.set("Жду, пока GTA снова окажется впереди — вслепую не кликаю.")
            # Проверяться часто: как только окно вернулось, бить надо сразу.
            self.next_action_at = now + self.REFOCUS_RECHECK_SECONDS
            # Этот промежуток ритмом уже не будет: мы в нём нарочно не били.
            self.rhythm_paused = True
            return
        if self.clicks_since_step and self.chops:
            # Прошлый клик ступеньки не дал — значит попал в глухое окно.
            # Условие про `chops` важно: пока первой ступеньки на дереве не
            # было, клики идут вслепую по холодному интервалу и попадают в
            # пустоту просто потому, что цепляться не за что.
            self.clean_streak = 0
            self.ready_floor = min(
                self.CHOP_READY_FLOOR_MAX,
                self.ready_floor + self.CHOP_READY_FLOOR_STEP,
            )
        if not send_left_click(random.uniform(*self.CLICK_HOLD_RANGE)):
            self.stop("Windows не принял удар топором.")
            return
        self.chop_clicks += 1
        self.clicks_since_step += 1
        self.last_click_at = now
        # Ступенька, если удар засчитан, придёт сама и перепланирует следующий.
        # Это окно — только на случай, когда клик выбросили.
        self.next_action_at = now + random.uniform(*self.retry_window())

    # ------------------------------------------------------------- обрубка

    def begin_collecting(self, now: float) -> None:
        self.target_clicks = 0
        self.target_rounds = 0
        self.attempts = []
        self.sightings = []
        self.last_look_at = 0.0
        self.table_missing = 0
        self.last_target_at = now
        self.next_action_at = now
        if self.saved_cursor is None:
            self.saved_cursor = cursor_position()
        self.enter("collecting", "Стол открыт. Ищу ветки на бревне.", "ОБРУБКА ВЕТОК")

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
            if self.same_place((x, y), point):
                return index, record
        return None

    def same_place(self, one: tuple[int, int], other: tuple[int, int]) -> bool:
        return (
            (one[0] - other[0]) ** 2 + (one[1] - other[1]) ** 2
            <= self.TARGET_SAME_DISTANCE ** 2
        )

    def remember_attempt(self, point: tuple[int, int], now: float) -> int:
        record = self.attempt_record(point)
        if record is None:
            self.attempts.append((point[0], point[1], 1, now))
            return 1
        index, (x, y, tries, _last) = record
        self.attempts[index] = (x, y, tries + 1, now)
        return tries + 1

    def retry_round(self, now: float) -> bool:
        """Заново разрешить попытки по найденным маскам.

        Стол всё ещё открыт, значит на бревне осталась ветка. Дешевле пройти
        по второму разу, чем стоять и смотреть.
        """
        if self.target_rounds >= self.MAX_TARGET_ROUNDS:
            return False
        self.target_rounds += 1
        self.attempts = []
        self.last_target_at = now
        return True

    def remember_sightings(self, detected: list, now: float) -> None:
        """Запомнить, где на этом столе вообще видели ветку."""
        for target in detected:
            record = None
            for sighting in self.sightings:
                if self.same_place((sighting[0], sighting[1]), (target.x, target.y)):
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
        """Ветка, которую видели своими глазами и не успели срубить.

        Со стола ветку снимает только наш клик, поэтому цель, пропавшая из
        виду раньше первого клика по ней, не срублена: её потеряло зрение.
        Заход по памяти стоит дешевле, чем оставленная ветка: промах по газону
        не делает ничего, а стол всё равно стоит открытым.

        Заход при этом один: цель, по которой уже кликали, из памяти выбывает.
        Пропала после клика — значит срублена, и гоняться за ней не за чем.
        """
        best = None
        for x, y, score, kind, frames, last in self.sightings:
            if frames < self.SIGHTING_MIN_FRAMES:
                continue
            if now - last > self.SIGHTING_MEMORY_SECONDS:
                continue
            if any(self.same_place((x, y), (target.x, target.y)) for target in detected):
                continue
            if self.attempt_record((x, y)) is not None:
                continue
            if best is None or score > best[2]:
                best = (x, y, score, kind)
        if best is None:
            return None
        return BranchTarget(best[0], best[1], best[2], best[3])

    def collect_step(self, now: float) -> None:
        if now < self.next_action_at:
            return
        captured = self.capture_client()
        if captured is None:
            self.stop("Окно GTA потерялось во время обрубки.")
            return
        frame, bounds = captured
        if not log_table_visible(frame):
            self.table_missing += 1
            if self.table_missing >= self.TABLE_MISSING_FRAMES:
                self.restore_cursor()
                self.enter(
                    "watching", "Стол закрылся. Иди к следующему дереву.",
                    "ЖДУ ДЕРЕВО",
                )
            return
        self.table_missing = 0
        if not self.game_is_foreground():
            self.status.set("GTA ушла из фокуса — по столу не кликаю.")
            self.next_action_at = now + self.REFOCUS_RECHECK_SECONDS
            return

        if now - self.last_look_at < self.BRANCH_LOOK_SECONDS:
            return
        self.last_look_at = now
        detected = find_branch_targets(frame)
        self.table_text.set(
            f"Веток на бревне: {len(detected)}" if detected
            else "Веток на бревне сейчас не вижу"
        )
        self.remember_sightings(detected, now)
        available = [
            target for target in detected
            if self.target_ready((target.x, target.y), now)
        ]
        if available:
            self.click_target(available[0], bounds, now)
            return
        lost = self.lost_sighting(detected, now)
        if lost is not None:
            self.click_target(lost, bounds, now, remembered=True)
            return
        if now - self.last_target_at >= self.TARGET_COOLDOWN_SECONDS and self.retry_round(now):
            self.status.set(
                f"Ветки не поддались — захожу ещё раз (круг {self.target_rounds})."
            )
            return
        if now - self.last_target_at >= self.TARGET_STALL_SECONDS:
            self.restore_cursor()
            self.enter(
                "manual", "Ветки не поддаются. Оставил стол тебе.", "РУЧНАЯ ПРОВЕРКА",
            )

    def click_target(self, target, bounds, now: float, *, remembered: bool = False) -> None:
        screen_x, screen_y = bounds[0] + target.x, bounds[1] + target.y
        moved = glide_cursor_to(
            screen_x, screen_y,
            duration_range=self.GLIDE_RANGE,
            bend_pixels=self.GLIDE_BEND_PIXELS,
        )
        if not moved or not send_left_click(random.uniform(*self.CLICK_HOLD_RANGE)):
            self.stop("Windows не принял клик по ветке.")
            return
        clicked_at = time.monotonic()
        attempt = self.remember_attempt((target.x, target.y), clicked_at)
        self.target_clicks += 1
        self.last_target_at = clicked_at
        self.next_action_at = clicked_at + random.uniform(*self.TARGET_SETTLE_RANGE)
        if attempt == 1:
            self.branches += 1
            self.refresh_session_view()
        self.stage.set(f"ОБРУБКА ВЕТОК · {self.target_clicks}")
        self.status.set(
            ("Ветка пропала из виду — пробую по последнему месту" if remembered
             else f"Нажал на {target.kind}")
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

    # --------------------------------------------------------------- цикл

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
        elif self.phase == "chopping":
            self.chop_step(now)
        elif self.phase == "collecting":
            self.collect_step(now)
        elif self.phase == "manual":
            self.manual_step(now)

    def watch_step(self, now: float) -> None:
        """Ждать дерево или стол, ничего не трогая.

        Признак «рубка идёт» — сама полоса, ровно как у Кварца в карьере:
        игра рисует её, когда дерево принято, и с этого мига удары идут в
        зачёт. Полоса зажигается пустой и первую секунду ударов не
        засчитывает — эти клики просто пропадают, и ждать ради них
        человеческого удара не стоит: за десять ударов дерева цена им один
        лишний взмах.
        """
        if now - self.last_watch_at < self.WATCH_INTERVAL:
            return
        self.last_watch_at = now
        captured = self.capture_client()
        if captured is None:
            return
        frame, _bounds = captured
        if log_table_visible(frame):
            self.begin_collecting(now)
            return
        if chopping_bar_fill(frame) is not None:
            self.begin_chopping(now)

    def manual_step(self, now: float) -> None:
        """Смотреть, не кликая, пока человек не закроет стол сам."""
        if now - self.last_watch_at < 0.2:
            return
        self.last_watch_at = now
        captured = self.capture_client()
        if captured is None:
            return
        frame, _bounds = captured
        if log_table_visible(frame):
            self.table_missing = 0
            return
        self.table_missing += 1
        if self.table_missing >= self.TABLE_MISSING_FRAMES:
            self.enter(
                "watching", "Стол закрыт вручную. Иди к следующему дереву.",
                "ЖДУ ДЕРЕВО",
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
