"""Shared visual building blocks for the GTA helper screens."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import customtkinter as ctk

from ..core import MUTED, SURFACE, SURFACE_ALT, TEXT

BORDER = "#303B50"
INPUT_BG = "#121A27"
NAV_BG = "#202A3A"
NAV_HOVER = "#2D3A4F"
OFFLINE = "#FF737D"


def module_header(
    parent: ctk.CTkFrame,
    *,
    code: str,
    title: str,
    subtitle: str,
    accent: str,
    on_back: Callable[[], None],
) -> ctk.CTkFrame:
    """Build the shared product-style header and return its action area."""
    header = ctk.CTkFrame(parent, fg_color="transparent")
    header.pack(fill="x", padx=42, pady=(30, 20))

    identity = ctk.CTkFrame(header, fg_color="transparent")
    identity.pack(side="left")
    ctk.CTkLabel(
        identity, text=code, width=46, height=46, corner_radius=14,
        fg_color=accent, text_color="#111722",
        font=ctk.CTkFont("Segoe UI", 13, "bold"),
    ).pack(side="left", padx=(0, 13))
    copy = ctk.CTkFrame(identity, fg_color="transparent")
    copy.pack(side="left")
    ctk.CTkLabel(
        copy, text="GTA HELPER  /  СЕКРЕТНЫЙ МОДУЛЬ",
        font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=accent,
    ).pack(anchor="w")
    ctk.CTkLabel(
        copy, text=title, font=ctk.CTkFont("Segoe UI", 22, "bold"), text_color=TEXT,
    ).pack(anchor="w", pady=(1, 0))
    ctk.CTkLabel(
        copy, text=subtitle, font=ctk.CTkFont("Segoe UI", 10), text_color=MUTED,
    ).pack(anchor="w")

    actions = ctk.CTkFrame(header, fg_color="transparent")
    actions.pack(side="right")
    ctk.CTkButton(
        actions, text="←  К котикам", command=on_back, width=126, height=38,
        corner_radius=12, fg_color=NAV_BG, hover_color=NAV_HOVER,
        font=ctk.CTkFont("Segoe UI", 10, "bold"),
    ).pack(side="right")
    return actions


def connection_panel(
    parent: ctk.CTkFrame,
    *,
    process: ctk.StringVar,
    connection: ctk.StringVar,
    accent: str,
    on_check: Callable[[], None],
    trailing: ctk.StringVar | None = None,
) -> ctk.CTkLabel:
    """Build the compact GTA connection strip and return its live indicator."""
    panel = ctk.CTkFrame(
        parent, fg_color=SURFACE, corner_radius=18,
        border_width=1, border_color=BORDER,
    )
    panel.pack(fill="x", pady=(0, 12))
    panel.grid_columnconfigure(2, weight=1)

    indicator = ctk.CTkLabel(
        panel, text="●", width=34, height=34, corner_radius=11,
        fg_color="#2A2632", font=ctk.CTkFont("Segoe UI", 16), text_color=OFFLINE,
    )
    indicator.grid(row=0, column=0, rowspan=2, padx=(16, 10), pady=14)
    ctk.CTkLabel(
        panel, text="ПОДКЛЮЧЕНИЕ К GTA",
        font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=accent,
    ).grid(row=0, column=1, padx=(0, 16), pady=(14, 0), sticky="w")
    ctk.CTkLabel(
        panel, textvariable=connection,
        font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=MUTED,
    ).grid(row=1, column=1, padx=(0, 16), pady=(0, 14), sticky="w")
    ctk.CTkEntry(
        panel, textvariable=process, height=38, border_width=1, border_color=BORDER,
        corner_radius=11, fg_color=INPUT_BG, font=ctk.CTkFont("Segoe UI", 12),
    ).grid(row=0, column=2, rowspan=2, padx=(0, 9), pady=13, sticky="ew")
    ctk.CTkButton(
        panel, text="Проверить", command=on_check, width=108, height=38,
        corner_radius=11, fg_color=accent, hover_color=accent,
        text_color="#111722", font=ctk.CTkFont("Segoe UI", 9, "bold"),
    ).grid(row=0, column=3, rowspan=2, padx=(0, 16), pady=13)
    if trailing is not None:
        ctk.CTkLabel(
            panel, textvariable=trailing, width=150,
            font=ctk.CTkFont("Segoe UI", 9, "bold"), text_color=accent,
        ).grid(row=0, column=4, rowspan=2, padx=(0, 16), pady=13)
    return indicator


def panel(
    parent: ctk.CTkFrame,
    title: str,
    accent: str,
    *,
    title_font_size: int = 9,
) -> ctk.CTkFrame:
    card = ctk.CTkFrame(
        parent, fg_color=SURFACE, corner_radius=20,
        border_width=1, border_color=BORDER,
    )
    ctk.CTkLabel(
        card, text=title,
        font=ctk.CTkFont("Segoe UI", title_font_size, "bold"),
        text_color=accent,
    ).pack(anchor="w", padx=18, pady=(16, 10))
    return card


def step_list(
    parent: ctk.CTkFrame,
    steps: Iterable[tuple[str, str, str]],
    accent: str,
) -> None:
    for number, heading, description in steps:
        row = ctk.CTkFrame(parent, fg_color=SURFACE_ALT, corner_radius=13)
        row.pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkLabel(
            row, text=number, width=28, height=28, corner_radius=9,
            fg_color=accent, text_color="#111722",
            font=ctk.CTkFont("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=(10, 10), pady=10)
        copy = ctk.CTkFrame(row, fg_color="transparent")
        copy.pack(side="left", fill="x", expand=True, pady=8)
        ctk.CTkLabel(
            copy, text=heading, font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=TEXT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            copy, text=description, font=ctk.CTkFont("Segoe UI", 8),
            text_color=MUTED, wraplength=325, justify="left",
        ).pack(anchor="w", pady=(1, 0))


def hotkey_bar(parent: ctk.CTkFrame, shortcuts: Iterable[tuple[str, str]], accent: str) -> None:
    bar = ctk.CTkFrame(parent, fg_color="#151D2A", corner_radius=13)
    bar.pack(fill="x", pady=(10, 0))
    ctk.CTkLabel(
        bar, text="ГОРЯЧИЕ КЛАВИШИ", font=ctk.CTkFont("Segoe UI", 8, "bold"), text_color=MUTED,
    ).pack(side="left", padx=(14, 12), pady=10)
    for key, description in shortcuts:
        ctk.CTkLabel(
            bar, text=key, width=30, height=22, corner_radius=7,
            fg_color="#2A3547", text_color=accent,
            font=ctk.CTkFont("Segoe UI", 8, "bold"),
        ).pack(side="left", pady=8)
        ctk.CTkLabel(
            bar, text=description, font=ctk.CTkFont("Segoe UI", 8), text_color=MUTED,
        ).pack(side="left", padx=(5, 14), pady=8)
