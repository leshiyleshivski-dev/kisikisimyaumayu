# PyInstaller build recipe for the single-file Windows application.
from pathlib import Path

from PyInstaller.building.build_main import Analysis, EXE, PYZ

block_cipher = None
here = Path(SPECPATH)

a = Analysis(
    [str(here / "kiski_clicker.pyw")],
    pathex=[str(here)],
    binaries=[],
    datas=[
        (str(here / "kiski kartinki"), "kiski kartinki"),
        (str(here / "sounds"), "sounds"),
        (str(here / "orange_cat.ico"), "."),
    ],
    hiddenimports=["customtkinter"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
    name="Кисикисимяумяу",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
    console=False,
    icon=str(here / "orange_cat.ico"),
)
