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
        (str(here / "assets" / "cats"), "assets/cats"),
        (str(here / "sounds"), "sounds"),
        (str(here / "assets" / "food"), "assets/food"),
        # Иконки девяти видов руды: их показывает статистика ORE HUNT.
        (str(here / "assets" / "ores"), "assets/ores"),
        (str(here / "assets" / "vision"), "assets/vision"),
        (str(here / "orange_cat.ico"), "."),
    ],
    hiddenimports=["customtkinter"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
# Кодеки видео тянут за собой 12 МБ, а приложение только снимает экран и
# разбирает кадры: ни VideoCapture, ни imshow в коде нет.
a.binaries = [entry for entry in a.binaries if "opencv_videoio_ffmpeg" not in entry[0]]

# .DS_Store — служебные файлы macOS. Своих у нас нет, эти два приезжают из
# customtkinter вместе с его иконками: пакет собран на маке, и хук тащит
# папку assets целиком. Для Windows-сборки это мусор.
a.datas = [entry for entry in a.datas if not entry[0].endswith(".DS_Store")]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
    name="Kisikisimyaumyau",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
    console=False,
    icon=str(here / "orange_cat.ico"),
)
