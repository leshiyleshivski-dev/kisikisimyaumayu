"""Compatibility entry point for the KisikiSimyaumyau desktop app.

Implementation lives in the `kisiki` package. Public Volt symbols are
re-exported so existing tools and tests importing this historical `.pyw` file
continue to work.
"""

from kisiki.app import KisikiApp
from kisiki.core import (
    PORT_DOWN, PORT_LEFT, PORT_RIGHT, PORT_UP,
    VK_A, VK_D, VK_S, VK_W,
)
from kisiki.modules import (
    BuilderModule, ElectricianModule, PhoneModule, RouletteModule,
)

__all__ = [
    "KisikiApp", "RouletteModule", "PhoneModule", "BuilderModule",
    "ElectricianModule", "PORT_UP", "PORT_RIGHT", "PORT_DOWN", "PORT_LEFT",
    "VK_W", "VK_A", "VK_S", "VK_D",
]


if __name__ == "__main__":
    KisikiApp().mainloop()
