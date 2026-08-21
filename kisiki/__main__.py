"""Allow `python -m kisiki` during development."""

from .app import KisikiApp


def main() -> None:
    KisikiApp().mainloop()


if __name__ == "__main__":
    main()
