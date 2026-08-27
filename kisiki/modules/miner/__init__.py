"""Помощник карьера ORE HUNT: сам бьёт камень и сам собирает руду со стола."""

from .module import MinerModule
from .stats import OreTally, fresh_daily_stats

__all__ = ["MinerModule", "OreTally", "fresh_daily_stats"]
