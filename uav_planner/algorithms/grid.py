"""Compatibility exports for grid-search planners.

New algorithm changes should be made in each algorithm package:
``dijkstra/``, ``astar/`` and ``dfs/``.
"""

from __future__ import annotations

from .astar import astar
from .dfs import dfs
from .dijkstra import dijkstra

__all__ = ["dijkstra", "astar", "dfs"]
