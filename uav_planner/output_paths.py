"""Output paths shared by visual artifacts from one interactive run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")


def sanitize_output_component(value, fallback="run"):
    """Return a Windows-safe directory-name component."""
    component = str(value or "").strip().replace("*", "Star")
    component = _INVALID_PATH_CHARS.sub("_", component)
    component = _WHITESPACE.sub("_", component).strip(" ._")
    return component or fallback


def difficulty_name_for_size(map_size):
    """Return a stable fallback difficulty name for programmatic callers."""
    return {30: "simple", 60: "medium", 90: "hard"}.get(
        int(map_size),
        f"{int(map_size)}x{int(map_size)}",
    )


@dataclass(frozen=True)
class RunOutputPaths:
    """All visual artifact paths belonging to one planning run."""

    directory: Path
    timestamp: str

    @property
    def static_result(self):
        return self.directory / f"uav_path_result_{self.timestamp}.png"

    @property
    def map_layers(self):
        return self.directory / f"uav_map_layers_{self.timestamp}.png"

    @property
    def animation(self):
        return self.directory / f"uav_path_animation_{self.timestamp}.gif"


def create_run_output_paths(algorithm_name, difficulty_name, timestamp=None, outputs_root=None):
    """Create one algorithm+difficulty+timestamp output directory."""
    timestamp = sanitize_output_component(
        timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
        fallback="timestamp",
    )
    algorithm = sanitize_output_component(algorithm_name, fallback="algorithm")
    difficulty = sanitize_output_component(difficulty_name, fallback="difficulty")
    root = Path(outputs_root) if outputs_root is not None else OUTPUTS_ROOT
    root.mkdir(parents=True, exist_ok=True)

    base_name = f"{algorithm}+{difficulty}+{timestamp}"
    suffix = 1
    while True:
        directory_name = base_name if suffix == 1 else f"{base_name}_{suffix}"
        directory = root / directory_name
        try:
            directory.mkdir()
            break
        except FileExistsError:
            suffix += 1

    return RunOutputPaths(directory=directory, timestamp=timestamp)
