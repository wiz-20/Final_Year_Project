"""Project paths: simulation code lives in sim/, figures in output/figures/."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"


def figure_path(filename: str) -> Path:
    """Absolute path for a plot PNG under output/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename
