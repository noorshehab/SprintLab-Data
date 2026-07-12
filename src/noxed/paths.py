"""Environment-resolved paths shared by every notebook and the Modal executor."""
import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "./data"))


def outcomes_dir() -> Path:
    d = Path(os.environ.get("OUTCOMES_DIR", "./outcomes"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir() -> Path:
    d = outcomes_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    """Large, reproducible intermediates shared across notebooks (e.g. the
    canonical-events table) -- distinct from outcomes/, which holds only
    small, curated, git-tracked artifacts."""
    d = Path(os.environ.get("CACHE_DIR", "./cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d
