"""
config.py -- Shared project paths and UI-safe constants.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "spotify_tracks.csv"
CLEAN_DATA_PATH: Path = DATA_DIR / "processed" / "spotify_tracks_clean.csv"
SAMPLE_DATA_PATH: Path = DATA_DIR / "sample" / "spotify_sample.csv"
MOOD_LEXICON_PATH: Path = DATA_DIR / "mood_lexicon.json"
LOG_PATH: Path = PROJECT_ROOT / "logs" / "validation_summary.txt"
STYLE_PATH: Path = PROJECT_ROOT / "ui" / "styles.css"

DEFAULT_SCATTER_SAMPLE: int = 5_000
DEFAULT_JOURNEY_STEPS: int = 18
DEFAULT_JOURNEY_CANDIDATES: int = 15
DEFAULT_TEMPO_WEIGHT: float = 0.30
DEFAULT_PLAYLIST_SIZE: int = 8
MOOD_AXIS_MIN: float = 0.0
MOOD_AXIS_MAX: float = 1.0
