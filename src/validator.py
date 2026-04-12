"""
validator.py — Data cleaning pipeline for SpotifyFeatures.csv.

Input:  data/raw/SpotifyFeatures.csv  (~232k rows, Kaggle)
Output: data/processed/SpotifyFeatures_clean.csv
        logs/validation_summary.txt

Run standalone:
    python src/validator.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────────

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "energy":       (0.0, 1.0),
    "valence":      (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "tempo":        (0.0, 250.0),
}

REQUIRED_COLUMNS: list[str] = [
    "track_name",
    "artist_name",   # normalised from "artists" (maharshipandya dataset)
    "genre",         # normalised from "track_genre" (maharshipandya dataset)
    "energy",
    "valence",
    "tempo",
    "acousticness",
]

# Column renames applied immediately after loading.
# Supports both the legacy zaheenhamidani dataset and the newer maharshipandya dataset.
_COLUMN_NORMALISE: dict[str, str] = {
    "artists":     "artist_name",
    "track_genre": "genre",
    "album_name":  "album",
}

# Module-level step log populated by run_pipeline()
_steps_log: list[dict[str, Any]] = []


# ── private helpers ────────────────────────────────────────────────────────────

def _record_step(step: int, name: str, before: int, after: int, detail: str = "") -> None:
    """Append one cleaning-step record to the module-level log."""
    _steps_log.append(
        {
            "step":    step,
            "name":    name,
            "before":  before,
            "after":   after,
            "removed": before - after,
            "detail":  detail,
        }
    )


def _load_raw(raw_path: Path) -> pd.DataFrame:
    """Load CSV and verify all required columns are present."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {raw_path}\n\n"
            "Download SpotifyFeatures.csv from Kaggle:\n"
            "  https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db\n"
            "Place it at:  data/raw/SpotifyFeatures.csv"
        )
    df = pd.read_csv(raw_path)
    # Normalise column names so both dataset variants work transparently.
    df = df.rename(columns=_COLUMN_NORMALISE)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from CSV: {missing}\n"
            f"Detected columns: {list(df.columns)}"
        )
    return df


def _step_remove_duplicates(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Drop rows that are exact duplicates on (track_name, artist_name)."""
    before = len(df)
    df = df.drop_duplicates(subset=["track_name", "artist_name"], keep="first")
    _record_step(step, "Remove duplicates (track_name + artist_name)", before, len(df))
    return df.reset_index(drop=True)


def _step_drop_null_features(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Drop any row where a feature column is null."""
    before = len(df)
    df = df.dropna(subset=list(FEATURE_RANGES.keys()))
    _record_step(step, "Drop rows with null feature values", before, len(df))
    return df.reset_index(drop=True)


def _step_clip_ranges(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Clip each feature column to its valid range; log how many values were adjusted."""
    clip_counts: dict[str, int] = {}
    df = df.copy()
    for col, (lo, hi) in FEATURE_RANGES.items():
        out_of_range = int(((df[col] < lo) | (df[col] > hi)).sum())
        clip_counts[col] = out_of_range
        df[col] = df[col].clip(lo, hi)
    detail = ", ".join(f"{col}×{cnt}" for col, cnt in clip_counts.items())
    _record_step(step, "Clip out-of-range feature values", len(df), len(df), detail)
    return df


def _step_normalise_tempo(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Min-max normalise tempo → tempo_norm in [0, 1]."""
    df = df.copy()
    t_min: float = float(df["tempo"].min())
    t_max: float = float(df["tempo"].max())
    denominator = (t_max - t_min) if t_max != t_min else 1.0
    df["tempo_norm"] = (df["tempo"] - t_min) / denominator
    detail = f"range {t_min:.2f}–{t_max:.2f} BPM → [0, 1]"
    _record_step(step, "Normalise tempo → tempo_norm", len(df), len(df), detail)
    return df


def _step_strip_whitespace(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Strip leading/trailing whitespace from string columns."""
    df = df.copy()
    for col in ["track_name", "artist_name", "genre"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    _record_step(step, "Strip whitespace from string columns", len(df), len(df))
    return df


def _write_summary(
    log_path: Path,
    raw_rows: int,
    clean_rows: int,
    genres: list[str],
) -> None:
    """Write a human-readable validation summary to disk."""
    retained_pct = (clean_rows / raw_rows * 100) if raw_rows > 0 else 0.0
    lines: list[str] = [
        "=== MoodTune Validation Report ===",
        f"Run at             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Raw rows           : {raw_rows:,}",
        "",
    ]
    for entry in _steps_log:
        prefix = f"  Step {entry['step']}: {entry['name']}"
        if entry["removed"] > 0:
            lines.append(f"{prefix}  →  removed {entry['removed']:,}")
        else:
            lines.append(prefix)
        if entry["detail"]:
            lines.append(f"         └─ {entry['detail']}")
    lines += [
        "",
        f"Clean rows         : {clean_rows:,}  ({retained_pct:.2f}% retained)",
        f"Genres found       : {len(genres)}",
        f"Genre list         : {', '.join(genres)}",
        "=================================",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")


# ── public API ─────────────────────────────────────────────────────────────────

def run_pipeline(
    raw_path: str | Path = "data/raw/spotify_tracks.csv",
    output_path: str | Path = "data/processed/spotify_tracks_clean.csv",
    log_path: str | Path = "logs/validation_summary.txt",
) -> pd.DataFrame:
    """
    Run the full data-cleaning pipeline on SpotifyFeatures.csv.

    Steps:
        1. Load raw CSV
        2. Remove duplicate tracks
        3. Drop rows with null feature values
        4. Clip feature values to valid ranges
        5. Normalise tempo → tempo_norm
        6. Strip whitespace from string columns

    Args:
        raw_path:    Path to the raw CSV. Raises FileNotFoundError if absent.
        output_path: Destination path for the cleaned CSV.
        log_path:    Destination path for the validation summary text file.

    Returns:
        Cleaned DataFrame (also written to output_path).

    Raises:
        FileNotFoundError: raw_path does not exist.
        ValueError: required columns are missing.
    """
    global _steps_log
    _steps_log = []

    raw_path    = Path(raw_path)
    output_path = Path(output_path)
    log_path    = Path(log_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_raw(raw_path)
    raw_rows = len(df)
    _record_step(1, "Load raw CSV", raw_rows, raw_rows, f"{len(df.columns)} columns")

    df = _step_remove_duplicates(df, step=2)
    df = _step_drop_null_features(df, step=3)
    df = _step_clip_ranges(df, step=4)
    df = _step_normalise_tempo(df, step=5)
    df = _step_strip_whitespace(df, step=6)

    df.to_csv(output_path, index=False)

    genres = sorted(df["genre"].dropna().unique().tolist())
    _write_summary(log_path, raw_rows, len(df), genres)

    return df


def get_cleaning_steps_log() -> list[dict[str, Any]]:
    """
    Return the step-by-step log from the most recent run_pipeline() call.

    Used by the Data Lab tab in the Streamlit UI to render the live demo.

    Returns:
        List of step dicts with keys: step, name, before, after, removed, detail.
        Empty list if run_pipeline() has not been called yet.
    """
    return list(_steps_log)


# ── standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parent.parent
    raw   = project_root / "data" / "raw"   / "spotify_tracks.csv"
    clean = project_root / "data" / "processed" / "spotify_tracks_clean.csv"
    log   = project_root / "logs" / "validation_summary.txt"

    print("MoodTune — Data Cleaning Pipeline")
    print("=" * 40)

    try:
        df = run_pipeline(raw, clean, log)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print("\nStep log:")
    for s in get_cleaning_steps_log():
        removed_str = f"  →  -{s['removed']:,}" if s["removed"] > 0 else ""
        print(f"  [{s['step']}] {s['name']}{removed_str}")
        if s["detail"]:
            print(f"       {s['detail']}")

    print(f"\n✓ Clean rows : {len(df):,}")
    print(f"  Saved to   : {clean}")
    print(f"  Log at     : {log}")
