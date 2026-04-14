"""Tests for src/validator.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import MAX_TEMPO_BPM
from src.validator import get_cleaning_steps_log, run_pipeline

# ── fixtures ───────────────────────────────────────────────────────────────────

def _make_csv(tmp_path: Path, **overrides) -> Path:
    """Write a minimal valid SpotifyFeatures CSV and return its path."""
    base = {
        "track_name":   ["Song A", "Song B", "Song C"],
        "artist_name":  ["Artist 1", "Artist 2", "Artist 3"],
        "genre":        ["pop", "rock", "jazz"],
        "energy":       [0.80, 0.50, 0.30],
        "valence":      [0.70, 0.40, 0.60],
        "tempo":        [120.0, 90.0, 60.0],
        "acousticness": [0.10, 0.50, 0.90],
    }
    base.update(overrides)
    path = tmp_path / "raw.csv"
    pd.DataFrame(base).to_csv(path, index=False)
    return path


# ── FileNotFoundError ──────────────────────────────────────────────────────────

def test_missing_csv_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        run_pipeline(
            raw_path=tmp_path / "nonexistent.csv",
            output_path=tmp_path / "out.csv",
            log_path=tmp_path / "log.txt",
        )


# ── output files ───────────────────────────────────────────────────────────────

def test_output_csv_and_log_are_created(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path)
    out = tmp_path / "out.csv"
    log = tmp_path / "log.txt"
    run_pipeline(raw, out, log)
    assert out.exists()
    assert log.exists()


def test_log_starts_with_header(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path)
    log = tmp_path / "log.txt"
    run_pipeline(raw, tmp_path / "out.csv", log)
    assert log.read_text().startswith("=== MoodTune Validation Report ===")


# ── duplicate removal ──────────────────────────────────────────────────────────

def test_duplicates_on_track_and_artist_are_removed(tmp_path: Path) -> None:
    raw = _make_csv(
        tmp_path,
        track_name=["Song A", "Song A", "Song B"],
        artist_name=["Artist 1", "Artist 1", "Artist 2"],
        genre=["pop", "pop", "rock"],
        energy=[0.8, 0.8, 0.5],
        valence=[0.7, 0.7, 0.4],
        tempo=[120.0, 120.0, 90.0],
        acousticness=[0.1, 0.1, 0.5],
    )
    result = run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    assert len(result) == 2


def test_different_artist_same_track_kept(tmp_path: Path) -> None:
    raw = _make_csv(
        tmp_path,
        track_name=["Song A", "Song A", "Song B"],
        artist_name=["Artist 1", "Artist 2", "Artist 3"],
    )
    result = run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    assert len(result) == 3


# ── null row dropping ──────────────────────────────────────────────────────────

def test_null_energy_row_is_dropped(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path, energy=[0.8, None, 0.3])
    result = run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    assert result["energy"].isna().sum() == 0
    assert len(result) == 2


def test_null_valence_row_is_dropped(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path, valence=[0.7, None, 0.6])
    result = run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    assert len(result) == 2


# ── range clipping ─────────────────────────────────────────────────────────────

def test_energy_above_1_is_clipped(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path, energy=[1.5, 0.5, 0.3])
    result = run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    assert result["energy"].max() <= 1.0


def test_energy_below_0_is_clipped(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path, energy=[-0.2, 0.5, 0.3])
    result = run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    assert result["energy"].min() >= 0.0


def test_tempo_above_250_is_clipped(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path, tempo=[300.0, 90.0, 60.0])
    result = run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    assert result["tempo"].max() <= MAX_TEMPO_BPM


def test_tempo_below_0_is_clipped(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path, tempo=[-10.0, 90.0, 60.0])
    result = run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    assert result["tempo"].min() >= 0.0


# ── tempo normalisation ────────────────────────────────────────────────────────

def test_tempo_norm_column_exists(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path)
    result = run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    assert "tempo_norm" in result.columns


def test_tempo_norm_within_unit_interval(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path, tempo=[60.0, 120.0, 180.0])
    result = run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    assert result["tempo_norm"].min() >= 0.0
    assert result["tempo_norm"].max() <= 1.0


def test_tempo_norm_min_is_zero_max_is_one(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path, tempo=[60.0, 120.0, 180.0])
    result = run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    assert abs(result["tempo_norm"].min() - 0.0) < 1e-9
    assert abs(result["tempo_norm"].max() - 1.0) < 1e-9


# ── step log structure ─────────────────────────────────────────────────────────

def test_step_log_has_six_entries(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path)
    run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    steps = get_cleaning_steps_log()
    assert len(steps) == 6


def test_step_log_has_required_keys(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path)
    run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    for entry in get_cleaning_steps_log():
        assert "step"    in entry
        assert "name"    in entry
        assert "before"  in entry
        assert "after"   in entry
        assert "removed" in entry


def test_step_log_removed_equals_before_minus_after(tmp_path: Path) -> None:
    raw = _make_csv(tmp_path)
    run_pipeline(raw, tmp_path / "out.csv", tmp_path / "log.txt")
    for entry in get_cleaning_steps_log():
        assert entry["removed"] == entry["before"] - entry["after"]
