"""Tests for src/journey.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.journey import build_journey_tree, generate_mood_journey, journey_to_dataframe


def _make_journey_df(size: int = 30) -> pd.DataFrame:
    """Create a synthetic track dataframe that spans the mood space."""
    return pd.DataFrame(
        {
            "track_name": [f"Track {idx}" for idx in range(size)],
            "artist_name": [f"Artist {idx % 5}" for idx in range(size)],
            "album": [f"Album {idx % 4}" for idx in range(size)],
            "genre": ["pop" if idx % 2 == 0 else "rock" for idx in range(size)],
            "valence": np.linspace(0.05, 0.95, size),
            "energy": np.linspace(0.10, 0.90, size),
            "tempo": np.linspace(70.0, 170.0, size),
            "tempo_norm": np.linspace(0.10, 0.90, size),
            "acousticness": np.linspace(0.80, 0.10, size),
            "danceability": np.linspace(0.20, 0.95, size),
            "popularity": np.linspace(10.0, 90.0, size),
        }
    )


def test_build_journey_tree_accepts_valid_dataframe() -> None:
    df = _make_journey_df()
    tree = build_journey_tree(df)
    assert hasattr(tree, "query")


def test_generate_mood_journey_returns_requested_number_of_tracks() -> None:
    df = _make_journey_df()
    tree = build_journey_tree(df)
    journey = generate_mood_journey(tree, df, start=(0.10, 0.10), target=(0.90, 0.90), n_steps=12)
    assert len(journey) == 12


def test_generate_mood_journey_never_reuses_the_same_track() -> None:
    df = _make_journey_df()
    tree = build_journey_tree(df)
    journey = generate_mood_journey(tree, df, start=(0.10, 0.10), target=(0.90, 0.90), n_steps=15)
    names = [track["track_name"] for track in journey]
    assert len(names) == len(set(names))


def test_generate_mood_journey_caps_steps_to_dataset_size() -> None:
    df = _make_journey_df(size=5)
    tree = build_journey_tree(df)
    journey = generate_mood_journey(tree, df, start=(0.20, 0.20), target=(0.80, 0.80), n_steps=20)
    assert len(journey) == 5


def test_generate_mood_journey_moves_toward_target_mood() -> None:
    df = _make_journey_df()
    tree = build_journey_tree(df)
    journey = generate_mood_journey(tree, df, start=(0.10, 0.10), target=(0.90, 0.90), n_steps=10)
    assert journey[0]["valence"] <= journey[-1]["valence"]
    assert journey[0]["energy"] <= journey[-1]["energy"]


def test_journey_to_dataframe_preserves_track_count() -> None:
    df = _make_journey_df()
    tree = build_journey_tree(df)
    journey = generate_mood_journey(tree, df, start=(0.30, 0.30), target=(0.70, 0.70), n_steps=8)
    frame = journey_to_dataframe(journey)
    assert len(frame) == 8
    assert "transition_score" in frame.columns
