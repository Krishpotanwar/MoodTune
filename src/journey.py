"""
journey.py -- Generate mood-transition playlists in valence-energy space.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLEAN_DATA_PATH, DEFAULT_JOURNEY_CANDIDATES, DEFAULT_JOURNEY_STEPS
from src.config import DEFAULT_TEMPO_WEIGHT, SAMPLE_DATA_PATH

MOOD_FEATURES: list[str] = ["valence", "energy"]
_TRACK_COLUMNS: list[str] = [
    "track_name",
    "artist_name",
    "album",
    "genre",
    "valence",
    "energy",
    "tempo",
    "tempo_norm",
    "acousticness",
    "danceability",
    "popularity",
    "track_id",
]


class JourneyTrack(TypedDict):
    """Serializable track payload used by the UI and tests."""

    track_name: str
    artist_name: str
    album: str
    genre: str
    valence: float
    energy: float
    tempo: float
    tempo_norm: float
    acousticness: float
    danceability: float
    popularity: float
    track_id: str
    waypoint_idx: int
    mood_distance: float
    transition_score: float


def _coerce_coordinate(point: tuple[float, float]) -> np.ndarray:
    """Clamp one (valence, energy) coordinate into the valid unit square."""
    return np.clip(np.asarray(point, dtype=float), 0.0, 1.0)


def _query_candidates(
    tree: KDTree,
    waypoint: np.ndarray,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return array-shaped nearest-neighbour results for one waypoint."""
    distances, indices = tree.query(waypoint, k=max(candidate_count, 1))
    return np.atleast_1d(distances), np.atleast_1d(indices)


def _tempo_norm_for_row(row: pd.Series) -> float:
    """Read a row's normalized tempo, falling back to raw BPM if needed."""
    if pd.notna(row.get("tempo_norm")):
        return float(row["tempo_norm"])
    tempo = float(row.get("tempo", 125.0))
    return float(np.clip(tempo / 250.0, 0.0, 1.0))


def _score_candidate(
    row: pd.Series,
    mood_distance: float,
    prev_tempo_norm: float | None,
    bpm_weight: float,
) -> float:
    """Blend mood accuracy with transition smoothness."""
    if prev_tempo_norm is None:
        return mood_distance
    tempo_jump = abs(_tempo_norm_for_row(row) - prev_tempo_norm)
    return mood_distance + (bpm_weight * tempo_jump)


def _build_track_payload(
    row: pd.Series,
    waypoint_idx: int,
    mood_distance: float,
    transition_score: float,
) -> JourneyTrack:
    """Convert a dataframe row into a journey track dict."""
    payload = {column: row.get(column, "") for column in _TRACK_COLUMNS}
    payload["track_name"] = str(payload["track_name"])
    payload["artist_name"] = str(payload["artist_name"])
    payload["album"] = str(payload["album"] or "")
    payload["genre"] = str(payload["genre"] or "")
    payload["track_id"] = str(payload["track_id"] or "")
    payload["valence"] = float(payload.get("valence", 0.0))
    payload["energy"] = float(payload.get("energy", 0.0))
    payload["tempo"] = float(payload.get("tempo", 0.0) or 0.0)
    payload["tempo_norm"] = float(_tempo_norm_for_row(row))
    payload["acousticness"] = float(payload.get("acousticness", 0.0) or 0.0)
    payload["danceability"] = float(payload.get("danceability", 0.0) or 0.0)
    payload["popularity"] = float(payload.get("popularity", 0.0) or 0.0)
    payload["waypoint_idx"] = waypoint_idx
    payload["mood_distance"] = mood_distance
    payload["transition_score"] = transition_score
    return payload  # type: ignore[return-value]


def build_journey_tree(df: pd.DataFrame) -> KDTree:
    """
    Build a KDTree over valence and energy for fast journey lookup.

    Args:
        df: Track dataframe with valence and energy columns.

    Returns:
        SciPy KDTree built on the dataframe's mood coordinates.
    """
    missing = [column for column in MOOD_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Journey dataframe missing columns: {missing}")
    return KDTree(df[MOOD_FEATURES].astype(float).to_numpy())


def generate_mood_journey(
    tree: KDTree,
    df: pd.DataFrame,
    start: tuple[float, float],
    target: tuple[float, float],
    n_steps: int = DEFAULT_JOURNEY_STEPS,
    candidates_per_step: int = DEFAULT_JOURNEY_CANDIDATES,
    bpm_weight: float = DEFAULT_TEMPO_WEIGHT,
) -> list[JourneyTrack]:
    """
    Generate an ordered playlist that drifts from start mood to target mood.

    Args:
        tree: KDTree built from the same dataframe.
        df: Track dataframe.
        start: Starting (valence, energy).
        target: Target (valence, energy).
        n_steps: Desired journey length.
        candidates_per_step: Number of nearby tracks to inspect per waypoint.
        bpm_weight: Penalty weight for large tempo jumps.

    Returns:
        Ordered list of journey tracks. Returns an empty list for empty data.
    """
    if df.empty or n_steps <= 0:
        return []

    actual_steps = min(int(n_steps), len(df))
    query_count = min(len(df), max(candidates_per_step * 3, 1))
    waypoints = np.linspace(_coerce_coordinate(start), _coerce_coordinate(target), actual_steps)

    used_indices: set[int] = set()
    journey: list[JourneyTrack] = []
    prev_tempo_norm: float | None = None

    for waypoint_idx, waypoint in enumerate(waypoints):
        distances, indices = _query_candidates(tree, waypoint, query_count)
        best_idx: int | None = None
        best_distance = 0.0
        best_score = float("inf")

        for mood_distance, candidate_idx in zip(distances, indices):
            if int(candidate_idx) in used_indices:
                continue
            row = df.iloc[int(candidate_idx)]
            transition_score = _score_candidate(row, float(mood_distance), prev_tempo_norm, bpm_weight)
            if transition_score < best_score:
                best_idx = int(candidate_idx)
                best_distance = float(mood_distance)
                best_score = transition_score

        if best_idx is None:
            continue

        used_indices.add(best_idx)
        chosen_row = df.iloc[best_idx]
        prev_tempo_norm = _tempo_norm_for_row(chosen_row)
        journey.append(
            _build_track_payload(
                chosen_row,
                waypoint_idx=waypoint_idx,
                mood_distance=best_distance,
                transition_score=best_score,
            )
        )

    return journey


def journey_to_dataframe(journey: list[JourneyTrack]) -> pd.DataFrame:
    """Convert a journey payload list into a dataframe for display."""
    return pd.DataFrame(journey)


def _load_demo_dataframe() -> pd.DataFrame:
    """Load whichever cleaned dataset is available for standalone checks."""
    for path in (CLEAN_DATA_PATH, SAMPLE_DATA_PATH):
        if Path(path).exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No cleaned or sample dataset found for journey verification.")


if __name__ == "__main__":
    print("MoodTune -- Journey Generator Verification")
    print("=" * 44)
    dataset = _load_demo_dataframe()
    tree = build_journey_tree(dataset)
    journey = generate_mood_journey(tree, dataset, start=(0.15, 0.20), target=(0.82, 0.86))
    print(f"Dataset rows: {len(dataset):,}")
    print(f"Journey size: {len(journey)}")
    preview = journey_to_dataframe(journey)[["track_name", "artist_name", "genre", "valence", "energy"]]
    print(preview.head(10).to_string(index=False))
