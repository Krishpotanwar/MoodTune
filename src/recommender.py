"""
recommender.py — NearestNeighbors cosine-similarity recommendation engine.

Loads the cleaned CSV produced by validator.py, fits a cosine model on the
four audio features, and returns the top-k closest tracks for any target vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ── constants ──────────────────────────────────────────────────────────────────

FEATURE_COLS: list[str] = ["energy", "valence", "tempo_norm", "acousticness"]
DEFAULT_TOP_N: int = 8
DEFAULT_CLEAN_CSV: Path = Path("data/processed/spotify_tracks_clean.csv")

# Output columns returned by recommend() (album added if present in data)
_BASE_OUTPUT_COLS: list[str] = [
    "track_name", "artist_name", "genre",
    *FEATURE_COLS,
    "similarity_pct",
]


# ── model ──────────────────────────────────────────────────────────────────────

def build_model(df: pd.DataFrame) -> NearestNeighbors:
    """
    Fit a NearestNeighbors model on the four audio feature columns.

    Args:
        df: DataFrame that must contain all columns in FEATURE_COLS.

    Returns:
        Fitted sklearn NearestNeighbors (cosine, brute-force).
    """
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(df[FEATURE_COLS].values.astype(float))
    return model


# ── recommendation ─────────────────────────────────────────────────────────────

def recommend(
    model: NearestNeighbors,
    df: pd.DataFrame,
    target_vector: list[float],
    k: int = DEFAULT_TOP_N,
    genre_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return the top-k tracks most similar to target_vector.

    Args:
        model:         Fitted NearestNeighbors (fitted on df with no genre filter).
        df:            Cleaned DataFrame matching the model.
        target_vector: [energy, valence, tempo_norm, acousticness], each in [0, 1].
        k:             Number of results to return.
        genre_filter:  If non-empty, restrict the search space to those genres
                       (re-fits a fresh model on the filtered subset).

    Returns:
        DataFrame sorted by similarity_pct (descending) with columns:
          track_name, artist_name, album (if present), genre,
          energy, valence, tempo_norm, acousticness, similarity_pct.
        Returns an empty DataFrame if genre_filter yields no rows.
    """
    if genre_filter:
        subset = df[df["genre"].isin(genre_filter)].copy().reset_index(drop=True)
        if subset.empty:
            return pd.DataFrame(columns=_BASE_OUTPUT_COLS)
        active_model = build_model(subset)
        query_df = subset
    else:
        active_model = model
        query_df = df

    actual_k = min(k, len(query_df))
    vec = np.array(target_vector, dtype=float).reshape(1, -1)
    distances, indices = active_model.kneighbors(vec, n_neighbors=actual_k)

    hits = query_df.iloc[indices[0]].copy().reset_index(drop=True)
    hits["similarity_pct"] = ((1.0 - distances[0]) * 100).round(1)

    output_cols = list(_BASE_OUTPUT_COLS)
    if "album" in hits.columns:
        output_cols.insert(2, "album")

    return (
        hits[[c for c in output_cols if c in hits.columns]]
        .sort_values("similarity_pct", ascending=False)
        .reset_index(drop=True)
    )


# ── standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parent.parent
    clean_path = project_root / "data" / "processed" / "SpotifyFeatures_clean.csv"

    if not clean_path.exists():
        print(f"ERROR: {clean_path} not found. Run src/validator.py first.")
        sys.exit(1)

    print("MoodTune — Recommender Verification")
    print("=" * 50)

    df = pd.read_csv(clean_path)
    model = build_model(df)
    print(f"Model fitted on {len(df):,} tracks  |  genres: {df['genre'].nunique()}")

    # Happy-energetic vector (Joyful quadrant)
    target = [0.90, 0.90, 0.90, 0.10]
    results = recommend(model, df, target, k=8)
    print(f"\nTop 8 for vector {target}:")
    print(results[["track_name", "artist_name", "genre", "similarity_pct"]].to_string(index=False))

    # Genre filter test
    genres = list(df["genre"].value_counts().head(2).index)
    filtered = recommend(model, df, target, k=5, genre_filter=genres)
    print(f"\nWith genre filter {genres}:")
    print(filtered[["track_name", "artist_name", "genre", "similarity_pct"]].to_string(index=False))
