"""Tests for src/recommender.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.recommender import FEATURE_COLS, build_model, recommend

# ── fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """50-row synthetic dataset with two genres."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "track_name":   [f"Track {i}"   for i in range(n)],
        "artist_name":  [f"Artist {i % 10}" for i in range(n)],
        "genre":        (["pop"] * 25) + (["rock"] * 25),
        "energy":       rng.uniform(0.0, 1.0, n),
        "valence":      rng.uniform(0.0, 1.0, n),
        "tempo_norm":   rng.uniform(0.0, 1.0, n),
        "acousticness": rng.uniform(0.0, 1.0, n),
    })


_TARGET = [0.8, 0.7, 0.6, 0.2]


# ── build_model ────────────────────────────────────────────────────────────────

def test_build_model_returns_fitted_object(sample_df: pd.DataFrame) -> None:
    from sklearn.neighbors import NearestNeighbors
    model = build_model(sample_df)
    assert isinstance(model, NearestNeighbors)


# ── result shape ───────────────────────────────────────────────────────────────

def test_returns_exactly_k_results(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET, k=8)
    assert len(results) == 8


def test_returns_fewer_than_k_when_dataset_is_smaller(sample_df: pd.DataFrame) -> None:
    tiny = sample_df.head(5).copy()
    model = build_model(tiny)
    results = recommend(model, tiny, _TARGET, k=8)
    assert len(results) == 5


def test_result_contains_required_columns(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET)
    for col in ("track_name", "artist_name", "genre", "similarity_pct"):
        assert col in results.columns


def test_all_feature_columns_present(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET)
    for col in FEATURE_COLS:
        assert col in results.columns


# ── similarity_pct ─────────────────────────────────────────────────────────────

def test_similarity_pct_within_0_100(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET)
    assert results["similarity_pct"].between(0.0, 100.0).all()


def test_results_sorted_descending_by_similarity(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET)
    assert results["similarity_pct"].is_monotonic_decreasing


# ── genre filter ───────────────────────────────────────────────────────────────

def test_genre_filter_restricts_results_to_genre(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET, k=8, genre_filter=["pop"])
    assert (results["genre"] == "pop").all()


def test_genre_filter_rock_only(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET, k=8, genre_filter=["rock"])
    assert (results["genre"] == "rock").all()


def test_empty_genre_filter_returns_multiple_genres(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET, k=8, genre_filter=[])
    assert results["genre"].nunique() > 1


def test_none_genre_filter_returns_multiple_genres(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET, k=8, genre_filter=None)
    assert results["genre"].nunique() > 1


def test_unknown_genre_returns_empty_dataframe(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET, genre_filter=["nonexistent_genre_xyz"])
    assert results.empty


def test_multi_genre_filter_returns_both_genres(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET, k=8, genre_filter=["pop", "rock"])
    assert set(results["genre"].unique()) <= {"pop", "rock"}


# ── edge cases ─────────────────────────────────────────────────────────────────

def test_identical_target_and_track(sample_df: pd.DataFrame) -> None:
    """Inserting a track identical to the target should get similarity_pct ≈ 100."""
    perfect_row = pd.DataFrame([{
        "track_name":   "Perfect Match",
        "artist_name":  "Test Artist",
        "genre":        "pop",
        "energy":       _TARGET[0],
        "valence":      _TARGET[1],
        "tempo_norm":   _TARGET[2],
        "acousticness": _TARGET[3],
    }])
    df_with_perfect = pd.concat([sample_df, perfect_row], ignore_index=True)
    model = build_model(df_with_perfect)
    results = recommend(model, df_with_perfect, _TARGET, k=1)
    assert results.iloc[0]["track_name"] == "Perfect Match"
    assert results.iloc[0]["similarity_pct"] > 99.0


def test_k_defaults_to_eight(sample_df: pd.DataFrame) -> None:
    model = build_model(sample_df)
    results = recommend(model, sample_df, _TARGET)
    assert len(results) == 8
