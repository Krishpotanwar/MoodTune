"""
data_loader.py — Full dataset loader with HuggingFace Hub fallback.

Priority order:
  1. data/processed/spotify_tracks_clean.csv  (local dev, fastest)
  2. data/raw/spotify_tracks.csv              (local dev, run validator inline)
  3. HuggingFace Hub download                 (Streamlit Cloud)
  4. data/sample/spotify_sample.csv           (last resort, 200 tracks with warning)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLEAN_DATA_PATH,
    HF_DATASET_SHA256_ALLOWLIST,
    MAX_TEMPO_BPM,
    RAW_DATA_PATH,
    SAMPLE_DATA_PATH,
)

# ── HuggingFace dataset details ────────────────────────────────────────────────

_HF_REPO_ID  = "maharshipandya/spotify-tracks-dataset"
_HF_FILENAME = "dataset.csv"

# Mapping from HuggingFace column names → app schema
_HF_COLUMN_MAP: dict[str, str] = {
    "track_name":  "track_name",
    "artists":     "artist_name",
    "track_genre": "genre",
    "popularity":  "popularity",
}

# Columns required by the app — all must be present after normalization
_REQUIRED_COLS: list[str] = [
    "track_name",
    "artist_name",
    "genre",
    "valence",
    "energy",
    "danceability",
    "acousticness",
    "tempo",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _normalise_hf_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename HuggingFace columns to match the app schema."""
    df = df.rename(columns=_HF_COLUMN_MAP)

    # Normalise tempo to [0,1] if it's in BPM range (typical: 0–250)
    if "tempo" in df.columns and df["tempo"].max() > 1.5:
        df = df.copy()
        df["tempo_norm"] = df["tempo"].clip(0, MAX_TEMPO_BPM) / MAX_TEMPO_BPM
    elif "tempo_norm" not in df.columns and "tempo" in df.columns:
        df = df.copy()
        df["tempo_norm"] = df["tempo"]

    # Drop rows missing any required column
    present = [c for c in _REQUIRED_COLS if c in df.columns]
    df = df.dropna(subset=present)

    return df.reset_index(drop=True)


def _validate_schema(df: pd.DataFrame, source: str) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[data_loader] {source}: missing columns {missing}")


def _sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 digest for a local file path."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ── main loader ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=None, show_spinner=False)
def load_full_dataset() -> pd.DataFrame:
    """
    Load the full 114k-track Spotify dataset.

    Uses @st.cache_data so the download only happens once per container
    lifecycle — fast on repeat visits.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with at minimum: track_name, artist_name, genre,
        valence, energy, danceability, acousticness, tempo, tempo_norm.
    """

    # 1. Local clean CSV -------------------------------------------------------
    if CLEAN_DATA_PATH.exists():
        df = pd.read_csv(CLEAN_DATA_PATH)
        _validate_schema(df, "local clean CSV")
        return df

    # 2. Local raw CSV → run validator ----------------------------------------
    if RAW_DATA_PATH.exists():
        with st.spinner("Processing raw dataset…"):
            try:
                from src.validator import run_pipeline  # local import to avoid circular dep
                df = run_pipeline(
                    raw_path=RAW_DATA_PATH,
                    output_path=CLEAN_DATA_PATH,
                )
                _validate_schema(df, "validator pipeline")
                return df
            except Exception as exc:
                st.warning(f"Validator failed ({exc}), falling through to HuggingFace.")

    # 3. HuggingFace Hub download ---------------------------------------------
    with st.spinner("Loading 114,000 tracks from HuggingFace Hub…"):
        try:
            from huggingface_hub import hf_hub_download  # optional dep

            local_path = hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=_HF_FILENAME,
                repo_type="dataset",
            )
            digest = _sha256_file(local_path)
            if digest not in HF_DATASET_SHA256_ALLOWLIST:
                st.warning(
                    "HuggingFace dataset checksum mismatch. "
                    f"Expected one of {len(HF_DATASET_SHA256_ALLOWLIST)} known hashes, got {digest}. "
                    "Continuing per warn-only policy."
                )
            df = pd.read_csv(local_path)
            df = _normalise_hf_df(df)
            _validate_schema(df, "HuggingFace Hub")
            return df
        except ImportError:
            st.warning("huggingface_hub not installed — falling back to sample dataset.")
        except Exception as exc:
            st.warning(f"HuggingFace download failed ({exc}) — falling back to sample dataset.")

    # 4. Sample fallback -------------------------------------------------------
    st.warning(
        "⚠️ Running on ~200-track sample. "
        "Full dataset unavailable on this environment."
    )
    df = pd.read_csv(SAMPLE_DATA_PATH)
    _validate_schema(df, "sample CSV")
    return df
