"""
spotify_client.py — Spotify Client Credentials API helper.

Read-only enrichment: given a track name + artist, fetches live Spotify data
(album art, preview URL, popularity, external link).  No user OAuth needed.

Secrets required (set in .streamlit/secrets.toml or environment):
  SPOTIFY_CLIENT_ID
  SPOTIFY_CLIENT_SECRET
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from typing import Any

import requests

# ── Auth ─────────────────────────────────────────────────────────────────────


def _get_credentials() -> tuple[str, str]:
    """Return (client_id, client_secret) from env or Streamlit secrets."""
    # Try Streamlit secrets first (Streamlit Cloud deployment)
    try:
        import streamlit as st  # type: ignore

        client_id = st.secrets["SPOTIFY_CLIENT_ID"]
        client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
        return client_id, client_secret
    except Exception:
        pass

    # Fallback to environment variables
    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise EnvironmentError(
            "Spotify credentials not found. "
            "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in "
            ".streamlit/secrets.toml or as environment variables."
        )
    return client_id, client_secret


@lru_cache(maxsize=1)
def get_token() -> str:
    """Fetch a Client Credentials access token (cached per process)."""
    client_id, client_secret = _get_credentials()
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {creds}"},
        data={"grant_type": "client_credentials"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["access_token"]


# ── Search ────────────────────────────────────────────────────────────────────


def search_track(name: str, artist: str) -> dict | None:
    """Search Spotify for a track by name + artist.

    Returns the first matching Spotify track object, or None if not found
    or if the API call fails.  Never raises — caller always checks for None.
    """
    try:
        token = get_token()
    except Exception:
        return None  # credentials not configured

    query = f"track:{name} artist:{artist}"
    try:
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": query, "type": "track", "limit": 1},
            timeout=8,
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    items = response.json().get("tracks", {}).get("items", [])
    return items[0] if items else None


# ── Live search (multi-result) ────────────────────────────────────────────────


def search_tracks_live(query: str, limit: int = 50) -> list[dict]:
    """Search Spotify by free-form query and return up to `limit` track objects.

    Each returned dict contains: id, name, artists (list), album, popularity,
    preview_url, external_urls.  Returns [] on failure.
    """
    try:
        token = get_token()
    except Exception:
        return []

    # Spotify max per request is 50
    limit = max(1, min(limit, 50))
    try:
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": query, "type": "track", "limit": limit},
            timeout=10,
        )
    except requests.RequestException:
        return []

    if response.status_code != 200:
        return []

    return response.json().get("tracks", {}).get("items", [])


def get_audio_features_batch(track_ids: list[str]) -> dict[str, dict]:
    """Fetch audio features for up to 100 track IDs in one request.

    Returns a dict mapping track_id → feature dict.
    Features include: valence, energy, danceability, tempo, acousticness,
    instrumentalness, liveness, speechiness, loudness.
    Missing tracks are silently skipped.
    """
    if not track_ids:
        return {}
    try:
        token = get_token()
    except Exception:
        return {}

    # API hard limit is 100 IDs per request
    ids_chunk = track_ids[:100]
    try:
        response = requests.get(
            "https://api.spotify.com/v1/audio-features",
            headers={"Authorization": f"Bearer {token}"},
            params={"ids": ",".join(ids_chunk)},
            timeout=10,
        )
    except requests.RequestException:
        return {}

    if response.status_code != 200:
        return {}

    features_list = response.json().get("audio_features", [])
    return {
        f["id"]: f
        for f in features_list
        if f is not None and "id" in f
    }


# Feature columns we read from local dataset fallback
_FEATURE_COLS = (
    "valence", "energy", "danceability", "tempo",
    "acousticness", "instrumentalness", "loudness",
)


def _enrich_with_local_features(
    tracks: list[dict], local_df: "Any | None",  # noqa: F821
) -> list[dict]:
    """Attach audio features from the local Kaggle dataset when available.

    Matches by track_id first, then by (lowercased track_name, first artist).
    Mutates tracks in place and returns it.
    """
    if local_df is None or local_df.empty:
        return tracks

    id_lookup = {str(tid): row for tid, row in local_df.set_index("track_id").iterrows()}
    name_lookup: dict[tuple[str, str], Any] = {}  # noqa: F821
    for _, row in local_df.iterrows():
        key = (
            str(row.get("track_name", "")).strip().lower(),
            str(row.get("artist_name", "")).split(",")[0].strip().lower(),
        )
        if key[0] and key not in name_lookup:
            name_lookup[key] = row

    for track in tracks:
        if any(track.get(col) is not None for col in _FEATURE_COLS):
            continue
        row = id_lookup.get(track.get("track_id", ""))
        if row is None:
            key = (
                track.get("track_name", "").strip().lower(),
                track.get("artist_name", "").split(",")[0].strip().lower(),
            )
            row = name_lookup.get(key)
        if row is None:
            continue
        for col in _FEATURE_COLS:
            if col in row.index:
                val = row[col]
                try:
                    track[col] = float(val) if val is not None else None
                except (TypeError, ValueError):
                    track[col] = None
    return tracks


def search_and_enrich(
    query: str,
    limit: int = 50,
    local_df: "Any | None" = None,  # noqa: F821
) -> list[dict]:
    """Search Spotify and attach audio features to each result.

    Spotify deprecated the Client-Credentials audio-features endpoint in late
    2024 for new apps; when that call returns empty, we fall back to matching
    tracks against the local Kaggle dataset by track_id (or name+artist).

    Returns a flat list of dicts with track metadata and audio features.
    Tracks without features are still included (features default to None).
    """
    tracks = search_tracks_live(query, limit=limit)
    if not tracks:
        return []

    track_ids = [t["id"] for t in tracks if t.get("id")]
    features_map = get_audio_features_batch(track_ids)

    result = []
    for track in tracks:
        tid = track.get("id", "")
        feats = features_map.get(tid, {})
        artists = [a["name"] for a in track.get("artists", [])]
        images = track.get("album", {}).get("images", [])
        result.append({
            "track_id":      tid,
            "track_name":    track.get("name", ""),
            "artist_name":   ", ".join(artists),
            "album":         track.get("album", {}).get("name", ""),
            "popularity":    track.get("popularity", 0),
            "preview_url":   track.get("preview_url"),
            "spotify_url":   track.get("external_urls", {}).get("spotify", ""),
            "album_art":     images[0]["url"] if images else None,
            "valence":       feats.get("valence"),
            "energy":        feats.get("energy"),
            "danceability":  feats.get("danceability"),
            "tempo":         feats.get("tempo"),
            "acousticness":  feats.get("acousticness"),
            "instrumentalness": feats.get("instrumentalness"),
            "loudness":      feats.get("loudness"),
        })

    # Fallback enrichment when Spotify features endpoint is blocked (403/deprecated)
    if not features_map:
        _enrich_with_local_features(result, local_df)
    return result


# ── Enrichment helper ─────────────────────────────────────────────────────────


def enrich_journey(journey: list[dict]) -> list[dict]:
    """Add Spotify metadata fields to each JourneyTrack dict in place.

    Fields added when found:
      spotify_url  — external Spotify link
      album_art    — URL of the largest album image
      preview_url  — 30-second MP3 preview (may be None even if track found)
      popularity   — 0–100 popularity score

    Returns the same list (mutated in place) for chaining.
    """
    for track in journey:
        sp = search_track(
            track.get("track_name", ""),
            track.get("artist_name", ""),
        )
        if sp:
            track["spotify_url"] = sp.get("external_urls", {}).get("spotify", "")
            images = sp.get("album", {}).get("images", [])
            track["album_art"] = images[0]["url"] if images else None
            track["preview_url"] = sp.get("preview_url")
            track["popularity"] = sp.get("popularity", 0)
    return journey
