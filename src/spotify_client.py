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
