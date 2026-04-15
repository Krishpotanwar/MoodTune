"""
app.py -- MoodTune Streamlit entry point.

Run from the project root:
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections.abc import Callable
from html import escape
from numbers import Real
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (  # noqa: E402
    CLEAN_DATA_PATH,
    DEFAULT_3D_SAMPLE,
    DEFAULT_JOURNEY_STEPS,
    DEFAULT_PLAYLIST_SIZE,
    LOG_PATH,
    RAW_DATA_PATH,
    STYLE_PATH,
)
from src.data_loader import load_full_dataset  # noqa: E402
from src.journey import (  # noqa: E402
    build_journey_tree,
    generate_mood_journey,
    journey_to_dataframe,
)
from src.mood_mapper import SURVEY_QUESTIONS, map_to_vector  # noqa: E402
from src.nlp_mood import text_to_mood_vector  # noqa: E402
from src.recommender import build_model, recommend  # noqa: E402
from src.validator import get_cleaning_steps_log, run_pipeline  # noqa: E402
from src.visualizer import (  # noqa: E402
    feature_correlation_heatmap,
    journey_progress_figure,
    mood_space_3d_figure,
    mood_space_figure,
)

st.set_page_config(
    page_title="MoodTune",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_BOOT_SCENE_SAMPLE_SIZE = 2_400
_AMBIENT_WORDS = (
    "sargam",
    "rhythm",
    "sur",
    "raag",
    "drop",
    "groove",
    "dil",
    "beat",
    "vibe",
    "journey",
    "tempo",
    "mood",
    "lehra",
    "bass",
    "echo",
    "taal",
)


# ── CSS injection ──────────────────────────────────────────────────────────────

def _inject_css() -> None:
    """Load and inject the custom stylesheet into the Streamlit app."""
    if STYLE_PATH.exists():
        css = STYLE_PATH.read_text(encoding="utf-8")
        st.html(f"<style>{css}</style>")


# ── cached data ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_recommender_model() -> NearestNeighbors:
    """Build the nearest-neighbour model once per dataset state."""
    return build_model(load_full_dataset())


@st.cache_resource(show_spinner=False)
def _get_journey_tree() -> KDTree:
    """Build the KDTree for the journey engine once per dataset state."""
    return build_journey_tree(load_full_dataset())


# ── session state ──────────────────────────────────────────────────────────────

_SELECTION_MODES: tuple[str, str] = ("Start mood", "Target mood")
_STATE_REPAIR_NOTIFIED_KEY = "_state_repairs_notified"
_STATE_DEFAULTS: dict[str, Any] = {
    "system_initialized": False,
    "journey_start": None,
    "journey_target": None,
    "journey_start_source": "Not set",
    "journey_target_source": "Not set",
    "selection_mode": "Start mood",
    "journey_steps": DEFAULT_JOURNEY_STEPS,
    "survey_step": 0,
    "survey_answers": {},
    "survey_result": None,
    "nlp_start_result": None,
    "nlp_target_result": None,
    "show_full_3d": False,
    _STATE_REPAIR_NOTIFIED_KEY: {},
}


def _default_value_for(key: str) -> Any:
    """Return a safe default value for a state key, copying mutables."""
    value = _STATE_DEFAULTS[key]
    if isinstance(value, dict):
        return dict(value)
    return value


def _state_set(key: str, value: Any) -> None:
    """Set one session-state value through a single helper."""
    st.session_state[key] = value


def _warn_state_repair(key: str, expected: str, actual: Any) -> None:
    """Warn once when an invalid session-state value is repaired."""
    notified = st.session_state.get(_STATE_REPAIR_NOTIFIED_KEY, {})
    if not isinstance(notified, dict):
        notified = {}
    if notified.get(key):
        return
    st.warning(
        f"Recovered invalid session value for '{key}'. "
        f"Expected {expected}, got {type(actual).__name__}. Reset to default."
    )
    notified[key] = True
    st.session_state[_STATE_REPAIR_NOTIFIED_KEY] = notified


def _restore_state_value(key: str, expected: str, is_valid: Callable[[Any], bool]) -> Any:
    """Return validated state value, repairing invalid or missing data."""
    default = _default_value_for(key)
    if key not in st.session_state:
        st.session_state[key] = default
        return default

    value = st.session_state[key]
    if is_valid(value):
        return value

    _warn_state_repair(key, expected, value)
    st.session_state[key] = default
    return default


def _is_finite_number(value: Any) -> bool:
    """Check if a value is a finite real number."""
    return isinstance(value, Real) and math.isfinite(float(value))


def _is_coordinate(value: Any) -> bool:
    """Check if a value is a coordinate-like pair."""
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and _is_finite_number(value[0])
        and _is_finite_number(value[1])
    )


def _is_survey_answers(value: Any) -> bool:
    """Check if survey answers are a safe dict[str, int in 1..4]."""
    return isinstance(value, dict) and all(
        isinstance(question, str)
        and isinstance(answer, int)
        and not isinstance(answer, bool)
        and 1 <= answer <= 4
        for question, answer in value.items()
    )


def _state_get_coordinate(key: str) -> tuple[float, float] | None:
    """Return one coordinate from state, validating and normalising it."""
    value = _restore_state_value(
        key,
        "None or (valence, energy) numeric coordinate",
        lambda item: item is None or _is_coordinate(item),
    )
    if value is None:
        return None
    normalised = (round(float(value[0]), 4), round(float(value[1]), 4))
    st.session_state[key] = normalised
    return normalised


def _state_get_source(key: str) -> str:
    """Return one source-label string from state."""
    return str(_restore_state_value(key, "string", lambda item: isinstance(item, str)))


def _state_get_selection_mode() -> str:
    """Return the validated current click-selection mode."""
    return str(
        _restore_state_value(
            "selection_mode",
            "'Start mood' or 'Target mood'",
            lambda item: isinstance(item, str) and item in _SELECTION_MODES,
        )
    )


def _state_get_journey_steps() -> int:
    """Return a validated journey-step count."""
    return int(
        _restore_state_value(
            "journey_steps",
            "integer in [8, 24]",
            lambda item: isinstance(item, int) and not isinstance(item, bool) and 8 <= item <= 24,
        )
    )


def _state_get_survey_step() -> int:
    """Return a validated survey step index."""
    return int(
        _restore_state_value(
            "survey_step",
            "non-negative integer",
            lambda item: isinstance(item, int) and not isinstance(item, bool) and item >= 0,
        )
    )


def _state_get_survey_answers() -> dict[str, int]:
    """Return validated survey answers."""
    value = _restore_state_value("survey_answers", "dict[str, int from 1 to 4]", _is_survey_answers)
    return dict(value)


def _state_get_optional_dict(key: str) -> dict[str, Any] | None:
    """Return a state payload that is either None or a dictionary."""
    value = _restore_state_value(key, "None or dict", lambda item: item is None or isinstance(item, dict))
    if value is None:
        return None
    return dict(value)


def _state_get_show_full_3d() -> bool:
    """Return whether 3D chart should render all points."""
    value = _restore_state_value("show_full_3d", "boolean", lambda item: isinstance(item, bool))
    return bool(value)


def _state_get_system_initialized() -> bool:
    """Return whether the immersive boot scene has been initialized."""
    value = _restore_state_value("system_initialized", "boolean", lambda item: isinstance(item, bool))
    return bool(value)


def _consume_boot_query_flag() -> bool:
    """Read query params once and flip system-initialized state when requested."""
    boot_flag = st.query_params.get("boot")
    if boot_flag != "1":
        return False
    _state_set("system_initialized", True)
    del st.query_params["boot"]
    return True


def _init_state() -> None:
    """Create every session_state key used by the app."""
    for key in _STATE_DEFAULTS:
        if key not in st.session_state:
            st.session_state[key] = _default_value_for(key)


# ── coordinate helpers ─────────────────────────────────────────────────────────

def _set_coordinate(key: str, coord: tuple[float, float], source: str) -> None:
    """Store one mood coordinate in session state with a readable source tag."""
    x_val, y_val = round(float(coord[0]), 4), round(float(coord[1]), 4)
    _state_set(key, (x_val, y_val))
    _state_set(f"{key}_source", source)


def _clear_coordinate(key: str) -> None:
    """Clear one stored mood coordinate and reset its source label."""
    _state_set(key, None)
    _state_set(f"{key}_source", "Not set")


def _reset_survey() -> None:
    """Clear the survey flow back to its initial state."""
    _state_set("survey_step", 0)
    _state_set("survey_answers", {})
    _state_set("survey_result", None)


def _quadrant_label(coord: tuple[float, float] | None) -> str:
    """Convert a coordinate into a human-readable mood label."""
    if coord is None:
        return "Not set"
    valence, energy = coord
    if energy >= 0.5 and valence >= 0.5:
        return "Joyful"
    if energy >= 0.5 and valence < 0.5:
        return "Intense"
    if energy < 0.5 and valence >= 0.5:
        return "Calm"
    return "Melancholic"


# ── chart event parsing ────────────────────────────────────────────────────────

def _coerce_mapping(obj: Any) -> dict[str, Any]:
    """Turn dict-like Streamlit selection objects into plain dictionaries."""
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except (TypeError, ValueError):
        return {}


def _extract_selected_points(chart_event: Any) -> list[tuple[float, float]]:
    """Pull selected Plotly points out of the Streamlit event payload."""
    if not chart_event:
        return []
    event_map = _coerce_mapping(chart_event)
    selection = event_map.get("selection", getattr(chart_event, "selection", {}))
    selection_map = _coerce_mapping(selection)
    raw_points = selection_map.get("points", getattr(selection, "points", []))

    coords: list[tuple[float, float]] = []
    for point in raw_points:
        point_map = _coerce_mapping(point)
        x_val = point_map.get("x", getattr(point, "x", None))
        y_val = point_map.get("y", getattr(point, "y", None))
        if x_val is not None and y_val is not None:
            coords.append((float(x_val), float(y_val)))
    return coords


def _apply_chart_selection(chart_event: Any) -> None:
    """Update the start or target mood from the latest chart selection."""
    selected = _extract_selected_points(chart_event)
    if not selected:
        return
    latest = selected[-1]
    if _state_get_selection_mode() == "Start mood":
        _set_coordinate("journey_start", latest, "Mood Space")
        return
    _set_coordinate("journey_target", latest, "Mood Space")


def _genre_hue(genre: str) -> int:
    """Map genre text to a stable hue value for the boot-scene particle color."""
    checksum = sum((idx + 1) * ord(ch) for idx, ch in enumerate(genre.lower()))
    return checksum % 360


def _boot_scene_points_json(df: pd.DataFrame, max_points: int = _BOOT_SCENE_SAMPLE_SIZE) -> str:
    """Create a compact JSON payload for the immersive boot-scene renderer."""
    sample_size = min(max_points, len(df))
    if sample_size <= 0:
        return "[]"
    sample = df.sample(n=sample_size, random_state=42).copy()
    if "danceability" not in sample.columns:
        sample["danceability"] = (sample["valence"] + sample["energy"]) / 2

    points: list[dict[str, float | int]] = []
    for valence, energy, danceability, genre in zip(
        sample["valence"],
        sample["energy"],
        sample["danceability"],
        sample["genre"].astype(str),
        strict=False,
    ):
        if not (_is_finite_number(valence) and _is_finite_number(energy) and _is_finite_number(danceability)):
            continue
        points.append({
            "x": round(float(valence) - 0.5, 4),
            "y": round(float(energy) - 0.5, 4),
            "z": round(float(danceability) - 0.5, 4),
            "h": _genre_hue(genre),
        })

    return json.dumps(points, separators=(",", ":"))


def _boot_scene_markup(points_json: str, total_tracks: int) -> str:
    """Build the immersive boot-scene HTML and JavaScript payload."""
    point_count = len(json.loads(points_json))
    markup = """
<style>
  html, body {
    margin: 0;
    height: 100%;
    width: 100%;
    overflow: hidden;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  .boot-scene-root {
    position: relative;
    width: 100%;
    height: 700px;
    background:
      radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0) 45%),
      radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0) 44%),
      #080808;
    overflow: hidden;
    color: #f0f0f0;
  }
  #boot-scene-canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
  .boot-scene-vignette {
    position: absolute;
    inset: 0;
    background: radial-gradient(circle, rgba(8, 8, 8, 0) 48%, rgba(8, 8, 8, 0.80) 100%);
    pointer-events: none;
  }
  .boot-scene-center {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: min(680px, 88vw);
    text-align: center;
    background: rgba(26, 26, 26, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 24px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: clamp(1.2rem, 2vw, 2rem) clamp(1rem, 2vw, 2.4rem);
    box-shadow: 0 18px 70px rgba(0, 0, 0, 0.7);
    z-index: 4;
  }
  .boot-scene-kicker {
    margin: 0 0 0.55rem;
    font-size: 0.72rem;
    color: rgba(255, 255, 255, 0.45);
    text-transform: uppercase;
    letter-spacing: 0.22em;
    font-weight: 700;
  }
  .boot-scene-center h1 {
    margin: 0;
    font-size: clamp(1.7rem, 4vw, 3rem);
    line-height: 1.1;
    font-weight: 800;
    color: #f0f0f0;
    text-shadow: 1px 1px 0 #555, 2px 2px 0 #333, 3px 3px 10px rgba(0,0,0,0.8);
  }
  .boot-scene-center p {
    margin: 0.85rem auto 0;
    max-width: 56ch;
    color: rgba(255, 255, 255, 0.55);
    font-size: clamp(0.88rem, 1.8vw, 1.02rem);
    line-height: 1.55;
  }
  .boot-scene-center code {
    color: rgba(255, 255, 255, 0.5);
    background: rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 0 6px;
  }
  .boot-init-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: 1.2rem;
    min-height: 52px;
    padding: 0.75rem 1.8rem;
    border-radius: 999px;
    text-decoration: none;
    color: #f0f0f0;
    font-weight: 700;
    letter-spacing: 0.03em;
    border: 1px solid rgba(255, 255, 255, 0.18);
    background: linear-gradient(180deg, rgba(240, 240, 240, 0.90), rgba(180, 180, 180, 0.90));
    box-shadow: 0 6px 0 rgba(0, 0, 0, 0.7), 0 18px 45px rgba(0, 0, 0, 0.5);
    transform: translateY(0);
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.2s ease;
  }
  .boot-init-button:hover {
    filter: brightness(1.06);
    transform: translateY(-1px);
  }
  .boot-init-button:active {
    transform: translateY(3px);
    box-shadow: 0 2px 0 rgba(0, 0, 0, 0.70), 0 10px 25px rgba(0, 0, 0, 0.40);
  }
  .boot-scene-meta {
    margin-top: 0.85rem !important;
    font-size: 0.76rem !important;
    color: rgba(255, 255, 255, 0.55) !important;
    letter-spacing: 0.05em;
  }
  @media (max-width: 768px) {
    .boot-scene-root { height: 620px; }
    .boot-scene-center {
      width: min(92vw, 620px);
      border-radius: 18px;
      padding: 1rem 1rem 1.2rem;
    }
    .boot-init-button {
      min-height: 48px;
      padding: 0.65rem 1.35rem;
    }
  }
  @media (max-width: 480px) {
    .boot-scene-root { height: 570px; }
    .boot-scene-center {
      width: 94vw;
      padding: 0.95rem 0.85rem 1rem;
      border-radius: 15px;
    }
    .boot-scene-kicker {
      letter-spacing: 0.16em;
      font-size: 0.62rem;
    }
    .boot-scene-center p {
      font-size: 0.82rem;
    }
    .boot-init-button {
      width: 100%;
      min-height: 46px;
      margin-top: 0.95rem;
    }
  }
</style>
<div class="boot-scene-root" role="region" aria-label="MoodTune immersive boot scene">
  <canvas id="boot-scene-canvas"></canvas>
  <div class="boot-scene-vignette"></div>
  <div class="boot-scene-center">
    <p class="boot-scene-kicker">MoodTune Immersive System</p>
    <h1>Explore the mood space</h1>
    <p>Scroll to zoom · Rotate to explore · Click to lock in</p>
    <a class="boot-init-button" id="boot-init-button" href="?boot=1" target="_top" rel="noopener">
      Initialize System
    </a>
    <p class="boot-scene-meta">Rendering __POINT_COUNT__ particles from __TRACK_COUNT__ tracks</p>
  </div>
</div>
<script>
(() => {
  const root = document.querySelector(".boot-scene-root");
  const canvas = document.getElementById("boot-scene-canvas");
  if (!root || !canvas) return;
  const button = document.getElementById("boot-init-button");
  if (button) {
    try {
      const topUrl = new URL(window.top.location.href);
      topUrl.searchParams.set("boot", "1");
      button.href = topUrl.pathname + "?" + topUrl.searchParams.toString();
    } catch (error) {
      const current = new URL(window.location.href);
      current.searchParams.set("boot", "1");
      button.href = current.pathname + "?" + current.searchParams.toString();
    }
  }

  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;
  const points = __POINTS_JSON__;
  if (!Array.isArray(points)) return;

  const cubeVertices = [
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
  ];
  const cubeEdges = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7],
  ];

  let width = 0;
  let height = 0;
  let cameraDistance = 1.0;
  let cameraTarget = 1.0;
  let pointerX = 0;
  let pointerY = 0;

  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    const bounds = root.getBoundingClientRect();
    width = Math.max(1, Math.floor(root.clientWidth || bounds.width || 1));
    height = Math.max(1, Math.floor(root.clientHeight || bounds.height || 1));
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  resize();
  window.addEventListener("resize", resize);
  if (typeof ResizeObserver !== "undefined") {
    const rootResizeObserver = new ResizeObserver(resize);
    rootResizeObserver.observe(root);
  }

  root.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const direction = Math.sign(event.deltaY || 0);
      cameraTarget = Math.min(4.8, Math.max(1.25, cameraTarget + direction * 0.18));
    },
    { passive: false }
  );

  root.addEventListener("pointermove", (event) => {
    const bounds = root.getBoundingClientRect();
    const px = (event.clientX - bounds.left) / Math.max(bounds.width, 1);
    const py = (event.clientY - bounds.top) / Math.max(bounds.height, 1);
    pointerX = (Math.max(0, Math.min(1, px)) - 0.5) * 2;
    pointerY = (Math.max(0, Math.min(1, py)) - 0.5) * 2;
  });

  const rotate = (point, rx, ry) => {
    const x1 = point.x * Math.cos(ry) - point.z * Math.sin(ry);
    const z1 = point.x * Math.sin(ry) + point.z * Math.cos(ry);
    const y2 = point.y * Math.cos(rx) - z1 * Math.sin(rx);
    const z2 = point.y * Math.sin(rx) + z1 * Math.cos(rx);
    return { x: x1, y: y2, z: z2, h: point.h };
  };

  const rotateVertex = (v, rx, ry) => {
    const x1 = v[0] * Math.cos(ry) - v[2] * Math.sin(ry);
    const z1 = v[0] * Math.sin(ry) + v[2] * Math.cos(ry);
    const y2 = v[1] * Math.cos(rx) - z1 * Math.sin(rx);
    const z2 = v[1] * Math.sin(rx) + z1 * Math.cos(rx);
    return { x: x1, y: y2, z: z2 };
  };

  const project = (p) => {
    const depth = Math.max(0.25, cameraDistance - p.z);
    const scale = 420 / depth;
    return {
      sx: width * 0.5 + p.x * scale,
      sy: height * 0.5 - p.y * scale,
      scale,
      z: p.z,
      h: p.h ?? 330,
    };
  };

  const drawCube = (rx, ry) => {
    const verts = cubeVertices.map((v) => project(rotateVertex(v, rx, ry)));
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = 1.1;
    for (const edge of cubeEdges) {
      const a = verts[edge[0]];
      const b = verts[edge[1]];
      ctx.beginPath();
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
    }
    ctx.restore();
  };

  const drawOrigin = (rx, ry) => {
    const center = project(rotate({ x: 0, y: 0, z: 0, h: 0 }, rx, ry));
    const glow = 8 + Math.max(0, 80 / Math.max(20, center.scale));
    ctx.save();
    const grad = ctx.createRadialGradient(center.sx, center.sy, 0, center.sx, center.sy, glow);
    grad.addColorStop(0, "rgba(255,255,255,0.80)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(center.sx, center.sy, glow, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  };

  let frame = 0;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion) {
    const rx = 0.34;
    const ry = 0.22;
    ctx.clearRect(0, 0, width, height);
    const bg = ctx.createLinearGradient(0, 0, width, height);
    bg.addColorStop(0, "rgba(7,4,13,0.94)");
    bg.addColorStop(1, "rgba(17,8,24,0.96)");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);
    drawCube(rx, ry);
    const projected = points
      .map((point) => project(rotate(point, rx, ry)))
      .sort((a, b) => a.z - b.z);
    for (const p of projected) {
      const size = Math.max(0.8, Math.min(3.3, p.scale * 0.018));
      ctx.beginPath();
      ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
      ctx.arc(p.sx, p.sy, size, 0, Math.PI * 2);
      ctx.fill();
    }
    drawOrigin(rx, ry);
    return;
  }

  const render = () => {
    frame += 1;
    const t = frame * 0.005;
    const rx = 0.34 + pointerY * 0.09;
    const ry = t + pointerX * 0.12;
    cameraDistance += (cameraTarget - cameraDistance) * 0.08;

    ctx.clearRect(0, 0, width, height);
    const bg = ctx.createLinearGradient(0, 0, width, height);
    bg.addColorStop(0, "rgba(7,4,13,0.94)");
    bg.addColorStop(1, "rgba(17,8,24,0.96)");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    drawCube(rx, ry);

    const projected = points
      .map((point) => project(rotate(point, rx, ry)))
      .sort((a, b) => a.z - b.z);

    for (const p of projected) {
      const size = Math.max(0.8, Math.min(3.6, p.scale * 0.018));
      const alpha = Math.max(0.18, Math.min(0.82, 0.12 + p.scale * 0.0012));
      ctx.beginPath();
      ctx.fillStyle = `hsla(${p.h}, 94%, 72%, ${alpha})`;
      ctx.arc(p.sx, p.sy, size, 0, Math.PI * 2);
      ctx.fill();
    }

    drawOrigin(rx, ry);
    requestAnimationFrame(render);
  };
  requestAnimationFrame(render);
})();
</script>
"""
    return (
        markup.replace("__POINTS_JSON__", points_json)
        .replace("__TRACK_COUNT__", f"{total_tracks:,}")
        .replace("__POINT_COUNT__", f"{point_count:,}")
    )


def _render_immersive_boot_scene(df: pd.DataFrame) -> None:
    """Render the immersive startup scene before entering the full app."""
    points_json = _boot_scene_points_json(df)
    components.html(_boot_scene_markup(points_json, len(df)), height=700, scrolling=False)
    st.caption(
        "The Start button is inside the scene. If the iframe blocks navigation,"
        " use the direct initialize button below."
    )
    left, center, right = st.columns([1.2, 1.0, 1.2])
    with center:
        if st.button("Initialize System", key="boot_initialize_direct", type="primary", use_container_width=True):
            _state_set("system_initialized", True)
            st.rerun()

def _ambient_word_positions(count: int) -> list[dict[str, str]]:
    """Generate deterministic absolute positions for low-opacity ambient words."""
    rng = random.Random(42)
    positioned: list[dict[str, str]] = []
    for idx in range(count):
        word = _AMBIENT_WORDS[idx % len(_AMBIENT_WORDS)]
        top = f"{rng.randint(4, 92)}vh"
        left = f"{rng.randint(2, 96)}vw"
        delay = f"{rng.uniform(-14.0, 0.0):.2f}s"
        duration = f"{rng.uniform(16.0, 32.0):.2f}s"
        positioned.append({
            "word": word,
            "top": top,
            "left": left,
            "delay": delay,
            "duration": duration,
        })
    return positioned


def _render_ambient_word_layer() -> None:
    """Attach floating, low-opacity music words behind the main content."""
    words = _ambient_word_positions(26)
    spans = "".join(
        (
            f'<span class="ambient-word" style="top:{escape(item["top"])};'
            f'left:{escape(item["left"])};animation-delay:{escape(item["delay"])};'
            f'animation-duration:{escape(item["duration"])}">{escape(item["word"])}</span>'
        )
        for item in words
    )
    st.html(f'<div id="ambient-word-layer" aria-hidden="true">{spans}</div>')


def _render_initialized_header(df: pd.DataFrame) -> None:
    """Render the header stats banner after initialization."""
    n_genres = df["genre"].nunique() if "genre" in df.columns else 0
    st.html(
        f"""
        <section class="v2-hero">
            <p class="v2-kicker">Mood Navigation Engine</p>
            <h1>MoodTune</h1>
            <p class="v2-subtitle">
                Map your mood, build a journey, and watch your emotional shift unfold through music.
            </p>
            <div class="v2-pill-row">
                <span class="v2-pill">🎵 {len(df):,} tracks</span>
                <span class="v2-pill">🎯 {DEFAULT_JOURNEY_STEPS} step default</span>
                <span class="v2-pill">🎸 {n_genres} genres</span>
                <span class="v2-pill">🧠 KDTree + NLP mapping</span>
            </div>
        </section>
        """
    )


# ── shared UI blocks ───────────────────────────────────────────────────────────

def _render_selection_summary() -> None:
    """Show the currently active mood coordinates and their source."""
    start = _state_get_coordinate("journey_start")
    target = _state_get_coordinate("journey_target")
    start_source = _state_get_source("journey_start_source")
    target_source = _state_get_source("journey_target_source")
    col_start, col_target, col_actions = st.columns([1.2, 1.2, 0.8])

    with col_start:
        if start is None:
            st.info("Start mood not set yet.")
        else:
            st.success(
                f"Start mood: **{_quadrant_label(start)}**\n\n"
                f"Valence `{start[0]:.2f}` · Energy `{start[1]:.2f}`\n\n"
                f"Source: {start_source}"
            )

    with col_target:
        if target is None:
            st.info("Target mood not set yet.")
        else:
            st.success(
                f"Target mood: **{_quadrant_label(target)}**\n\n"
                f"Valence `{target[0]:.2f}` · Energy `{target[1]:.2f}`\n\n"
                f"Source: {target_source}"
            )

    with col_actions:
        st.radio(
            "Next chart click sets:",
            options=["Start mood", "Target mood"],
            key="selection_mode",
            horizontal=False,
        )
        if st.button("Clear start", use_container_width=True):
            _clear_coordinate("journey_start")
            st.rerun()
        if st.button("Clear target", use_container_width=True):
            _clear_coordinate("journey_target")
            st.rerun()


def _render_text_mapper() -> None:
    """Render the NLP start/target mood mapping controls."""
    left, right = st.columns(2)

    with left:
        st.markdown("#### Describe your starting mood")
        start_text = st.text_area(
            "Start mood text",
            key="nlp_start_text",
            height=110,
            placeholder="I just finished a long exam and want to zone out",
            label_visibility="collapsed",
        )
        if st.button("Map text to start mood", key="nlp_start_button", use_container_width=True):
            result = text_to_mood_vector(start_text)
            _state_set("nlp_start_result", result)
            _set_coordinate("journey_start", result["coordinate"], "NLP")

        result = _state_get_optional_dict("nlp_start_result")
        if result:
            matched_words = result.get("matched_words", [])
            if not isinstance(matched_words, list):
                matched_words = []
            confidence = result.get("confidence", 0.0)
            if not _is_finite_number(confidence):
                confidence = 0.0
            st.caption(
                f"Matched: {', '.join(str(word) for word in matched_words) or 'none'} | "
                f"confidence {float(confidence):.2f}"
            )

    with right:
        st.markdown("#### Describe your target mood")
        target_text = st.text_area(
            "Target mood text",
            key="nlp_target_text",
            height=110,
            placeholder="Take me from drained to bright and energized",
            label_visibility="collapsed",
        )
        if st.button("Map text to target mood", key="nlp_target_button", use_container_width=True):
            result = text_to_mood_vector(target_text)
            _state_set("nlp_target_result", result)
            _set_coordinate("journey_target", result["coordinate"], "NLP")

        result = _state_get_optional_dict("nlp_target_result")
        if result:
            matched_words = result.get("matched_words", [])
            if not isinstance(matched_words, list):
                matched_words = []
            confidence = result.get("confidence", 0.0)
            if not _is_finite_number(confidence):
                confidence = 0.0
            st.caption(
                f"Matched: {', '.join(str(word) for word in matched_words) or 'none'} | "
                f"confidence {float(confidence):.2f}"
            )


def _render_survey_question(step: int) -> None:
    """Render one survey question with back/next navigation."""
    question = SURVEY_QUESTIONS[step - 1]
    labels = [option["label"] for option in question["options"]]
    answers = _state_get_survey_answers()
    previous = answers.get(question["id"])
    default_index = previous - 1 if previous else 0

    st.progress(step / 4, text=f"Survey step {step} of 4")
    selected = st.radio(
        question["question"],
        options=labels,
        index=default_index,
        key=f"survey_step_{step}",
    )

    col_back, _, col_next = st.columns([1, 1, 1])
    with col_back:
        if step > 1 and st.button("Back", key=f"survey_back_{step}", use_container_width=True):
            _state_set("survey_step", max(0, _state_get_survey_step() - 1))
            st.rerun()
    with col_next:
        if st.button("Next", key=f"survey_next_{step}", use_container_width=True, type="primary"):
            answers = _state_get_survey_answers()
            answers[question["id"]] = labels.index(selected) + 1
            _state_set("survey_answers", answers)
            _state_set("survey_step", _state_get_survey_step() + 1)
            st.rerun()


def _render_survey_result(df: pd.DataFrame) -> None:
    """Render the classic survey outcome and optional instant recommendations."""
    answers = _state_get_survey_answers()
    missing = [question_id for question_id in ("q1", "q2", "q3", "q4") if question_id not in answers]
    if missing:
        st.warning(f"Survey answers are incomplete: missing {', '.join(missing)}. Please complete all steps.")
        return
    result = map_to_vector(answers["q1"], answers["q2"], answers["q3"], answers["q4"])
    _state_set("survey_result", result)

    st.success(
        f"{result['mood_emoji']} Survey mood: **{result['mood_label']}** | "
        f"Valence `{result['coordinate'][0]:.2f}` · Energy `{result['coordinate'][1]:.2f}`"
    )
    col_start, col_target, col_reset = st.columns(3)
    with col_start:
        if st.button("Use survey as start mood", use_container_width=True):
            _set_coordinate("journey_start", result["coordinate"], "Survey")
    with col_target:
        if st.button("Use survey as target mood", use_container_width=True):
            _set_coordinate("journey_target", result["coordinate"], "Survey")
    with col_reset:
        if st.button("Reset survey", use_container_width=True):
            _reset_survey()
            st.rerun()

    with st.spinner("Finding instant matches..."):
        instant = recommend(_get_recommender_model(), df, result["vector"], k=DEFAULT_PLAYLIST_SIZE)
    st.markdown("##### Classic instant matches")
    st.dataframe(
        instant[["track_name", "artist_name", "genre", "similarity_pct"]],
        hide_index=True,
        use_container_width=True,
    )


def _render_survey_fallback(df: pd.DataFrame) -> None:
    """Render the guided survey flow inside an expander."""
    survey_step = _state_get_survey_step()
    with st.expander("Classic survey fallback", expanded=survey_step > 0):
        if survey_step == 0:
            st.write("Prefer a guided flow? The original four-question survey still works.")
            if st.button("Start survey", type="primary"):
                _state_set("survey_step", 1)
                st.rerun()
            return

        if 1 <= survey_step <= 4:
            _render_survey_question(survey_step)
            return

        _render_survey_result(df)


# ── journey helpers ────────────────────────────────────────────────────────────

def _build_journey_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the current mood journey as a dataframe, with Spotify enrichment."""
    start = _state_get_coordinate("journey_start")
    target = _state_get_coordinate("journey_target")
    if start is None or target is None:
        return pd.DataFrame()
    journey = generate_mood_journey(
        _get_journey_tree(),
        df,
        start=start,
        target=target,
        n_steps=_state_get_journey_steps(),
    )
    # Optional Spotify enrichment — app works fine without it
    try:
        from src.spotify_client import enrich_journey  # noqa: PLC0415
        journey = enrich_journey(journey)
    except Exception:
        pass
    return journey_to_dataframe(journey)


def _spotify_search_url(track_name: str, artist_name: str) -> str:
    """Build a Spotify web search URL for a track."""
    query = quote_plus(f"{track_name} {artist_name}")
    return f"https://open.spotify.com/search/{query}"


def _song_card_markup(
    step_num: int,
    track_name: str,
    artist: str,
    genre: str,
    valence: float,
    energy: float,
    album_art: str | None = None,
    preview_url: str | None = None,
) -> str:
    """Build escaped song-card HTML so dataset text is never rendered as raw HTML."""
    safe_track_name = escape(track_name, quote=True)
    safe_artist = escape(artist, quote=True)
    safe_genre = escape(genre, quote=True)

    art_html = (
        f'<img class="song-card-art" src="{escape(album_art, quote=True)}" alt="album art" loading="lazy">'
        if album_art
        else '<div class="song-card-art" style="display:flex;align-items:center;justify-content:center;font-size:1.5rem;">🎵</div>'
    )
    preview_html = (
        f'<div class="song-card-preview"><audio controls preload="none" src="{escape(preview_url, quote=True)}"></audio></div>'
        if preview_url
        else ""
    )
    return f"""
                    <div class="song-card glass-card">
                        <div style="display:flex;gap:12px;align-items:flex-start;">
                            {art_html}
                            <div style="flex:1;min-width:0;">
                                <div class="song-card-step">Step {step_num}</div>
                                <div class="song-card-title">{safe_track_name}</div>
                                <div class="song-card-artist">{safe_artist}</div>
                                <div class="song-card-genre">{safe_genre}</div>
                            </div>
                        </div>
                        <div class="song-card-badges" style="margin-top:10px;">
                            <span class="stat-badge">V {valence:.2f}</span>
                            <span class="stat-badge">E {energy:.2f}</span>
                        </div>
                        {preview_html}
                    </div>
                    """


def _render_song_cards(journey_df: pd.DataFrame) -> None:
    """Render the journey playlist as a 3-column card grid."""
    rows = [journey_df.iloc[i:i + 3] for i in range(0, len(journey_df), 3)]
    for row_df in rows:
        cols = st.columns(3)
        for col, (_, track) in zip(cols, row_df.iterrows(), strict=False):
            step_num = int(track["waypoint_idx"]) + 1
            track_name = str(track.get("track_name", "Unknown"))
            artist = str(track.get("artist_name", ""))
            genre = str(track.get("genre", ""))
            valence = float(track.get("valence", 0.0))
            energy = float(track.get("energy", 0.0))

            with col:
                album_art = str(track.get("album_art", "")) or None
                preview_url = str(track.get("preview_url", "")) or None
                spotify_url = str(track.get("spotify_url", "")) or None
                st.html(_song_card_markup(
                    step_num, track_name, artist, genre, valence, energy,
                    album_art=album_art, preview_url=preview_url,
                ))
                link_url = spotify_url or _spotify_search_url(track_name, artist)
                st.link_button("Open Spotify", link_url, use_container_width=True)


# ── tab renderers ──────────────────────────────────────────────────────────────

def _render_mood_space_tab(df: pd.DataFrame) -> None:
    """Render the interactive mood-space explorer."""
    # hero stats
    n_genres = df["genre"].nunique() if "genre" in df.columns else 0
    st.html(
        f"""
        <div class="hero-stats">
            <span class="stat-badge">🎵 {len(df):,} tracks</span>
            <span class="stat-badge">🎯 {DEFAULT_JOURNEY_STEPS}-step journey</span>
            <span class="stat-badge">🧠 KDTree + NLP</span>
            <span class="stat-badge">🎸 {n_genres} genres</span>
        </div>
        """
    )

    st.markdown("## Mood Space Studio")
    st.caption(
        "Set your start and target mood using song particles, text input, or the survey. "
        f"{len(df):,} tracks ready to explore."
    )
    _render_selection_summary()

    st.markdown("### Map your mood from text")
    _render_text_mapper()

    st.markdown("### Mood-space explorer")
    st.caption(
        "Select real song points from the chart. "
        "The radio control determines whether the next click sets the start or target mood."
    )
    figure = mood_space_figure(
        df,
        start_coord=_state_get_coordinate("journey_start"),
        target_coord=_state_get_coordinate("journey_target"),
    )
    chart_event = st.plotly_chart(
        figure,
        use_container_width=True,
        key="mood_space_chart",
        on_select="rerun",
        selection_mode=("points",),
        config={"scrollZoom": False, "displaylogo": False},
    )
    _apply_chart_selection(chart_event)
    st.caption(
        "2D chart: each track is mapped by valence (x-axis) and energy (y-axis). "
        "Start and target markers appear after selection."
    )

    # 3D Mood Space
    with st.expander("Immersive 3D Mood Cube — rotate • zoom • explore", expanded=False):
        preview_sample = min(DEFAULT_3D_SAMPLE, len(df))
        showing_full_3d = _state_get_show_full_3d()
        if len(df) > preview_sample and not showing_full_3d:
            if st.button(
                f"Aur points dikhao ({len(df):,} total)",
                key="show_more_3d",
                use_container_width=True,
            ):
                _state_set("show_full_3d", True)
                st.rerun()
        if len(df) > preview_sample and showing_full_3d:
            if st.button(
                f"Fast preview par wapas ({preview_sample:,} points)",
                key="show_less_3d",
                use_container_width=True,
            ):
                _state_set("show_full_3d", False)
                st.rerun()
        sample_size = len(df) if showing_full_3d else preview_sample

        st.caption(
            "Three axes: **Valence** (sad → happy) · **Energy** (chill → intense) · "
            "**Danceability** (still → groove). Genre encoded by both colour and marker shape."
        )
        with st.spinner("Building 3D mood space..."):
            fig_3d = mood_space_3d_figure(df, sample_size=sample_size)
        st.plotly_chart(
            fig_3d,
            use_container_width=True,
            config={"displaylogo": False},
        )
        st.caption(
            "3D chart: each point is a song. "
            "Colour and marker shape both encode genre for accessibility."
        )

    _render_survey_fallback(df)


def _render_journey_tab(df: pd.DataFrame) -> None:
    """Render the generated journey playlist and supporting charts."""
    st.markdown("## Journey Playlist | Mood Safar")

    start = _state_get_coordinate("journey_start")
    target = _state_get_coordinate("journey_target")
    if start is None or target is None:
        st.warning("Set both a start and target mood in the Mood Space tab to generate a journey.")
        return

    st.slider(
        "Journey length | Safar steps",
        min_value=8,
        max_value=24,
        key="journey_steps",
        help="A longer playlist gives a smoother emotional transition.",
    )

    with st.spinner("Generating your mood journey..."):
        journey_df = _build_journey_dataframe(df)

    if journey_df.empty:
        st.error("Failed to generate a journey from the current selection.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Start mood", _quadrant_label(start))
    c2.metric("Target mood", _quadrant_label(target))
    c3.metric("Avg transition score", f"{journey_df['transition_score'].mean():.3f}")

    mood_fig = mood_space_figure(
        df,
        start_coord=start,
        target_coord=target,
        journey_df=journey_df,
    )
    st.plotly_chart(
        mood_fig,
        use_container_width=True,
        config={"scrollZoom": False, "displaylogo": False},
    )
    st.caption(
        "Chart description: path generated through selected tracks from start mood to target mood."
    )

    st.plotly_chart(journey_progress_figure(journey_df), use_container_width=True)
    st.caption(
        "Chart description: valence and energy progression at each journey step."
    )

    st.markdown("### Your mood journey")
    _render_song_cards(journey_df)



def _render_live_search_tab() -> None:
    """Live Spotify search: user query → fetch tracks → plot valence × energy scatter."""
    st.markdown("## Live Search — Spotify Mood Map")
    st.caption(
        "Type any query (artist, song, mood, genre) to fetch live Spotify tracks "
        "and see them plotted in mood space. Requires Spotify API credentials."
    )

    # ── credentials check ────────────────────────────────────────────────────
    try:
        from src.spotify_client import get_token, search_and_enrich  # noqa: PLC0415
        get_token()
        creds_ok = True
    except Exception as exc:
        st.warning(
            f"Spotify credentials not configured: {exc}. "
            "Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to "
            ".streamlit/secrets.toml or environment variables to enable this tab."
        )
        return

    # ── search input ──────────────────────────────────────────────────────────
    col_query, col_limit = st.columns([4, 1])
    with col_query:
        query = st.text_input(
            "Search Spotify",
            placeholder="e.g. sad rainy night, Taylor Swift, lo-fi chill, gym workout",
            key="live_search_query",
            label_visibility="collapsed",
        )
    with col_limit:
        limit = st.selectbox("Results", [20, 30, 50], index=1, key="live_search_limit")

    if not query:
        st.info("Enter a search query above to fetch and plot live Spotify tracks.")
        return

    # ── fetch ─────────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching {limit} tracks from Spotify..."):
        tracks = search_and_enrich(query, limit=limit)

    if not tracks:
        st.error("No results returned. Check your query or Spotify credentials.")
        return

    import pandas as pd  # noqa: PLC0415
    import plotly.express as px  # noqa: PLC0415

    df_live = pd.DataFrame(tracks)

    # Tracks with audio features
    df_with_feats = df_live.dropna(subset=["valence", "energy"]).copy()
    n_total = len(df_live)
    n_plotted = len(df_with_feats)

    st.caption(
        f"Found **{n_total}** tracks · **{n_plotted}** have audio features and appear on the chart."
    )

    # ── scatter plot ──────────────────────────────────────────────────────────
    if df_with_feats.empty:
        st.warning("None of the returned tracks have audio features. Try a different query.")
    else:
        df_with_feats["hover_label"] = (
            df_with_feats["track_name"] + " — " + df_with_feats["artist_name"]
        )
        df_with_feats["popularity_size"] = df_with_feats["popularity"].fillna(20).clip(10, 100)

        fig = px.scatter(
            df_with_feats,
            x="valence",
            y="energy",
            size="popularity_size",
            color="danceability",
            color_continuous_scale="Greys",
            hover_name="hover_label",
            hover_data={
                "valence":          ":.2f",
                "energy":           ":.2f",
                "danceability":     ":.2f",
                "tempo":            ":.0f",
                "popularity":       True,
                "popularity_size":  False,
                "hover_label":      False,
            },
            labels={
                "valence":      "Valence (sad → happy)",
                "energy":       "Energy (chill → intense)",
                "danceability": "Danceability",
            },
            title=f'Live Spotify: "{query}"',
            size_max=28,
        )
        fig.update_layout(
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font_color="#f0f0f0",
            title_font_size=16,
            xaxis=dict(
                range=[-0.05, 1.05],
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.15)",
                tickfont_color="#f0f0f0",
            ),
            yaxis=dict(
                range=[-0.05, 1.05],
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.15)",
                tickfont_color="#f0f0f0",
            ),
            coloraxis_colorbar=dict(
                title="Dance",
                tickfont_color="#f0f0f0",
                titlefont_color="#f0f0f0",
            ),
            margin=dict(l=40, r=20, t=50, b=40),
            height=520,
        )
        # Add quadrant labels
        for text, x, y in [
            ("Calm / Sad",   0.10, 0.10),
            ("Energetic / Sad",  0.10, 0.92),
            ("Calm / Happy",     0.92, 0.10),
            ("Energetic / Happy",0.87, 0.92),
        ]:
            fig.add_annotation(
                x=x, y=y, text=text,
                showarrow=False,
                font=dict(size=10, color="rgba(255,255,255,0.30)"),
                xref="x", yref="y",
            )

        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # ── track list ────────────────────────────────────────────────────────────
    st.markdown("### Tracks")
    rows = [df_live.iloc[i:i + 3] for i in range(0, len(df_live), 3)]
    for row_df in rows:
        cols = st.columns(3)
        for col, (_, track) in zip(cols, row_df.iterrows(), strict=False):
            track_name  = str(track.get("track_name", "Unknown"))
            artist      = str(track.get("artist_name", ""))
            album_art   = str(track.get("album_art", "")) or None
            preview_url = str(track.get("preview_url", "")) or None
            spotify_url = str(track.get("spotify_url", "")) or None
            valence     = track.get("valence")
            energy      = track.get("energy")
            popularity  = int(track.get("popularity") or 0)

            val_str = f"{valence:.2f}" if valence is not None else "—"
            ene_str = f"{energy:.2f}" if energy is not None else "—"

            art_html = (
                f'<img class="song-card-art" src="{escape(album_art, quote=True)}" '
                f'alt="album art" loading="lazy" style="width:52px;height:52px;border-radius:8px;object-fit:cover;">'
                if album_art
                else '<div style="width:52px;height:52px;border-radius:8px;background:rgba(255,255,255,0.05);'
                     'display:flex;align-items:center;justify-content:center;font-size:1.3rem;">🎵</div>'
            )
            preview_html = (
                f'<audio controls preload="none" src="{escape(preview_url, quote=True)}" '
                f'style="width:100%;height:28px;margin-top:8px;filter:invert(1) opacity(0.7);"></audio>'
                if preview_url
                else ""
            )
            card_html = f"""
<div class="song-card glass-card" style="margin-bottom:0.7rem;">
  <div style="display:flex;gap:10px;align-items:flex-start;">
    {art_html}
    <div style="flex:1;min-width:0;">
      <div class="song-card-title" style="font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{escape(track_name, quote=True)}</div>
      <div class="song-card-artist" style="font-size:0.85rem;opacity:0.65;">{escape(artist, quote=True)}</div>
      <div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px;">
        <span class="stat-badge">V {val_str}</span>
        <span class="stat-badge">E {ene_str}</span>
        <span class="stat-badge">⭐ {popularity}</span>
      </div>
    </div>
  </div>
  {preview_html}
</div>"""
            with col:
                st.html(card_html)
                if spotify_url:
                    st.link_button("Open Spotify", spotify_url, use_container_width=True)


def _render_data_lab_tab() -> None:
    """Render the professor-facing cleaning pipeline walkthrough."""
    st.markdown("## Data Lab")
    st.caption("Step-by-step walkthrough of the raw CSV cleaning pipeline.")

    if not RAW_DATA_PATH.exists():
        st.info(
            "Demo mode is active. "
            "Add the full Kaggle CSV at `data/raw/spotify_tracks.csv` to run the cleaning pipeline live."
        )
        return

    if st.button("Run validation pipeline", type="primary"):
        raw_df = pd.read_csv(RAW_DATA_PATH)

        with st.spinner("Running cleaning pipeline..."):
            clean_df = run_pipeline(RAW_DATA_PATH, CLEAN_DATA_PATH, LOG_PATH)
        steps = get_cleaning_steps_log()

        load_full_dataset.clear()
        _get_recommender_model.clear()
        _get_journey_tree.clear()

        st.success(f"Pipeline complete. {len(clean_df):,} clean rows are ready.")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Raw rows", f"{len(raw_df):,}")
        metric_cols[1].metric("Clean rows", f"{len(clean_df):,}")
        metric_cols[2].metric("Retention", f"{len(clean_df) / len(raw_df) * 100:.1f}%")
        metric_cols[3].metric("Genres", str(clean_df["genre"].nunique()))

        st.markdown("### Cleaning steps")
        for step in steps:
            with st.expander(f"Step {step['step']}: {step['name']}", expanded=step["removed"] > 0):
                cols = st.columns(3)
                cols[0].metric("Before", f"{step['before']:,}")
                cols[1].metric("After", f"{step['after']:,}")
                cols[2].metric("Removed", f"{step['removed']:,}")
                if step["detail"]:
                    st.caption(step["detail"])

        st.markdown("### Before vs after")
        left, right = st.columns(2)
        left.dataframe(raw_df.head(5), use_container_width=True)
        right.dataframe(clean_df.head(5), use_container_width=True)

        st.markdown("### Feature correlation snapshot")
        try:
            st.pyplot(feature_correlation_heatmap(clean_df))
        except Exception as e:
            st.error(f"Correlation heatmap failed to render: {e}")

        if LOG_PATH.exists():
            with st.expander("Validation log"):
                st.code(LOG_PATH.read_text(encoding="utf-8"), language="text")


def _render_how_it_works_tab(df: pd.DataFrame) -> None:
    """Render the professor-demo explanation tab."""
    st.markdown("## How It Works")
    st.markdown(
        f"""
MoodTune turns **{len(df):,} songs** into a navigable emotional map.

1. **Input layer**
   Survey answers, mood-space selections, and natural-language text all map to a `(valence, energy)` coordinate.
2. **Journey engine**
   A SciPy `KDTree` indexes the full dataset. The app interpolates waypoints between your start and target moods, then picks nearby songs while penalising abrupt tempo jumps.
3. **3D Mood Space**
   All {len(df):,} tracks plotted across Valence × Energy × Danceability. Drag to rotate, scroll to zoom.
4. **Visualization**
   The playlist is drawn as a visible path through mood space so the transition is easy to explain and demo.
"""
    )

    st.code(
        """Input Layer
[Survey] ──────────┐
[Mood Space] ──────┼──► (valence, energy) start + target
[NLP Text] ────────┘
          │
          ▼
generate_mood_journey()
          │
          ▼
Ordered journey playlist + path visualization""",
        language="text",
    )

    st.markdown("### Journey scoring")
    st.code(
        "transition_score = mood_distance + 0.30 * abs(tempo_norm_current - tempo_norm_previous)",
        language="python",
    )

    st.markdown("### Why this is different")
    st.write(
        "Most student recommenders only return tracks that match one static mood. "
        "MoodTune instead treats music discovery like path planning through emotional space."
    )


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Render the full MoodTune application."""
    _inject_css()
    _init_state()
    if _consume_boot_query_flag():
        st.rerun()

    with st.spinner("Loading dataset..."):
        df = load_full_dataset()

    if not _state_get_system_initialized():
        _render_immersive_boot_scene(df)
        return

    _render_ambient_word_layer()
    _render_initialized_header(df)

    st.markdown(
        """
<button class="theme-toggle" onclick="
  var b = document.documentElement;
  b.dataset.theme = b.dataset.theme === 'light' ? '' : 'light';
" title="Toggle light/dark theme">&#9728;</button>
<style>
.theme-toggle {
  position: fixed; top: 16px; right: 16px; z-index: 9999;
  background: var(--glass-bg, rgba(255,255,255,0.04));
  border: 1px solid var(--glass-border, rgba(255,255,255,0.10));
  color: var(--text-primary, #f0f0f0);
  border-radius: 50%; width: 36px; height: 36px;
  cursor: pointer; font-size: 1rem; line-height: 1;
  backdrop-filter: blur(20px);
  transition: background 0.2s ease;
}
.theme-toggle:hover { background: rgba(255,255,255,0.12); }
</style>
""",
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "Mood Space",
        "Journey",
        "Live Search",
        "Data Lab",
        "How It Works",
    ])
    with tabs[0]:
        _render_mood_space_tab(df)
    with tabs[1]:
        _render_journey_tab(df)
    with tabs[2]:
        _render_live_search_tab()
    with tabs[3]:
        _render_data_lab_tab()
    with tabs[4]:
        _render_how_it_works_tab(df)


if __name__ == "__main__":
    main()
