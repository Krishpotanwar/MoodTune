"""
app.py -- MoodTune Streamlit entry point.

Run from the project root:
    streamlit run ui/app.py
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from html import escape
from numbers import Real
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
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
    """Generate the current mood journey as a dataframe."""
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
) -> str:
    """Build escaped song-card HTML so dataset text is never rendered as raw HTML."""
    safe_track_name = escape(track_name, quote=True)
    safe_artist = escape(artist, quote=True)
    safe_genre = escape(genre, quote=True)
    return f"""
                    <div class="song-card">
                        <div class="song-card-step">Step {step_num}</div>
                        <div class="song-card-title">{safe_track_name}</div>
                        <div class="song-card-artist">{safe_artist}</div>
                        <div class="song-card-genre">{safe_genre}</div>
                        <div class="song-card-badges">
                            <span class="stat-badge">V {valence:.2f}</span>
                            <span class="stat-badge">E {energy:.2f}</span>
                        </div>
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
                st.html(_song_card_markup(step_num, track_name, artist, genre, valence, energy))
                search_url = _spotify_search_url(track_name, artist)
                st.link_button("Open Spotify", search_url, use_container_width=True)


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

    st.title("MoodTune")
    st.caption(
        "Set a start mood and a target mood using song particles, free text, or the classic survey. "
        f"{len(df):,} tracks are ready for exploration."
    )
    _render_selection_summary()

    st.markdown("### Natural-language mood mapping")
    _render_text_mapper()

    st.markdown("### Mood-space explorer")
    st.caption(
        "Select real song points in the chart. "
        "Use the radio control above to choose whether the next click sets the start or target mood."
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
        "Chart description: this 2D chart maps each track by valence (x-axis) and energy (y-axis). "
        "Start and target markers appear when selected."
    )

    # 3D Mood Space
    with st.expander("3D Mood Space — rotate, zoom, explore", expanded=False):
        preview_sample = min(DEFAULT_3D_SAMPLE, len(df))
        showing_full_3d = _state_get_show_full_3d()
        if len(df) > preview_sample and not showing_full_3d:
            if st.button(f"Show more points ({len(df):,} total)", key="show_more_3d", use_container_width=True):
                _state_set("show_full_3d", True)
                st.rerun()
        if len(df) > preview_sample and showing_full_3d:
            if st.button(
                f"Back to faster preview ({preview_sample:,} points)",
                key="show_less_3d",
                use_container_width=True,
            ):
                _state_set("show_full_3d", False)
                st.rerun()
        sample_size = len(df) if showing_full_3d else preview_sample

        st.caption(
            "Three axes: **Valence** (sad → happy) · **Energy** (chill → intense) · "
            "**Danceability** (still → groove). Genre uses both colour and marker symbols."
        )
        with st.spinner("Building 3D mood space..."):
            fig_3d = mood_space_3d_figure(df, sample_size=sample_size)
        st.plotly_chart(
            fig_3d,
            use_container_width=True,
            config={"displaylogo": False},
        )
        st.caption(
            "Chart description: each point is a song in 3D mood space. "
            "Colour and marker shape both encode genre groups for accessibility."
        )

    _render_survey_fallback(df)


def _render_journey_tab(df: pd.DataFrame) -> None:
    """Render the generated journey playlist and supporting charts."""
    st.markdown("## Journey Playlist")

    start = _state_get_coordinate("journey_start")
    target = _state_get_coordinate("journey_target")
    if start is None or target is None:
        st.warning("Set both a start mood and a target mood in the Mood Space tab to generate a journey.")
        return

    st.slider(
        "Journey length",
        min_value=8,
        max_value=24,
        key="journey_steps",
        help="Longer playlists create smoother transitions through the mood space.",
    )

    with st.spinner("Generating your mood journey..."):
        journey_df = _build_journey_dataframe(df)

    if journey_df.empty:
        st.error("No journey could be generated from the current selection.")
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
        "Chart description: this shows the generated path from start mood to target mood through selected tracks."
    )

    st.plotly_chart(journey_progress_figure(journey_df), use_container_width=True)
    st.caption(
        "Chart description: line chart of valence and energy progression at each playlist step."
    )

    st.markdown("### Your journey")
    _render_song_cards(journey_df)


def _render_data_lab_tab() -> None:
    """Render the professor-facing cleaning pipeline walkthrough."""
    st.markdown("## Data Lab")
    st.caption("This tab demonstrates the raw CSV cleaning pipeline that prepares the recommendation dataset.")

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
        st.pyplot(feature_correlation_heatmap(clean_df))

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

    with st.spinner("Loading dataset..."):
        df = load_full_dataset()

    tabs = st.tabs(["Mood Space", "Journey", "Data Lab", "How It Works"])
    with tabs[0]:
        _render_mood_space_tab(df)
    with tabs[1]:
        _render_journey_tab(df)
    with tabs[2]:
        _render_data_lab_tab()
    with tabs[3]:
        _render_how_it_works_tab(df)


if __name__ == "__main__":
    main()
