"""
app.py -- MoodTune Streamlit entry point.

Run from the project root:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CLEAN_DATA_PATH,
    DEFAULT_JOURNEY_STEPS,
    DEFAULT_PLAYLIST_SIZE,
    LOG_PATH,
    RAW_DATA_PATH,
    STYLE_PATH,
)
from src.data_loader import load_full_dataset
from src.journey import build_journey_tree, generate_mood_journey, journey_to_dataframe
from src.mood_mapper import SURVEY_QUESTIONS, map_to_vector
from src.nlp_mood import text_to_mood_vector
from src.recommender import build_model, recommend
from src.validator import get_cleaning_steps_log, run_pipeline
from src.visualizer import (
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
def _get_recommender_model():
    """Build the nearest-neighbour model once per dataset state."""
    return build_model(load_full_dataset())


@st.cache_resource(show_spinner=False)
def _get_journey_tree():
    """Build the KDTree for the journey engine once per dataset state."""
    return build_journey_tree(load_full_dataset())


# ── session state ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    """Create every session_state key used by the app."""
    defaults: dict[str, Any] = {
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── coordinate helpers ─────────────────────────────────────────────────────────

def _set_coordinate(key: str, coord: tuple[float, float], source: str) -> None:
    """Store one mood coordinate in session state with a readable source tag."""
    x_val, y_val = round(float(coord[0]), 4), round(float(coord[1]), 4)
    st.session_state[key] = (x_val, y_val)
    st.session_state[f"{key}_source"] = source


def _clear_coordinate(key: str) -> None:
    """Clear one stored mood coordinate and reset its source label."""
    st.session_state[key] = None
    st.session_state[f"{key}_source"] = "Not set"


def _reset_survey() -> None:
    """Clear the survey flow back to its initial state."""
    st.session_state["survey_step"] = 0
    st.session_state["survey_answers"] = {}
    st.session_state["survey_result"] = None


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
    if st.session_state["selection_mode"] == "Start mood":
        _set_coordinate("journey_start", latest, "Mood Space")
        return
    _set_coordinate("journey_target", latest, "Mood Space")


# ── shared UI blocks ───────────────────────────────────────────────────────────

def _render_selection_summary() -> None:
    """Show the currently active mood coordinates and their source."""
    start = st.session_state["journey_start"]
    target = st.session_state["journey_target"]
    col_start, col_target, col_actions = st.columns([1.2, 1.2, 0.8])

    with col_start:
        if start is None:
            st.info("Start mood not set yet.")
        else:
            st.success(
                f"Start mood: **{_quadrant_label(start)}**\n\n"
                f"Valence `{start[0]:.2f}` · Energy `{start[1]:.2f}`\n\n"
                f"Source: {st.session_state['journey_start_source']}"
            )

    with col_target:
        if target is None:
            st.info("Target mood not set yet.")
        else:
            st.success(
                f"Target mood: **{_quadrant_label(target)}**\n\n"
                f"Valence `{target[0]:.2f}` · Energy `{target[1]:.2f}`\n\n"
                f"Source: {st.session_state['journey_target_source']}"
            )

    with col_actions:
        st.radio(
            "Next chart click sets:",
            options=["Start mood", "Target mood"],
            key="selection_mode",
            horizontal=False,
        )
        if st.button("Clear start", width="stretch"):
            _clear_coordinate("journey_start")
            st.rerun()
        if st.button("Clear target", width="stretch"):
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
        if st.button("Map text to start mood", key="nlp_start_button", width="stretch"):
            result = text_to_mood_vector(start_text)
            st.session_state["nlp_start_result"] = result
            _set_coordinate("journey_start", result["coordinate"], "NLP")

        result = st.session_state.get("nlp_start_result")
        if result:
            st.caption(
                f"Matched: {', '.join(result['matched_words']) or 'none'} | "
                f"confidence {result['confidence']:.2f}"
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
        if st.button("Map text to target mood", key="nlp_target_button", width="stretch"):
            result = text_to_mood_vector(target_text)
            st.session_state["nlp_target_result"] = result
            _set_coordinate("journey_target", result["coordinate"], "NLP")

        result = st.session_state.get("nlp_target_result")
        if result:
            st.caption(
                f"Matched: {', '.join(result['matched_words']) or 'none'} | "
                f"confidence {result['confidence']:.2f}"
            )


def _render_survey_question(step: int) -> None:
    """Render one survey question with back/next navigation."""
    question = SURVEY_QUESTIONS[step - 1]
    labels = [option["label"] for option in question["options"]]
    previous = st.session_state["survey_answers"].get(question["id"])
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
        if step > 1 and st.button("Back", key=f"survey_back_{step}", width="stretch"):
            st.session_state["survey_step"] -= 1
            st.rerun()
    with col_next:
        if st.button("Next", key=f"survey_next_{step}", width="stretch", type="primary"):
            st.session_state["survey_answers"][question["id"]] = labels.index(selected) + 1
            st.session_state["survey_step"] += 1
            st.rerun()


def _render_survey_result(df: pd.DataFrame) -> None:
    """Render the classic survey outcome and optional instant recommendations."""
    answers = st.session_state["survey_answers"]
    result = map_to_vector(answers["q1"], answers["q2"], answers["q3"], answers["q4"])
    st.session_state["survey_result"] = result

    st.success(
        f"{result['mood_emoji']} Survey mood: **{result['mood_label']}** | "
        f"Valence `{result['coordinate'][0]:.2f}` · Energy `{result['coordinate'][1]:.2f}`"
    )
    col_start, col_target, col_reset = st.columns(3)
    with col_start:
        if st.button("Use survey as start mood", width="stretch"):
            _set_coordinate("journey_start", result["coordinate"], "Survey")
    with col_target:
        if st.button("Use survey as target mood", width="stretch"):
            _set_coordinate("journey_target", result["coordinate"], "Survey")
    with col_reset:
        if st.button("Reset survey", width="stretch"):
            _reset_survey()
            st.rerun()

    with st.spinner("Finding instant matches..."):
        instant = recommend(_get_recommender_model(), df, result["vector"], k=DEFAULT_PLAYLIST_SIZE)
    st.markdown("##### Classic instant matches")
    st.dataframe(
        instant[["track_name", "artist_name", "genre", "similarity_pct"]],
        hide_index=True,
        width="stretch",
    )


def _render_survey_fallback(df: pd.DataFrame) -> None:
    """Render the guided survey flow inside an expander."""
    with st.expander("Classic survey fallback", expanded=st.session_state["survey_step"] > 0):
        if st.session_state["survey_step"] == 0:
            st.write("Prefer a guided flow? The original four-question survey still works.")
            if st.button("Start survey", type="primary"):
                st.session_state["survey_step"] = 1
                st.rerun()
            return

        if 1 <= st.session_state["survey_step"] <= 4:
            _render_survey_question(st.session_state["survey_step"])
            return

        _render_survey_result(df)


# ── journey helpers ────────────────────────────────────────────────────────────

def _build_journey_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the current mood journey as a dataframe."""
    start = st.session_state["journey_start"]
    target = st.session_state["journey_target"]
    if start is None or target is None:
        return pd.DataFrame()
    journey = generate_mood_journey(
        _get_journey_tree(),
        df,
        start=start,
        target=target,
        n_steps=st.session_state["journey_steps"],
    )
    return journey_to_dataframe(journey)


def _spotify_search_url(track_name: str, artist_name: str) -> str:
    """Build a Spotify web search URL for a track."""
    query = f"{track_name} {artist_name}".replace(" ", "+")
    return f"https://open.spotify.com/search/{query}"


def _render_song_cards(journey_df: pd.DataFrame) -> None:
    """Render the journey playlist as a 3-column card grid."""
    rows = [journey_df.iloc[i:i + 3] for i in range(0, len(journey_df), 3)]
    for row_df in rows:
        cols = st.columns(3)
        for col, (_, track) in zip(cols, row_df.iterrows()):
            step_num = int(track["waypoint_idx"]) + 1
            track_name = str(track.get("track_name", "Unknown"))
            artist = str(track.get("artist_name", ""))
            genre = str(track.get("genre", ""))
            valence = float(track.get("valence", 0.0))
            energy = float(track.get("energy", 0.0))

            with col:
                st.html(
                    f"""
                    <div class="song-card">
                        <div class="song-card-step">Step {step_num}</div>
                        <div class="song-card-title">{track_name}</div>
                        <div class="song-card-artist">{artist}</div>
                        <div class="song-card-genre">{genre}</div>
                        <div class="song-card-badges">
                            <span class="stat-badge">V {valence:.2f}</span>
                            <span class="stat-badge">E {energy:.2f}</span>
                        </div>
                    </div>
                    """
                )
                search_url = _spotify_search_url(track_name, artist)
                st.link_button("Open Spotify", search_url, width="stretch")


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
        start_coord=st.session_state["journey_start"],
        target_coord=st.session_state["journey_target"],
    )
    chart_event = st.plotly_chart(
        figure,
        width="stretch",
        key="mood_space_chart",
        on_select="rerun",
        selection_mode=("points",),
        config={"scrollZoom": False, "displaylogo": False},
    )
    _apply_chart_selection(chart_event)

    # 3D Mood Space
    with st.expander("3D Mood Space — rotate, zoom, explore", expanded=False):
        st.caption(
            "Three axes: **Valence** (sad → happy) · **Energy** (chill → intense) · "
            "**Danceability** (still → groove). Coloured by genre using the Plasma colorscale."
        )
        with st.spinner("Building 3D mood space..."):
            fig_3d = mood_space_3d_figure(df)
        st.plotly_chart(
            fig_3d,
            width="stretch",
            config={"displaylogo": False},
        )

    _render_survey_fallback(df)


def _render_journey_tab(df: pd.DataFrame) -> None:
    """Render the generated journey playlist and supporting charts."""
    st.markdown("## Journey Playlist")

    if st.session_state["journey_start"] is None or st.session_state["journey_target"] is None:
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
    c1.metric("Start mood", _quadrant_label(st.session_state["journey_start"]))
    c2.metric("Target mood", _quadrant_label(st.session_state["journey_target"]))
    c3.metric("Avg transition score", f"{journey_df['transition_score'].mean():.3f}")

    mood_fig = mood_space_figure(
        df,
        start_coord=st.session_state["journey_start"],
        target_coord=st.session_state["journey_target"],
        journey_df=journey_df,
    )
    st.plotly_chart(
        mood_fig,
        width="stretch",
        config={"scrollZoom": False, "displaylogo": False},
    )

    st.plotly_chart(journey_progress_figure(journey_df), width="stretch")

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
        left.dataframe(raw_df.head(5), width="stretch")
        right.dataframe(clean_df.head(5), width="stretch")

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
