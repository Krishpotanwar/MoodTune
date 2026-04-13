"""
visualizer.py — Chart generation functions for MoodTune.

All functions return figure objects (never call .show()) so Streamlit
can render them with st.plotly_chart / st.pyplot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend required for Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_SCATTER_SAMPLE, MOOD_AXIS_MAX, MOOD_AXIS_MIN

# ── theme constants ────────────────────────────────────────────────────────────

_BG_MAIN  = "#0f0f1a"
_BG_CARD  = "#1a1a2e"
_ACCENT   = "#a855f7"
_CYAN     = "#06b6d4"
_CORAL    = "#ff6b6b"
_MINT     = "#7ae7c7"
_GOLD     = "#f7b538"

_FEATURE_LABELS: list[str] = ["Energy", "Valence", "Tempo", "Acousticness"]
_FEATURE_KEYS:   list[str] = ["energy", "valence", "tempo_norm", "acousticness"]

_QUADRANT_COLORS: dict[str, str] = {
    "Joyful":    "#f59e0b",
    "Angry":     "#ef4444",
    "Relaxed":   "#10b981",
    "Depressed": "#6366f1",
}

_QUADRANT_LABELS: list[tuple[float, float, str]] = [
    (0.75, 0.75, "Joyful 😄"),
    (0.25, 0.75, "Angry 😤"),
    (0.75, 0.25, "Relaxed 😌"),
    (0.25, 0.25, "Depressed 😔"),
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _quadrant_label(energy: float, valence: float) -> str:
    """Return the mood quadrant name for a given (energy, valence) pair."""
    if energy >= 0.5 and valence >= 0.5:
        return "Joyful"
    if energy >= 0.5 and valence < 0.5:
        return "Angry"
    if energy < 0.5 and valence >= 0.5:
        return "Relaxed"
    return "Depressed"


# ── Chart 1 ────────────────────────────────────────────────────────────────────

def mood_vs_recommended_bar(
    user_vector: dict,
    recommendations: pd.DataFrame,
) -> go.Figure:
    """
    Grouped bar chart: user target features vs average of recommended tracks.

    Args:
        user_vector:     dict with keys energy, valence, tempo_norm, acousticness.
        recommendations: DataFrame from recommender.recommend().

    Returns:
        Plotly Figure with two bar groups ("Your Mood" and "Avg Recommended").
    """
    user_vals: list[float] = [user_vector[k] for k in _FEATURE_KEYS]
    avg_vals:  list[float] = [float(recommendations[k].mean()) for k in _FEATURE_KEYS]

    fig = go.Figure(data=[
        go.Bar(
            name="Your Mood",
            x=_FEATURE_LABELS,
            y=user_vals,
            marker_color=_ACCENT,
            text=[f"{v:.2f}" for v in user_vals],
            textposition="outside",
        ),
        go.Bar(
            name="Avg Recommended",
            x=_FEATURE_LABELS,
            y=avg_vals,
            marker_color=_CYAN,
            text=[f"{v:.2f}" for v in avg_vals],
            textposition="outside",
        ),
    ])
    fig.update_layout(
        barmode="group",
        title="Your Mood vs. Average of Recommended Tracks",
        paper_bgcolor=_BG_MAIN,
        plot_bgcolor=_BG_CARD,
        font=dict(color="white", size=13),
        legend=dict(bgcolor=_BG_CARD, bordercolor="#444"),
        yaxis=dict(range=[0, 1.15], gridcolor="#333", title="Score (0–1)"),
        xaxis=dict(title="Audio Feature"),
        margin=dict(t=60, b=40),
    )
    return fig


# ── Chart 2 ────────────────────────────────────────────────────────────────────

def feature_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Seaborn correlation heatmap for the four audio features
    (plus danceability if present in the dataset).

    Args:
        df: Cleaned DataFrame from validator.run_pipeline().

    Returns:
        Matplotlib Figure (pass to st.pyplot()).
    """
    cols = list(_FEATURE_KEYS)
    if "danceability" in df.columns:
        cols.append("danceability")

    corr = df[cols].corr()
    tick_labels = [c.replace("_norm", "").replace("_", " ").title() for c in cols]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(_BG_MAIN)
    ax.set_facecolor(_BG_MAIN)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
        linewidths=0.5,
        linecolor="#333",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Audio Feature Correlations", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white", labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="white")
    plt.setp(ax.get_yticklabels(), rotation=0,  color="white")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white")

    plt.tight_layout()
    return fig


# ── Chart 3 ────────────────────────────────────────────────────────────────────

def mood_cluster_scatter(
    df: pd.DataFrame,
    user_vector: dict,
    recommendations: pd.DataFrame,
) -> go.Figure:
    """
    Scatter plot of valence (x) vs energy (y).

    - Grey dots: 2 000-row random sample from the full cleaned dataset.
    - Coloured dots: the 8 recommended tracks, colour-coded by mood quadrant.
    - Gold star: the user's target point.
    - Dashed lines divide the four quadrants at (0.5, 0.5).

    Args:
        df:              Cleaned DataFrame.
        user_vector:     dict with energy, valence keys.
        recommendations: DataFrame from recommender.recommend().

    Returns:
        Plotly Figure.
    """
    sample = df.sample(n=min(2_000, len(df)), random_state=42)
    fig = go.Figure()

    # ── background sample ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=sample["valence"],
        y=sample["energy"],
        mode="markers",
        marker=dict(color="#555", opacity=0.25, size=4),
        name="All tracks",
        hoverinfo="skip",
    ))

    # ── quadrant dividers ──────────────────────────────────────────────────────
    fig.add_vline(x=0.5, line_dash="dash", line_color="#555", opacity=0.6)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#555", opacity=0.6)

    # ── quadrant labels ────────────────────────────────────────────────────────
    for x_pos, y_pos, label in _QUADRANT_LABELS:
        fig.add_annotation(
            x=x_pos, y=y_pos, text=label,
            showarrow=False,
            font=dict(color="#777", size=11),
            xanchor="center",
        )

    # ── recommended tracks ─────────────────────────────────────────────────────
    seen_quadrants: set[str] = set()
    for _, row in recommendations.iterrows():
        quadrant = _quadrant_label(float(row["energy"]), float(row["valence"]))
        show_legend = quadrant not in seen_quadrants
        seen_quadrants.add(quadrant)

        name = str(row.get("track_name", ""))
        artist = str(row.get("artist_name", ""))
        fig.add_trace(go.Scatter(
            x=[row["valence"]],
            y=[row["energy"]],
            mode="markers+text",
            marker=dict(
                color=_QUADRANT_COLORS[quadrant],
                size=12,
                line=dict(color="white", width=1.2),
            ),
            text=[name[:22] + ("…" if len(name) > 22 else "")],
            textposition="top center",
            textfont=dict(size=8, color="white"),
            name=quadrant,
            legendgroup=quadrant,
            showlegend=show_legend,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Artist: {artist}<br>"
                f"Energy: {row['energy']:.2f} | Valence: {row['valence']:.2f}<br>"
                f"Quadrant: {quadrant}<extra></extra>"
            ),
        ))

    # ── user star ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[user_vector["valence"]],
        y=[user_vector["energy"]],
        mode="markers",
        marker=dict(symbol="star", color="#f59e0b", size=20,
                    line=dict(color="white", width=1.5)),
        name="You",
        hovertemplate=(
            "<b>Your Mood</b><br>"
            f"Energy: {user_vector['energy']:.2f}<br>"
            f"Valence: {user_vector['valence']:.2f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Mood Quadrant Map — Where Do the Recommendations Land?",
        xaxis=dict(title="Valence (Positivity) →", range=[0, 1], gridcolor="#333"),
        yaxis=dict(title="Energy →",               range=[0, 1], gridcolor="#333"),
        paper_bgcolor=_BG_MAIN,
        plot_bgcolor=_BG_CARD,
        font=dict(color="white"),
        legend=dict(bgcolor=_BG_CARD, bordercolor="#444"),
        margin=dict(t=60, b=50),
    )
    return fig


def _sample_mood_space(df: pd.DataFrame, sample_size: int = DEFAULT_SCATTER_SAMPLE) -> pd.DataFrame:
    """Subsample the full dataset for responsive mood-space rendering."""
    if len(df) <= sample_size:
        return df.copy()
    return df.sample(n=sample_size, random_state=42).copy()


def _coord_trace(
    coord: tuple[float, float],
    label: str,
    color: str,
    symbol: str,
) -> go.Scatter:
    """Create one highlighted mood coordinate marker."""
    return go.Scatter(
        x=[coord[0]],
        y=[coord[1]],
        mode="markers+text",
        marker=dict(size=16, color=color, symbol=symbol, line=dict(color="white", width=1.5)),
        text=[label],
        textposition="top center",
        name=label,
        hovertemplate=f"<b>{label}</b><br>Valence: {coord[0]:.2f}<br>Energy: {coord[1]:.2f}<extra></extra>",
    )


def mood_space_figure(
    df: pd.DataFrame,
    start_coord: tuple[float, float] | None = None,
    target_coord: tuple[float, float] | None = None,
    journey_df: pd.DataFrame | None = None,
    sample_size: int = DEFAULT_SCATTER_SAMPLE,
) -> go.Figure:
    """
    Build the interactive mood-space scatter used by the new UI.

    Args:
        df: Full track dataframe.
        start_coord: Optional selected starting mood.
        target_coord: Optional selected target mood.
        journey_df: Optional generated journey dataframe.
        sample_size: Render sample size for browser performance.

    Returns:
        Plotly figure ready for Streamlit rendering.
    """
    sample = _sample_mood_space(df, sample_size=sample_size)
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=sample["valence"],
            y=sample["energy"],
            mode="markers",
            marker=dict(
                size=5,
                color=sample["valence"],
                colorscale="Turbo",
                opacity=0.28,
                showscale=False,
            ),
            customdata=np.stack([sample["track_name"], sample["artist_name"]], axis=1),
            name="Songs",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Artist: %{customdata[1]}<br>"
                "Valence: %{x:.2f}<br>"
                "Energy: %{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0.5, line_dash="dot", line_color="rgba(255,255,255,0.20)")
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.20)")

    for x_pos, y_pos, label in _QUADRANT_LABELS:
        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=label,
            showarrow=False,
            font=dict(color="rgba(255,255,255,0.45)", size=11),
        )

    if journey_df is not None and not journey_df.empty:
        fig = add_journey_path(fig, journey_df)
    if start_coord is not None:
        fig.add_trace(_coord_trace(start_coord, "START", _MINT, "circle"))
    if target_coord is not None:
        fig.add_trace(_coord_trace(target_coord, "TARGET", _CORAL, "diamond"))

    fig.update_layout(
        title="Mood Space Explorer",
        xaxis=dict(title="Valence (Positivity)", range=[MOOD_AXIS_MIN, MOOD_AXIS_MAX], gridcolor="#2e3446"),
        yaxis=dict(title="Energy", range=[MOOD_AXIS_MIN, MOOD_AXIS_MAX], gridcolor="#2e3446"),
        paper_bgcolor=_BG_MAIN,
        plot_bgcolor=_BG_CARD,
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.25)", bordercolor="rgba(255,255,255,0.12)"),
        margin=dict(t=60, b=40, l=40, r=20),
        height=560,
        dragmode="select",
        clickmode="event+select",
    )
    return fig


def add_journey_path(fig: go.Figure, journey_df: pd.DataFrame) -> go.Figure:
    """Overlay the generated journey path and nodes on top of a mood-space plot."""
    if journey_df.empty:
        return fig

    hover_text = (
        "<b>" + journey_df["track_name"].astype(str) + "</b><br>"
        + "Artist: " + journey_df["artist_name"].astype(str) + "<br>"
        + "Step: " + (journey_df["waypoint_idx"] + 1).astype(str)
    )
    color_scale = np.linspace(0.0, 1.0, len(journey_df))

    fig.add_trace(
        go.Scatter(
            x=journey_df["valence"],
            y=journey_df["energy"],
            mode="lines+markers",
            line=dict(color="rgba(255,255,255,0.45)", width=2, dash="dot"),
            marker=dict(
                size=11,
                color=color_scale,
                colorscale=[[0.0, _MINT], [0.5, _GOLD], [1.0, _CORAL]],
                line=dict(color="white", width=1),
            ),
            text=hover_text,
            name="Journey",
            hovertemplate="%{text}<extra></extra>",
        )
    )
    return fig


def journey_progress_figure(journey_df: pd.DataFrame) -> go.Figure:
    """Plot how valence and energy evolve across journey steps."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=journey_df["waypoint_idx"] + 1,
            y=journey_df["valence"],
            mode="lines+markers",
            name="Valence",
            line=dict(color=_GOLD, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=journey_df["waypoint_idx"] + 1,
            y=journey_df["energy"],
            mode="lines+markers",
            name="Energy",
            line=dict(color=_CYAN, width=3),
        )
    )
    fig.update_layout(
        title="Journey Drift Across Playlist Steps",
        xaxis=dict(title="Journey Step"),
        yaxis=dict(title="Mood Score", range=[0, 1]),
        paper_bgcolor=_BG_MAIN,
        plot_bgcolor=_BG_CARD,
        font=dict(color="white"),
        legend=dict(bgcolor=_BG_CARD, bordercolor="#444"),
        margin=dict(t=60, b=40),
    )
    return fig


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "track_name": ["A", "B", "C", "D"],
            "artist_name": ["One", "Two", "Three", "Four"],
            "valence": [0.20, 0.40, 0.65, 0.88],
            "energy": [0.18, 0.38, 0.62, 0.90],
        }
    )
    fig = mood_space_figure(sample_df, start_coord=(0.20, 0.18), target_coord=(0.88, 0.90))
    print(f"Visualiser verification figure with {len(fig.data)} traces created successfully.")
