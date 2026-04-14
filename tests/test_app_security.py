"""Security and state regression tests for ui/app.py."""
from __future__ import annotations

from typing import Any, cast

import ui.app as app


def test_song_card_markup_escapes_html() -> None:
    markup = app._song_card_markup(
        step_num=1,
        track_name='<script>alert("x")</script>',
        artist='<img src=x onerror=alert(1)>',
        genre="<b>pop</b>",
        valence=0.5,
        energy=0.6,
    )

    assert "<script>" not in markup
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in markup
    assert "&lt;img src=x onerror=alert(1)&gt;" in markup
    assert "&lt;b&gt;pop&lt;/b&gt;" in markup


def test_state_coordinate_repair_recovers_invalid_data() -> None:
    app.st.session_state = cast(Any, {
        "journey_start": "broken",
        "_state_repairs_notified": {},
    })
    repaired = app._state_get_coordinate("journey_start")
    assert repaired is None
    assert app.st.session_state["journey_start"] is None


def test_state_selection_mode_repair_uses_default() -> None:
    app.st.session_state = cast(Any, {
        "selection_mode": "INVALID",
        "_state_repairs_notified": {},
    })
    mode = app._state_get_selection_mode()
    assert mode == "Start mood"


def test_state_survey_answers_repair_uses_default() -> None:
    app.st.session_state = cast(Any, {
        "survey_answers": {"q1": 99},
        "_state_repairs_notified": {},
    })
    answers = app._state_get_survey_answers()
    assert answers == {}


def test_state_optional_dict_rejects_non_dict() -> None:
    app.st.session_state = cast(Any, {
        "nlp_start_result": "bad",
        "_state_repairs_notified": {},
    })
    value = app._state_get_optional_dict("nlp_start_result")
    assert value is None


def test_spotify_search_url_encodes_query() -> None:
    url = app._spotify_search_url("A/B Track", "Artist & Co")
    assert url == "https://open.spotify.com/search/A%2FB+Track+Artist+%26+Co"


def test_state_show_full_3d_repair_uses_default_false() -> None:
    app.st.session_state = cast(Any, {
        "show_full_3d": "yes",
        "_state_repairs_notified": {},
    })
    assert app._state_get_show_full_3d() is False
