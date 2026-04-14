"""Tests for src/nlp_mood.py."""
from __future__ import annotations

import json
from pathlib import Path

from src.nlp_mood import load_mood_lexicon, text_to_mood_vector


def test_text_to_mood_vector_detects_high_energy_positive_language() -> None:
    result = text_to_mood_vector("I feel happy, excited, and energetic today")
    assert result["valence"] > 0.75
    assert result["energy"] > 0.75
    assert "happy" in result["matched_words"]


def test_text_to_mood_vector_detects_calm_language() -> None:
    result = text_to_mood_vector("I want something calm and peaceful to unwind")
    assert result["valence"] > 0.55
    assert result["energy"] < 0.30
    assert "calm" in result["matched_words"]


def test_text_to_mood_vector_falls_back_to_neutral_when_no_terms_match() -> None:
    result = text_to_mood_vector("zxqv lorem ipsum no known mood words")
    assert result["coordinate"] == (0.5, 0.5)
    assert result["confidence"] == 0.0


def test_custom_lexicon_file_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps({"sunrise": [0.77, 0.44]}), encoding="utf-8")
    loaded = load_mood_lexicon(path)
    assert loaded["sunrise"] == (0.77, 0.44)


def test_text_to_mood_vector_uses_custom_lexicon(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps({"deep focus": [0.61, 0.32]}), encoding="utf-8")
    result = text_to_mood_vector("Need deep focus for work", lexicon_path=path)
    assert result["coordinate"] == (0.61, 0.32)
    assert result["matched_words"] == ["deep focus"]


def test_load_mood_lexicon_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    try:
        load_mood_lexicon(path)
        raise AssertionError("Expected ValueError for non-object lexicon JSON.")
    except ValueError as exc:
        assert "JSON object" in str(exc)


def test_load_mood_lexicon_rejects_invalid_coordinate_shape(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps({"focus": [0.5]}), encoding="utf-8")
    try:
        load_mood_lexicon(path)
        raise AssertionError("Expected ValueError for malformed coordinate list.")
    except ValueError as exc:
        assert "list of two numbers" in str(exc)


def test_load_mood_lexicon_rejects_out_of_bounds_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps({"focus": [1.2, 0.2]}), encoding="utf-8")
    try:
        load_mood_lexicon(path)
        raise AssertionError("Expected ValueError for out-of-range coordinate values.")
    except ValueError as exc:
        assert "within [0.0, 1.0]" in str(exc)
