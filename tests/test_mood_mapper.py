"""Tests for src/mood_mapper.py."""
from __future__ import annotations

import pytest

from src.mood_mapper import SURVEY_QUESTIONS, map_to_vector

_VALID = (1, 2, 3, 4)


# ── return shape ───────────────────────────────────────────────────────────────

def test_returns_all_expected_keys() -> None:
    result = map_to_vector(1, 1, 1, 1)
    expected = {"energy", "valence", "tempo_norm", "acousticness",
                "vector", "mood_label", "mood_emoji"}
    assert expected <= result.keys()


def test_vector_has_four_elements() -> None:
    result = map_to_vector(2, 3, 1, 4)
    assert len(result["vector"]) == 4


# ── value ranges ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("q1,q2,q3,q4", [
    (1, 1, 1, 1),
    (4, 4, 4, 4),
    (2, 3, 1, 4),
    (3, 2, 4, 1),
])
def test_all_feature_values_in_unit_interval(q1: int, q2: int, q3: int, q4: int) -> None:
    result = map_to_vector(q1, q2, q3, q4)
    for key in ("energy", "valence", "tempo_norm", "acousticness"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of [0, 1]: {result[key]}"


# ── mood quadrant labels ───────────────────────────────────────────────────────

@pytest.mark.parametrize("q1,q2,expected_label", [
    (4, 4, "Joyful"),     # Pumped + Great → high energy + high valence
    (4, 1, "Angry"),      # Pumped + Very negative → high energy + low valence
    (1, 4, "Relaxed"),    # Exhausted + Great → low energy + high valence
    (1, 1, "Depressed"),  # Exhausted + Very negative → low energy + low valence
])
def test_mood_quadrant_label(q1: int, q2: int, expected_label: str) -> None:
    result = map_to_vector(q1, q2, 2, 2)
    assert result["mood_label"] == expected_label


def test_mood_emoji_is_one_of_valid_set() -> None:
    for q1 in _VALID:
        for q2 in _VALID:
            result = map_to_vector(q1, q2, 2, 2)
            assert result["mood_emoji"] in ("😄", "😤", "😌", "😔")


# ── score ordering ─────────────────────────────────────────────────────────────

def test_q1_energy_increases_with_answer() -> None:
    scores = [map_to_vector(q, 2, 2, 2)["energy"] for q in _VALID]
    assert scores == sorted(scores)


def test_q2_valence_increases_with_answer() -> None:
    scores = [map_to_vector(2, q, 2, 2)["valence"] for q in _VALID]
    assert scores == sorted(scores)


def test_q3_tempo_increases_with_answer() -> None:
    scores = [map_to_vector(2, 2, q, 2)["tempo_norm"] for q in _VALID]
    assert scores == sorted(scores)


def test_q4_acousticness_decreases_with_answer() -> None:
    # Alone (1) → most acoustic; Big crowd (4) → least acoustic
    scores = [map_to_vector(2, 2, 2, q)["acousticness"] for q in _VALID]
    assert scores == sorted(scores, reverse=True)


# ── invalid inputs ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [0, 5, -1, 99, "2", 1.5])
def test_raises_value_error_for_invalid_q1(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        map_to_vector(bad, 2, 2, 2)


@pytest.mark.parametrize("bad", [0, 5, -1])
def test_raises_value_error_for_invalid_q2(bad: int) -> None:
    with pytest.raises(ValueError):
        map_to_vector(2, bad, 2, 2)


# ── survey question definitions ────────────────────────────────────────────────

def test_survey_has_four_questions() -> None:
    assert len(SURVEY_QUESTIONS) == 4


def test_each_question_has_four_options() -> None:
    for q in SURVEY_QUESTIONS:
        assert len(q["options"]) == 4


def test_each_option_has_label_and_score() -> None:
    for q in SURVEY_QUESTIONS:
        for opt in q["options"]:
            assert "label" in opt
            assert "score" in opt
            assert 0.0 <= opt["score"] <= 1.0


# ── all 256 combinations are valid ────────────────────────────────────────────

def test_all_256_answer_combinations_produce_valid_results() -> None:
    for q1 in _VALID:
        for q2 in _VALID:
            for q3 in _VALID:
                for q4 in _VALID:
                    result = map_to_vector(q1, q2, q3, q4)
                    assert len(result["vector"]) == 4
                    assert result["mood_label"] in (
                        "Joyful", "Angry", "Relaxed", "Depressed"
                    )
