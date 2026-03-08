import pytest

from config import (
    calculate_clip_generation_cost,
    calculate_custom_clip_generation_cost,
    calculate_video_analysis_cost,
    normalize_clip_generation_credits,
    normalize_custom_clip_generation_credits,
)


@pytest.mark.parametrize(
    ("duration_seconds", "expected_credits"),
    [
        (0, 0),
        (60, 1),
        (61, 2),
        (3599, 60),
        (3601, 61),
    ],
)
def test_calculate_video_analysis_cost_rounds_up_started_minutes(
    duration_seconds: int, expected_credits: int
):
    assert calculate_video_analysis_cost(duration_seconds) == expected_credits


def test_calculate_clip_generation_cost_returns_base_cost_regardless_of_tier():
    assert calculate_clip_generation_cost(False, "basic") == 3
    assert calculate_clip_generation_cost(True, "basic") == 3
    assert calculate_clip_generation_cost(False, "pro") == 3
    assert calculate_clip_generation_cost(True, "pro") == 3
    assert calculate_clip_generation_cost(False, "enterprise") == 3
    assert calculate_clip_generation_cost(True, "enterprise") == 3
    assert calculate_clip_generation_cost(False, "free") == 3
    assert calculate_clip_generation_cost(True, "free") == 3


def test_calculate_custom_clip_generation_cost_never_drops_below_base_cost():
    assert calculate_custom_clip_generation_cost(False, "basic") == 3
    assert calculate_custom_clip_generation_cost(True, "basic") == 3
    assert calculate_custom_clip_generation_cost(False, "pro") == 3
    assert calculate_custom_clip_generation_cost(True, "pro") == 3
    assert calculate_custom_clip_generation_cost(False, "free") == 3
    assert calculate_custom_clip_generation_cost(True, "free") == 3


@pytest.mark.parametrize(
    ("raw_credits", "expected"),
    [
        (None, 3),
        (0, 3),
        (-2, 3),
        ("", 3),
        ("2", 3),
        (3, 3),
        (4, 4),
    ],
)
def test_normalize_custom_clip_generation_credits(raw_credits: object, expected: int):
    assert normalize_custom_clip_generation_credits(raw_credits) == expected


@pytest.mark.parametrize(
    ("raw_credits", "expected"),
    [
        (None, 3),
        (0, 3),
        (-10, 3),
        (1, 3),
        (3, 3),
        (6, 6),
    ],
)
def test_normalize_clip_generation_credits_enforces_minimum_charge(
    raw_credits: object, expected: int
):
    assert normalize_clip_generation_credits(raw_credits) == expected


@pytest.mark.parametrize(
    ("raw_credits", "expected"),
    [
        (None, 0),
        (0, 0),
        (-10, 0),
        (1, 1),
        (3, 3),
    ],
)
def test_normalize_clip_generation_credits_allows_free_generation_when_minimum_is_zero(
    raw_credits: object, expected: int
):
    assert normalize_clip_generation_credits(raw_credits, minimum_credits=0) == expected
