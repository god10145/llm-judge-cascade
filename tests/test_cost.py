"""Tests for the cost tracker and price-table helpers."""
from __future__ import annotations

import math

import pytest

from llm_judge_cascade.cost import (
    MODEL_PRICING,
    CostTracker,
    estimate_cost,
    estimate_cost_from_chars,
    estimate_tokens_from_chars,
)


def test_pricing_table_has_default_cascade_models() -> None:
    for model in (
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-7",
        "gemini/gemini-3.1-pro-preview",
    ):
        assert model in MODEL_PRICING
        in_rate, out_rate = MODEL_PRICING[model]
        assert in_rate > 0
        assert out_rate > 0


def test_estimate_tokens_from_chars_heuristic() -> None:
    assert estimate_tokens_from_chars("") == 0
    assert estimate_tokens_from_chars("a") == 1
    assert estimate_tokens_from_chars("hello world") == max(1, len("hello world") // 4)


def test_estimate_cost_known_model() -> None:
    cost = estimate_cost("anthropic/claude-opus-4-7", 1_000_000, 1_000_000)
    assert math.isclose(cost, 15.0 + 75.0)


def test_estimate_cost_unknown_model_is_zero() -> None:
    assert estimate_cost("vendor/no-such-model", 1_000_000, 1_000_000) == 0.0


def test_estimate_cost_from_chars_uses_div4_heuristic() -> None:
    cost = estimate_cost_from_chars("anthropic/claude-haiku-4-5-20251001", 4000, 400)
    expected = (1000 / 1_000_000) * 1.0 + (100 / 1_000_000) * 5.0
    assert math.isclose(cost, expected)


def test_cost_tracker_accumulates() -> None:
    tracker = CostTracker()
    cost1 = tracker.record_call("anthropic/claude-haiku-4-5-20251001", 1_000_000, 0)
    cost2 = tracker.record_call("anthropic/claude-sonnet-4-6", 0, 1_000_000)
    assert math.isclose(cost1, 1.0)
    assert math.isclose(cost2, 15.0)
    assert math.isclose(tracker.total_usd, 16.0)
    assert tracker.n_calls == 2
    assert math.isclose(tracker.per_model_usd["anthropic/claude-haiku-4-5-20251001"], 1.0)
    assert math.isclose(tracker.per_model_usd["anthropic/claude-sonnet-4-6"], 15.0)


def test_cost_tracker_record_chars_round_trip() -> None:
    tracker = CostTracker()
    cost = tracker.record_chars("anthropic/claude-opus-4-7", 4000, 400)
    # 4000 chars -> 1000 tok input, 400 chars -> 100 tok output
    expected = (1000 / 1_000_000) * 15.0 + (100 / 1_000_000) * 75.0
    assert math.isclose(cost, expected)


def test_cost_tracker_over_cap_behavior() -> None:
    tracker = CostTracker()
    assert tracker.over_cap(None) is False
    tracker.record_call("anthropic/claude-opus-4-7", 1_000_000, 1_000_000)
    assert tracker.over_cap(50.0) is True
    assert tracker.over_cap(1000.0) is False


@pytest.mark.parametrize(
    "input_chars,output_chars",
    [(0, 0), (1, 1), (3, 3), (4, 4), (1000, 0), (0, 1000)],
)
def test_estimate_cost_from_chars_nonnegative(input_chars: int, output_chars: int) -> None:
    cost = estimate_cost_from_chars(
        "anthropic/claude-sonnet-4-6", input_chars, output_chars
    )
    assert cost >= 0.0
