"""Tests for the Cascade controller (no real network calls)."""
from __future__ import annotations

import json

from llm_judge_cascade.cascade import (
    Cascade,
    CascadeRunResult,
    CascadeTier,
    make_default_cascade,
)
from llm_judge_cascade.client import ChatCompletionResult, JudgeClientConfig
from llm_judge_cascade.judges import JudgeVerdict


def _cfg() -> JudgeClientConfig:
    return JudgeClientConfig(api_base="http://x/v1", api_key="k")


def _make_judge_reply(verdict: str, confidence: float, **extra: object) -> str:
    payload = {"verdict": verdict, "confidence": confidence, "issues": []}
    payload.update(extra)
    return json.dumps(payload)


class _ScriptedChat:
    """Returns scripted ChatCompletionResults keyed by model name."""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(
        self,
        messages: list[dict],
        model: str,
        config: JudgeClientConfig,
        max_tokens: int | None,
    ) -> ChatCompletionResult:
        self.calls.append(model)
        if model not in self.responses:
            return ChatCompletionResult(content="", latency_ms=10, error="no script")
        return ChatCompletionResult(content=self.responses[model], latency_ms=20)


def _cost_fixed(usd: float):
    def fn(_model: str, _in_chars: int, _out_chars: int) -> float:
        return usd

    return fn


def _three_tier_cascade(
    chat: _ScriptedChat,
    cost_fn=None,
) -> Cascade:
    return make_default_cascade(_cfg(), chat_fn=chat, cost_fn=cost_fn)


# ─── happy paths ────────────────────────────────────────────────────────────


def test_confident_sound_stops_at_first_tier() -> None:
    chat = _ScriptedChat(
        {
            "anthropic/claude-haiku-4-5-20251001": _make_judge_reply("SOUND", 0.95),
        }
    )
    cascade = _three_tier_cascade(chat, cost_fn=_cost_fixed(0.01))
    result = cascade.judge(prompt="p", response="r")

    assert isinstance(result, CascadeRunResult)
    assert result.final_verdict is JudgeVerdict.SOUND
    assert result.final_confidence == 0.95
    assert chat.calls == ["anthropic/claude-haiku-4-5-20251001"]
    assert result.escalation_path == ["anthropic/claude-haiku-4-5-20251001"]
    assert result.total_cost_usd == 0.01
    assert len(result.tier_results) == 1


def test_uncertain_haiku_escalates_to_sonnet() -> None:
    chat = _ScriptedChat(
        {
            "anthropic/claude-haiku-4-5-20251001": _make_judge_reply("UNCERTAIN", 0.4),
            "anthropic/claude-sonnet-4-6": _make_judge_reply("SOUND", 0.88),
        }
    )
    cascade = _three_tier_cascade(chat)
    result = cascade.judge(prompt="p", response="r")

    assert chat.calls == [
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-sonnet-4-6",
    ]
    assert result.final_verdict is JudgeVerdict.SOUND
    assert result.escalation_path == chat.calls


def test_low_confidence_escalates_even_when_decisive() -> None:
    chat = _ScriptedChat(
        {
            "anthropic/claude-haiku-4-5-20251001": _make_judge_reply("SOUND", 0.5),
            "anthropic/claude-sonnet-4-6": _make_judge_reply("SOUND", 0.9),
        }
    )
    cascade = _three_tier_cascade(chat)
    result = cascade.judge(prompt="p", response="r")
    assert chat.calls == [
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-sonnet-4-6",
    ]
    assert result.final_verdict is JudgeVerdict.SOUND


def test_tier_disagreement_escalates_to_opus() -> None:
    """Haiku is decisively SOUND at low confidence, Sonnet says FABRICATED;
    that disagreement forces Opus to break the tie."""
    chat = _ScriptedChat(
        {
            "anthropic/claude-haiku-4-5-20251001": _make_judge_reply("SOUND", 0.5),
            "anthropic/claude-sonnet-4-6": _make_judge_reply("FABRICATED", 0.85),
            "anthropic/claude-opus-4-7": _make_judge_reply("FABRICATED", 0.95),
        }
    )
    cascade = _three_tier_cascade(chat)
    result = cascade.judge(prompt="p", response="r")
    assert chat.calls == [
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-7",
    ]
    assert result.final_verdict is JudgeVerdict.FABRICATED
    assert result.final_confidence == 0.95


def test_sonnet_escalates_on_flawed_per_default_preset() -> None:
    """Sonnet's default tier has escalate_if_flawed=True; Haiku does not."""
    chat = _ScriptedChat(
        {
            "anthropic/claude-haiku-4-5-20251001": _make_judge_reply("UNCERTAIN", 0.2),
            "anthropic/claude-sonnet-4-6": _make_judge_reply("FLAWED", 0.9),
            "anthropic/claude-opus-4-7": _make_judge_reply("FLAWED", 0.92),
        }
    )
    cascade = _three_tier_cascade(chat)
    result = cascade.judge(prompt="p", response="r")
    assert chat.calls[-1] == "anthropic/claude-opus-4-7"
    assert result.final_verdict is JudgeVerdict.FLAWED


def test_haiku_flawed_high_conf_stops_unless_configured() -> None:
    """Haiku tier defaults to escalate_if_flawed=False, so FLAWED at high
    confidence should terminate the cascade."""
    chat = _ScriptedChat(
        {
            "anthropic/claude-haiku-4-5-20251001": _make_judge_reply("FLAWED", 0.9),
        }
    )
    cascade = _three_tier_cascade(chat)
    result = cascade.judge(prompt="p", response="r")
    assert chat.calls == ["anthropic/claude-haiku-4-5-20251001"]
    assert result.final_verdict is JudgeVerdict.FLAWED


# ─── failure modes ──────────────────────────────────────────────────────────


def test_transport_error_records_issue_and_keeps_going() -> None:
    chat = _ScriptedChat(
        {
            "anthropic/claude-sonnet-4-6": _make_judge_reply("SOUND", 0.95),
        }
    )

    def chat_fn(messages, model, config, max_tokens):
        if model == "anthropic/claude-haiku-4-5-20251001":
            return ChatCompletionResult(content="", latency_ms=5, error="HTTP 500: boom")
        return chat(messages, model, config, max_tokens)

    cascade = make_default_cascade(_cfg(), chat_fn=chat_fn)
    result = cascade.judge(prompt="p", response="r")
    assert result.tier_results[0].verdict is JudgeVerdict.UNCERTAIN
    assert any("transport_error" in issue for issue in result.tier_results[0].issues)
    assert result.final_verdict is JudgeVerdict.SOUND
    assert chat.calls[-1] == "anthropic/claude-sonnet-4-6"


def test_total_cost_accumulates_across_tiers() -> None:
    chat = _ScriptedChat(
        {
            "anthropic/claude-haiku-4-5-20251001": _make_judge_reply("UNCERTAIN", 0.3),
            "anthropic/claude-sonnet-4-6": _make_judge_reply("SOUND", 0.9),
        }
    )
    cascade = _three_tier_cascade(chat, cost_fn=_cost_fixed(0.01))
    result = cascade.judge(prompt="p", response="r")
    assert result.total_cost_usd == 0.02


def test_empty_tiers_rejected() -> None:
    try:
        Cascade(tiers=[], config=_cfg())
    except ValueError:
        return
    raise AssertionError("Cascade with no tiers should have raised")


def test_custom_two_tier_cascade() -> None:
    tiers = [
        CascadeTier(model="cheap", min_confidence_to_stop=0.6),
        CascadeTier(model="expensive", min_confidence_to_stop=0.0),
    ]
    chat = _ScriptedChat(
        {
            "cheap": _make_judge_reply("SOUND", 0.95),
        }
    )
    cascade = Cascade(tiers, _cfg(), chat_fn=chat)
    result = cascade.judge(prompt="p", response="r")
    assert chat.calls == ["cheap"]
    assert result.final_verdict is JudgeVerdict.SOUND
