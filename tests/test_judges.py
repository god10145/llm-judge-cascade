"""Tests for judge result parsing and message construction."""
from __future__ import annotations

import json

from llm_judge_cascade.judges import (
    DEFAULT_JUDGE_PROMPT_GENERIC,
    JudgeResult,
    JudgeVerdict,
    build_judge_messages,
    parse_judge_response,
)


def test_parse_clean_json() -> None:
    raw = json.dumps(
        {
            "verdict": "SOUND",
            "confidence": 0.91,
            "issues": [],
            "annotation": "Looks fine.",
            "rewritten_response": None,
        }
    )
    result = parse_judge_response(raw, model="claude-haiku-4-5-20251001", latency_ms=420)
    assert isinstance(result, JudgeResult)
    assert result.verdict is JudgeVerdict.SOUND
    assert result.confidence == 0.91
    assert result.issues == []
    assert result.annotation == "Looks fine."
    assert result.rewritten_response is None
    assert result.model == "claude-haiku-4-5-20251001"
    assert result.latency_ms == 420


def test_parse_fenced_json() -> None:
    raw = (
        "Here is my evaluation:\n\n```json\n"
        + json.dumps(
            {
                "verdict": "FLAWED",
                "confidence": 0.6,
                "issues": ["Skips a reasoning step."],
                "annotation": "Decent but rushed.",
                "rewritten_response": "A clearer answer.",
            }
        )
        + "\n```\n"
    )
    result = parse_judge_response(raw)
    assert result.verdict is JudgeVerdict.FLAWED
    assert result.confidence == 0.6
    assert result.issues == ["Skips a reasoning step."]
    assert result.annotation == "Decent but rushed."
    assert result.rewritten_response == "A clearer answer."


def test_parse_brace_grep_fallback() -> None:
    raw = (
        "Sure thing! Here you go:\n"
        '{"verdict": "FABRICATED", "confidence": 0.88, "issues": ["Invents a source."]}'
        "\n\nLet me know if you need more."
    )
    result = parse_judge_response(raw)
    assert result.verdict is JudgeVerdict.FABRICATED
    assert result.confidence == 0.88
    assert result.issues == ["Invents a source."]


def test_parse_malformed_returns_uncertain() -> None:
    result = parse_judge_response("the model just wrote prose, no JSON at all")
    assert result.verdict is JudgeVerdict.UNCERTAIN
    assert result.confidence == 0.0
    assert any("judge_parse_failed" in issue for issue in result.issues)


def test_parse_missing_fields_defaults_safely() -> None:
    raw = json.dumps({"verdict": "sound"})
    result = parse_judge_response(raw)
    assert result.verdict is JudgeVerdict.SOUND
    assert result.confidence == 0.0
    assert result.issues == []
    assert result.annotation is None
    assert result.rewritten_response is None


def test_verdict_unknown_collapses_to_uncertain() -> None:
    raw = json.dumps({"verdict": "definitely-bad", "confidence": 0.9})
    result = parse_judge_response(raw)
    assert result.verdict is JudgeVerdict.UNCERTAIN


def test_confidence_clamped_to_unit_interval() -> None:
    raw = json.dumps({"verdict": "SOUND", "confidence": 4.2})
    assert parse_judge_response(raw).confidence == 1.0
    raw_neg = json.dumps({"verdict": "SOUND", "confidence": -0.5})
    assert parse_judge_response(raw_neg).confidence == 0.0


def test_issues_accepts_scalar_string() -> None:
    raw = json.dumps({"verdict": "FLAWED", "confidence": 0.5, "issues": "single complaint"})
    assert parse_judge_response(raw).issues == ["single complaint"]


def test_build_judge_messages_has_system_and_user() -> None:
    messages = build_judge_messages(
        prompt="Summarize this article.",
        response="A short summary.",
        judge_system_prompt=DEFAULT_JUDGE_PROMPT_GENERIC,
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "agent_input" in messages[1]["content"]
    assert "agent_response" in messages[1]["content"]


def test_build_judge_messages_includes_extra_context() -> None:
    messages = build_judge_messages(
        prompt="?",
        response="!",
        judge_system_prompt="be brief",
        extra_context={"source_url": "example.com"},
    )
    assert "extra_context" in messages[1]["content"]
    assert "example.com" in messages[1]["content"]


def test_to_dict_round_trip() -> None:
    result = JudgeResult(
        verdict=JudgeVerdict.SOUND,
        confidence=0.75,
        issues=["minor nit"],
        annotation="OK",
        rewritten_response=None,
        raw_response="{}",
        model="m",
        latency_ms=10,
    )
    d = result.to_dict()
    assert d["verdict"] == "SOUND"
    assert d["confidence"] == 0.75
    assert d["issues"] == ["minor nit"]
