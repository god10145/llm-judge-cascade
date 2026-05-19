"""Tests for the CLI argument parsing and preset resolution."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_judge_cascade.cascade import Cascade, CascadeRunResult
from llm_judge_cascade.cli import _resolve_cascade, main
from llm_judge_cascade.client import ChatCompletionResult, JudgeClientConfig
from llm_judge_cascade.judges import JudgeVerdict


def _scripted_chat(payload: dict):
    body = json.dumps(payload)

    def fn(messages, model, config, max_tokens):
        return ChatCompletionResult(content=body, latency_ms=5, error=None)

    return fn


def test_resolve_default_preset() -> None:
    config = JudgeClientConfig(api_base="x", api_key="y")
    cascade = _resolve_cascade("haiku-sonnet-opus")(config)
    assert isinstance(cascade, Cascade)
    assert len(cascade.tiers) == 3
    assert cascade.tiers[0].model == "anthropic/claude-haiku-4-5-20251001"
    assert cascade.tiers[-1].model == "anthropic/claude-opus-4-7"


def test_resolve_single_preset() -> None:
    config = JudgeClientConfig(api_base="x", api_key="y")
    cascade = _resolve_cascade("single:anthropic/claude-opus-4-7")(config)
    assert len(cascade.tiers) == 1
    assert cascade.tiers[0].model == "anthropic/claude-opus-4-7"
    assert cascade.tiers[0].min_confidence_to_stop == 0.0


def test_resolve_panel_preset_runs_all_tiers() -> None:
    config = JudgeClientConfig(api_base="x", api_key="y")
    cascade = _resolve_cascade("panel:m1,m2,m3")(config)
    assert [t.model for t in cascade.tiers] == ["m1", "m2", "m3"]

    cascade._chat_fn = _scripted_chat({"verdict": "SOUND", "confidence": 0.99})
    result = cascade.judge(prompt="p", response="r")
    assert isinstance(result, CascadeRunResult)
    assert result.escalation_path == ["m1", "m2", "m3"]
    assert result.final_verdict is JudgeVerdict.SOUND


def test_resolve_unknown_preset_raises() -> None:
    with pytest.raises(ValueError):
        _resolve_cascade("nope")


def test_resolve_single_requires_model() -> None:
    with pytest.raises(ValueError):
        _resolve_cascade("single:")


def test_resolve_panel_requires_at_least_one_model() -> None:
    with pytest.raises(ValueError):
        _resolve_cascade("panel:")


def test_main_missing_api_key_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps({"id": "x", "prompt": "p", "response": "r"}) + "\n")
    out = tmp_path / "out.jsonl"
    monkeypatch.delenv("MY_KEY", raising=False)
    rc = main(
        [
            "run",
            "--input",
            str(inp),
            "--output",
            str(out),
            "--api-base",
            "http://x/v1",
            "--api-key-env",
            "MY_KEY",
            "--cascade",
            "single:m",
        ]
    )
    assert rc == 2


def test_main_unknown_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "value")
    out = tmp_path / "out.jsonl"
    rc = main(
        [
            "run",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(out),
            "--api-base",
            "http://x/v1",
            "--api-key-env",
            "MY_KEY",
            "--cascade",
            "single:m",
        ]
    )
    assert rc == 2


def test_main_end_to_end_with_mocked_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inp = tmp_path / "in.jsonl"
    inp.write_text(
        json.dumps({"id": "r1", "prompt": "What is 2+2?", "response": "4"})
        + "\n"
        + json.dumps({"id": "r2", "prompt": "Capital of France?", "response": "Paris"})
        + "\n"
    )
    out = tmp_path / "out.jsonl"
    monkeypatch.setenv("MY_KEY", "secret")

    def fake_chat(messages, model, config, max_tokens):
        return ChatCompletionResult(
            content=json.dumps({"verdict": "SOUND", "confidence": 0.95}),
            latency_ms=12,
            error=None,
        )

    with patch("llm_judge_cascade.cascade.chat_completion", side_effect=fake_chat):
        rc = main(
            [
                "run",
                "--input",
                str(inp),
                "--output",
                str(out),
                "--api-base",
                "http://x/v1",
                "--api-key-env",
                "MY_KEY",
                "--cascade",
                "single:vendor/model",
            ]
        )

    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        payload = json.loads(line)
        assert payload["judgments"]["final_verdict"] == "SOUND"
        assert payload["judgments"]["final_confidence"] == 0.95
