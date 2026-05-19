"""Tests for JSONL dataset I/O."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from llm_judge_cascade.cascade import CascadeRunResult
from llm_judge_cascade.dataset import (
    DecisionRecord,
    load_decisions,
    save_decisions_with_judgments,
)
from llm_judge_cascade.judges import JudgeResult, JudgeVerdict

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_decisions.jsonl"


def test_synthetic_fixture_loads() -> None:
    records = list(load_decisions(FIXTURE))
    assert len(records) == 20
    assert all(isinstance(r, DecisionRecord) for r in records)
    assert all(r.prompt and r.response for r in records)
    sound = [r for r in records if r.metadata.get("expected_verdict") == "SOUND"]
    assert len(sound) >= 7


def test_load_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps({"id": "ok", "prompt": "p", "response": "r"})
        + "\n"
        + "not json at all\n"
        + json.dumps({"id": "missing_response", "prompt": "p"})
        + "\n"
        + json.dumps({"id": "ok2", "prompt": "p2", "response": "r2"})
        + "\n",
        encoding="utf-8",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = list(load_decisions(path))
    assert [r.id for r in records] == ["ok", "ok2"]


def test_load_emits_warning_on_malformed(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("totally bogus line\n", encoding="utf-8")
    with pytest.warns(UserWarning):
        list(load_decisions(path))


def test_save_round_trip(tmp_path: Path) -> None:
    record = DecisionRecord(
        id="r1",
        prompt="What is 2+2?",
        response="4",
        metadata={"source": "synthetic"},
    )
    run = CascadeRunResult(
        final_verdict=JudgeVerdict.SOUND,
        final_confidence=0.91,
        tier_results=[
            JudgeResult(
                verdict=JudgeVerdict.SOUND,
                confidence=0.91,
                issues=[],
                annotation="ok",
                rewritten_response=None,
                raw_response="{}",
                model="m",
                latency_ms=10,
            )
        ],
        total_latency_ms=10,
        total_cost_usd=0.0123,
        escalation_path=["m"],
    )
    out = tmp_path / "judged.jsonl"
    n = save_decisions_with_judgments(out, [(record, run)])
    assert n == 1

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["id"] == "r1"
    assert payload["prompt"] == "What is 2+2?"
    assert payload["source"] == "synthetic"
    j = payload["judgments"]
    assert j["final_verdict"] == "SOUND"
    assert j["final_confidence"] == 0.91
    assert j["escalation_path"] == ["m"]
    assert j["tier_results"][0]["model"] == "m"


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    record = DecisionRecord(id="x", prompt="p", response="r")
    run = CascadeRunResult(
        final_verdict=JudgeVerdict.SOUND,
        final_confidence=0.8,
        tier_results=[],
        total_latency_ms=0,
        total_cost_usd=0.0,
        escalation_path=[],
    )
    out = tmp_path / "nested" / "subdir" / "out.jsonl"
    save_decisions_with_judgments(out, [(record, run)])
    assert out.exists()
