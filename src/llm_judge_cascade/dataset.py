"""JSONL I/O for decision records and their cascade judgments."""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

if TYPE_CHECKING:
    from llm_judge_cascade.cascade import CascadeRunResult


@dataclass
class DecisionRecord:
    """One (prompt, response) pair plus free-form metadata."""

    id: str
    prompt: str
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DecisionRecord":
        if "prompt" not in raw or "response" not in raw:
            raise ValueError("DecisionRecord requires 'prompt' and 'response' fields")
        record_id = str(raw.get("id") or "")
        nested = raw.get("metadata")
        if isinstance(nested, dict):
            metadata = dict(nested)
            for k, v in raw.items():
                if k in {"id", "prompt", "response", "metadata"}:
                    continue
                metadata.setdefault(k, v)
        else:
            metadata = {
                k: v for k, v in raw.items() if k not in {"id", "prompt", "response"}
            }
        return cls(
            id=record_id,
            prompt=str(raw["prompt"]),
            response=str(raw["response"]),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "prompt": self.prompt, "response": self.response}
        for k, v in self.metadata.items():
            if k not in out:
                out[k] = v
        return out


def load_decisions(path: Path | str) -> Iterator[DecisionRecord]:
    """Stream DecisionRecords from a JSONL file.

    Malformed lines are warned about and skipped rather than raising, so a
    single bad line doesn't kill a large batch.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                warnings.warn(
                    f"{path}:{lineno}: skipping malformed JSON ({exc.msg})",
                    stacklevel=2,
                )
                continue
            if not isinstance(raw, dict):
                warnings.warn(
                    f"{path}:{lineno}: skipping non-object JSONL line",
                    stacklevel=2,
                )
                continue
            try:
                yield DecisionRecord.from_dict(raw)
            except ValueError as exc:
                warnings.warn(
                    f"{path}:{lineno}: skipping invalid record ({exc})",
                    stacklevel=2,
                )


def _judgments_block(run_result: "CascadeRunResult") -> dict[str, Any]:
    return {
        "final_verdict": run_result.final_verdict.value,
        "final_confidence": run_result.final_confidence,
        "total_latency_ms": run_result.total_latency_ms,
        "total_cost_usd": run_result.total_cost_usd,
        "escalation_path": list(run_result.escalation_path),
        "tier_results": [tr.to_dict() for tr in run_result.tier_results],
    }


def save_decisions_with_judgments(
    path: Path | str,
    records: Iterable[tuple[DecisionRecord, "CascadeRunResult"]],
) -> int:
    """Write augmented JSONL where each line carries the original record + judgments.

    Returns the number of records written. Parent directories are created if
    they don't already exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for record, run_result in records:
            payload = record.to_dict()
            payload["judgments"] = _judgments_block(run_result)
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


__all__ = ["DecisionRecord", "load_decisions", "save_decisions_with_judgments"]


# Avoid an "unused" complaint from linters that can't see TYPE_CHECKING imports.
_ = (asdict, sys)
