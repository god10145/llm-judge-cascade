"""Judge result types, default judge prompts, and response parsing.

The judge prompts here are domain-neutral: they evaluate whether an arbitrary
agent response engages with its prompt, fabricates anything, and reasons
coherently. Swap in a domain-specific system prompt for code review, legal
analysis, customer support, etc., while reusing this module's JSON schema.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JudgeVerdict(str, Enum):
    """Top-level classification a judge can assign to an agent response."""

    SOUND = "SOUND"
    FLAWED = "FLAWED"
    FABRICATED = "FABRICATED"
    UNCERTAIN = "UNCERTAIN"

    @classmethod
    def coerce(cls, raw: Any) -> "JudgeVerdict":
        """Map a string (case-insensitive) to a verdict; unknown → UNCERTAIN."""
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, str):
            return cls.UNCERTAIN
        key = raw.strip().upper()
        for v in cls:
            if v.value == key:
                return v
        return cls.UNCERTAIN


@dataclass
class JudgeResult:
    """A single judge model's verdict on one (prompt, response) pair."""

    verdict: JudgeVerdict
    confidence: float
    issues: list[str] = field(default_factory=list)
    annotation: str | None = None
    rewritten_response: str | None = None
    raw_response: str = ""
    model: str = ""
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "issues": list(self.issues),
            "annotation": self.annotation,
            "rewritten_response": self.rewritten_response,
            "raw_response": self.raw_response,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }


# ─── parsing ───────────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a possibly-fenced, possibly-noisy LLM reply."""
    if not text:
        return None
    body = text.strip()
    if body.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", body)
        if fenced:
            body = fenced.group(1).strip()
    try:
        loaded = json.loads(body)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        return loaded
    match = re.search(r"\{[\s\S]*\}", body)
    if not match:
        return None
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _coerce_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN
        return default
    return max(0.0, min(1.0, value))


def _coerce_str_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _coerce_optional_str(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def parse_judge_response(
    raw: str,
    *,
    model: str = "",
    latency_ms: int = 0,
) -> JudgeResult:
    """Parse a judge LLM reply into a JudgeResult.

    Robust to markdown fences, leading commentary, and partially-malformed
    JSON. Anything we can't recover from collapses to UNCERTAIN with the
    failure recorded under ``issues``.
    """
    obj = _extract_json(raw)
    if obj is None:
        return JudgeResult(
            verdict=JudgeVerdict.UNCERTAIN,
            confidence=0.0,
            issues=["judge_parse_failed: no JSON object found"],
            raw_response=raw,
            model=model,
            latency_ms=latency_ms,
        )
    return JudgeResult(
        verdict=JudgeVerdict.coerce(obj.get("verdict")),
        confidence=_coerce_float(obj.get("confidence")),
        issues=_coerce_str_list(obj.get("issues")),
        annotation=_coerce_optional_str(obj.get("annotation")),
        rewritten_response=_coerce_optional_str(obj.get("rewritten_response")),
        raw_response=raw,
        model=model,
        latency_ms=latency_ms,
    )


# ─── default judge prompts ─────────────────────────────────────────────────

_JSON_SCHEMA_BLOCK = """\
Respond with a single JSON object, and nothing else. The object MUST have
exactly these top-level keys:

  - "verdict": one of "SOUND", "FLAWED", "FABRICATED", "UNCERTAIN"
  - "confidence": a float between 0.0 and 1.0 indicating how sure you are
    of the verdict
  - "issues": an array of short strings (may be empty) listing specific
    problems you observed. Each issue should be one concrete claim, not a
    paragraph.
  - "annotation": a short free-form explanation (1-3 sentences) of how you
    arrived at the verdict. Use null if there is nothing to add.
  - "rewritten_response": if and only if you believe a better response is
    obvious AND would meaningfully change behavior in downstream training,
    provide a rewritten version. Otherwise use null. Do not rewrite for
    style alone.

Verdict meanings:

  - SOUND: the response engages with the prompt, is internally consistent,
    and does not fabricate facts beyond what the prompt provides.
  - FLAWED: the response is on-topic but contains weak reasoning, missing
    steps, or partial inaccuracies that a careful reader would catch.
  - FABRICATED: the response cites specific facts, sources, numbers, or
    objects that are not supported by the prompt's context.
  - UNCERTAIN: you cannot tell from the prompt alone, or the response is
    too short or too ambiguous to evaluate."""


_WORKED_EXAMPLES = """\
Worked example, SOUND:

  Prompt: "Summarize this paragraph about photosynthesis in two sentences:
  Plants use sunlight to convert carbon dioxide and water into glucose
  and oxygen. This process occurs in chloroplasts and is driven by the
  pigment chlorophyll."

  Response: "Plants convert carbon dioxide and water into glucose and
  oxygen using sunlight, a process called photosynthesis. It happens in
  chloroplasts and depends on the pigment chlorophyll."

  Judgment:
    {"verdict": "SOUND", "confidence": 0.92, "issues": [],
     "annotation": "Captures both key ideas from the source without
     adding outside facts.", "rewritten_response": null}

Worked example, FLAWED with fabrication:

  Prompt: "Classify this support ticket as billing, technical, or other:
  'My printer is making a grinding noise when I print.'"

  Response: "This is a billing issue, likely related to a recent invoice
  dispute. I recommend escalating to the accounts team in Region 7."

  Judgment:
    {"verdict": "FABRICATED", "confidence": 0.95,
     "issues": ["Invents an invoice dispute that is not in the ticket.",
                "Invents a Region 7 routing destination."],
     "annotation": "The ticket describes a hardware symptom; classifying
     it as billing and inventing routing details is not grounded in the
     input.",
     "rewritten_response": "Technical."}"""


DEFAULT_JUDGE_PROMPT_GENERIC = f"""\
You are a careful, skeptical evaluator. You will be shown the user's input
to an AI agent and the agent's response. Your job is to evaluate the
response along three axes:

1. Engagement: does the response actually address the user's input, or
   does it drift, refuse, or change the topic?
2. Grounding: does the response cite anything (facts, numbers, names,
   sources, objects) that is not present in the user's input or in any
   provided context? Fabricated grounding is the single most important
   failure mode to catch.
3. Internal consistency: does the chain of reasoning hold together? Are
   conclusions supported by the steps that preceded them, and do the
   steps contradict each other?

You are not evaluating whether the response is the "best possible" answer.
You are evaluating whether it is trustworthy enough that a downstream
system could safely train on it as a positive example. A response that is
short and correct is SOUND. A response that is long, fluent, and confident
but invents specifics is FABRICATED.

Be specific in the "issues" array. "Reasoning is weak" is not useful;
"Claims the document is 200 pages when the prompt does not state a page
count" is useful.

{_JSON_SCHEMA_BLOCK}

{_WORKED_EXAMPLES}
"""


DEFAULT_JUDGE_PROMPT_REASONING_QUALITY = f"""\
You are evaluating the *reasoning quality* of an AI agent's response,
focusing on whether the response shows its work in a way that another
reader could verify.

For each response, ask:

1. Does the response cite specific evidence from the user's input, or does
   it gesture vaguely at categories ("the data suggests", "trends are
   bearish") without naming the fields, values, or quotes that justify
   the claim?
2. When the response makes a numerical or factual claim, can you point
   to the line in the user's input that supports it?
3. Are the inferential steps explicit, or does the response leap from
   premise to conclusion in a way you would have to take on faith?
4. Does the response handle competing evidence, or does it cherry-pick?

Score the response as SOUND only if a careful reviewer could trace each
substantive claim back to the input. Score it as FLAWED if the reasoning
is on-topic but skips steps or uses vague gestures. Score it as
FABRICATED if it invents specific evidence (numbers, fields, quotes,
identifiers) that are not in the input. Use UNCERTAIN if the input is
too thin to evaluate.

{_JSON_SCHEMA_BLOCK}

{_WORKED_EXAMPLES}
"""


DEFAULT_JUDGE_PROMPT_FABRICATION_DETECTION = f"""\
You have one job: determine whether the agent's response invents anything
that is not present in the user's input or its provided context.

Specifically scan for:

- Named entities (people, places, organizations, products) that do not
  appear in the input.
- Numeric claims (prices, counts, percentages, dates, identifiers) that
  are not in the input.
- Cited sources (papers, URLs, document titles, sections) that are not
  in the input.
- Internal references ("as noted above", "per the table", "Region 7")
  that point at content the input does not contain.
- Confident assertions about state that the input does not describe.

If the response is fully grounded in the input, return SOUND. If it
contains any unsupported specifics, return FABRICATED and list each
fabricated item separately in "issues". Do not penalize stylistic
embellishment or paraphrasing — only invented substantive content.

If you cannot tell (for example, the input is itself a question without
context), return UNCERTAIN rather than guessing.

{_JSON_SCHEMA_BLOCK}

{_WORKED_EXAMPLES}
"""


# ─── message construction ──────────────────────────────────────────────────


def build_judge_messages(
    prompt: str,
    response: str,
    judge_system_prompt: str,
    extra_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build the messages list to send to a judge model.

    The user turn is a structured block so the judge sees the boundary
    between the agent's input and the agent's output explicitly.
    """
    body = {"agent_input": prompt, "agent_response": response}
    if extra_context:
        body["extra_context"] = extra_context
    user_content = (
        "Evaluate the following (agent_input, agent_response) pair. "
        "Respond with the JSON object specified in your system prompt.\n\n"
        + json.dumps(body, ensure_ascii=False, indent=2, default=str)
    )
    return [
        {"role": "system", "content": judge_system_prompt},
        {"role": "user", "content": user_content},
    ]
