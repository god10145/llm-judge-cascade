"""Cascade controller: run a list of judge tiers, escalating on uncertainty.

The cascade pattern keeps cost down: a cheap model handles obviously sound
or obviously broken cases by itself, and only the genuinely ambiguous
records pay for a more expensive judge. ``make_default_cascade()`` ships a
Haiku → Sonnet → Opus preset that matches the Chimera homelab's LiteLLM
config; callers can build their own ordering with ``CascadeTier`` instances
for other providers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from llm_judge_cascade.client import (
    ChatCompletionResult,
    JudgeClientConfig,
    chat_completion,
)
from llm_judge_cascade.cost import estimate_cost_from_chars
from llm_judge_cascade.judges import (
    DEFAULT_JUDGE_PROMPT_GENERIC,
    JudgeResult,
    JudgeVerdict,
    build_judge_messages,
    parse_judge_response,
)

# ─── injectable transport types ────────────────────────────────────────────

ChatCompletionFn = Callable[
    [list[dict], str, JudgeClientConfig, int | None],
    ChatCompletionResult,
]
"""Signature for a chat completion callable. Inject a fake in tests."""

CostEstimatorFn = Callable[[str, int, int], float]
"""(model, input_chars, output_chars) -> usd. Replaced by cost.py in commit 5."""


def _default_cost(model: str, in_chars: int, out_chars: int) -> float:
    return estimate_cost_from_chars(model, in_chars, out_chars)


# ─── public dataclasses ────────────────────────────────────────────────────


@dataclass
class CascadeTier:
    """One step in the cascade."""

    model: str
    judge_prompt: str = DEFAULT_JUDGE_PROMPT_GENERIC
    escalate_if_uncertain: bool = True
    escalate_if_flawed: bool = False
    min_confidence_to_stop: float = 0.7
    max_tokens: int | None = None


@dataclass
class CascadeRunResult:
    """Aggregate outcome of running a cascade against one (prompt, response) pair."""

    final_verdict: JudgeVerdict
    final_confidence: float
    tier_results: list[JudgeResult] = field(default_factory=list)
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0
    escalation_path: list[str] = field(default_factory=list)


# ─── controller ────────────────────────────────────────────────────────────


class Cascade:
    """Run an ordered list of judge tiers, escalating when warranted.

    Stopping rules (evaluated after each tier):

    1. If the tier's verdict is UNCERTAIN and ``escalate_if_uncertain`` is
       set, escalate.
    2. If the tier's verdict is FLAWED and ``escalate_if_flawed`` is set,
       escalate.
    3. If the tier's confidence is below ``min_confidence_to_stop``,
       escalate.
    4. If the tier disagrees with the previous tier's verdict, escalate.
    5. Otherwise stop and use this tier's verdict as the final answer.

    If we exhaust the cascade, the last tier's verdict wins.
    """

    def __init__(
        self,
        tiers: list[CascadeTier],
        config: JudgeClientConfig,
        *,
        chat_fn: ChatCompletionFn | None = None,
        cost_fn: CostEstimatorFn | None = None,
    ) -> None:
        if not tiers:
            raise ValueError("Cascade requires at least one tier")
        self.tiers = list(tiers)
        self.config = config
        self._chat_fn = chat_fn if chat_fn is not None else chat_completion
        self._cost_fn = cost_fn if cost_fn is not None else _default_cost

    def judge(
        self,
        prompt: str,
        response: str,
        extra_context: dict | None = None,
    ) -> CascadeRunResult:
        tier_results: list[JudgeResult] = []
        escalation_path: list[str] = []
        total_latency = 0
        total_cost = 0.0

        prev: JudgeResult | None = None
        for tier in self.tiers:
            escalation_path.append(tier.model)
            messages = build_judge_messages(
                prompt=prompt,
                response=response,
                judge_system_prompt=tier.judge_prompt,
                extra_context=extra_context,
            )
            chat = self._chat_fn(messages, tier.model, self.config, tier.max_tokens)
            judge_result = parse_judge_response(
                chat.content if chat.error is None else "",
                model=tier.model,
                latency_ms=chat.latency_ms,
            )
            if chat.error is not None:
                judge_result.issues.append(f"transport_error: {chat.error}")
            tier_results.append(judge_result)
            total_latency += chat.latency_ms

            in_chars = sum(len(m.get("content", "")) for m in messages)
            out_chars = len(chat.content or "")
            total_cost += float(self._cost_fn(tier.model, in_chars, out_chars))

            if self._should_stop(judge_result, prev, tier):
                break
            prev = judge_result

        final = tier_results[-1]
        return CascadeRunResult(
            final_verdict=final.verdict,
            final_confidence=final.confidence,
            tier_results=tier_results,
            total_latency_ms=total_latency,
            total_cost_usd=total_cost,
            escalation_path=escalation_path,
        )

    @staticmethod
    def _should_stop(
        current: JudgeResult,
        prev: JudgeResult | None,
        tier: CascadeTier,
    ) -> bool:
        if current.verdict is JudgeVerdict.UNCERTAIN and tier.escalate_if_uncertain:
            return False
        if current.verdict is JudgeVerdict.FLAWED and tier.escalate_if_flawed:
            return False
        if current.confidence < tier.min_confidence_to_stop:
            return False
        if (
            prev is not None
            and prev.verdict is not JudgeVerdict.UNCERTAIN
            and prev.verdict is not current.verdict
        ):
            return False
        return True


# ─── preset ────────────────────────────────────────────────────────────────


def make_default_cascade(
    config: JudgeClientConfig,
    *,
    chat_fn: ChatCompletionFn | None = None,
    cost_fn: CostEstimatorFn | None = None,
) -> Cascade:
    """Default 3-tier cascade: Haiku 4.5 → Sonnet 4.6 → Opus 4.7.

    Haiku handles the easy cases. Sonnet is the middle arbiter — it
    escalates on uncertain or flawed verdicts. Opus is the final tier and
    accepts whatever it returns.
    """
    tiers = [
        CascadeTier(
            model="anthropic/claude-haiku-4-5-20251001",
            judge_prompt=DEFAULT_JUDGE_PROMPT_GENERIC,
            escalate_if_uncertain=True,
            escalate_if_flawed=False,
            min_confidence_to_stop=0.7,
        ),
        CascadeTier(
            model="anthropic/claude-sonnet-4-6",
            judge_prompt=DEFAULT_JUDGE_PROMPT_GENERIC,
            escalate_if_uncertain=True,
            escalate_if_flawed=True,
            min_confidence_to_stop=0.75,
        ),
        CascadeTier(
            model="anthropic/claude-opus-4-7",
            judge_prompt=DEFAULT_JUDGE_PROMPT_GENERIC,
            escalate_if_uncertain=False,
            escalate_if_flawed=False,
            min_confidence_to_stop=0.0,
        ),
    ]
    return Cascade(tiers, config, chat_fn=chat_fn, cost_fn=cost_fn)
