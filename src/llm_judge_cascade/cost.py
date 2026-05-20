"""Per-model cost estimation for cascade runs.

The MODEL_PRICING table tracks current approximate list prices in USD per
million tokens. **These prices need periodic refresh** as providers update
their rates; treat the table as a default that callers can override per
model. ``CostTracker`` accumulates costs across many cascade runs so a
batch runner can enforce a cost cap.

Tokens-per-character is a cheap heuristic (``len(text) // 4``); pass real
token counts to ``estimate_cost`` if you have them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Prices in USD per 1,000,000 tokens (input, output).
# Update periodically as providers change list prices. Last refreshed: 2026-05.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4-5-20251001": (1.00, 5.00),
    "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "anthropic/claude-opus-4-7": (15.00, 75.00),
    "gemini/gemini-3.1-pro-preview": (1.25, 5.00),
    "gemini/gemini-3.5-flash": (0.075, 0.30),
    "minimax/MiniMax-M2.7": (0.30, 1.20),
}


def estimate_tokens_from_chars(text: str) -> int:
    """Rough heuristic: roughly four characters per token for English text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a single call, using MODEL_PRICING.

    Unknown models cost $0; track them in a separate place if you need to
    surface "we don't have a price for X" warnings.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    in_rate, out_rate = pricing
    return (input_tokens / 1_000_000.0) * in_rate + (output_tokens / 1_000_000.0) * out_rate


def estimate_cost_from_chars(model: str, input_chars: int, output_chars: int) -> float:
    """Convenience: estimate tokens from character counts, then cost."""
    in_tok = max(1, input_chars // 4) if input_chars else 0
    out_tok = max(1, output_chars // 4) if output_chars else 0
    return estimate_cost(model, in_tok, out_tok)


@dataclass
class CostTracker:
    """Accumulate spending across many calls.

    ``record_call`` is the low-level API; ``record_chars`` is a convenience
    for callers that have characters rather than tokens.
    """

    total_usd: float = 0.0
    per_model_usd: dict[str, float] = field(default_factory=dict)
    n_calls: int = 0

    def record_call(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = estimate_cost(model, input_tokens, output_tokens)
        self.total_usd += cost
        self.per_model_usd[model] = self.per_model_usd.get(model, 0.0) + cost
        self.n_calls += 1
        return cost

    def record_chars(self, model: str, input_chars: int, output_chars: int) -> float:
        in_tok = max(1, input_chars // 4) if input_chars else 0
        out_tok = max(1, output_chars // 4) if output_chars else 0
        return self.record_call(model, in_tok, out_tok)

    def over_cap(self, cap_usd: float | None) -> bool:
        return cap_usd is not None and self.total_usd >= cap_usd


__all__ = [
    "MODEL_PRICING",
    "CostTracker",
    "estimate_cost",
    "estimate_cost_from_chars",
    "estimate_tokens_from_chars",
]
