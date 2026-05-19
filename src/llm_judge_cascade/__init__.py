"""llm-judge-cascade: cost-tiered LLM-as-judge cascade for AI agent evaluation."""
from __future__ import annotations

__version__ = "0.1.0"

from llm_judge_cascade.cascade import (
    Cascade,
    CascadeRunResult,
    CascadeTier,
    make_default_cascade,
)
from llm_judge_cascade.client import (
    ChatCompletionResult,
    JudgeClientConfig,
    chat_completion,
)
from llm_judge_cascade.cost import (
    MODEL_PRICING,
    CostTracker,
    estimate_cost,
    estimate_tokens_from_chars,
)
from llm_judge_cascade.dataset import (
    DecisionRecord,
    load_decisions,
    save_decisions_with_judgments,
)
from llm_judge_cascade.judges import (
    DEFAULT_JUDGE_PROMPT_FABRICATION_DETECTION,
    DEFAULT_JUDGE_PROMPT_GENERIC,
    DEFAULT_JUDGE_PROMPT_REASONING_QUALITY,
    JudgeResult,
    JudgeVerdict,
    build_judge_messages,
    parse_judge_response,
)

__all__ = [
    "__version__",
    "Cascade",
    "CascadeRunResult",
    "CascadeTier",
    "ChatCompletionResult",
    "CostTracker",
    "DEFAULT_JUDGE_PROMPT_FABRICATION_DETECTION",
    "DEFAULT_JUDGE_PROMPT_GENERIC",
    "DEFAULT_JUDGE_PROMPT_REASONING_QUALITY",
    "DecisionRecord",
    "JudgeClientConfig",
    "JudgeResult",
    "JudgeVerdict",
    "MODEL_PRICING",
    "build_judge_messages",
    "chat_completion",
    "estimate_cost",
    "estimate_tokens_from_chars",
    "load_decisions",
    "make_default_cascade",
    "parse_judge_response",
    "save_decisions_with_judgments",
]
