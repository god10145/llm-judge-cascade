"""Judge every record in a JSONL file with one model.

Run:

    export OPENAI_API_KEY=...
    python examples/01_basic_replay.py tests/fixtures/synthetic_decisions.jsonl
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from llm_judge_cascade import (
    Cascade,
    CascadeTier,
    DEFAULT_JUDGE_PROMPT_GENERIC,
    JudgeClientConfig,
    load_decisions,
)

MODEL = "anthropic/claude-haiku-4-5-20251001"


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: 01_basic_replay.py <input.jsonl>\n")
        return 2
    input_path = Path(sys.argv[1])
    config = JudgeClientConfig(
        api_base=os.environ.get("LLM_JUDGE_API_BASE", "https://api.openai.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    cascade = Cascade(
        tiers=[CascadeTier(model=MODEL, judge_prompt=DEFAULT_JUDGE_PROMPT_GENERIC)],
        config=config,
    )
    for record in load_decisions(input_path):
        result = cascade.judge(prompt=record.prompt, response=record.response)
        print(
            f"{record.id:>16} | {result.final_verdict.value:<11} "
            f"| conf={result.final_confidence:.2f} | ${result.total_cost_usd:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
