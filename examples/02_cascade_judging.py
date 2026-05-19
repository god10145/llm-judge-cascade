"""Judge a JSONL file with the default Haiku -> Sonnet -> Opus cascade.

Compares cascade cost against a hypothetical "blind 3-judge" baseline that
would have run every record through all three models regardless. The
cascade only escalates on uncertainty or disagreement, so most records
should stop at Haiku.

Run:

    export LITELLM_MASTER_KEY=...
    python examples/02_cascade_judging.py tests/fixtures/synthetic_decisions.jsonl
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

from llm_judge_cascade import (
    JudgeClientConfig,
    load_decisions,
    make_default_cascade,
)
from llm_judge_cascade.cost import estimate_cost_from_chars


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: 02_cascade_judging.py <input.jsonl>\n")
        return 2
    input_path = Path(sys.argv[1])
    config = JudgeClientConfig(
        api_base=os.environ.get("LLM_JUDGE_API_BASE", "http://litellm:4000/v1"),
        api_key=os.environ["LITELLM_MASTER_KEY"],
    )
    cascade = make_default_cascade(config)

    total_cost = 0.0
    blind_cost = 0.0
    stopped_at = Counter()
    n = 0

    for record in load_decisions(input_path):
        n += 1
        result = cascade.judge(prompt=record.prompt, response=record.response)
        total_cost += result.total_cost_usd
        stopped_at[result.escalation_path[-1]] += 1
        for tier in cascade.tiers:
            blind_cost += estimate_cost_from_chars(
                tier.model,
                input_chars=len(record.prompt) + len(record.response),
                output_chars=400,
            )
        print(
            f"{record.id:>16} | path: {' -> '.join(result.escalation_path):<70} "
            f"| final={result.final_verdict.value:<11} "
            f"conf={result.final_confidence:.2f}"
        )

    print()
    print(f"Processed {n} record(s).")
    print(f"Cascade total cost     : ${total_cost:.4f}")
    print(f"Blind 3-judge estimate : ${blind_cost:.4f}")
    if blind_cost > 0:
        print(f"Savings                : {1.0 - total_cost / blind_cost:.1%}")
    print()
    print("Records stopped at:")
    for model, count in stopped_at.most_common():
        print(f"  {model:<48} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
