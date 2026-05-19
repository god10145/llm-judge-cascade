# llm-judge-cascade

**Cost-tiered LLM-as-judge cascade for evaluating AI agent outputs and
generating preference data.** Point it at a JSONL file of `(prompt,
response)` pairs your agent produced, and it walks each record through
a chain of judge models — cheap first, expensive only when necessary —
to flag confabulation, score reasoning quality, and emit rewritten
responses you can use as training data. Works with any
OpenAI-compatible endpoint.

## Why this exists

AI agents confabulate. They cite numbers that are not in the prompt,
reference documents that do not exist, follow plausible-looking reasoning
chains to conclusions that the underlying evidence does not support, and
do all of this with high confidence and fluent prose. Catching this at
inference time with runtime guardrails is hard; the agent's behavior
often only looks wrong in hindsight, after a careful reader compares the
response against the prompt's actual content.

Post-hoc LLM-as-judge evaluation is the natural fix: take a logged batch
of `(prompt, response)` pairs and ask a second, capable model to grade
them. The judge can re-read the prompt slowly, check the response for
made-up facts, and assign a verdict that downstream systems can act on —
filtering training data, emitting preference pairs for DPO, or surfacing
the bad cases to a human reviewer.

The bottleneck is cost. A frontier judge on every record gets expensive
fast, and most records are obviously fine. `llm-judge-cascade` solves
that with a small cascade: a cheap judge handles the easy cases by
itself, and only the genuinely uncertain or contested records pay for
escalation to a more expensive model. The result is roughly the
accuracy of a frontier judge at roughly the cost of a cheap one.

## 30-second example

```python
from llm_judge_cascade import (
    JudgeClientConfig,
    load_decisions,
    make_default_cascade,
)

config = JudgeClientConfig(
    api_base="https://api.openai.com/v1",
    api_key="sk-...",
)
cascade = make_default_cascade(config)

for record in load_decisions("agent_decisions.jsonl"):
    result = cascade.judge(prompt=record.prompt, response=record.response)
    print(
        f"{record.id}: {result.final_verdict.value} "
        f"({result.final_confidence:.2f}) via {result.escalation_path}, "
        f"cost=${result.total_cost_usd:.4f}"
    )
```

Or from the command line:

```bash
judge-cascade run \
  --input agent_decisions.jsonl \
  --output judged.jsonl \
  --api-base https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --cascade haiku-sonnet-opus
```

## Installation

When this package is on PyPI:

```bash
pip install llm-judge-cascade
```

In the meantime, install from a local checkout:

```bash
git clone https://github.com/kceekremeier/llm-judge-cascade.git
cd llm-judge-cascade
pip install -e ".[dev]"
```

Requires Python 3.11 or newer.

## The cascade pattern

```
  +----------------+      stop if confident,
  | Tier 1: Haiku  | --->  unambiguous verdict
  +--------+-------+
           |
           | escalate on UNCERTAIN, low
           | confidence, or disagreement
           v
  +----------------+
  | Tier 2: Sonnet | --->  stop, or...
  +--------+-------+
           |
           v
  +----------------+
  | Tier 3: Opus   | --->  final arbiter
  +----------------+
```

In a realistic 70/20/10 escalation split (where Tier 1 handles 70% of
records on its own, Tier 2 handles another 20%, and Tier 3 the remaining
10%), the cascade costs about **18% of a blind 3-judge ensemble** while
preserving frontier-model accuracy on the hard cases. See
[`docs/cost_analysis.md`](docs/cost_analysis.md) for the worked numbers.

## Built-in cascades

| Preset | When to use | Typical models |
|---|---|---|
| `haiku-sonnet-opus` | Default cost-tiered cascade for general AI agent evaluation. | Haiku 4.5, Sonnet 4.6, Opus 4.7 |
| `single:<model>` | One model, one pass. Sanity check, or as a baseline before turning on the full cascade. | Any single model alias |
| `panel:<m1>,<m2>,...` | Force every model to judge every record. Useful for measuring cross-model disagreement as preference data, not for production batches. | Any comma-separated list |

You can build your own cascade by composing `CascadeTier` instances and
passing them to `Cascade(...)` directly. Each tier independently
controls its judge prompt, confidence threshold, and escalation flags.

## Customizing judge prompts

The package ships three built-in domain-neutral judge prompts:

- `DEFAULT_JUDGE_PROMPT_GENERIC` — evaluates engagement, grounding, and
  internal consistency together. The default for every tier.
- `DEFAULT_JUDGE_PROMPT_REASONING_QUALITY` — stricter; demands every
  substantive claim be traceable back to the input.
- `DEFAULT_JUDGE_PROMPT_FABRICATION_DETECTION` — narrowest; flags any
  invented fact, citation, or named entity.

To plug in a domain-specific prompt, pass it through
`CascadeTier(judge_prompt=...)`. The cascade controller does not depend
on the prompt content, only on the JSON output schema. See
[`docs/judge_prompts.md`](docs/judge_prompts.md) for a worked example
that swaps in a custom code-review judge prompt.

## Output format

Each line in the augmented JSONL written by the CLI is the original
record plus a `judgments` block:

```json
{
  "id": "decision-12",
  "prompt": "Summarize this article...",
  "response": "The author argues that...",
  "metadata": {"source": "internal"},
  "judgments": {
    "final_verdict": "FABRICATED",
    "final_confidence": 0.88,
    "total_latency_ms": 1430,
    "total_cost_usd": 0.0163,
    "escalation_path": [
      "anthropic/claude-haiku-4-5-20251001",
      "anthropic/claude-sonnet-4-6"
    ],
    "tier_results": [
      {
        "verdict": "UNCERTAIN",
        "confidence": 0.4,
        "issues": ["..."],
        "annotation": "...",
        "rewritten_response": null,
        "model": "anthropic/claude-haiku-4-5-20251001",
        "latency_ms": 410,
        "raw_response": "..."
      },
      {
        "verdict": "FABRICATED",
        "confidence": 0.88,
        "issues": ["Invents a publication date."],
        "annotation": "...",
        "rewritten_response": "A more grounded summary...",
        "model": "anthropic/claude-sonnet-4-6",
        "latency_ms": 1020,
        "raw_response": "..."
      }
    ]
  }
}
```

Each `tier_result` carries the model, latency, raw response, and any
proposed `rewritten_response`, so downstream tools can inspect every
hop the cascade took.

## Extracting preference pairs for DPO

When a judge believes a better answer is obvious, it can return a
`rewritten_response`. The cascade preserves these on every tier, and the
`examples/03_extract_preference_pairs.py` script walks a judged JSONL,
picks the highest-confidence rewrite per record, and emits
`(prompt, chosen, rejected)` triples ready for Direct Preference
Optimization training:

```bash
python examples/03_extract_preference_pairs.py judged.jsonl preferences.jsonl
```

This turns post-hoc judge disagreement with your agent into free
preference data, without requiring human labeling.

## API compatibility

`chat_completion` posts to `{api_base}/chat/completions` with an
OpenAI-shaped body. Anything that speaks the OpenAI chat completions
wire format works:

- OpenAI
- LiteLLM gateway (single endpoint, many providers)
- vLLM
- Together, Anyscale, Fireworks, Groq
- Local servers like llama.cpp and Ollama-compatible runners
- Self-hosted endpoints reachable over HTTP

The client deliberately does not pass `temperature`; some newer judge
models reject it. Override `max_tokens` per call if you need to.

## Limitations

- **Judges have their own biases.** A cascade of LLM judges agreeing
  with each other is not the same thing as ground truth. For
  high-stakes decisions, route disagreements to a human reviewer.
- **Reward hacking is real.** If you fine-tune an agent against a
  specific judge prompt, expect the agent to learn the prompt's
  shortcuts over time. Cycle prompts and judges periodically.
- **This is offline, batch evaluation.** The cascade is designed to
  judge logged decisions after the fact, not to guard runtime
  inference. Per-record latency is on the order of seconds, not
  milliseconds.
- **Pricing drifts.** The `MODEL_PRICING` table is a best-effort
  default. Refresh it as providers change list prices.

## License

MIT. See [`LICENSE`](LICENSE).
