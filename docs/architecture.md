# Architecture

`llm-judge-cascade` is a small library with a single job: take a JSONL
file of `(prompt, response)` pairs that an AI agent produced, and decide
for each one whether the response is trustworthy, flawed, fabricated, or
ambiguous, while spending as little API budget as possible.

## Cascade flow

```
+---------------------+
| JSONL of decisions  |
+----------+----------+
           |
           v
+---------------------+
| Cascade.judge()     |
+----------+----------+
           |
           v
+---------------------+      stop if confident,
| Tier 1: cheap model | ---> in-agreement verdict
+----------+----------+
           |
           | escalate on UNCERTAIN,
           | low confidence, or
           | disagreement
           v
+---------------------+
| Tier 2: mid model   | ---> stop, or...
+----------+----------+
           |
           v
+---------------------+
| Tier 3: strong model| --> final tier always wins
+----------+----------+
           |
           v
+---------------------+
| CascadeRunResult    |
| - final_verdict     |
| - final_confidence  |
| - tier_results[]    |
| - escalation_path[] |
| - total_cost_usd    |
| - total_latency_ms  |
+---------------------+
```

## Why multi-tier instead of blind ensemble?

A blind 3-judge ensemble pays for the most expensive model on every
record, regardless of whether the record was easy. In practice, most agent
responses are obviously sound or obviously broken, and a cheap judge
handles them with high confidence. Only a small minority warrant the
expensive judge.

The cascade encodes this as an explicit stopping rule:

1. Run the first tier.
2. If the verdict is `UNCERTAIN`, or the confidence is below the tier's
   `min_confidence_to_stop`, escalate.
3. If a flag like `escalate_if_flawed` matches the verdict, escalate.
4. If the new verdict disagrees with the previous tier's verdict (and the
   previous tier was not itself `UNCERTAIN`), escalate.
5. Otherwise stop and return the current tier's verdict.
6. If the cascade is exhausted, the last tier's verdict is final.

Treating a previous `UNCERTAIN` as "not a disagreement" matters: the
previous tier explicitly abstained, so the new tier breaking the tie is
not a contradiction.

## The `JudgeResult` contract

Every judge — regardless of underlying model — is expected to emit a JSON
object with the same five fields:

- `verdict`: one of `SOUND`, `FLAWED`, `FABRICATED`, `UNCERTAIN`.
- `confidence`: a float in `[0.0, 1.0]`.
- `issues`: a list of short, concrete strings naming specific problems.
- `annotation`: a 1-3 sentence free-form explanation, or `null`.
- `rewritten_response`: an alternative response, or `null`. Used by the
  DPO preference-pair extractor.

`parse_judge_response` is permissive — it strips markdown fences, falls
back to brace-grep extraction if the model wrapped the JSON in prose, and
collapses any unrecoverable response to `UNCERTAIN` with the failure
recorded under `issues`. The cascade controller does not depend on a
specific judge prompt; it only depends on this output schema.

## Custom cascades

`make_default_cascade(config)` returns a Haiku to Sonnet to Opus chain that
matches the homelab LiteLLM aliases. To build a different ordering — for
example, an open-weights cheap tier with a frontier final arbiter —
construct a list of `CascadeTier` instances and instantiate `Cascade`
directly. Each tier independently controls its judge prompt, max tokens,
confidence threshold, and escalation flags.
