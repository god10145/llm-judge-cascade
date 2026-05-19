# Cost analysis

The point of the cascade pattern is to spend less. This page walks through
a worked example using the four list prices baked into `MODEL_PRICING` at
the time of writing (per million tokens; refresh as providers change
their rates):

| Model | Input | Output |
|---|---|---|
| Haiku 4.5 | $1.00 | $5.00 |
| Sonnet 4.6 | $3.00 | $15.00 |
| Opus 4.7 | $15.00 | $75.00 |
| Gemini 3.1 Pro Preview | $1.25 | $5.00 |

## Cost per judgment

Assume the average judge call consumes 1500 input tokens (the agent's
prompt plus response plus the judge system prompt) and 200 output tokens
(the JSON verdict block). Per-model cost is then:

| Model | Input cost | Output cost | Total |
|---|---|---|---|
| Haiku | $0.0015 | $0.0010 | $0.0025 |
| Sonnet | $0.0045 | $0.0030 | $0.0075 |
| Opus | $0.0225 | $0.0150 | $0.0375 |

## Two judging strategies

**Blind 3-judge:** every record is judged by Haiku, Sonnet, and Opus.
Cost per record: $0.0025 + $0.0075 + $0.0375 = `$0.0475`.

**Cascade:** Haiku handles most records alone. Assume realistic escalation
rates — 70% stop at Haiku, 20% escalate to Sonnet, 10% reach Opus.
Expected cost per record:

```
0.70 * 0.0025
+ 0.20 * (0.0025 + 0.0075)
+ 0.10 * (0.0025 + 0.0075 + 0.0375)
= 0.00175 + 0.002 + 0.00475
= $0.0085
```

That is **roughly 18% of the blind 3-judge cost**, with the same
worst-case Opus accuracy on the hard records.

## Worked example: 2000 records per day

| Strategy | Cost per record | Daily cost | Monthly (30d) |
|---|---|---|---|
| Single Opus | $0.0375 | $75.00 | $2,250 |
| Blind 3-judge | $0.0475 | $95.00 | $2,850 |
| Cascade (70/20/10) | $0.0085 | $17.00 | $510 |

At lower volume:

| Records / day | Blind 3-judge | Cascade |
|---|---|---|
| 100 | $4.75 | $0.85 |
| 1,000 | $47.50 | $8.50 |
| 10,000 | $475.00 | $85.00 |

## Tuning escalation thresholds

The 70/20/10 split is an estimate; your real distribution depends on
how often the cheap judge is confidently right. If you find the cascade
escalating too often, lower the cheaper tier's `min_confidence_to_stop`
or turn off `escalate_if_flawed`. If you find it stopping on borderline
cases, raise the threshold.

`Cascade.judge()` records the actual escalation path on every run, so
after a batch you can count how often each tier was the last one called
and tune from data rather than guesswork. The `02_cascade_judging.py`
example prints exactly this distribution at the end of a run.

## Caveats

- **Prices drift.** Refresh `MODEL_PRICING` periodically; treat the
  numbers in the table as defaults that callers can override.
- **Token counts are approximate.** `estimate_tokens_from_chars` uses a
  4-chars-per-token heuristic; for billing-critical numbers, pull the
  real token counts from your provider's API response.
- **The cheap tier still costs something.** Even if Haiku handles
  90% of records, it costs more than zero. Don't run a cascade against
  data you wouldn't have judged at all otherwise.
