# Examples

Three small scripts demonstrating typical usage of `llm-judge-cascade`.
All examples expect a JSONL file of decisions, one record per line, with
at minimum an `id`, `prompt`, and `response` field. The synthetic fixture
under `tests/fixtures/synthetic_decisions.jsonl` is a good starting point.

| Script | What it shows |
|---|---|
| [`01_basic_replay.py`](01_basic_replay.py) | Judge every record with a single model and print verdicts. |
| [`02_cascade_judging.py`](02_cascade_judging.py) | Run the default Haiku to Sonnet to Opus cascade and compare cost against a hypothetical blind 3-judge baseline. |
| [`03_extract_preference_pairs.py`](03_extract_preference_pairs.py) | Walk a judged JSONL and emit DPO-style `(prompt, chosen, rejected)` triples wherever a judge supplied a rewritten response. |

## Environment variables

The examples read API credentials from environment variables so nothing
secret has to live in argv.

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | Used by `01_basic_replay.py`. Replace with your provider's key. |
| `LITELLM_MASTER_KEY` | Used by `02_cascade_judging.py` when pointing at a LiteLLM gateway. |
| `LLM_JUDGE_API_BASE` | Optional override for the API base URL. |

## Typical end-to-end flow

```bash
# 1. Judge a JSONL of (prompt, response) pairs.
judge-cascade run \
  --input my_agent_decisions.jsonl \
  --output judged.jsonl \
  --api-base https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --cascade haiku-sonnet-opus

# 2. Extract preference pairs for downstream DPO training.
python examples/03_extract_preference_pairs.py judged.jsonl preferences.jsonl
```
