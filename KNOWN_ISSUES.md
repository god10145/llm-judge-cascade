# Known issues

Tracked here, not yet filed against the eventual public repo (no remote yet). Fix or strike through as resolved.

## 1. Cost charged on transport-level failures

**Severity:** Low (small inaccuracy in spend reporting; no functional impact)
**Discovered:** 2026-05-20 during first integration smoke-test

**Reproduction:** point `--api-base` at an unreachable host (e.g., `http://litellm:4000/v1` from outside the Docker network). Tool fails with `ConnectionError`/`NameResolutionError` in `issues`, returns empty `raw_response`, but `total_cost_usd` reports a non-zero value (~$0.003 for Sonnet pricing applied to estimated input tokens).

**Root cause:** in `cost.py` / `cascade.py`, cost is estimated from input/output token counts at the *start* of the call. When the call fails, output tokens are zero but input tokens are still counted at the per-million rate. The pricing pipeline doesn't have a "was this call successful?" gate.

**Suggested fix:** in `cascade.py`'s tier-result handling, skip cost estimation entirely if `tier_result.raw_response == ""` or `tier_result.issues` contains a `transport_error:*`. Alternatively, in `cost.py`, accept the `JudgeResult` and short-circuit to `0.0` when those conditions hold.

**Test that should exist after fix:** mock client that returns transport error → assert `total_cost_usd == 0.0`.

---

*(empty for now; add as found)*
