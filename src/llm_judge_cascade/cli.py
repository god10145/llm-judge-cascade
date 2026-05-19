"""Command-line interface: ``judge-cascade run --input ... --output ...``.

Supports three modes via ``--cascade``:

- ``haiku-sonnet-opus`` (default cost-tiered cascade)
- ``single:<model>`` (one model, one pass per record)
- ``panel:<model1>,<model2>,...`` (every model judges every record)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a hard dependency
    def tqdm(it, **_kw):  # type: ignore[no-redef]
        return it

from llm_judge_cascade.cascade import (
    Cascade,
    CascadeRunResult,
    CascadeTier,
    make_default_cascade,
)
from llm_judge_cascade.client import JudgeClientConfig
from llm_judge_cascade.dataset import (
    DecisionRecord,
    load_decisions,
    save_decisions_with_judgments,
)
from llm_judge_cascade.judges import DEFAULT_JUDGE_PROMPT_GENERIC

# ─── cascade resolution ────────────────────────────────────────────────────

CascadeFactory = "callable[[JudgeClientConfig], Cascade]"


def _single_model_cascade(model: str) -> CascadeFactory:
    def factory(config: JudgeClientConfig) -> Cascade:
        return Cascade(
            tiers=[
                CascadeTier(
                    model=model,
                    judge_prompt=DEFAULT_JUDGE_PROMPT_GENERIC,
                    escalate_if_uncertain=False,
                    escalate_if_flawed=False,
                    min_confidence_to_stop=0.0,
                )
            ],
            config=config,
        )

    return factory


def _panel_cascade(models: list[str]) -> CascadeFactory:
    """Force every tier to run by demanding impossible-to-meet confidence."""
    def factory(config: JudgeClientConfig) -> Cascade:
        return Cascade(
            tiers=[
                CascadeTier(
                    model=m,
                    judge_prompt=DEFAULT_JUDGE_PROMPT_GENERIC,
                    escalate_if_uncertain=True,
                    escalate_if_flawed=True,
                    min_confidence_to_stop=1.01,
                )
                for m in models
            ],
            config=config,
        )

    return factory


def _resolve_cascade(spec: str) -> CascadeFactory:
    spec = spec.strip()
    if spec == "haiku-sonnet-opus":
        return make_default_cascade
    if spec.startswith("single:"):
        model = spec[len("single:") :].strip()
        if not model:
            raise ValueError("--cascade single:<model> requires a model name")
        return _single_model_cascade(model)
    if spec.startswith("panel:"):
        models = [m.strip() for m in spec[len("panel:") :].split(",") if m.strip()]
        if not models:
            raise ValueError("--cascade panel:<m1>,<m2>,... requires at least one model")
        return _panel_cascade(models)
    raise ValueError(
        f"Unrecognized --cascade preset: {spec!r}. "
        "Use 'haiku-sonnet-opus', 'single:<model>', or 'panel:<m1>,<m2>,...'."
    )


# ─── batch runner ──────────────────────────────────────────────────────────


def _run_batch(
    records: Iterable[DecisionRecord],
    cascade: Cascade,
    *,
    cost_cap_usd: float | None,
    verbose: bool,
) -> Iterable[tuple[DecisionRecord, CascadeRunResult]]:
    total_usd = 0.0
    capped = False
    for record in tqdm(records, desc="Judging", unit="rec"):
        if capped:
            break
        result = cascade.judge(prompt=record.prompt, response=record.response)
        total_usd += result.total_cost_usd
        if verbose:
            verdicts = " -> ".join(tr.verdict.value for tr in result.tier_results)
            sys.stderr.write(
                f"  [{record.id or '?'}] {verdicts}  "
                f"final={result.final_verdict.value} "
                f"conf={result.final_confidence:.2f} "
                f"cost=${result.total_cost_usd:.4f} "
                f"(running total ${total_usd:.4f})\n"
            )
        yield record, result
        if cost_cap_usd is not None and total_usd >= cost_cap_usd:
            sys.stderr.write(
                f"\nCost cap reached: accumulated ${total_usd:.4f} >= "
                f"${cost_cap_usd:.4f}. Stopping after current record.\n"
            )
            capped = True


# ─── argparse + main ───────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judge-cascade",
        description=(
            "Run a cost-tiered LLM-as-judge cascade over a JSONL file of "
            "(prompt, response) pairs and emit an augmented JSONL with "
            "per-record judgments."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Judge a JSONL file of decisions")
    run.add_argument("--input", required=True, type=Path, help="Input JSONL path")
    run.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    run.add_argument(
        "--api-base",
        required=True,
        help="Base URL of an OpenAI-compatible API (e.g. https://api.openai.com/v1)",
    )
    run.add_argument(
        "--api-key-env",
        required=True,
        help="Name of the environment variable holding the API key",
    )
    run.add_argument(
        "--cascade",
        default="haiku-sonnet-opus",
        help=(
            "Cascade preset: 'haiku-sonnet-opus' (default), "
            "'single:<model>', or 'panel:<m1>,<m2>,...'"
        ),
    )
    run.add_argument(
        "--cost-cap-usd",
        type=float,
        default=None,
        help="Stop processing once cumulative cost reaches this many USD",
    )
    run.add_argument("--timeout-sec", type=int, default=90, help="Per-call timeout")
    run.add_argument(
        "--max-tokens", type=int, default=2000, help="Default max_tokens per call"
    )
    run.add_argument(
        "--verbose", action="store_true", help="Stream per-record progress to stderr"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 2

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.stderr.write(
            f"Environment variable {args.api_key_env!r} is unset or empty.\n"
        )
        return 2

    if not args.input.exists():
        sys.stderr.write(f"Input file does not exist: {args.input}\n")
        return 2

    try:
        cascade_factory = _resolve_cascade(args.cascade)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    config = JudgeClientConfig(
        api_base=args.api_base,
        api_key=api_key,
        timeout_sec=args.timeout_sec,
        default_max_tokens=args.max_tokens,
    )
    cascade = cascade_factory(config)

    records = load_decisions(args.input)
    pairs = _run_batch(
        records,
        cascade,
        cost_cap_usd=args.cost_cap_usd,
        verbose=args.verbose,
    )
    n = save_decisions_with_judgments(args.output, pairs)

    sys.stderr.write(
        f"Wrote {n} record(s) to {args.output}. "
        "See per-record 'judgments' block for verdicts and costs.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
