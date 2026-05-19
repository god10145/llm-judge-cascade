"""Convert judged JSONL into DPO-style preference pairs.

For each record where any tier produced a ``rewritten_response``, emits
``(prompt, chosen=rewritten, rejected=original)``. The chosen string is
taken from the highest-confidence tier that supplied a rewrite.

Run after one of the judging examples:

    python examples/03_extract_preference_pairs.py judged.jsonl preferences.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def best_rewritten(tier_results: list[dict]) -> tuple[str, str] | None:
    candidates = [
        (tr.get("rewritten_response"), float(tr.get("confidence") or 0.0), tr.get("model", ""))
        for tr in tier_results
        if tr.get("rewritten_response")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    chosen, _conf, model = candidates[0]
    return chosen, model


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: 03_extract_preference_pairs.py <judged.jsonl> <prefs.jsonl>\n")
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    n_in = 0
    n_out = 0
    with src.open("r", encoding="utf-8") as fh_in, dst.open("w", encoding="utf-8") as fh_out:
        for line in fh_in:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            payload = json.loads(line)
            tier_results = payload.get("judgments", {}).get("tier_results", [])
            best = best_rewritten(tier_results)
            if best is None:
                continue
            chosen, judge_model = best
            fh_out.write(
                json.dumps(
                    {
                        "id": payload.get("id"),
                        "prompt": payload["prompt"],
                        "chosen": chosen,
                        "rejected": payload["response"],
                        "source_judge_model": judge_model,
                        "final_verdict": payload.get("judgments", {}).get("final_verdict"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_out += 1
    sys.stderr.write(
        f"Read {n_in} judged record(s); wrote {n_out} preference pair(s) to {dst}.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
