# Judge prompts

`llm-judge-cascade` ships three built-in judge prompts, each optimized for
a slightly different failure mode. All three share the same JSON output
schema, so they are interchangeable inside a `CascadeTier`.

## Built-in prompts

### `DEFAULT_JUDGE_PROMPT_GENERIC`

The all-purpose default. Asks the judge to evaluate three things at once:

- **Engagement:** does the response address the user's input, or does it
  drift?
- **Grounding:** does the response cite facts, names, or numbers that
  are not in the input?
- **Internal consistency:** does the reasoning chain hold together?

Use this when you do not know in advance what kind of failure modes are
most common in your agent's output. It is the default for every tier in
`make_default_cascade`.

### `DEFAULT_JUDGE_PROMPT_REASONING_QUALITY`

A stricter variant that focuses on whether the response shows its work.
Demands that every substantive claim be traceable back to the input.
Useful when:

- The agent is supposed to do explicit reasoning (planning, analysis,
  multi-step explanations).
- You care about which justifications survive scrutiny, not just whether
  the final answer is correct.

### `DEFAULT_JUDGE_PROMPT_FABRICATION_DETECTION`

A narrower variant that flags any specific fact, citation, number, or
named entity in the response that is not in the input. Useful when:

- You expect short, factual responses (classifications, extractions).
- You want a high recall fabrication detector for filtering training
  data before fine-tuning.

## JSON output schema (all three prompts)

Every built-in prompt instructs the judge to emit a single JSON object:

```json
{
  "verdict": "SOUND | FLAWED | FABRICATED | UNCERTAIN",
  "confidence": 0.0,
  "issues": ["..."],
  "annotation": "...",
  "rewritten_response": null
}
```

`parse_judge_response` is permissive about formatting; it strips markdown
fences and falls back to brace-grep extraction so the cascade does not
fail on minor prompt drift.

## Writing a custom judge prompt

Sometimes a domain-specific prompt produces better verdicts than a
generic one. The pattern is:

1. Start from one of the built-ins as a template.
2. Replace the engagement / grounding / consistency framing with domain
   criteria that matter for your use case.
3. Keep the JSON schema and worked examples block intact.
4. Pass the new prompt into a `CascadeTier(judge_prompt=...)`.

### Example: judging code-review comments

Imagine an agent that writes code-review comments on diffs. You want to
catch comments that misread the diff, invent line numbers, or recommend
non-existent functions.

```python
from llm_judge_cascade import Cascade, CascadeTier, JudgeClientConfig

CODE_REVIEW_JUDGE_PROMPT = """\
You are evaluating an AI code reviewer's comment on a diff.

The user's input contains a code diff (lines prefixed + or -) and any
accompanying context the reviewer was given. The agent's response is a
single review comment.

Evaluate three things:

1. Diff fidelity: does the comment refer to lines, functions, or
   constructs that actually appear in the diff? Hallucinated line
   numbers and made-up function names are FABRICATED.
2. Actionability: does the comment give the author a clear next step,
   or is it vague? Vague comments are FLAWED.
3. Severity calibration: does the comment escalate trivial issues to
   "this is a bug" levels? Miscalibrated severity is FLAWED.

(Then include the standard JSON schema and worked-example blocks from
the built-in prompts.)
"""

config = JudgeClientConfig(api_base="...", api_key="...")
cascade = Cascade(
    tiers=[
        CascadeTier(
            model="anthropic/claude-haiku-4-5-20251001",
            judge_prompt=CODE_REVIEW_JUDGE_PROMPT,
            min_confidence_to_stop=0.7,
        ),
        CascadeTier(
            model="anthropic/claude-opus-4-7",
            judge_prompt=CODE_REVIEW_JUDGE_PROMPT,
            min_confidence_to_stop=0.0,
        ),
    ],
    config=config,
)
```

### Mixing prompts across tiers

Nothing requires every tier to use the same prompt. A reasonable pattern
is to use a permissive prompt at the cheap tier (catch the obvious cases
fast) and a stricter prompt at the final arbiter (do the careful work
only on records that survived the cheap filter). The cascade controller
will hand each tier's prompt to that tier's model.

## Things judge prompts cannot fix

- **Judge bias.** The judge has its own training-data preferences. If
  the underlying model is bad at, say, evaluating poetry, no prompt
  rewrite will save you.
- **Reward hacking by the agent.** If the agent has been trained against
  the same judge, expect to see goodharting. Cycle judges and refresh
  the prompts when this becomes visible.
- **Domain expertise the judge does not have.** The judge cannot
  evaluate medical advice if it does not know the medicine.
