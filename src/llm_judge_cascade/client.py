"""HTTP client for OpenAI-compatible /v1/chat/completions endpoints.

Works against any provider that speaks the OpenAI chat completions wire
format: OpenAI, LiteLLM, vLLM, Together, Anyscale, etc. The client is
deliberately small and does not pass ``temperature`` (some newer judge
models reject the field) — if you need to override sampling, pass it
through ``max_tokens`` or extend ``extra_body`` in a subclass.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests


@dataclass
class JudgeClientConfig:
    """Connection settings for an OpenAI-compatible endpoint.

    ``api_base`` should be the root of the API (e.g. ``https://api.openai.com/v1``
    or ``http://litellm:4000/v1``). The client appends ``/chat/completions``.
    """

    api_base: str
    api_key: str
    timeout_sec: int = 90
    default_max_tokens: int = 2000


@dataclass
class ChatCompletionResult:
    """Outcome of a single chat completion call."""

    content: str
    latency_ms: int
    error: str | None = None


def chat_completion(
    messages: list[dict],
    model: str,
    config: JudgeClientConfig,
    max_tokens: int | None = None,
) -> ChatCompletionResult:
    """POST to ``{api_base}/chat/completions`` and return the assistant content.

    Never raises on HTTP or network errors — they are surfaced via
    ``ChatCompletionResult.error`` so a caller orchestrating many requests
    can decide whether to retry, skip, or abort.
    """
    url = f"{config.api_base.rstrip('/')}/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else config.default_max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    t0 = time.time()
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=config.timeout_sec)
    except Exception as exc:
        latency_ms = int((time.time() - t0) * 1000)
        return ChatCompletionResult(content="", latency_ms=latency_ms, error=repr(exc))

    latency_ms = int((time.time() - t0) * 1000)
    if resp.status_code != 200:
        snippet = resp.text[:300] if resp.text else ""
        return ChatCompletionResult(
            content="",
            latency_ms=latency_ms,
            error=f"HTTP {resp.status_code}: {snippet}",
        )

    try:
        data = resp.json()
    except ValueError as exc:
        return ChatCompletionResult(
            content="",
            latency_ms=latency_ms,
            error=f"non_json_response: {exc!r}",
        )

    choices = data.get("choices") or []
    if not choices:
        return ChatCompletionResult(
            content="",
            latency_ms=latency_ms,
            error=f"empty_choices: {data!r}",
        )
    message = choices[0].get("message") or {}
    content = str(message.get("content") or "").strip()
    return ChatCompletionResult(content=content, latency_ms=latency_ms, error=None)
