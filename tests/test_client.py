"""Tests for the HTTP client wrapper (no real network calls)."""
from __future__ import annotations

from unittest.mock import patch

from llm_judge_cascade.client import (
    ChatCompletionResult,
    JudgeClientConfig,
    chat_completion,
)


class _FakeResp:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _config() -> JudgeClientConfig:
    return JudgeClientConfig(api_base="http://example.test/v1", api_key="k", timeout_sec=5)


def test_successful_completion() -> None:
    payload = {"choices": [{"message": {"content": "  hello there  "}}]}
    with patch(
        "llm_judge_cascade.client.requests.post", return_value=_FakeResp(200, payload)
    ) as posted:
        result = chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            model="some-model",
            config=_config(),
        )
    assert isinstance(result, ChatCompletionResult)
    assert result.content == "hello there"
    assert result.error is None
    assert result.latency_ms >= 0

    call = posted.call_args
    assert call.args[0] == "http://example.test/v1/chat/completions"
    body = call.kwargs["json"]
    assert body["model"] == "some-model"
    assert body["max_tokens"] == 2000
    assert "temperature" not in body
    assert call.kwargs["headers"]["Authorization"] == "Bearer k"


def test_max_tokens_override() -> None:
    payload = {"choices": [{"message": {"content": "ok"}}]}
    with patch(
        "llm_judge_cascade.client.requests.post", return_value=_FakeResp(200, payload)
    ) as posted:
        chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            model="m",
            config=_config(),
            max_tokens=42,
        )
    assert posted.call_args.kwargs["json"]["max_tokens"] == 42


def test_http_error_surfaced() -> None:
    with patch(
        "llm_judge_cascade.client.requests.post",
        return_value=_FakeResp(500, text="upstream broke"),
    ):
        result = chat_completion(
            messages=[{"role": "user", "content": "x"}],
            model="m",
            config=_config(),
        )
    assert result.content == ""
    assert result.error is not None
    assert "HTTP 500" in result.error
    assert "upstream broke" in result.error


def test_empty_choices() -> None:
    with patch(
        "llm_judge_cascade.client.requests.post",
        return_value=_FakeResp(200, {"choices": []}),
    ):
        result = chat_completion(messages=[], model="m", config=_config())
    assert result.error is not None
    assert "empty_choices" in result.error


def test_network_exception() -> None:
    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ConnectionError("dns blew up")

    with patch("llm_judge_cascade.client.requests.post", side_effect=boom):
        result = chat_completion(messages=[], model="m", config=_config())
    assert result.content == ""
    assert result.error is not None
    assert "ConnectionError" in result.error


def test_api_base_trailing_slash_stripped() -> None:
    payload = {"choices": [{"message": {"content": "ok"}}]}
    cfg = JudgeClientConfig(api_base="http://example.test/v1/", api_key="k")
    with patch(
        "llm_judge_cascade.client.requests.post", return_value=_FakeResp(200, payload)
    ) as posted:
        chat_completion(messages=[], model="m", config=cfg)
    assert posted.call_args.args[0] == "http://example.test/v1/chat/completions"
