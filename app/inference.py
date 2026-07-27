"""LLM access through Hugging Face Inference Providers, via the OpenAI SDK.

Thin on purpose: this module knows how to send a chat completion and nothing about the
agent loop or MCP. Swapping provider is an env var change (LLM_BASE_URL, LLM_MODEL).
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessage

from app.config import get_settings

# Support answers must be reproducible and dull. Low but not zero, because some routed
# providers reject temperature=0 outright.
TEMPERATURE = 0.2
MAX_TOKENS = 900

_client: AsyncOpenAI | None = None


class InferenceError(RuntimeError):
    """The LLM call failed - bad credentials, provider outage, rate limit, timeout."""


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.hf_token,
            timeout=settings.request_timeout_s,
            max_retries=1,
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def complete(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> ChatCompletionMessage:
    """One chat completion. Returns the assistant message, tool calls included."""
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    try:
        response = await client().chat.completions.create(**kwargs)
    except OpenAIError as exc:
        raise InferenceError(f"{type(exc).__name__}: {exc}") from exc

    if not response.choices:
        raise InferenceError("LLM returned no choices.")
    return response.choices[0].message
