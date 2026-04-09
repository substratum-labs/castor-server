"""LLM adapter: LiteLLM → castor-kernel LLMSyscall bridge.

Translates the Anthropic-style tool/message format used by the agent loop
into LiteLLM ``acompletion`` calls, and wraps the result back into
kernel-compatible structures.
"""

from __future__ import annotations

import logging
from typing import Any

import litellm

from castor_server.config import settings

logger = logging.getLogger("castor_server.llm_adapter")


async def litellm_chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    **kwargs: Any,
) -> dict[str, Any]:
    """Call LiteLLM acompletion and return an Anthropic-compatible response dict."""
    litellm_model = settings.litellm_model_map.get(model, model)

    call_kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }
    if tools:
        call_kwargs["tools"] = tools

    logger.info("llm_call model=%s messages=%d", litellm_model, len(messages))
    response = await litellm.acompletion(**call_kwargs)

    choice = response.choices[0]
    result: dict[str, Any] = {
        "role": "assistant",
        "content": [],
        "stop_reason": choice.finish_reason,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    }

    if choice.message.content:
        result["content"].append({"type": "text", "text": choice.message.content})

    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            import json

            result["content"].append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                }
            )

    return result


async def litellm_chat_for_kernel(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    **kwargs: Any,
) -> dict[str, Any]:
    """Kernel-compatible LLM callable.

    Same as ``litellm_chat`` but with a signature that matches what
    ``LLMSyscall.infer()`` forwards as keyword arguments through the proxy.
    """
    return await litellm_chat(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
