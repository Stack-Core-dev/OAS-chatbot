import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

_resolved_model: str | None = None


class LLMError(RuntimeError):
    pass


async def list_models() -> list[str]:
    try:
        # llama.cpp:
        # async with httpx.AsyncClient(base_url=settings.llama_base_url, timeout=10.0) as client:
        async with httpx.AsyncClient(base_url=settings.geniex_base_url, timeout=10.0) as client:
            response = await client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        # raise LLMError(f"llama.cpp server unreachable at {settings.llama_base_url}: {exc}") from exc
        raise LLMError(f"GenieX server unreachable at {settings.geniex_base_url}: {exc}") from exc

    return [model["id"] for model in data.get("data", [])]


async def resolve_model() -> str:
    global _resolved_model
    if _resolved_model is not None:
        return _resolved_model

    models = await list_models()
    if not models:
        raise LLMError("GenieX server reports no available models")

    if settings.geniex_model in models:
        _resolved_model = settings.geniex_model
        logger.info("Using GenieX model: %s", _resolved_model)
    else:
        _resolved_model = models[0]
        logger.warning(
            "Configured model '%s' not found; falling back to available model '%s' (available: %s)",
            settings.geniex_model,
            _resolved_model,
            ", ".join(models),
        )

    return _resolved_model


async def chat(messages: list[dict[str, str]]) -> str:
    model = await resolve_model()
    payload = {"model": model, "messages": messages, "stream": False}

    try:
        # llama.cpp:
        # async with httpx.AsyncClient(
        #     base_url=settings.llama_base_url, timeout=settings.llama_timeout
        # ) as client:
        async with httpx.AsyncClient(
            base_url=settings.geniex_base_url, timeout=settings.geniex_timeout
        ) as client:
            response = await client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise LLMError(f"GenieX request failed: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected response from GenieX: {data}") from exc


async def stream_chat(messages: list[dict[str, str]]) -> AsyncIterator[tuple[str, str]]:
    """Yield ("reasoning" | "delta", text) tuples from GenieX."""
    model = await resolve_model()
    payload = {"model": model, "messages": messages, "stream": True}

    try:
        # llama.cpp:
        # async with httpx.AsyncClient(
        #     base_url=settings.llama_base_url, timeout=settings.llama_timeout
        # ) as client:
        async with httpx.AsyncClient(
            base_url=settings.geniex_base_url, timeout=settings.geniex_timeout
        ) as client:
            async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    if raw == "[DONE]":
                        break
                    choices = json.loads(raw).get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        yield ("reasoning", reasoning)
                    content = delta.get("content")
                    if content:
                        yield ("delta", content)
    except httpx.HTTPError as exc:
        raise LLMError(f"GenieX request failed: {exc}") from exc
