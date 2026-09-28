"""Optional JSON fallback calls for Groq-compatible audit stages."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def is_rate_limit_error(error: Exception) -> bool:
    return (
        getattr(error, "status_code", None) == 429
        or getattr(error, "code", None) == 429
        or "rate limit" in str(error).lower()
        or "429" in str(error)
    )


def _post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        timeout = float(os.getenv("PROVIDER_REQUEST_TIMEOUT", "30"))
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Provider HTTP {error.code}: {detail}") from error


def request_fallback(messages: list[dict[str, str]], max_tokens: int) -> tuple[str, str]:
    """Try Kimi then Gemini through OpenRouter using one API key."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("KIMI_API_KEY")
    if not api_key:
        raise RuntimeError("Groq is rate-limited and OPENROUTER_API_KEY is not configured")

    configured_models = os.getenv("OPENROUTER_MODELS")
    if configured_models:
        models = [model.strip() for model in configured_models.split(",") if model.strip()]
    else:
        models = [
            "qwen/qwen3.8-27b:free",
            "google/gemma-4-31b-it:free",
            "nvidia/nemotron-3.5-lightning:free",
            "inclusionai/ling-3.0-flash-sante:free",
        ]
    model_errors = []
    for model in models:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            result = _post_json(
                "https://openrouter.ai/api/v1/chat/completions",
                payload,
                {
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/",
                    "X-Title": "SPEG Audit Scorecard Automation",
                },
            )
            content = result["choices"][0]["message"].get("content")
            if content:
                return content, f"openrouter/{model}"
            raise ValueError(f"OpenRouter returned an empty response for {model}")
        except Exception as error:
            model_errors.append(f"{model}: {error}")
            message = str(error).lower()
            if not (
                is_rate_limit_error(error)
                or "empty response" in message
                or "http 400" in message
                or "http 404" in message
                or "http 402" in message
            ):
                raise

    details = " | ".join(model_errors)
    raise RuntimeError(f"All configured OpenRouter models failed: {details}")


def request_with_fallback(client, model: str, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, str]:
    """Use OpenRouter only when configured, otherwise use Groq then fallback."""
    if os.getenv("OPENROUTER_ONLY", "false").lower() == "true":
        return request_fallback(messages, max_tokens)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content
        if content:
            return content, f"groq/{model}"
        return request_fallback(messages, max_tokens)
    except Exception as error:
        if not is_rate_limit_error(error):
            raise
        return request_fallback(messages, max_tokens)
    return request_fallback(messages, max_tokens)