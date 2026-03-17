"""Multi-provider LLM configuration for GenAI Agents tutorials.

Provides a unified ``get_llm()`` helper that returns a LangChain
``ChatOpenAI`` instance wired to the requested provider.  Every
OpenAI-compatible provider (including MiniMax) is supported through
the same interface.

Supported providers
-------------------
- **openai** – OpenAI GPT models (default)
- **minimax** – MiniMax M2.5 / M2.5-highspeed (204K context, OpenAI-compatible)

Usage::

    from utils.llm_provider import get_llm

    # Default – OpenAI
    llm = get_llm()

    # MiniMax M2.5
    llm = get_llm(provider="minimax")

    # MiniMax high-speed variant
    llm = get_llm(provider="minimax", model="MiniMax-M2.5-highspeed")

Environment variables
---------------------
Set the API key for the provider you want to use:

- ``OPENAI_API_KEY`` – for OpenAI
- ``MINIMAX_API_KEY`` – for MiniMax (obtain at https://www.minimaxi.com/)
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "default_model": "MiniMax-M2.5",
    },
}


def list_providers() -> list[str]:
    """Return the names of all registered providers."""
    return list(PROVIDERS.keys())


def get_provider_info(provider: str) -> dict[str, Any]:
    """Return configuration metadata for *provider*.

    Raises ``ValueError`` if the provider is unknown.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Available providers: {list_providers()}"
        )
    return dict(PROVIDERS[provider])


def get_llm(
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs: Any,
) -> ChatOpenAI:
    """Return a :class:`ChatOpenAI` instance configured for *provider*.

    Parameters
    ----------
    provider:
        Name of the provider (see :data:`PROVIDERS`).
    model:
        Model identifier.  Falls back to the provider's default model when
        ``None``.
    temperature:
        Sampling temperature.  Automatically clamped to ``0.01`` for MiniMax,
        whose API rejects ``0``.
    max_tokens:
        Maximum number of tokens to generate.
    **kwargs:
        Extra keyword arguments forwarded to :class:`ChatOpenAI`.
    """
    load_dotenv()

    config = get_provider_info(provider)  # raises on bad name

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise EnvironmentError(
            f"Missing API key: please set the {config['api_key_env']!r} "
            f"environment variable."
        )

    resolved_model = model or config["default_model"]
    base_url = config["base_url"]

    # MiniMax rejects temperature == 0; clamp to a near-zero value.
    if provider == "minimax" and temperature <= 0:
        temperature = 0.01

    params: dict[str, Any] = {
        "model": resolved_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": api_key,
    }
    if base_url is not None:
        params["base_url"] = base_url
    params.update(kwargs)

    return ChatOpenAI(**params)
