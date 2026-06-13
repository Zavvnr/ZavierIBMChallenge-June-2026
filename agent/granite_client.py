"""
agent/granite_client.py

Shared IBM Granite client for the commentary agent.

Granite is served behind an OpenAI-compatible API — locally via LM Studio or
Ollama, or hosted on watsonx.ai's OpenAI-compatible endpoint — so we talk to it
with the `openai` SDK pointed at GRANITE_BASE_URL. Both the text-generation loop
(agent/commentary_agent.py) and the context-retrieval embedder
(agent/mcp_client.py) import from here, which keeps all Granite configuration in
one place and avoids a circular import between those two modules.

Env vars used (names only; this module never reads .env files directly):
    GRANITE_BASE_URL    — OpenAI-compatible endpoint, e.g. http://localhost:1234/v1
                          (LM Studio) or http://localhost:11434/v1 (Ollama)
    GRANITE_API_KEY     — API key/placeholder; LM Studio uses the literal "lm-studio"
    GRANITE_MODEL_ID    — chat model id, e.g. granite-4-h-tiny
    GRANITE_EMBED_MODEL — (optional) embedding model id for RAG / context retrieval
"""
from __future__ import annotations

import os

# Defaults mirror the project's .env so the modules still work if a var is unset.
DEFAULT_MODEL_ID = "granite-4-h-tiny"
# A 768-dim multilingual Granite embedder, matching the commentary languages and
# the dimensionality the Atlas vector index was created with.
DEFAULT_EMBED_MODEL = "granite-embedding-278m-multilingual"


def model_id() -> str:
    """Return the configured Granite chat model id (or the project default)."""
    return os.getenv("GRANITE_MODEL_ID", DEFAULT_MODEL_ID)


def embed_model_id() -> str:
    """Return the configured Granite embedding model id (or the project default)."""
    return os.getenv("GRANITE_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def build_granite_client():
    """
    Return an OpenAI-compatible client pointed at the local/hosted Granite endpoint.

    Raises SystemExit with an actionable message if the SDK is missing or
    GRANITE_BASE_URL is unset, so the CLI fails clearly instead of deep in a stack
    trace. Callers that must stay alive (the live commentary loop) wrap their use
    of this in try/except and degrade gracefully.
    """
    try:
        from openai import OpenAI  # lazy import — only needed for a real (non-mock) run
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("openai not installed — `pip install openai`") from exc

    base_url = os.getenv("GRANITE_BASE_URL")
    if not base_url:
        raise SystemExit(
            "Set GRANITE_BASE_URL (e.g. http://localhost:1234/v1 for LM Studio) "
            "or run with --mock. See .env."
        )
    # LM Studio / Ollama accept any non-empty key; default to LM Studio's placeholder.
    api_key = os.getenv("GRANITE_API_KEY") or "lm-studio"
    return OpenAI(base_url=base_url, api_key=api_key)
