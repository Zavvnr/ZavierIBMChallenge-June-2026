"""
agent/langflow_client.py — call a deployed Langflow flow over its REST API.

Thin client for the optional Langflow orchestration layer. The flow itself is built
in the Langflow UI (it can wrap agent.explainer via the custom component in flows/),
and is referenced here by LANGFLOW_FLOW_ID. We just POST the question to it.

Everything is best-effort: if Langflow isn't configured or is unreachable, callers
(agent.explainer.answer) fall back to the in-process explainer, so the feature never
hard-depends on a running Langflow server.

Env vars (names only; never reads .env files directly):
    LANGFLOW_BASE_URL  — e.g. http://localhost:7860
    LANGFLOW_FLOW_ID   — the deployed flow's id
    LANGFLOW_API_KEY   — sent as the x-api-key header (Langflow >= 1.5)
"""
from __future__ import annotations

import os

import requests


def is_configured() -> bool:
    """True when there's enough config to attempt a Langflow call."""
    return bool(os.getenv("LANGFLOW_BASE_URL") and os.getenv("LANGFLOW_FLOW_ID"))


def run_flow(input_value: str, timeout: float = 30.0) -> str:
    """POST `input_value` to the configured Langflow flow and return the text output.

    Raises if Langflow isn't configured or on any HTTP error, so callers can fall back.
    """
    base = (os.getenv("LANGFLOW_BASE_URL") or "").rstrip("/")
    flow_id = os.getenv("LANGFLOW_FLOW_ID")
    if not (base and flow_id):
        raise RuntimeError("Langflow not configured (set LANGFLOW_BASE_URL + LANGFLOW_FLOW_ID).")

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("LANGFLOW_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    payload = {"input_value": input_value, "input_type": "chat", "output_type": "chat"}

    resp = requests.post(f"{base}/api/v1/run/{flow_id}", json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return _extract_message(resp.json())


def _extract_message(data: dict) -> str:
    """Pull the chat message text out of a Langflow /run response (best-effort)."""
    try:
        outputs = data["outputs"][0]["outputs"][0]
        message = (outputs.get("results") or {}).get("message") or {}
        if isinstance(message, dict) and message.get("text"):
            return str(message["text"]).strip()
        msgs = outputs.get("messages")
        if isinstance(msgs, list) and msgs:
            return str(msgs[0].get("message", "")).strip()
    except (KeyError, IndexError, TypeError):
        pass
    return ""
