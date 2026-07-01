"""MATE preflight diagnostics — a fail-safe "doctor" for the demo dependencies.

One place that answers "why isn't it working?" before you go hunting: it checks the
Granite endpoint (server up + chat/embedding models loaded), the Google TTS key, the
StatsBomb open-data, the Laws-of-the-Game index, and the vision clip — and reports a
clear ok / warn / fail per item with an actionable hint.

Design rules (match the rest of MATE):
  * FAIL-SAFE: every check is guarded; a broken check degrades to a result, never a crash.
  * SECRET-SAFE: env vars are checked by NAME only — values are never read into the output.
  * All model access goes through ``agent.granite_client`` (no other providers).

Use it from the command line (``python -m diagnostics``) or programmatically via
``run_all()`` (the web app's ``/api/health`` does the latter for its status banner).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from agent.granite_client import build_granite_client, embed_model_id, model_id

REPO_ROOT = Path(__file__).resolve().parent.parent

# Network checks fail fast so the doctor (and the page banner) never hang on a dead host.
GRANITE_PING_TIMEOUT_S = 4.0
STATSBOMB_TIMEOUT_S = 5.0

# Laws index filenames — mirror context/retrieve.py (kept here so this stays import-light).
_LAWS_FILES = ("laws_chunks.json", "laws_vectors.npy")

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class CheckResult:
    """One dependency's health: a name, a status, and a human hint (never a secret)."""

    name: str
    status: str            # OK | WARN | FAIL
    detail: str = ""       # what's wrong / what to do — safe to display
    required: bool = False  # True = needed for real (non-mock) commentary


def _loaded(target: str, model_ids: List[str]) -> bool:
    """True if a configured model id loosely matches one the server reports.

    LM Studio / Ollama often append a quant or namespace (``ibm/granite-4-h-tiny-GGUF``),
    so an exact match is too strict; a substring either way is the right test.
    """
    needle = (target or "").lower()
    return any(needle and (needle in mid.lower() or mid.lower() in needle) for mid in model_ids)


def _probe_openai() -> tuple:
    """Tell 'openai missing' apart from 'openai installed but failing to import'.

    ``build_granite_client`` reports every import failure as "openai not installed", which
    hides a corrupted install — e.g. a compiled dependency with a DLL load error, common
    when a venv lives in a cloud-synced folder. Returns ``(importable, detail)``; ``detail``
    is empty when the import works.
    """
    import importlib
    import importlib.util
    try:
        spec = importlib.util.find_spec("openai")
    except BaseException:
        spec = None
    if spec is None:
        return False, "openai not installed — `pip install openai`"
    try:
        importlib.import_module("openai")
    except BaseException as exc:
        return False, (
            f"openai present but import failed ({type(exc).__name__}: {exc}) — likely a "
            "corrupted install; try `pip install --force-reinstall --no-cache-dir openai`"
        )
    return True, ""


def check_granite(client=None) -> List[CheckResult]:
    """Granite endpoint reachable, and the chat + embedding models actually loaded."""
    chat_name, embed_name = "Granite chat", "Granite embeddings"
    hint = "start LM Studio and load the models (or tick Fast demo for offline)"

    if client is None:  # surface the real import error before the generic client build
        importable, detail = _probe_openai()
        if not importable:
            return [
                CheckResult(chat_name, FAIL, detail, required=True),
                CheckResult(embed_name, WARN, "needs the Granite client too"),
            ]

    try:
        client = client or build_granite_client()
    except BaseException as exc:  # SystemExit when GRANITE_BASE_URL/SDK is missing
        reason = str(exc) or exc.__class__.__name__
        return [
            CheckResult(chat_name, FAIL, f"client unavailable: {reason}", required=True),
            CheckResult(embed_name, WARN, "needs the Granite client too"),
        ]

    try:
        pinger = client
        try:
            pinger = client.with_options(timeout=GRANITE_PING_TIMEOUT_S)
        except Exception:
            pass  # a test double or older SDK without with_options — ping anyway
        listing = pinger.models.list()
        data = getattr(listing, "data", None) or listing
        model_ids = [getattr(m, "id", "") for m in data]
    except BaseException as exc:
        reason = str(exc) or exc.__class__.__name__
        return [
            CheckResult(chat_name, FAIL, f"unreachable: {reason}; {hint}", required=True),
            CheckResult(embed_name, WARN, "could not query the server"),
        ]

    chat, embed = model_id(), embed_model_id()
    results = []
    if _loaded(chat, model_ids):
        results.append(CheckResult(chat_name, OK, f"'{chat}' loaded", required=True))
    else:
        results.append(CheckResult(
            chat_name, FAIL, f"server up but '{chat}' not loaded — load it in LM Studio",
            required=True))
    if _loaded(embed, model_ids):
        results.append(CheckResult(embed_name, OK, f"'{embed}' loaded"))
    else:
        results.append(CheckResult(
            embed_name, WARN, f"'{embed}' not loaded — the explainer/RAG will be limited"))
    return results


def check_tts_key() -> CheckResult:
    """Is GOOGLE_TTS_API_KEY present? (Checked by name only — the value is never read out.)"""
    if os.getenv("GOOGLE_TTS_API_KEY"):
        return CheckResult("Google TTS key", OK, "GOOGLE_TTS_API_KEY is set")
    return CheckResult("Google TTS key", WARN,
                       "GOOGLE_TTS_API_KEY not set — audio falls back to text-only")


def check_statsbomb() -> CheckResult:
    """Is the StatsBomb open-data reachable (the demo match's first-use fetch)?"""
    try:
        import requests
        from data_extraction.loader import OPEN_DATA_BASE
        resp = requests.get(f"{OPEN_DATA_BASE}/competitions.json",
                            timeout=STATSBOMB_TIMEOUT_S, stream=True)
        resp.close()
        if resp.status_code < 400:
            return CheckResult("StatsBomb data", OK, "open-data reachable")
        return CheckResult("StatsBomb data", WARN,
                           f"HTTP {resp.status_code}; cached matches still work")
    except Exception as exc:
        return CheckResult("StatsBomb data", WARN,
                           f"unreachable ({type(exc).__name__}); cached matches still work")


def check_laws_index(index_dir=None) -> CheckResult:
    """Is the Laws-of-the-Game vector index built (needed by the explainer/RAG)?"""
    if index_dir is None:
        try:
            from context.retrieve import INDEX_DIR
            index_dir = Path(INDEX_DIR)
        except Exception:
            index_dir = REPO_ROOT / "context" / "index"
    index_dir = Path(index_dir)
    if all((index_dir / name).exists() for name in _LAWS_FILES):
        return CheckResult("Laws index", OK, "index built")
    return CheckResult("Laws index", WARN,
                       "not built — run `python -m context.ingest_laws` for the explainer")


def check_vision_events(events_path=None) -> CheckResult:
    """Is a vision clip's events.json present (the optional 'Vision clip' demo)?"""
    events = Path(events_path) if events_path else REPO_ROOT / "data" / "vision" / "events.json"
    if events.exists():
        return CheckResult("Vision clip", OK, "data/vision/events.json present")
    return CheckResult("Vision clip", WARN,
                       "no data/vision/events.json — optional, only for the vision demo")


def _guard(name: str, producer: Callable[[], List[CheckResult]],
           required: bool = False) -> List[CheckResult]:
    """Run a check, converting any unexpected error into a result so the doctor never crashes."""
    try:
        return list(producer())
    except BaseException as exc:  # a misbehaving check must not break the whole report
        return [CheckResult(name, FAIL if required else WARN,
                            f"check error: {type(exc).__name__}: {exc}", required)]


def run_all(granite_client=None) -> List[CheckResult]:
    """Run every check and return the results in display order. Never raises."""
    results: List[CheckResult] = []
    results += _guard("Granite", lambda: check_granite(granite_client), required=True)
    results += _guard("Google TTS key", lambda: [check_tts_key()])
    results += _guard("StatsBomb data", lambda: [check_statsbomb()])
    results += _guard("Laws index", lambda: [check_laws_index()])
    results += _guard("Vision clip", lambda: [check_vision_events()])
    return results


def overall_status(results: List[CheckResult]) -> str:
    """Worst status across results: fail if any fail, else warn if any warn, else ok."""
    if any(r.status == FAIL for r in results):
        return FAIL
    if any(r.status == WARN for r in results):
        return WARN
    return OK
