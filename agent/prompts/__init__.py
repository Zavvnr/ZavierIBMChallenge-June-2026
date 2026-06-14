"""
agent.prompts — assembles the commentary prompts from the .md files in this package.

The .md fragments (faithfulness, energy, pacing, language, agent_explainer) hold the
prompt CONTENT and are the single source of truth. This module only loads and
composes them into the functions the agent / pipeline / web call; it never edits
them. Drop a new fragment in this folder and reference it here to extend a prompt.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent

# Language code -> display name. Used by the web language picker and the prompts.
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "id": "Indonesian",
    "fr": "French", "pt": "Portuguese", "de": "German", "it": "Italian",
}
SUPPORTED_LANGUAGE_CODES = sorted(LANGUAGE_NAMES)

# Bare language -> a default BCP-47 locale, so the agent's language also resolves
# against the TTS voice maps (which are keyed by locale, e.g. "es-ES").
_DEFAULT_LOCALE = {
    "en": "en-US", "es": "es-ES", "id": "id-ID", "fr": "fr-FR",
    "pt": "pt-BR", "de": "de-DE", "it": "it-IT",
}


@lru_cache(maxsize=None)
def _fragment(name: str) -> str:
    """Read one prompt .md fragment (cached); empty string if it's missing."""
    try:
        return (_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def normalize_language(language: str | None = "en") -> str:
    """Return a BCP-47 locale (e.g. 'es-ES') so prompts and TTS voices agree.

    Accepts either a bare code ('es') or a full locale ('es-ES'); a bare code is
    mapped to its default locale, an already-qualified locale is kept as-is.
    """
    code = (language or "en").strip()
    if "-" in code:
        return code
    return _DEFAULT_LOCALE.get(code.lower(), code)


def language_display_name(code: str | None) -> str:
    """Human-readable language name from a bare or BCP-47 code (e.g. 'es-ES' -> 'Spanish')."""
    base = (code or "en").split("-")[0].lower()
    return LANGUAGE_NAMES.get(base, code or "")


def system_prompt(language: str) -> str:
    """Assemble the lead/analyst system prompt for `language` from the .md fragments."""
    name = language_display_name(language)
    parts = [
        "You are a live football commentator for MlangCast.",
        _fragment("faithfulness"),
        _fragment("energy"),
        _fragment("pacing"),
        _fragment("language").replace("{language_name}", name),
    ]
    return "\n\n".join(p for p in parts if p)


def explainer_system_prompt(language: str) -> str:
    """System prompt for the on-demand third (explainer/Q&A) commentator."""
    name = language_display_name(language)
    parts = [
        f"You are the tactical explainer for MlangCast. Answer the viewer's question "
        f"in {name}, grounded only in the match data and the rules.",
        _fragment("faithfulness"),
        _fragment("agent_explainer"),
        _fragment("language").replace("{language_name}", name),
    ]
    return "\n\n".join(p for p in parts if p)


def _event_facts(event: dict) -> str:
    """A compact, faithful description of a single event (only stated facts)."""
    etype = (event.get("type") or {}).get("name", "event")
    bits = [etype]
    team = (event.get("team") or {}).get("name", "")
    player = (event.get("player") or {}).get("name", "")
    if team:
        bits.append(f"team={team}")
    if player:
        bits.append(f"player={player}")
    if etype == "Shot":
        outcome = ((event.get("shot") or {}).get("outcome") or {}).get("name", "")
        if outcome:
            bits.append(f"outcome={outcome}")
    elif etype == "Pass":
        recipient = ((event.get("pass") or {}).get("recipient") or {}).get("name", "")
        if recipient:
            bits.append(f"to={recipient}")
    return "; ".join(bits)


def build_event_prompt(event: dict, state: dict, context: dict | None = None) -> str:
    """The per-event user prompt: one event + match state + any retrieved context.

    Instructs the model to reply exactly ``NO_COMMENT`` when nothing is worth
    saying, which the agent treats as "stay silent".
    """
    lines = [
        "Comment on THIS event only, using just the facts below. If nothing is worth "
        "saying right now, reply with exactly: NO_COMMENT",
        f"EVENT: {_event_facts(event)}",
        f"MATCH STATE: {state or {}}",
    ]
    if context:
        lines.append(f"CONTEXT: {context}")
    return "\n".join(lines)
