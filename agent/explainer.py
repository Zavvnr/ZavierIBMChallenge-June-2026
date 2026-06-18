"""
agent/explainer.py — the third commentator (on-demand tactical Q&A).

Answers a viewer's question, grounded in (a) the match event/state and (b) the Laws
of the Game retrieved from the local index (context.retrieve). Composes the explainer
system prompt (agent.prompts) and calls Granite. `client` and `retriever` are
injectable so it's testable with no endpoint and no index.

This is the Python engine; a Langflow flow can later wrap it, with this as the
in-process fallback.
"""
from __future__ import annotations

from typing import Callable, Optional

from agent import prompts


def _format_event(event: Optional[dict]) -> str:
    """A compact, faithful description of the event the question is about."""
    if not event:
        return "(no specific event in focus)"
    clock = f"{event.get('minute', 0):02d}:{event.get('second', 0):02d}"
    etype = (event.get("type") or {}).get("name", "event")
    parts = [f"[{clock}] {etype}"]
    team = (event.get("team") or {}).get("name", "")
    player = (event.get("player") or {}).get("name", "")
    if team:
        parts.append(team)
    if player:
        parts.append(player)
    if etype == "Shot":
        outcome = ((event.get("shot") or {}).get("outcome") or {}).get("name", "")
        if outcome:
            parts.append(f"outcome={outcome}")
    return " — ".join(parts)


def build_explainer_prompt(question: str, event: Optional[dict] = None,
                           laws: Optional[list] = None, state: Optional[dict] = None) -> str:
    """The user prompt: the question + the event in focus + retrieved Laws + match state."""
    lines = [f"QUESTION: {question}", f"EVENT IN FOCUS: {_format_event(event)}"]
    if state:
        lines.append(f"MATCH STATE: {state}")
    if laws:
        lines.append("RELEVANT LAWS OF THE GAME:")
        for i, chunk in enumerate(laws, 1):
            text = chunk["text"] if isinstance(chunk, dict) else str(chunk)
            lines.append(f"[{i}] {text}")
    else:
        lines.append("RELEVANT LAWS OF THE GAME: (none retrieved)")
    lines.append(
        "Answer the question grounded ONLY in the event and the laws above. If the "
        "laws shown don't cover it, say so rather than guessing. Briefly cite the "
        "relevant law or event."
    )
    return "\n".join(lines)


def explain(question: str, event: Optional[dict] = None, state: Optional[dict] = None,
            language: str = "en", k: int = 4,
            client=None, retriever: Optional[Callable] = None) -> str:
    """Answer one question via Granite, grounded in retrieved Laws + the event.

    Retrieval is best-effort: if it fails (or no index is built), the explainer
    still answers from the event alone. `client`/`retriever` are injectable for tests.
    """
    from agent.granite_client import model_id

    if retriever is None:
        from context.retrieve import retrieve as retriever
    try:
        laws = retriever(question, k=k)
    except Exception:
        laws = []  # no index / retrieval error -> answer from the event alone

    lang = prompts.normalize_language(language)
    system = prompts.explainer_system_prompt(lang)
    user = build_explainer_prompt(question, event=event, laws=laws, state=state)

    if client is None:
        from agent.granite_client import build_granite_client
        client = build_granite_client()
    resp = client.chat.completions.create(
        model=model_id(),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.4,  # lower temperature: factual, grounded answers
    )
    return (resp.choices[0].message.content or "").strip()


def answer(question: str, event: Optional[dict] = None, state: Optional[dict] = None,
           language: str = "en", k: int = 4) -> str:
    """Answer via the Langflow flow if configured, else the in-process explainer.

    Langflow is best-effort orchestration (the IBM-stack layer): any failure —
    unconfigured, unreachable, or an empty result — falls back to explain(), so the
    explainer always responds. This is what /api/ask calls.
    """
    from agent import langflow_client
    if langflow_client.is_configured():
        try:
            text = langflow_client.run_flow(question)
            if text:
                return text
        except Exception:
            pass  # Langflow down/misconfigured -> in-process fallback below
    return explain(question, event=event, state=state, language=language, k=k)
