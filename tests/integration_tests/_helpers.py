"""Shared fixtures for integration tests.

Integration tests wire the REAL components together (replayer + agent + pipeline +
TTS + web). The only things faked are external boundaries: the Granite HTTP endpoint
and Google Cloud TTS. Because ``agent.prompts`` is not implemented yet, ``ensure_prompts``
installs a minimal stand-in so the real CommentaryAgent can be constructed — it is a
no-op once the real ``agent.prompts`` exposes these functions.
"""
from __future__ import annotations

import types


def ensure_prompts():
    """Provide the minimal agent.prompts API the real agent needs (if absent)."""
    import agent.prompts as p
    if not hasattr(p, "normalize_language"):
        p.normalize_language = lambda lang="en": (lang or "en")[:2].lower()
    if not hasattr(p, "system_prompt"):
        p.system_prompt = lambda lang: f"You are a faithful football commentator. Respond in {lang}."
    if not hasattr(p, "build_event_prompt"):
        p.build_event_prompt = lambda ev, state, context=None: f"Event={ev}; State={state}; Ctx={context or {}}"
    if not hasattr(p, "LANGUAGE_NAMES"):
        p.LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "id": "Indonesian"}
    if not hasattr(p, "SUPPORTED_LANGUAGE_CODES"):
        p.SUPPORTED_LANGUAGE_CODES = ["en", "es", "id"]
    return p


def build_up_with_goal():
    """An ordered window: a progressive pass, a skip-type, a saved shot, then a goal."""
    return [
        {"index": 1, "period": 1, "timestamp": "00:00:01.000", "minute": 0, "second": 1,
         "type": {"name": "Pass"}, "team": {"name": "Argentina"},
         "player": {"name": "Rodrigo De Paul"}, "pass": {"end_location": [88, 30]}},
        {"index": 2, "period": 1, "timestamp": "00:00:03.000", "minute": 0, "second": 3,
         "type": {"name": "Pressure"}, "team": {"name": "France"}},
        {"index": 3, "period": 1, "timestamp": "00:00:20.000", "minute": 0, "second": 20,
         "type": {"name": "Shot"}, "team": {"name": "Argentina"},
         "player": {"name": "Lionel Messi"}, "shot": {"outcome": {"name": "Saved"}}},
        {"index": 4, "period": 1, "timestamp": "00:00:40.000", "minute": 0, "second": 40,
         "type": {"name": "Shot"}, "team": {"name": "Argentina"},
         "player": {"name": "Lionel Messi"}, "shot": {"outcome": {"name": "Goal"}}},
    ]


class FakeGranite:
    """Stands in for the OpenAI-compatible Granite client (no network)."""

    def __init__(self, content="Granite commentary line."):
        self.content = content
        self.calls = 0

        def _create(**kwargs):
            self.calls += 1
            self.last_kwargs = kwargs
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.content))]
            )

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))
