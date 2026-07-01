"""Curated + Wikipedia match briefings: the stakes and storylines a prepared commentator
knows before kickoff, so the opening scene-set is grounded ("Messi's likely last World
Cup") instead of generic.

This is editorial scene-setting context ONLY — never the StatsBomb event feed, which is
always fetched live via ``data_extraction.loader``. Curated entries cover the demo match;
any other match falls back to a best-effort Wikipedia article summary (cached), then to
nothing (the opening simply stays generic). Mirrors the curated+Wikipedia pattern in
``agent.player_facts``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "match_briefings"

# Curated briefings keyed by StatsBomb match id. Editorial context, NOT event data.
# Core idea: Real commentators know the stakes and storylines before kickoff, so the opening scene-set is grounded 
# ("Messi's likely last World Cup") instead of generic. Mirrors the curated+Wikipedia pattern in agent.player_facts.
CURATED = {
    "3869685": {                        # 2022 FIFA World Cup Final — Argentina vs France
        "headline": "the 2022 World Cup final",
        "stakes": ("Argentina chase a third world title and Lionel Messi the one prize that "
                   "has eluded him; France, the holders, go for back-to-back crowns."),
        "storylines": [
            "Likely Messi's final World Cup.",
            "Kylian Mbappe and Messi — Paris Saint-Germain team-mates, now rivals.",
            "France bidding to retain the trophy they won in 2018.",
        ],
        "watch": "Messi and Julian Alvarez for Argentina; Mbappe and Griezmann for France.",
        "source_url": "",
    },
}


def _slug(text: str) -> str:
    """Filesystem-safe slug for a cache key."""
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "match"


def _lang(language: str) -> str:
    """The base language code (es-ES -> es) used for cache folders."""
    return (language or "en").split("-")[0]


def _read_cache(key: str, language: str) -> Optional[dict]:
    """Read a cached Wikipedia-derived briefing from disk, or None."""
    path = CACHE_DIR / _lang(language) / f"{_slug(key)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_cache(key: str, language: str, briefing: dict) -> None:
    """Persist a briefing so future runs read it offline (best-effort)."""
    path = CACHE_DIR / _lang(language) / f"{_slug(key)}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(briefing, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _candidate_titles(competition: str, stage: str, year: str) -> list:
    """Plausible Wikipedia article titles for a match, most specific first."""
    comp, st = (competition or "").strip(), (stage or "").strip()
    titles = []
    if comp and year and "final" in st.lower():
        titles.append(f"{year} {comp} Final")
    if comp and "final" in st.lower():
        titles.append(f"{comp} Final")
    if comp and year:
        titles.append(f"{year} {comp}")
    if comp:
        titles.append(comp)
    seen, out = set(), []
    for title in titles:
        if title and title not in seen:
            seen.add(title)
            out.append(title)
    return out


def _from_wikipedia(competition, stage, year, language, fetcher) -> Optional[dict]:
    """Best-effort briefing from the most specific resolvable Wikipedia article."""
    from profiles.wiki_client import fetch_summary
    for title in _candidate_titles(competition, stage, year):
        summary = fetch_summary(title, language, fetcher=fetcher, use_cache=False)
        extract = (summary or {}).get("extract", "").strip()
        if extract:
            first_two = " ".join(re.split(r"(?<=[.!?])\s", extract)[:2])
            return {
                "headline": title,
                "stakes": first_two[:360],
                "storylines": [],
                "watch": "",
                "source_url": (summary or {}).get("url", ""),
            }
    return None


def briefing_for(match_id=None, home="", away="", competition="", stage="", year="",
                 language="en", fetcher=None, use_cache=True) -> Optional[dict]:
    """Return a match briefing (curated first, then a cached Wikipedia fallback), or None.

    Keys: headline, stakes, storylines (list), watch, source_url. Network (Wikipedia) only
    happens on a cache miss for a non-curated match; ``fetcher`` is injectable for tests.
    """
    key = str(match_id) if match_id not in (None, "") else f"{home}-{away}"
    if key in CURATED:
        return dict(CURATED[key])
    if use_cache:
        cached = _read_cache(key, language)
        if cached:
            return cached
    brief = _from_wikipedia(competition, stage, year, language, fetcher)
    if brief and use_cache:
        _write_cache(key, language, brief)
    return brief


def note_text(briefing: Optional[dict], max_storylines: int = 2) -> str:
    """Condense a briefing into a short background string for the opening prompt."""
    if not briefing:
        return ""
    parts = []
    if briefing.get("stakes"):
        parts.append(briefing["stakes"].strip())
    storylines = [s.strip() for s in (briefing.get("storylines") or []) if s.strip()]
    if storylines:
        parts.append(" ".join(storylines[:max_storylines]))
    return " ".join(parts).strip()
