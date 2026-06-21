"""Fetch grounded player facts from Wikipedia's public REST API (no key, no secret).

Returns a short factual summary + a photo URL for a player, in any supported language
(Wikipedia has per-language editions). Everything is best-effort: a network or lookup
failure returns None so callers degrade gracefully. Results are cached under
data/cache/profiles/ so repeated runs are fast and work offline afterwards.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "profiles"
REST_SUMMARY = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"

# Our language codes -> Wikipedia subdomain (identical for the 12 we support).
_WIKI_LANG = {
    "en": "en", "de": "de", "es": "es", "fr": "fr", "ja": "ja", "pt": "pt",
    "ar": "ar", "cs": "cs", "it": "it", "ko": "ko", "nl": "nl", "zh": "zh",
}


def _wiki_lang(language: Optional[str]) -> str:
    """Map a bare/BCP-47 code to a Wikipedia subdomain (default 'en')."""
    base = (language or "en").split("-")[0].lower()
    return _WIKI_LANG.get(base, "en")


def _slug(name: str) -> str:
    """Filesystem-safe cache key for a player name."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().replace(" ", "_")).strip("_") or "_"


def _default_fetcher(url: str) -> Optional[dict]:
    """Fetch + decode JSON from Wikipedia's REST API (lazy requests import)."""
    import requests  # already a project dependency
    resp = requests.get(
        url,
        headers={"accept": "application/json", "user-agent": "MATE/1.0 (educational project)"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_summary(name: str, language: str = "en",
                  fetcher: Optional[Callable[[str], Optional[dict]]] = None,
                  use_cache: bool = True) -> Optional[dict]:
    """Return {title, extract, description, photo_url, url, lang} for a player, or None.

    `fetcher` is injectable for offline tests; the default hits Wikipedia's REST summary
    endpoint. Returns None on any failure or when no usable summary is found.
    """
    if not name:
        return None
    lang = _wiki_lang(language)
    cache_path = CACHE_DIR / lang / f"{_slug(name)}.json"
    if use_cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    url = REST_SUMMARY.format(lang=lang, title=name.strip().replace(" ", "_"))
    try:
        data = (fetcher or _default_fetcher)(url)
    except Exception:
        return None
    if not data or "extract" not in data or not data.get("extract"):
        return None

    summary = {
        "title": data.get("title", name),
        "extract": data.get("extract", ""),
        "description": data.get("description", ""),
        "photo_url": (data.get("thumbnail") or {}).get("source", ""),
        "url": ((data.get("content_urls") or {}).get("desktop") or {}).get("page", ""),
        "lang": lang,
    }
    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        except Exception:
            pass
    return summary
