"""Build a grounded, multilingual player profile (the player-explainer feature).

Combines what we KNOW (StatsBomb position + the player's involvement in the match) with a
factual Wikipedia summary, then asks Granite to write a short profile in the target
language — strictly from the provided facts, never inventing. Fail-safe at every step:
no Wikipedia -> a match-only note; no Granite -> the raw factual summary.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

from agent import prompts
from profiles import wiki_client

# Disk cache for finished profiles, mirroring agent.player_facts: a click (or the live
# loop) becomes a file read instead of a Granite call — so profiles can be pre-warmed
# before commentary and never contend with it for the model.
PROFILE_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "profiles"


def _slug(name: str) -> str:
    """Filesystem-safe slug for a player name."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "player"


def _cache_path(name: str, language: str, cache_dir: Path) -> Path:
    """Where a player's profile is cached for a given language."""
    return Path(cache_dir) / language / f"{_slug(name)}.json"


def _read_cache(name: str, language: str, cache_dir: Path) -> Optional[dict]:
    """Return a previously cached profile, or None."""
    path = _cache_path(name, language, cache_dir)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_cache(name: str, language: str, profile: dict, cache_dir: Path) -> None:
    """Persist a profile so future clicks/runs read it offline (best-effort)."""
    path = _cache_path(name, language, cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _build_prompt(name: str, language_name: str, summary: Optional[dict],
                  position: str, involvement: str) -> str:
    """Assemble the grounded, anti-hallucination profile prompt."""
    facts = [f"Player: {name}"]
    if position:
        facts.append(f"Position in this match: {position}")
    if summary:
        if summary.get("description"):
            facts.append(f"Wikipedia description: {summary['description']}")
        facts.append(f"Wikipedia summary: {summary['extract']}")
    if involvement:
        facts.append(f"Involvement so far in this match: {involvement}")
    facts_block = "\n".join(facts)
    return (
        f"Write a short player profile in {language_name}, using ONLY the facts below. "
        f"Cover, when the facts support it: background, current club, position, and what the "
        f"player is famous for; then one sentence on what to watch for when they are on the "
        f"ball. Do NOT invent clubs, honours, or statistics. Omit anything the facts don't "
        f"mention. Keep it to about four short sentences.\n\nFACTS:\n{facts_block}"
    )


def _generate(prompt: str, client=None) -> Optional[str]:
    """Call Granite for the profile text; return None on any failure (fail-safe)."""
    try:
        from agent.granite_client import build_granite_client, model_id
        client = client or build_granite_client()
        resp = client.chat.completions.create(
            model=model_id(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None


def build_profile(name: str, language: str = "en", involvement: str = "", position: str = "",
                  *, granite_client=None, fetcher=None, use_cache: bool = True,
                  cache_dir=None) -> dict:
    """Return a grounded profile dict for a player in the target language.

    Keys: name, language, photo_url, source_url, grounded (bool), profile (text). When
    ``use_cache`` is set, a finished biographical profile is read from / written to the
    on-disk cache, so a pre-warmed player is served without any Granite call.
    """
    language = prompts.normalize_language(language)
    cache_dir = Path(cache_dir) if cache_dir is not None else PROFILE_CACHE_DIR
    if use_cache:
        cached = _read_cache(name, language, cache_dir)
        if cached:
            return cached
    language_name = prompts.language_display_name(language)
    summary = wiki_client.fetch_summary(name, language, fetcher=fetcher, use_cache=use_cache)

    result = {
        "name": name,
        "language": language,
        "photo_url": (summary or {}).get("photo_url", ""),
        "source_url": (summary or {}).get("url", ""),
        "grounded": bool(summary),
        "profile": "",
    }

    text = _generate(_build_prompt(name, language_name, summary, position, involvement),
                     granite_client)
    if text:
        result["profile"] = text
    elif summary:                       # Granite unavailable -> fall back to the factual summary
        result["profile"] = summary["extract"]
    else:                               # nothing grounded -> honest, minimal note
        bits = [b for b in ((f"Position: {position}" if position else ""), involvement) if b]
        result["profile"] = "; ".join(bits) or f"No profile information available for {name}."

    # Cache the biographical profile, but only when it has real content — never cache the
    # empty "no info" note, so an auto-prewarm before Granite/Wikipedia is up can't poison it.
    if use_cache and not involvement and (text or summary):
        _write_cache(name, language, result, cache_dir)
    return result


def prewarm_profiles(players: Iterable[str], language: str = "en", *, granite_client=None,
                     fetcher=None, cache_dir=None) -> int:
    """Generate and cache a profile for each player, skipping ones already cached.

    Returns the number freshly generated. Run this before commentary so that clicking a
    player during the live demo is a cache read — zero Granite calls, zero contention.
    """
    language = prompts.normalize_language(language)
    cache_dir = Path(cache_dir) if cache_dir is not None else PROFILE_CACHE_DIR
    written, seen = 0, set()
    for raw in players:
        name = (raw or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        if _read_cache(name, language, cache_dir):
            continue
        build_profile(name, language, granite_client=granite_client, fetcher=fetcher,
                      use_cache=True, cache_dir=cache_dir)
        written += 1
    return written


def main(argv: Optional[list] = None) -> int:
    """CLI: build and print one player's profile (needs network + Granite)."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Grounded player profile (Wikipedia + Granite).")
    parser.add_argument("--player", required=True)
    parser.add_argument("--language", default="en", choices=prompts.SUPPORTED_LANGUAGE_CODES)
    parser.add_argument("--position", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    profile = build_profile(args.player, args.language, position=args.position)
    if args.json:
        print(json.dumps(profile, ensure_ascii=False, indent=2))
    else:
        print(f"{profile['name']} ({profile['language']}) — grounded={profile['grounded']}")
        print(profile["profile"])
        print(f"photo:  {profile['photo_url'] or '(none)'}")
        print(f"source: {profile['source_url'] or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
