"""Curated + Wikipedia-augmented player dossiers for the analyst's color commentary.

When the analyst occasionally spotlights a player, it speaks from real facts — role, club,
what they're famous for, and a stylistic tendency — instead of inventing. Facts come from a
small hand-curated set (fast, offline, for marquee players) plus an on-disk cache that can
be pre-warmed from Wikipedia (``profiles.wiki_client``) for everyone else.

``facts_for`` never touches the network — it reads curated entries + the cache only — so it
is safe inside the live commentary loop. ``prewarm`` does the (cached) Wikipedia fetches
ahead of time. The curated dossiers are reference data, like a broadcaster's prep sheet:
edit them freely.

    python -m agent.player_facts --match-id 3869685 --language en   # pre-warm both squads
    python -m agent.player_facts --players "Lionel Messi,Kylian Mbappé"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "player_facts"

# Hand-curated dossiers (stable facts; clubs as widely known, with history). Keyed by
# lowercased full name and matched on last name too. Edit/extend freely.
CURATED = {
    "lionel messi": {
        "role": "forward / playmaker", "club": "Inter Miami (ex-Barcelona, PSG)",
        "known_for": "dribbling, vision and free-kicks; in the greatest-of-all-time conversation",
        "tendency": "drops deep and drifts in off the right to combine, then threads the final ball"},
    "ángel di maría": {
        "role": "winger", "club": "Benfica (ex-Real Madrid, PSG, Juventus)",
        "known_for": "turning up in finals", "tendency": "cuts inside from the left to shoot or cross"},
    "julián álvarez": {
        "role": "forward", "club": "Atlético Madrid (ex-Manchester City, River Plate)",
        "known_for": "relentless pressing and movement", "tendency": "runs the channels and presses from the front"},
    "rodrigo de paul": {
        "role": "central midfielder", "club": "Atlético Madrid",
        "known_for": "energy and ball-carrying", "tendency": "links defence to attack, covers huge ground"},
    "enzo fernández": {
        "role": "central midfielder", "club": "Chelsea (ex-Benfica, River Plate)",
        "known_for": "breakout star of the 2022 World Cup", "tendency": "deep playmaker with a long passing range"},
    "alexis mac allister": {
        "role": "central midfielder", "club": "Liverpool (ex-Brighton)",
        "known_for": "composure beyond his years", "tendency": "progresses the ball and arrives late in the box"},
    "emiliano martínez": {
        "role": "goalkeeper", "club": "Aston Villa",
        "known_for": "penalty-shootout heroics and mind games", "tendency": "commanding, aggressive off his line"},
    "nicolás otamendi": {
        "role": "centre-back", "club": "Benfica",
        "known_for": "experience and aggression", "tendency": "steps out to meet runners early"},
    "kylian mbappé": {
        "role": "forward", "club": "Real Madrid (ex-PSG)",
        "known_for": "explosive pace and finishing; 2022 final hat-trick", "tendency": "attacks the left channel in behind"},
    "antoine griezmann": {
        "role": "forward / playmaker", "club": "Atlético Madrid",
        "known_for": "selfless link play and big-game goals", "tendency": "drops into midfield to create"},
    "olivier giroud": {
        "role": "striker", "club": "ex-AC Milan; France's record scorer",
        "known_for": "hold-up play and aerial threat", "tendency": "plays back-to-goal and brings others in"},
    "ousmane dembélé": {
        "role": "winger", "club": "PSG (ex-Barcelona)",
        "known_for": "two-footed dribbling", "tendency": "takes on the full-back and whips it in"},
    "aurélien tchouaméni": {
        "role": "defensive midfielder", "club": "Real Madrid",
        "known_for": "shielding the back line", "tendency": "breaks up play and recycles possession"},
    "theo hernández": {
        "role": "left-back", "club": "AC Milan",
        "known_for": "marauding runs", "tendency": "overlaps high and drives into the final third"},
    "cristian romero": {
        "role": "centre-back", "club": "Tottenham",
        "known_for": "front-foot, aggressive defending", "tendency": "steps out early to intercept"},
    "lisandro martínez": {
        "role": "centre-back", "club": "Manchester United",
        "known_for": "left-footed ball-playing and bite", "tendency": "steps in and starts attacks"},
    "lautaro martínez": {
        "role": "striker", "club": "Inter Milan",
        "known_for": "sharp movement and finishing", "tendency": "plays off the last shoulder"},
    "hugo lloris": {
        "role": "goalkeeper", "club": "LAFC (ex-Tottenham)",
        "known_for": "France's most-capped player and captain", "tendency": "sweeps behind a high line"},
    "raphaël varane": {
        "role": "centre-back", "club": "ex-Real Madrid & Manchester United",
        "known_for": "calm under pressure and reading the game", "tendency": "defends on the front foot"},
    "jules koundé": {
        "role": "full-back / centre-back", "club": "Barcelona",
        "known_for": "recovery pace and versatility", "tendency": "tucks in or overlaps from the right"},
    "dayot upamecano": {
        "role": "centre-back", "club": "Bayern Munich",
        "known_for": "pace and power on the cover", "tendency": "steps up to win it early"},
    "adrien rabiot": {
        "role": "midfielder", "club": "Marseille (ex-Juventus)",
        "known_for": "long-striding left-footer", "tendency": "arrives late in the box"},
    "kingsley coman": {
        "role": "winger", "club": "Bayern Munich",
        "known_for": "one-v-one pace", "tendency": "takes on the full-back down the flank"},
    "cristiano ronaldo": {
        "role": "forward", "club": "Al-Nassr (ex-Manchester United, Real Madrid, Juventus)",
        "known_for": "prolific scoring and aerial threat", "tendency": "attacks the back post"},
    "neymar": {
        "role": "forward", "club": "Santos (ex-Barcelona, PSG)",
        "known_for": "dribbling and flair", "tendency": "drifts left to take defenders on"},
    "kevin de bruyne": {
        "role": "midfielder", "club": "Manchester City",
        "known_for": "elite passing range and crossing", "tendency": "picks the killer pass from the right half-space"},
    "erling haaland": {
        "role": "striker", "club": "Manchester City",
        "known_for": "relentless box movement and finishing", "tendency": "runs in behind the last line"},
    "harry kane": {
        "role": "striker", "club": "Bayern Munich (ex-Tottenham)",
        "known_for": "clinical finishing and deep link play", "tendency": "drops to create, then attacks the box"},
    "vinícius júnior": {
        "role": "winger", "club": "Real Madrid",
        "known_for": "direct dribbling", "tendency": "isolates the full-back on the left"},
    "jude bellingham": {
        "role": "midfielder", "club": "Real Madrid (ex-Borussia Dortmund)",
        "known_for": "all-action box-to-box play", "tendency": "makes late runs into the area"},
    "rodri": {
        "role": "defensive midfielder", "club": "Manchester City",
        "known_for": "2024 Ballon d'Or; tempo control", "tendency": "screens the defence and recycles play"},
    "robert lewandowski": {
        "role": "striker", "club": "Barcelona (ex-Bayern Munich)",
        "known_for": "penalty-box poaching", "tendency": "finds space between centre-backs"},
    "mohamed salah": {
        "role": "winger", "club": "Liverpool",
        "known_for": "goals from the right", "tendency": "cuts inside to shoot far post"},
    "luka modrić": {
        "role": "midfielder", "club": "Real Madrid",
        "known_for": "press resistance and long passing", "tendency": "drops to dictate, sprays it wide"},
    "son heung-min": {
        "role": "forward", "club": "Tottenham",
        "known_for": "two-footed finishing on the break", "tendency": "attacks space in transition"},
    "pedri": {
        "role": "midfielder", "club": "Barcelona",
        "known_for": "press resistance and line-breaking passes", "tendency": "receives between the lines"},
}


def _norm(name: str) -> str:
    """Lowercased, single-spaced name for lookup."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _last(name: str) -> str:
    """Last-name token of a name (for looser matching)."""
    parts = _norm(name).split(" ")
    return parts[-1] if parts and parts[0] else ""


def _slug(name: str) -> str:
    """Filesystem-safe cache key for a player name."""
    return re.sub(r"[^a-z0-9]+", "_", _norm(name)).strip("_") or "_"


def _lang(language: Optional[str]) -> str:
    """Bare language code (e.g. 'es-ES' -> 'es')."""
    return (language or "en").split("-")[0].lower()


def facts_for(player: str, language: str = "en") -> Optional[dict]:
    """Return a dossier dict for a player, or None — curated + cache only, no network."""
    if not player:
        return None
    name = _norm(player)
    dossier = CURATED.get(name)
    if dossier is None:
        last = _last(player)
        for key, value in CURATED.items():
            if key.split(" ")[-1] == last:
                dossier = value
                break
    if dossier is None:
        dossier = _read_cache(player, language)
    return dict(dossier) if dossier else None


def note_text(facts: Optional[dict]) -> str:
    """A compact one-line dossier string (for CLI / logging)."""
    if not facts:
        return ""
    order = ("role", "club", "known_for", "tendency", "note")
    bits = [f"{key.replace('_', ' ')}: {facts[key]}" for key in order if facts.get(key)]
    return "; ".join(bits)


def _read_cache(player: str, language: str) -> Optional[dict]:
    """Read a previously pre-warmed Wikipedia dossier from disk, or None."""
    path = CACHE_DIR / _lang(language) / f"{_slug(player)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_cache(player: str, language: str, facts: dict) -> None:
    """Persist a dossier so future runs (and the live loop) read it offline."""
    path = CACHE_DIR / _lang(language) / f"{_slug(player)}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _condense(summary: dict) -> dict:
    """Turn a Wikipedia summary into a compact dossier dict."""
    extract = (summary.get("extract") or "").strip()
    first_sentence = re.split(r"(?<=[.!?])\s", extract)[0] if extract else ""
    return {
        "known_for": summary.get("description") or "",
        "note": first_sentence[:280],
        "source": summary.get("url", ""),
    }


def prewarm(players: Iterable[str], language: str = "en", fetcher=None) -> int:
    """Cache dossiers for players not already curated/cached, from Wikipedia. Returns count.

    Network happens here (via ``profiles.wiki_client``, which itself caches), never in the
    live loop. ``fetcher`` is injectable for offline tests.
    """
    from profiles.wiki_client import fetch_summary

    written = 0
    for player in players:
        if not player or facts_for(player, language):
            continue
        summary = fetch_summary(player, language, fetcher=fetcher, use_cache=False)
        if not summary:
            continue
        _write_cache(player, language, _condense(summary))
        written += 1
    return written


def _match_players(match_id: Optional[int]) -> list:
    """Both squads' player names for a match (best-effort, via the line-up engine)."""
    try:
        from data_extraction.lineups import fetch_lineups
        names = []
        for team in fetch_lineups(match_id):
            names += [p.name for p in team.starting_xi] + [p.name for p in team.substitutes]
        return names
    except Exception:
        return []


def main(argv: Optional[list] = None) -> int:
    """CLI: pre-warm player dossiers for a match's squads or an explicit list."""
    parser = argparse.ArgumentParser(description="Pre-warm player dossiers (curated + Wikipedia).")
    parser.add_argument("--match-id", type=int, default=None, help="Pre-warm both squads of a match.")
    parser.add_argument("--players", default="", help="Comma-separated player names.")
    parser.add_argument("--language", default="en")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    players = [p.strip() for p in args.players.split(",") if p.strip()] or _match_players(args.match_id)
    if not players:
        print("No players (give --players or --match-id).")
        return 1
    written = prewarm(players, args.language)
    print(f"Pre-warmed {written} new dossier(s); {len(players)} players checked.")
    for name in players[:22]:
        facts = facts_for(name, args.language)
        print(f"  {name}: {note_text(facts) or '(no facts)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
