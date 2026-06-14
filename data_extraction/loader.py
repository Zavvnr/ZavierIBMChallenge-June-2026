"""Download + cache a StatsBomb Open Data match (see data_extraction/__init__.py)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

import argparse
import json
import requests
import sys


OPEN_DATA_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


# Cached matches live here (repo-root data/cache). Git-ignored (see .gitignore) and
# shared with data_replayer / data_pipeline, which read from the same location.
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


# Default demo match: FIFA World Cup 2022 Final — Argentina 3-3 France
# Verified against StatsBomb open-data matches/43/106.json:
#   match_id 3869685 | competition_id 43 (FIFA World Cup) | season_id 106 (2022)
#   home Argentina (779) vs away France (771).
DEFAULT_DEMO_MATCH_ID = 3869685
WORLD_CUP_2022 = {"competition_id": 43, "season_id": 106}


# Canonical demo match (verified). Discover others with `--list-matches`.
DEMO_MATCHES = {
    3869685: "FIFA World Cup 2022 Final — Argentina 3-3 France (pens), 2022-12-18",
    # Can add more here as desired, but note that the open-data set is heavily skewed toward major 
    # leagues and international tournaments, so there are no EFL Championship matches and no Ligue 2 matches, for example.
}


STATSBOMB_ATTRIBUTION = (
    "Match data provided by StatsBomb Open Data "
    "(https://github.com/statsbomb/open-data). "
    "Per the StatsBomb public data user agreement, StatsBomb must be credited "
    "as the data source for any output derived from this data."
)


# --------------------------------------------------------------------------- #
# Low-level fetch                                                             #
# --------------------------------------------------------------------------- #
def _get_json(url: str) -> Any:
    """Fetch one StatsBomb Open Data JSON document and return the decoded body."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_competitions() -> list[dict]:
    """Every competition/season available in the open-data set."""
    return _get_json(f"{OPEN_DATA_BASE}/competitions.json")


def list_matches(competition_id: int, season_id: int) -> list[dict]:
    """All matches for one competition+season."""
    return _get_json(f"{OPEN_DATA_BASE}/matches/{competition_id}/{season_id}.json")


def download_events(match_id: int) -> list[dict]:
    """Full nested event stream for a match (the important one)."""
    return _get_json(f"{OPEN_DATA_BASE}/events/{match_id}.json")


def download_lineups(match_id: int) -> list[dict]:
    """Both teams' lineups (names, numbers, positions)."""
    return _get_json(f"{OPEN_DATA_BASE}/lineups/{match_id}.json")


def find_match_meta(
    match_id: int,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
) -> Optional[dict]:
    """
    Best-effort match metadata (teams, score, date, competition).

    The open-data set has no events->match index, so we look in a matches file.
    Pass competition_id/season_id if you know them; otherwise we try the
    World Cup 2022 default and give up quietly if it isn't there.
    """
    candidates = []
    if competition_id is not None and season_id is not None:
        candidates.append((competition_id, season_id))
    candidates.append((WORLD_CUP_2022["competition_id"], WORLD_CUP_2022["season_id"]))

    for comp, season in candidates:
        try:
            for match in list_matches(comp, season):
                if match.get("match_id") == match_id:
                    return match
        except requests.RequestException:
            continue
    return None


# --------------------------------------------------------------------------- #
# Cache                                                                       #
# --------------------------------------------------------------------------- #
def match_dir(match_id: int) -> Path:
    """Return the cache directory path for one StatsBomb match id."""
    return CACHE_DIR / str(match_id)


def _write_json(path: Path, payload: Any) -> None:
    """Write a payload as pretty UTF-8 JSON for repeatable local cache files."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_match(
    match_id: int,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,) -> Path:
    """
    Download events + lineups (+ best-effort meta) and cache them under
    data/cache/<match_id>/. Returns the directory.
    """
    dest = match_dir(match_id)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading events for match {match_id} ...")
    events = download_events(match_id)
    _write_json(dest / "events.json", events)

    print("Downloading lineups ...")
    lineups = download_lineups(match_id)
    _write_json(dest / "lineups.json", lineups)

    meta = find_match_meta(match_id, competition_id, season_id)
    if meta:
        _write_json(dest / "meta.json", meta)

    # Keep the attribution requirement physically next to the data.
    (dest / "ATTRIBUTION.txt").write_text(STATSBOMB_ATTRIBUTION + "\n", encoding="utf-8")

    print(f"Cached {len(events)} events -> {dest}")
    print(STATSBOMB_ATTRIBUTION)
    return dest


def load_cached_events(match_id: int) -> list[dict]:
    """Read previously cached events. Raises if the match hasn't been downloaded."""
    path = match_dir(match_id) / "events.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached events for match {match_id}. "
            f"Run: python -m data_extraction.loader --match-id {match_id}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_cached_meta(match_id: int) -> Optional[dict]:
    """Read cached match metadata when available, otherwise return None."""
    path = match_dir(match_id) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fetch_events(match_id: Optional[int] = None, use_cache: bool = True) -> list[dict]:
    """
    Return a match's events, fetched DIRECTLY from the StatsBomb Open Data API.

    Nothing is bundled in the repo: the first fetch downloads from the API and (by
    default) caches it under data/cache/<id>/events.json so later runs are fast and
    offline. `match_id` None means the default demo match (World Cup 2022 Final),
    so callers can offer a zero-argument "demo" without hardcoding any data.
    """
    mid = int(match_id) if match_id else DEFAULT_DEMO_MATCH_ID
    if use_cache:
        try:
            return load_cached_events(mid)
        except FileNotFoundError:
            pass
    events = download_events(mid)  # direct StatsBomb API call
    if use_cache:
        dest = match_dir(mid)
        dest.mkdir(parents=True, exist_ok=True)
        _write_json(dest / "events.json", events)
    return events


# --------------------------------------------------------------------------- #
# Optional statsbombpy backend (discovery / sanity-check)                     #
# --------------------------------------------------------------------------- #
def statsbombpy_events(match_id: int):
    """
    Pull events via statsbombpy (returns a flattened pandas DataFrame).
    Useful for quick inspection; the agent uses the nested JSON above instead.
    """
    try:
        from statsbombpy import sb  # noqa: WPS433 (optional import)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("statsbombpy not installed — `pip install statsbombpy`") from exc
    return sb.events(match_id=match_id)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _print_competitions() -> None:
    """Print every available StatsBomb competition/season for CLI discovery."""
    for c in list_competitions():
        print(
            f"comp {c['competition_id']:>4} / season {c['season_id']:>4}  "
            f"{c.get('competition_name','?')} — {c.get('season_name','?')}"
        )


def _print_matches(competition_id: int, season_id: int) -> None:
    """Print every match in one competition/season for CLI discovery."""
    for m in list_matches(competition_id, season_id):
        home = m.get("home_team", {}).get("home_team_name", "?")
        away = m.get("away_team", {}).get("away_team_name", "?")
        print(
            f"match {m['match_id']:>8}  {m.get('match_date','?')}  "
            f"{home} {m.get('home_score','?')}-{m.get('away_score','?')} {away}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    """Run the loader CLI for discovery or match caching."""
    parser = argparse.ArgumentParser(description="Download/cache a StatsBomb open-data match.")
    parser.add_argument("--match-id", type=int, default=None, help="Match to download and cache.")
    parser.add_argument("--competition-id", type=int, default=None)
    parser.add_argument("--season-id", type=int, default=None)
    parser.add_argument("--list-competitions", action="store_true")
    parser.add_argument("--list-matches", action="store_true",
                        help="List matches for --competition-id/--season-id.")
    args = parser.parse_args(argv)

    try:
        if args.list_competitions:
            _print_competitions()
            return 0

        if args.list_matches:
            if args.competition_id is None or args.season_id is None:
                parser.error("--list-matches requires --competition-id and --season-id")
            _print_matches(args.competition_id, args.season_id)
            return 0

        match_id = args.match_id or DEFAULT_DEMO_MATCH_ID
        if not args.match_id:
            print(f"No --match-id given; using default demo match {match_id} "
                  f"({DEMO_MATCHES.get(match_id, 'unknown')}).")
        cache_match(match_id, args.competition_id, args.season_id)
        return 0
    except requests.RequestException as exc:
        print(f"Network error talking to StatsBomb open-data: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
