"""Go/no-go commentary spike: events -> one Granite call -> commentary (see spike/__init__.py)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

from agent.prompts import LANGUAGE_NAMES  # single source of truth for supported languages

# Event types that carry no narrative weight on their own — drop them so the
# ~18-event window the model sees is dense with meaningful play.
SKIP_TYPES = {"Ball Receipt*", "Pressure", "Carry", "Half Start", "Half End"}


# --------------------------------------------------------------------------- #
# Load events                                                                 #
# --------------------------------------------------------------------------- #
def load_events(match_id: Optional[int]) -> list[dict]:
    """Fetch events from the StatsBomb API (cached); None = the default demo match."""
    from data_extraction.loader import fetch_events
    return fetch_events(match_id)


def select_window(events: list[dict], start: int, count: int, dense: bool) -> list[dict]:
    """Take a contiguous slice, optionally dropping low-signal event types."""
    window = events[start:start + count]
    if dense:
        window = [e for e in window if e.get("type", {}).get("name") not in SKIP_TYPES]
    return window


# --------------------------------------------------------------------------- #
# Format events for the model                                                 #
# --------------------------------------------------------------------------- #
def describe_event(ev: dict) -> str:
    """One compact, faithful line per event — only facts present in the data."""
    minute = ev.get("minute", 0)
    second = ev.get("second", 0)
    etype = ev.get("type", {}).get("name", "?")
    team = ev.get("team", {}).get("name", "?")
    player = (ev.get("player") or {}).get("name", "")
    clock = f"{minute:02d}:{second:02d}"
    head = f"[{clock}] {etype} — {team}"
    if player:
        head += f" / {player}"

    detail = ""
    if etype == "Pass" and "pass" in ev:
        p = ev["pass"]
        recipient = (p.get("recipient") or {}).get("name", "")
        outcome = (p.get("outcome") or {}).get("name", "Complete")
        tech = (p.get("technique") or {}).get("name", "")
        bits = [b for b in (f"to {recipient}" if recipient else "", outcome, tech) if b]
        detail = " (" + ", ".join(bits) + ")" if bits else ""
    elif etype == "Shot" and "shot" in ev:
        s = ev["shot"]
        outcome = (s.get("outcome") or {}).get("name", "?")
        body = (s.get("body_part") or {}).get("name", "")
        xg = s.get("statsbomb_xg")
        bits = [outcome]
        if body:
            bits.append(body)
        if xg is not None:
            bits.append(f"xG {xg:.2f}")
        detail = " (" + ", ".join(bits) + ")"
    elif etype == "Dribble" and "dribble" in ev:
        detail = f" ({(ev['dribble'].get('outcome') or {}).get('name', '?')})"
    elif etype == "Goal Keeper" and "goalkeeper" in ev:
        detail = f" ({(ev['goalkeeper'].get('type') or {}).get('name', '?')})"

    return head + detail


def build_prompt(events: list[dict], language: str) -> str:
    lang_name = LANGUAGE_NAMES.get(language, language)
    feed = "\n".join(describe_event(e) for e in events)
    return f"""You are a live football commentator. Below is a short, ordered window of
real match events (newest last), already filtered to the meaningful ones.

Produce natural running commentary in {lang_name}. Rules:
- Comment ONLY on what the events state. Do NOT invent goals, names, scores,
  cards, or details that are not in the data.
- Don't narrate every line — group the build-up and land the big moments.
- Match the energy to the play: calm in midfield, loud for the shot and goal.
- Write the commentary in {lang_name} only.

EVENTS:
{feed}

Commentary ({lang_name}):"""


# --------------------------------------------------------------------------- #
# Granite call (reuses the shared agent.granite_client)                        #
# --------------------------------------------------------------------------- #
def call_granite(prompt: str) -> str:
    """Send the assembled prompt to Granite (OpenAI-compatible) and return the text."""
    from agent.granite_client import build_granite_client, model_id
    client = build_granite_client()
    resp = client.chat.completions.create(
        model=model_id(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
    )
    return (resp.choices[0].message.content or "").strip()


GO_NO_GO_CHECKLIST = """
------------------------------------------------------------------
GO / NO-GO — judge the output above:
  [ ] Faithful?    No invented goals, names, cards, or scores.
  [ ] Right language?  Entirely in the requested language.
  [ ] Paced?       Build-up grouped, big moments emphasised.
  [ ] Listenable?  Reads like a commentator, not a data dump.
If yes -> GO: proceed to replayer + agent.
If no  -> NO-GO: iterate on the prompt before building the pipeline.
------------------------------------------------------------------"""


def main(argv: Optional[list[str]] = None) -> int:
    """Run the go/no-go CLI from event selection through commentary output."""
    parser = argparse.ArgumentParser(description="go/no-go commentary spike.")
    parser.add_argument("--match-id", type=int, default=None,
                        help="Use a cached match instead of the bundled sample.")
    parser.add_argument("--language", default=os.getenv("DEFAULT_LANGUAGE", "en"),
                        choices=sorted(LANGUAGE_NAMES))
    parser.add_argument("--start", type=int, default=0, help="Window start index.")
    parser.add_argument("--count", type=int, default=18, help="Events to include (~15-20).")
    parser.add_argument("--all-types", action="store_true",
                        help="Keep low-signal events (passes' receipts, pressure, carries).")
    parser.add_argument("--mock", action="store_true",
                        help="Print the assembled prompt without calling Granite (offline).")
    args = parser.parse_args(argv)

    # Try to load local API settings only when a real Granite call is requested.
    if not args.mock:
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO / ".env")
        except ImportError:
            pass

    events = select_window(
        load_events(args.match_id), args.start, args.count, dense=not args.all_types
    )
    if not events:
        raise SystemExit("No events in the selected window — adjust --start/--count.")

    prompt = build_prompt(events, args.language)

    print(f"# {len(events)} events -> {LANGUAGE_NAMES[args.language]}  "
          f"(match={'sample' if args.match_id is None else args.match_id})\n")

    if args.mock:
        print("----- PROMPT (mock, no API call) -----")
        print(prompt)
        return 0

    print("----- GRANITE COMMENTARY -----")
    print(call_granite(prompt))
    print(GO_NO_GO_CHECKLIST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
