"""The vision -> commentary integration contract.

The commentary agent and replayer consume StatsBomb-shaped event dicts. The vision
pipeline must emit the SAME shape so its events drop straight into
``data_replayer.replay()`` and ``agent.commentary_agent.CommentaryAgent`` with no
adapter. This module is the single place that knows that shape, so the contract lives
in one file. Coordinates stay on StatsBomb's 120x80 pitch so the agent's importance /
in-the-box / progressive-pass heuristics keep working unchanged.
"""
from __future__ import annotations

from typing import Optional, Sequence

# StatsBomb pitch dimensions: length (x, 0..120) by width (y, 0..80).
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


def seconds_to_clock(elapsed_s: float, period: int = 1) -> dict:
    """Map elapsed seconds within a period to StatsBomb clock fields.

    ``minute`` is continuous across halves (the agent relies on that for pacing math),
    ``timestamp`` is 'HH:MM:SS.mmm' within the period (the replayer parses it for waits).
    """
    elapsed_s = max(0.0, float(elapsed_s))
    base_minute = 0 if period <= 1 else 45  # 2nd-half clock continues from 45'
    minute = base_minute + int(elapsed_s // 60)
    second = int(elapsed_s % 60)
    millis = int(round((elapsed_s - int(elapsed_s)) * 1000))
    hh = int(elapsed_s // 3600)
    mm = int((elapsed_s % 3600) // 60)
    ss = int(elapsed_s % 60)
    return {
        "period": period,
        "minute": minute,
        "second": second,
        "timestamp": f"{hh:02d}:{mm:02d}:{ss:02d}.{millis:03d}",
    }


def _named(name: Optional[str]) -> Optional[dict]:
    """StatsBomb wraps enumerations as {'name': ...}; return that or None."""
    return {"name": name} if name else None


def _xy(loc: Optional[Sequence[float]]) -> Optional[list]:
    """Round a pitch coordinate pair to a 2-element list, or None."""
    if loc is None:
        return None
    return [round(float(loc[0]), 2), round(float(loc[1]), 2)]


def shot(outcome: str, end_location=None, xg: Optional[float] = None,
         body_part: Optional[str] = None) -> dict:
    """Build a StatsBomb ``shot`` sub-dict (outcome 'Goal'/'Saved'/'Off T'/...)."""
    s: dict = {"outcome": _named(outcome)}
    if end_location is not None:
        s["end_location"] = _xy(end_location)
    if xg is not None:
        s["statsbomb_xg"] = round(float(xg), 3)
    if body_part:
        s["body_part"] = _named(body_part)
    return s


def pass_(recipient: Optional[str] = None, end_location=None,
          outcome: str = "Complete") -> dict:
    """Build a StatsBomb ``pass`` sub-dict. A completed pass carries no ``outcome`` key
    (matching StatsBomb), so only incomplete outcomes are recorded."""
    p: dict = {}
    if recipient:
        p["recipient"] = _named(recipient)
    if end_location is not None:
        p["end_location"] = _xy(end_location)
    if outcome and outcome != "Complete":
        p["outcome"] = _named(outcome)
    return p


def make_event(index: int, etype: str, *, team: Optional[str] = None,
               player: Optional[str] = None, location=None,
               elapsed_s: float = 0.0, period: int = 1, **extra) -> dict:
    """Assemble one StatsBomb-shaped event dict.

    ``etype`` is the type name ('Pass', 'Shot', ...). ``extra`` carries an already-shaped
    type sub-dict, e.g. ``make_event(1, 'Shot', **{'shot': shot('Goal')})``.
    """
    ev: dict = {"index": int(index), "type": {"name": etype}}
    ev.update(seconds_to_clock(elapsed_s, period))
    if team:
        ev["team"] = {"name": team}
    if player:
        ev["player"] = {"name": player}
    loc = _xy(location)
    if loc is not None:
        ev["location"] = loc
    ev.update({k: v for k, v in extra.items() if v is not None})
    return ev
