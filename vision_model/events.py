"""Infer discrete match events from a stream of per-frame detections.

The semantic layer: watch the ball and players over time and emit StatsBomb-shaped
events (via ``vision_model.schema``) when something narratable happens. The rules are
deliberately simple MVP heuristics — possession is 'nearest player to the ball', a
hand-over between team-mates is a Pass, and a ball that reaches the goal mouth is a
Shot/Goal. They are approximate by design; see README for the path to real inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from vision_model.detector import Detection, FrameDetections
from vision_model.schema import PITCH_LENGTH, make_event, pass_, shot

POSSESSION_RADIUS = 3.0          # pitch units: how close a player must be to 'have' the ball
GOAL_MOUTH = (36.0, 44.0)        # width band (y) of the goal, centred on 40
GOAL_LINE = PITCH_LENGTH - 0.5   # x at/after which the ball counts as over the line


def _dist(a, b) -> float:
    """Euclidean distance between two (x, y) pitch points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def nearest_player(fd: FrameDetections) -> Optional[Detection]:
    """The player in possession this frame (closest to the ball, within reach), or None."""
    ball = fd.ball()
    if ball is None or ball.xy_pitch is None:
        return None
    candidates = [p for p in fd.players() if p.xy_pitch is not None]
    if not candidates:
        return None
    closest = min(candidates, key=lambda d: _dist(d.xy_pitch, ball.xy_pitch))
    return closest if _dist(closest.xy_pitch, ball.xy_pitch) <= POSSESSION_RADIUS else None


@dataclass
class _Possessor:
    """Who held the ball, for comparing frame-to-frame."""

    track_id: Optional[int]
    team: Optional[str]
    name: str
    xy: tuple


class EventBuilder:
    """Turn a sequence of frames into events, tracking possession across frames."""

    def __init__(self, player_names: Optional[dict] = None):
        """``player_names`` optionally maps track_id -> display name (e.g. from a roster
        or jersey-number OCR); otherwise players are named 'Player {track_id}'."""
        self._names = player_names or {}
        self._prev: Optional[_Possessor] = None
        self._goal_done = False
        self._index = 0
        self._events: List[dict] = []

    def _name(self, d: Detection) -> str:
        """Resolve a display name for a detected player."""
        if d.track_id is None:
            return "Player"
        return self._names.get(d.track_id, f"Player {d.track_id}")

    def _emit(self, etype: str, **kwargs) -> None:
        """Append one event with the next running index."""
        self._index += 1
        self._events.append(make_event(self._index, etype, **kwargs))

    def observe(self, fd: FrameDetections) -> None:
        """Update possession from one frame, emitting a Pass or a Shot/Goal if warranted."""
        ball = fd.ball()

        # 1) Goal: ball reaches the goal mouth past the line -> Shot/Goal, attributed
        #    to the last known possessor. Emitted once.
        if (ball is not None and ball.xy_pitch is not None
                and not self._goal_done and self._prev is not None):
            bx, by = ball.xy_pitch
            if bx >= GOAL_LINE and GOAL_MOUTH[0] <= by <= GOAL_MOUTH[1]:
                self._emit(
                    "Shot", team=self._prev.team, player=self._prev.name,
                    location=self._prev.xy, elapsed_s=fd.elapsed_s,
                    **{"shot": shot("Goal", end_location=(bx, by))},
                )
                self._goal_done = True
                return

        # 2) Possession: nearest player to the ball.
        current_det = nearest_player(fd)
        if current_det is None:
            return
        current = _Possessor(
            current_det.track_id, current_det.team,
            self._name(current_det), current_det.xy_pitch,
        )
        prev = self._prev
        if prev is not None and current.track_id != prev.track_id:
            # Same-team hand-over -> a completed pass from prev to current.
            if prev.team is not None and current.team == prev.team:
                self._emit(
                    "Pass", team=prev.team, player=prev.name,
                    location=prev.xy, elapsed_s=fd.elapsed_s,
                    **{"pass": pass_(recipient=current.name, end_location=current.xy)},
                )
            # A change of team is a turnover; left for a future event type.
        self._prev = current

    @property
    def events(self) -> List[dict]:
        """The events inferred so far, in emission order."""
        return self._events
