"""Tactical movement-pattern detection over a window of FrameDetections.

Patterns detected (all heuristic — no torch required):
  man_marking    — a defender shadows a specific attacker closely over time.
  drag_defenders — an attacker draws multiple defenders out of position,
                   creating space behind the defensive line.
  crowd_box      — three or more attackers cluster in the penalty area.
  overlap_run    — a wide player makes a deep advancing run behind the line.
  high_press     — the defending team's shape is pushed unusually high,
                   leaving space for a counter-attack.

Call ``analyse(frames, attacking_team, defending_team)`` to get a
``TacticalReport`` for the window. This is the offline layer: no torch, no GPU.

Intended use — goal explanation:
    window = frames[-75:]   # ~3 s of footage before the goal (at 25 fps)
    report = analyse(window, "Blue", "Red")
    # report.key_movement -> "drag_defenders"
    # report.observations[0].description -> human-readable tactical summary
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vision_model.detector import Detection, FrameDetections

# Penalty area boundaries on the StatsBomb 120×80 pitch (attacking/high-x end).
PENALTY_AREA_X = 102.0
PENALTY_AREA_Y_LOW = 18.0
PENALTY_AREA_Y_HIGH = 62.0

# Thresholds for each pattern.
MAN_MARK_RADIUS = 6.0    # pitch units: max defender–attacker gap for marking
MIN_MARK_FRAMES = 4      # frames the pair must stay close to confirm marking
DRAG_MIN_MOVE = 8.0      # min attacker displacement (pitch units) to count as drag
DRAG_MIN_FOLLOWERS = 2   # min defenders that follow the attacker
CROWD_BOX_MIN = 3        # min attackers in the box to call it "crowding"
OVERLAP_ADVANCE = 12.0   # min x advance (pitch units) for an overlap run
HIGH_PRESS_X = 60.0      # defending team's mean x above this = high press


@dataclass
class TacticalObservation:
    """A single detected tactical pattern with a commentary-ready description."""

    pattern: str                              # e.g. "drag_defenders"
    description: str                          # suitable for the explainer / commentary
    confidence: float                         # 0.0–1.0
    players: List[str] = field(default_factory=list)   # player ids / names involved


@dataclass
class TacticalReport:
    """Summary of tactical patterns across a window of frames."""

    formation_before: Optional[str]           # formation string at window start
    formation_after: Optional[str]            # formation string at window end
    observations: List[TacticalObservation]
    key_movement: Optional[str]               # pattern name with highest confidence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _players_for_team(fd: FrameDetections, team: str) -> List[Detection]:
    """Player detections for one team that have pitch coordinates."""
    return [d for d in fd.players() if d.team == team and d.xy_pitch is not None]


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two (x, y) pitch points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------

def _detect_crowd_box(
    frames: List[FrameDetections], attacking_team: str
) -> Optional[TacticalObservation]:
    """Detect if the attacking team is flooding the penalty area."""
    peak = 0
    for fd in frames:
        in_box = [
            d for d in _players_for_team(fd, attacking_team)
            if (d.xy_pitch[0] >= PENALTY_AREA_X
                and PENALTY_AREA_Y_LOW <= d.xy_pitch[1] <= PENALTY_AREA_Y_HIGH)
        ]
        peak = max(peak, len(in_box))

    if peak < CROWD_BOX_MIN:
        return None

    confidence = min(1.0, peak / 5.0)  # 5+ attackers in box → full confidence
    return TacticalObservation(
        pattern="crowd_box",
        description=(
            f"{peak} attackers crowded the penalty area, stretching the defensive "
            "shape and forcing the goalkeeper to choose between multiple threats."
        ),
        confidence=confidence,
    )


def _detect_man_marking(
    frames: List[FrameDetections], attacking_team: str, defending_team: str
) -> Optional[TacticalObservation]:
    """Detect a defender shadowing a specific attacker closely across frames."""
    pair_counts: Dict[Tuple[Optional[int], Optional[int]], int] = {}

    for fd in frames:
        for att in _players_for_team(fd, attacking_team):
            for def_ in _players_for_team(fd, defending_team):
                if _dist(att.xy_pitch, def_.xy_pitch) <= MAN_MARK_RADIUS:
                    key = (att.track_id, def_.track_id)
                    pair_counts[key] = pair_counts.get(key, 0) + 1

    if not pair_counts:
        return None

    best_pair, count = max(pair_counts.items(), key=lambda kv: kv[1])
    if count < MIN_MARK_FRAMES:
        return None

    confidence = min(1.0, count / max(1, len(frames)))
    att_id, def_id = best_pair
    return TacticalObservation(
        pattern="man_marking",
        description=(
            f"Defender {def_id} was tight-marking attacker {att_id}, "
            "tracking their movement across the pitch and denying them space to turn."
        ),
        confidence=confidence,
        players=[f"att_{att_id}", f"def_{def_id}"],
    )


def _detect_drag_defenders(
    frames: List[FrameDetections], attacking_team: str, defending_team: str
) -> Optional[TacticalObservation]:
    """Detect an attacker pulling defenders out of position to open space.

    The run is confirmed when: the attacker displaces by ≥ DRAG_MIN_MOVE pitch
    units AND at least DRAG_MIN_FOLLOWERS defenders move in the same direction.
    """
    if len(frames) < 2:
        return None

    first_fd, last_fd = frames[0], frames[-1]

    att_first = {d.track_id: d for d in _players_for_team(first_fd, attacking_team)}
    att_last = {d.track_id: d for d in _players_for_team(last_fd, attacking_team)}
    def_first = {d.track_id: d for d in _players_for_team(first_fd, defending_team)}
    def_last = {d.track_id: d for d in _players_for_team(last_fd, defending_team)}

    best_drag: Optional[Tuple[Optional[int], int, float]] = None  # (att_id, followers, dist)

    for tid in set(att_first) & set(att_last):
        p0 = att_first[tid].xy_pitch
        p1 = att_last[tid].xy_pitch
        move = _dist(p0, p1)
        if move < DRAG_MIN_MOVE:
            continue

        # Movement vector of the attacker.
        dx_att = p1[0] - p0[0]
        dy_att = p1[1] - p0[1]

        # Count defenders whose movement vector has a positive dot product
        # with the attacker's vector (i.e. they followed the same direction).
        followers = 0
        for dtid in set(def_first) & set(def_last):
            dx_def = def_last[dtid].xy_pitch[0] - def_first[dtid].xy_pitch[0]
            dy_def = def_last[dtid].xy_pitch[1] - def_first[dtid].xy_pitch[1]
            if dx_att * dx_def + dy_att * dy_def > 0:
                followers += 1

        if followers >= DRAG_MIN_FOLLOWERS:
            if best_drag is None or move > best_drag[2]:
                best_drag = (tid, followers, move)

    if best_drag is None:
        return None

    tid, followers, move = best_drag
    confidence = min(1.0, followers / 4.0)
    return TacticalObservation(
        pattern="drag_defenders",
        description=(
            f"Attacker {tid} made a {move:.0f} m dragging run that pulled "
            f"{followers} defender(s) out of position, opening space behind the line."
        ),
        confidence=confidence,
        players=[f"att_{tid}"],
    )


def _detect_overlap_run(
    frames: List[FrameDetections], attacking_team: str
) -> Optional[TacticalObservation]:
    """Detect a wide player making a deep overlapping run into the final third."""
    if len(frames) < 2:
        return None

    first_fd, last_fd = frames[0], frames[-1]
    att_first = {d.track_id: d for d in _players_for_team(first_fd, attacking_team)}
    att_last = {d.track_id: d for d in _players_for_team(last_fd, attacking_team)}

    best: Optional[Tuple[Optional[int], float]] = None
    for tid in set(att_first) & set(att_last):
        x0 = att_first[tid].xy_pitch[0]
        x1 = att_last[tid].xy_pitch[0]
        advance = x1 - x0
        # Wide run: significant x advance AND the player started wide (y close to touchline).
        y0 = att_first[tid].xy_pitch[1]
        is_wide = y0 < 20.0 or y0 > 60.0
        if advance >= OVERLAP_ADVANCE and is_wide:
            if best is None or advance > best[1]:
                best = (tid, advance)

    if best is None:
        return None

    tid, advance = best
    confidence = min(1.0, advance / 25.0)
    return TacticalObservation(
        pattern="overlap_run",
        description=(
            f"Player {tid} made an overlapping run of {advance:.0f} m along the "
            "flank, arriving late into the penalty area to create an overload."
        ),
        confidence=confidence,
        players=[f"att_{tid}"],
    )


def _detect_high_press(
    frames: List[FrameDetections], defending_team: str
) -> Optional[TacticalObservation]:
    """Detect if the defending team has pushed its shape unusually high."""
    avg_x_values = []
    for fd in frames:
        defenders = _players_for_team(fd, defending_team)
        if defenders:
            avg_x_values.append(
                sum(d.xy_pitch[0] for d in defenders) / len(defenders)
            )

    if not avg_x_values:
        return None

    mean_x = sum(avg_x_values) / len(avg_x_values)
    if mean_x < HIGH_PRESS_X:
        return None

    confidence = min(1.0, (mean_x - HIGH_PRESS_X) / 20.0)
    return TacticalObservation(
        pattern="high_press",
        description=(
            f"The defending team was pressing high (avg. position x≈{mean_x:.0f}), "
            "leaving significant space in behind for a through-ball or counter-attack."
        ),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def analyse(
    frames: List[FrameDetections],
    attacking_team: str,
    defending_team: str,
    formation_before: Optional[str] = None,
    formation_after: Optional[str] = None,
) -> TacticalReport:
    """Analyse a window of frames and return a ``TacticalReport``.

    ``formation_before`` / ``formation_after`` are optional strings from the
    formation classifier and are passed through unchanged.  If no frames are
    provided an empty report is returned.
    """
    if not frames:
        return TacticalReport(
            formation_before=formation_before,
            formation_after=formation_after,
            observations=[],
            key_movement=None,
        )

    observations: List[TacticalObservation] = []

    for detector in (
        lambda: _detect_crowd_box(frames, attacking_team),
        lambda: _detect_man_marking(frames, attacking_team, defending_team),
        lambda: _detect_drag_defenders(frames, attacking_team, defending_team),
        lambda: _detect_overlap_run(frames, attacking_team),
        lambda: _detect_high_press(frames, defending_team),
    ):
        obs = detector()
        if obs is not None:
            observations.append(obs)

    key = (
        max(observations, key=lambda o: o.confidence).pattern
        if observations else None
    )

    return TacticalReport(
        formation_before=formation_before,
        formation_after=formation_after,
        observations=observations,
        key_movement=key,
    )
