"""Adapt a SoccerNet Game State Reconstruction (GSR) clip into MATE match events.

SoccerNet-GSR ships, for each 30-second broadcast clip, a ``Labels-GameState.json`` in
which every player and the ball is already projected onto a 2D pitch (the "minimap").
That lets us skip the weak link in the vision pipeline -- pixel->pitch calibration --
and feed ground-truth positions straight into the existing event inference, so the
resulting StatsBomb-shaped events drop into ``data_replayer.replay()`` and the
commentary agent unchanged.

GSR pitch coordinates are in metres with the origin at the centre spot (x along the
105 m length in ~[-52.5, 52.5], y across the 68 m width in ~[-34, 34]); we map them onto
StatsBomb's 0..120 x 0..80 pitch so the agent's in-the-box / progressive heuristics keep
working. Format and coordinate convention: see the official sn-gamestate repo
(``sn_gamestate/calibration/bbox2pitch.py`` and ``sn_gamestate/visualization/pitch.py``).

CLI (clip -> events -> replay):
    python -m vision_model.soccernet_gsr <Labels-GameState.json> \\
        --home Argentina --away France --save-events data/vision/events.json
    python -m vision_model.soccernet_gsr <Labels-GameState.json> --commentary --mock

Note: the goal-mouth Shot heuristic in ``events.py`` only fires for the right-hand goal
(StatsBomb +x), so a clip attacking the left goal yields passes but no Shot event.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vision_model import frames as frames_mod
from vision_model.detector import BALL, PLAYER, Detection, StubDetector
from vision_model.pipeline import save_events, video_to_events
from vision_model.schema import PITCH_LENGTH, PITCH_WIDTH

# Official SoccerNet pitch model (metres), origin at the centre spot.
GSR_PITCH_LENGTH_M = 105.0
GSR_PITCH_WIDTH_M = 68.0

DEFAULT_FPS = 25.0                        # SoccerNet broadcast clips are 25 fps.
_PLAYER_ROLES = {"player", "goalkeeper"}  # a keeper is a player for possession/passing
_BALL_ROLE = "ball"


def gsr_to_statsbomb(x_m: float, y_m: float) -> Tuple[float, float]:
    """Map a GSR pitch point (metres, centre origin) onto StatsBomb 120x80 coordinates."""
    sb_x = (x_m + GSR_PITCH_LENGTH_M / 2.0) / GSR_PITCH_LENGTH_M * PITCH_LENGTH
    sb_y = (y_m + GSR_PITCH_WIDTH_M / 2.0) / GSR_PITCH_WIDTH_M * PITCH_WIDTH
    # Clamp: off-pitch projections must not push coordinates outside the agent's pitch.
    sb_x = min(max(sb_x, 0.0), PITCH_LENGTH)
    sb_y = min(max(sb_y, 0.0), PITCH_WIDTH)
    return (round(sb_x, 2), round(sb_y, 2))


def _attributes(ann: dict) -> dict:
    """The annotation's attributes sub-dict (role/team/jersey), or an empty dict."""
    attrs = ann.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _pitch_xy(ann: dict) -> Optional[Tuple[float, float]]:
    """StatsBomb (x, y) for an annotation's bottom-middle pitch point, or None if absent."""
    bbox_pitch = ann.get("bbox_pitch")
    if not isinstance(bbox_pitch, dict):
        return None
    x, y = bbox_pitch.get("x_bottom_middle"), bbox_pitch.get("y_bottom_middle")
    if x is None or y is None:
        return None
    return gsr_to_statsbomb(float(x), float(y))


def _jersey(ann: dict) -> Optional[int]:
    """Jersey number as an int, or None when it isn't annotated."""
    jersey = _attributes(ann).get("jersey")
    try:
        return int(jersey)
    except (TypeError, ValueError):
        return None


def _frame_order(images: List[dict]) -> Tuple[Dict[object, int], int]:
    """Map each image_id to a 0-based frame index in temporal (file_name) order."""
    labeled = [im for im in images if im.get("file_name") is not None]
    labeled.sort(key=lambda im: im["file_name"])
    order = {im.get("image_id"): idx for idx, im in enumerate(labeled)}
    return order, len(labeled)


def build_script(
    labels: dict,
    team_names: Optional[Dict[str, str]] = None,
) -> Tuple[List[List[Detection]], Dict[int, str], Tuple[str, str]]:
    """Turn parsed GSR labels into a per-frame detection script and player-name map.

    Returns ``(script, player_names, (home, away))`` where ``script[i]`` is the list of
    ``Detection`` for frame ``i`` (ball + players carrying StatsBomb ``xy_pitch``),
    ``player_names`` maps track_id -> jersey label, and home/away are the display names
    chosen for the GSR 'left'/'right' teams.
    """
    names = team_names or {}
    home = names.get("left") or "Left Team"
    away = names.get("right") or "Right Team"
    team_label = {"left": home, "right": away}

    order, n_frames = _frame_order(labels.get("images", []))
    script: List[List[Detection]] = [[] for _ in range(n_frames)]
    player_names: Dict[int, str] = {}

    for ann in labels.get("annotations", []):
        frame_idx = order.get(ann.get("image_id"))
        if frame_idx is None:
            continue
        xy = _pitch_xy(ann)
        if xy is None:                    # off-pitch / failed projection -> drop
            continue
        role = _attributes(ann).get("role")
        if role == _BALL_ROLE:
            script[frame_idx].append(Detection(BALL, xy_pitch=xy))
            continue
        if role not in _PLAYER_ROLES:
            continue                      # referees / 'other' aren't players for commentary
        track_id = ann.get("track_id")
        if track_id is None:
            continue
        track_id = int(track_id)
        team = team_label.get(_attributes(ann).get("team"))
        script[frame_idx].append(Detection(PLAYER, track_id=track_id, team=team, xy_pitch=xy))
        if track_id not in player_names:
            jersey = _jersey(ann)
            if jersey is not None:
                player_names[track_id] = f"#{jersey}"

    return script, player_names, (home, away)


def events_from_labels(
    labels: dict,
    team_names: Optional[Dict[str, str]] = None,
    fps: float = DEFAULT_FPS,
) -> List[dict]:
    """Build StatsBomb-shaped events from already-parsed GSR labels.

    Replays the per-frame positions through the existing offline pipeline
    (``StubDetector`` -> ``video_to_events``), so no calibration, torch or OpenCV is
    needed -- the positions are already on the pitch.
    """
    script, player_names, _ = build_script(labels, team_names)
    frame_iter = frames_mod.synthetic_clip(len(script), fps=fps)
    detector = StubDetector(script)
    return video_to_events(frame_iter, detector=detector, player_names=player_names)


def gsr_to_events(
    labels_path,
    team_names: Optional[Dict[str, str]] = None,
    fps: float = DEFAULT_FPS,
) -> List[dict]:
    """Read a clip's Labels-GameState.json and return StatsBomb-shaped match events."""
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    return events_from_labels(labels, team_names=team_names, fps=fps)


def main(argv: Optional[list] = None) -> int:
    """Convert a GSR clip to events.json, and optionally replay it as commentary."""
    parser = argparse.ArgumentParser(
        description="SoccerNet GSR clip -> MATE match events (StatsBomb schema)."
    )
    parser.add_argument("labels", help="Path to a clip's Labels-GameState.json.")
    parser.add_argument("--home", default="Left Team",
                        help="Display name for the GSR 'left' team.")
    parser.add_argument("--away", default="Right Team",
                        help="Display name for the GSR 'right' team.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Clip frame rate.")
    parser.add_argument("--save-events", dest="save_events", default="data/vision/events.json",
                        help="Where to write events JSON (web app lists this as 'Vision clip').")
    parser.add_argument("--commentary", action="store_true",
                        help="Replay the events through the commentary pipeline.")
    parser.add_argument("--mock", action="store_true",
                        help="Offline commentary (no Granite) when used with --commentary.")
    parser.add_argument("--language", default="en", help="Commentary language code.")
    args = parser.parse_args(argv)

    events = gsr_to_events(
        args.labels, team_names={"left": args.home, "right": args.away}, fps=args.fps,
    )
    if args.save_events:
        save_events(events, args.save_events)
        print(f"Wrote {len(events)} events -> {args.save_events}")

    if args.commentary:
        if not args.mock:  # make GRANITE_* available for a real replay (mirrors pipeline.main)
            try:
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).resolve().parent.parent / ".env")
            except ImportError:
                pass
        from vision_model.pipeline import _print_commentary
        _print_commentary(events, language=args.language, mock=args.mock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
