"""Orchestrate the vision pipeline: frames -> detector -> pitch mapping -> events.

``video_to_events`` returns StatsBomb-shaped event dicts that drop straight into
``data_replayer.replay()`` and ``agent.commentary_agent``. The default detector is the
offline stub; pass a real one (``build_detector('yolo', ...)``) for broadcast video.

``with_tactical_context`` enriches Shot/Goal events with a ``vision_context`` dict
that describes the formation and tactical patterns in the seconds before the goal.
This context is consumed by the explainer when answering "how did they create the chance?".

CLI (Tier A — clip -> events -> replay):
    python -m vision_model.pipeline --video clip.mp4 --save-events clip.json      # extract once
    python -m vision_model.pipeline --events clip.json --commentary --language es # replay (Granite)
    python -m vision_model.pipeline --stub --commentary --mock                    # offline demo
"""
from __future__ import annotations

import argparse
import itertools
import json
from typing import Iterable, List, Optional, Tuple, Union

from vision_model import frames as frames_mod
from vision_model.detector import (
    BALL, PLAYER, Detection, Detector, FrameDetections, StubDetector, build_detector,
)
from vision_model.events import EventBuilder
from vision_model.pitch import LinearPitchMapper

# How many seconds of footage to analyse before a goal when building tactical context.
_GOAL_WINDOW_S = 3.0


def video_to_events(
    frame_iter: Iterable[Tuple[int, float, object]],
    detector: Optional[Detector] = None,
    mapper: Optional[LinearPitchMapper] = None,
    player_names: Optional[dict] = None,
    return_frames: bool = False,
) -> Union[List[dict], Tuple[List[dict], List[FrameDetections]]]:
    """Run the pipeline over a frame iterator and return the inferred events.

    ``mapper`` is needed for real footage (pixels -> pitch); the stub's detections
    already carry pitch coordinates, so it can be omitted there.

    When ``return_frames=True`` the function returns ``(events, frames)`` so callers
    can pass the frame list to ``with_tactical_context`` for goal analysis.
    """
    detector = detector or build_detector("stub")
    builder = EventBuilder(player_names=player_names)
    all_frames: List[FrameDetections] = []
    for frame_idx, elapsed_s, frame in frame_iter:
        detections = detector.detect(frame)
        fd = FrameDetections(frame_idx, elapsed_s, detections)
        if mapper is not None:
            mapper.map_frame(fd)
        builder.observe(fd)
        if return_frames:
            all_frames.append(fd)
    if return_frames:
        return builder.events, all_frames
    return builder.events


def with_tactical_context(
    events: List[dict],
    frames: List[FrameDetections],
    attacking_team: str,
    defending_team: str,
    formation_predictor=None,
    window_s: float = _GOAL_WINDOW_S,
) -> List[dict]:
    """Attach tactical analysis to Shot/Goal events in ``events``.

    For each goal, the function takes a ``window_s``-second window of
    ``FrameDetections`` before the goal, runs formation detection and tactical
    pattern analysis, and attaches the result as a ``vision_context`` dict.
    Non-goal events are returned unchanged.

    This method is the bridge between the vision pipeline and the explainer:
    when a user asks "how did they create the chance?", the explainer can read
    ``event['vision_context']['tactical_patterns']`` to ground its answer.
    """
    from vision_model import tactics
    from vision_model.formation import StubFormationPredictor

    predictor = formation_predictor or StubFormationPredictor()
    enriched: List[dict] = []

    for ev in events:
        is_goal = (
            ev.get("type", {}).get("name") == "Shot"
            and ev.get("shot", {}).get("outcome", {}).get("name") == "Goal"
        )
        if not is_goal:
            enriched.append(ev)
            continue

        goal_elapsed_s = ev.get("minute", 0) * 60.0 + ev.get("second", 0)
        window_start_s = max(0.0, goal_elapsed_s - window_s)
        window = [
            fd for fd in frames
            if window_start_s <= fd.elapsed_s <= goal_elapsed_s
        ]

        # Sample formation from the start and end of the window.
        formation_before = _formation_for_frame(window[0], attacking_team, predictor) if window else None
        formation_after = _formation_for_frame(window[-1], attacking_team, predictor) if window else None

        report = tactics.analyse(
            window, attacking_team, defending_team,
            formation_before=formation_before,
            formation_after=formation_after,
        )

        ev = dict(ev)  # shallow copy — don't mutate the original
        ev["vision_context"] = {
            "formation_before": report.formation_before,
            "formation_after": report.formation_after,
            "tactical_patterns": [
                {
                    "pattern": obs.pattern,
                    "description": obs.description,
                    "confidence": round(obs.confidence, 3),
                }
                for obs in report.observations
            ],
            "key_movement": report.key_movement,
        }
        enriched.append(ev)

    return enriched


def _formation_for_frame(fd: FrameDetections, team: str, predictor) -> Optional[str]:
    """Extract attacking player positions from one frame and predict their formation."""
    positions = [
        d.xy_pitch for d in fd.players()
        if d.team == team and d.xy_pitch is not None
    ]
    if not positions:
        return None
    return predictor.predict(positions)


def from_video(path: str, weights: str = "yolov8n.pt", stride: int = 3) -> List[dict]:
    """Convenience: real broadcast video -> events (needs ultralytics + opencv)."""
    frame_iter = frames_mod.video_frames(path, stride=stride)
    try:
        first = next(frame_iter)
    except StopIteration:
        return []
    height, width = first[2].shape[:2]
    mapper = LinearPitchMapper(width, height)
    detector = build_detector("yolo", weights=weights)
    return video_to_events(itertools.chain([first], frame_iter), detector=detector, mapper=mapper)


def build_demo() -> Tuple[Iterable[Tuple[int, float, int]], Detector]:
    """A tiny scripted clip (no video, no model): a forward pass, then a goal.

    Returns (frame_iter, detector) for the stub pipeline — shared by the CLI demo and
    the tests so there's one source of truth for the example.
    """
    def frame(ball_xy, players) -> List[Detection]:
        dets = [Detection(BALL, xy_pitch=ball_xy)]
        for track_id, xy in players:
            dets.append(Detection(PLAYER, track_id=track_id, team="Blue", xy_pitch=xy))
        return dets

    a1, a2 = (60.0, 40.0), (100.0, 41.0)
    script = [
        frame((61.0, 40.0), [(10, a1), (7, a2)]),     # 0: #10 has the ball
        frame((61.0, 40.0), [(10, a1), (7, a2)]),     # 1
        frame((100.0, 41.0), [(10, a1), (7, a2)]),    # 2: ball at #7  -> Pass #10 -> #7
        frame((110.0, 40.0), [(10, a1), (7, a2)]),    # 3: ball travelling to goal
        frame((119.8, 40.0), [(10, a1), (7, a2)]),    # 4: ball in the goal mouth -> Goal
    ]
    return frames_mod.synthetic_clip(len(script)), StubDetector(script)


def demo_events() -> List[dict]:
    """Run the built-in stub demo end to end and return its events."""
    frame_iter, detector = build_demo()
    return video_to_events(frame_iter, detector=detector)


def save_events(events: List[dict], path: str) -> str:
    """Write events to JSON so a clip can be processed once, then replayed many times."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(events, fh, ensure_ascii=False, indent=2)
    return path


def load_events(path: str) -> List[dict]:
    """Read events written by ``save_events`` (or any StatsBomb-shaped events JSON)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _teams(events: List[dict]) -> Tuple[str, str]:
    """First two distinct team names in the stream — grounds the opening scene-setter."""
    names: List[str] = []
    for ev in events:
        name = (ev.get("team") or {}).get("name")
        if name and name not in names:
            names.append(name)
        if len(names) == 2:
            break
    return (names[0] if names else "", names[1] if len(names) > 1 else "")


def _print_commentary(events: List[dict], language: str = "en", mock: bool = False) -> None:
    """Replay events through the REAL commentary pipeline and print each line.

    This is the Tier A 'replay' step: the same ``stream_commentary`` the web app uses, so
    vision events are narrated exactly like the StatsBomb feed. ``mock=True`` keeps it
    offline (no Granite); otherwise it calls the configured Granite endpoint.
    """
    from data_pipeline.commentary_pipeline import stream_commentary

    home, away = _teams(events)
    for item in stream_commentary(events, language=language, speed=0.0, mock=mock,
                                  match_context={"home": home, "away": away}):
        ev = item.event or {}
        print(f"{ev.get('minute', 0):02d}:{ev.get('second', 0):02d}  {item.text}")


def main(argv: Optional[list] = None) -> int:
    """Run the vision pipeline from the command line (stub demo or a real video)."""
    parser = argparse.ArgumentParser(description="Vision -> match events (StatsBomb schema).")
    parser.add_argument("--video", help="Path to a broadcast video (needs ultralytics + opencv).")
    parser.add_argument("--events", help="Replay events from a saved JSON file (skip vision).")
    parser.add_argument("--stub", action="store_true", help="Run the offline scripted demo.")
    parser.add_argument("--save-events", dest="save_events", default=None,
                        help="Write the extracted events to this JSON file (for replay).")
    parser.add_argument("--commentary", action="store_true",
                        help="Replay the events through the commentary pipeline.")
    parser.add_argument("--mock", action="store_true",
                        help="Offline commentary (no Granite) when used with --commentary.")
    parser.add_argument("--language", default="en", help="Commentary language code.")
    parser.add_argument("--stride", type=int, default=3, help="Process every Nth video frame.")
    args = parser.parse_args(argv)

    if args.events:
        events = load_events(args.events)
    elif args.video:
        events = from_video(args.video, stride=args.stride)
    else:
        if not args.stub:
            print("No --video/--events given; running the offline --stub demo.")
        events = demo_events()

    if args.save_events:
        save_events(events, args.save_events)
        print(f"Wrote {len(events)} events -> {args.save_events}")

    if args.commentary:
        if not args.mock:  # local convenience so GRANITE_* is available for a real replay
            try:
                from pathlib import Path
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).resolve().parent.parent / ".env")
            except ImportError:
                pass
        _print_commentary(events, language=args.language, mock=args.mock)
    elif not args.save_events:
        for event in events:
            print(json.dumps(event, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
