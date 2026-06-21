# vision_model — broadcast video → match events

MATE's stretch module. The commentary system runs on a StatsBomb event feed; this
package's job is to produce those **same events from video**, so the replayer →
commentary → TTS chain and the explainer all keep working unchanged.

It is a **scaffold**: every stage runs offline today with a stub, and each stub names
the real component that replaces it. The point is a clean, tested skeleton you can grow
one stage at a time — the genuinely hard parts (homography, team ID, naming) are marked.

## Pipeline

```
 frames ──▶ detector ──▶ [track IDs] ──▶ [team assign] ──▶ pitch mapping ──▶ event inference ──▶ events
 frames.py  detector.py                                     pitch.py          events.py        (schema.py)
```

`pipeline.video_to_events()` wires it together and returns StatsBomb-shaped event dicts.

## Quick start

Offline — no model, no video, no extra installs:

```
python -m vision_model.pipeline --stub --commentary
```

That runs a tiny scripted clip (a forward pass, then a goal) through the pipeline and
then through the mock commentary agent, proving the events are drop-in compatible.

Real footage (needs the optional vision extras):

```
pip install ultralytics opencv-python
python -m vision_model.pipeline --video clip.mp4
```

## The integration contract (`schema.py`)

Events must match the shape the agent already consumes, so they need no adapter:

- ordering: `index`, `period`, `minute`, `second`, `timestamp` (`HH:MM:SS.mmm`)
- semantics: `type.name`, `team.name`, `player.name`, `location` `[x, y]`
- type sub-dicts: `shot.outcome.name` (+ `end_location`, `statsbomb_xg`), `pass.recipient.name` (+ `end_location`)
- coordinates stay on StatsBomb's **120 × 80** pitch, so the agent's `importance()` and
  the in-the-box / progressive-pass heuristics work unchanged.

## Real vs. stubbed

| Stage | Stub (now) | Real (drop-in) |
|---|---|---|
| Frames | `frames.synthetic_clip` (frame indices) | `frames.video_frames` (OpenCV) |
| Detection | `StubDetector` (scripted) | `YoloDetector` (Ultralytics YOLO) |
| Tracking | track IDs come from the script | YOLO `model.track` (ByteTrack/BoTSORT) |
| Team assignment | preset `team` on detections | jersey-colour clustering (k-means on crops) |
| Pitch mapping | `LinearPitchMapper` (scale) | calibrated homography (`cv2.findHomography`) |
| Player names | `Player {track_id}` | jersey-number OCR + roster lookup |
| Events | nearest-player possession heuristics | learned spatiotemporal event model |

## Roadmap (the hard parts, in rough order)

1. Real detection + tracking: YOLO + `model.track` for persistent IDs.
2. Pitch homography: detect pitch lines / known points, calibrate per shot, re-estimate on camera cuts.
3. Team assignment: cluster jersey colours into two teams (+ referee/keeper handling).
4. Event inference: replace the nearest-player heuristics with possession-spell logic
   (pass vs. carry vs. shot), then consider a learned model.
5. Player identity: jersey-number OCR mapped to a team roster for real names.
6. Calibrate against StatsBomb: run on a match that also has event data and compare.

## Notes

- This is the hardest module and may stay partial for the MVP — that's expected. The
  scaffold is built so commentary already works end-to-end on stub events, and each real
  component slots in behind the same interface without touching the rest of MATE.
- The vision extras (`ultralytics`, `opencv-python`) are optional and imported lazily;
  nothing here breaks if they aren't installed.
