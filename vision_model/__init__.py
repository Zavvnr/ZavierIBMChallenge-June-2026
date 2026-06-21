"""
vision_model — turn broadcast video into the match events the agent already consumes.

This is MATE's stretch module. Today the commentary runs on a StatsBomb event feed
(`data_extraction.loader.fetch_events`). The goal here is to produce those same events
from video instead, so the rest of the system (replayer -> commentary -> TTS, and the
explainer) works unchanged.

Pipeline (see `pipeline.video_to_events`):

    frames -> detector -> [track ids] -> [team assign] -> pitch mapping -> event inference -> [tactical context]
      frames.py   detector.py                              pitch.py        events.py          tactics.py / formation.py

Formation recognition (see `formation.py`):
    Player (x, y) positions -> preprocess -> FormationNet (PyTorch MLP) -> "4-3-3" etc.
    Offline: StubFormationPredictor replays scripted formations with no torch.
    Training: `trainer.py` generates synthetic data and trains the MLP.

Tactical analysis (see `tactics.py`):
    A window of FrameDetections -> TacticalReport with observations like
    drag_defenders, man_marking, crowd_box, overlap_run, high_press.
    `pipeline.with_tactical_context` attaches these to goal events so the
    explainer can answer "how did they create the chance?"

Every stage has an offline, dependency-free path so the package imports and its tests
run with no torch / ultralytics / OpenCV and no video file:
  - `detector.StubDetector` replays scripted detections (real path: `YoloDetector`).
  - `frames.synthetic_clip` yields fake frames (real path: `frames.video_frames`).
  - synthetic detections carry pitch coordinates directly; `pitch.LinearPitchMapper`
    maps real pixels (placeholder for a calibrated homography).

The integration contract lives in `schema.py`: events are StatsBomb-shaped
(`index`, `period`, `minute`, `second`, `timestamp`, `type.name`, `team.name`,
`player.name`, `location` on the 120x80 pitch, plus `shot` / `pass` sub-dicts), so they
drop straight into `data_replayer.replay()` and `agent.commentary_agent`.

Quick start (offline):
    python -m vision_model.pipeline --stub --commentary

Real footage (needs the vision extras):
    pip install ultralytics opencv-python
    python -m vision_model.pipeline --video clip.mp4

See README.md for what is real vs. stubbed and the roadmap.
"""
