"""Frame sources for the pipeline: a synthetic clip for offline tests, and a lazy
OpenCV reader for real video files.
"""
from __future__ import annotations

from typing import Iterator, Tuple


def synthetic_clip(n_frames: int, fps: float = 25.0) -> Iterator[Tuple[int, float, int]]:
    """Yield (frame_idx, elapsed_s, frame) for a fake clip.

    The 'frame' is just the index, which ``StubDetector`` uses to look up scripted
    detections — no pixels, no OpenCV — so tests and the offline demo stay dependency-free.
    """
    fps = fps or 25.0
    for i in range(n_frames):
        yield i, i / fps, i


def video_frames(path: str, stride: int = 1) -> Iterator[Tuple[int, float, "object"]]:
    """Yield (frame_idx, elapsed_s, frame_ndarray) from a video file via OpenCV.

    OpenCV is imported lazily so the package imports without it. ``stride`` processes
    every Nth frame to keep inference affordable on long clips.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only with the extra installed
        raise SystemExit(
            "vision_model: opencv-python not installed (`pip install opencv-python`)."
        ) from exc

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"vision_model: could not open video {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, stride) == 0:
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()
