"""Detection layer: the data types, a Detector protocol, an offline stub, and an
optional YOLO adapter (imported lazily so the package works without torch/ultralytics).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable

PLAYER = "player"
BALL = "ball"


@dataclass
class Detection:
    """One detected object. ``bbox`` is pixels (x1, y1, x2, y2); ``xy_pitch`` is the
    (x, y) position on the 120x80 pitch, filled in later by the pitch mapper."""

    cls: str
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    conf: float = 1.0
    track_id: Optional[int] = None
    team: Optional[str] = None
    xy_pitch: Optional[tuple] = None

    @property
    def cx(self) -> float:
        """Bounding-box centre x in pixels."""
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        """Bounding-box centre y in pixels (feet ~ bbox bottom, but centre is fine here)."""
        return (self.bbox[1] + self.bbox[3]) / 2.0


@dataclass
class FrameDetections:
    """Every detection in one video frame, with its position on the timeline."""

    frame_idx: int
    elapsed_s: float
    detections: List[Detection] = field(default_factory=list)

    def ball(self) -> Optional[Detection]:
        """The ball detection for this frame, if one was found."""
        for d in self.detections:
            if d.cls == BALL:
                return d
        return None

    def players(self) -> List[Detection]:
        """All player detections for this frame."""
        return [d for d in self.detections if d.cls == PLAYER]


@runtime_checkable
class Detector(Protocol):
    """Anything that turns a single frame into a list of detections."""

    def detect(self, frame) -> List[Detection]:
        """Return detections for one frame."""
        ...


class StubDetector:
    """Deterministic synthetic detector — no model, no GPU, no video.

    It ignores pixels and replays a scripted list of detections per frame (indexed by
    frame number), so the whole pipeline and its tests run offline without ultralytics
    or torch. The synthetic frame source yields the frame index as the ``frame``.
    """

    def __init__(self, script: List[List[Detection]]):
        self._script = script

    def detect(self, frame) -> List[Detection]:
        """Look up the scripted detections for this frame index."""
        idx = frame if isinstance(frame, int) else getattr(frame, "idx", 0)
        if 0 <= idx < len(self._script):
            return list(self._script[idx])
        return []


class YoloDetector:
    """Optional real detector using Ultralytics YOLO. Imported lazily so the package
    (stub path + tests) works without it installed.

    Maps COCO 'person' -> PLAYER and 'sports ball' -> BALL. For persistent IDs, a real
    deployment would use ``model.track`` (ByteTrack/BoTSORT); here we expose plain
    detection and leave tracking to the pipeline (see README roadmap).
    """

    _COCO = {0: PLAYER, 32: BALL}  # person, sports ball

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised only with the extra installed
            raise SystemExit(
                "vision_model: ultralytics not installed. Install the vision extras "
                "(`pip install ultralytics opencv-python`) or use the stub detector."
            ) from exc
        self._model = YOLO(weights)
        self._conf = conf

    def detect(self, frame) -> List[Detection]:
        """Run YOLO on one frame and return player/ball detections."""
        out: List[Detection] = []
        for result in self._model(frame, conf=self._conf, verbose=False):
            for box in result.boxes:
                cls_id = int(box.cls)
                if cls_id not in self._COCO:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                out.append(Detection(self._COCO[cls_id], (x1, y1, x2, y2), float(box.conf)))
        return out


def build_detector(kind: str = "stub", **kwargs) -> Detector:
    """Return a detector by name: 'stub' (offline) or 'yolo' (real, needs ultralytics)."""
    if kind == "stub":
        return StubDetector(kwargs.get("script", []))
    if kind == "yolo":
        return YoloDetector(**{k: v for k, v in kwargs.items() if k != "script"})
    raise ValueError(f"unknown detector kind: {kind!r}")
