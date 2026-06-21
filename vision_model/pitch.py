"""Map image-space detections onto the StatsBomb 120x80 pitch.

A real implementation calibrates a homography from the broadcast's pitch lines to pitch
coordinates (``cv2.findHomography`` on 4+ correspondences). For the scaffold we provide
a simple linear scaler from frame size to the pitch, plus the seam to drop a real
homography in later. Detections that already carry ``xy_pitch`` (e.g. the synthetic stub)
are left untouched.
"""
from __future__ import annotations

from vision_model.detector import FrameDetections
from vision_model.schema import PITCH_LENGTH, PITCH_WIDTH


class LinearPitchMapper:
    """Placeholder mapping: scale pixel (cx, cy) from a frame of WxH onto 120x80.

    This is NOT a homography — it assumes a flat, axis-aligned overhead-ish view and is
    only good enough to exercise the pipeline. Replace with a calibrated homography for
    real broadcast footage (see README).
    """

    def __init__(self, frame_w: float, frame_h: float):
        self.frame_w = float(frame_w) or 1.0
        self.frame_h = float(frame_h) or 1.0

    def to_pitch(self, x: float, y: float) -> tuple:
        """Scale one pixel coordinate to a clamped (x, y) on the 120x80 pitch."""
        px = max(0.0, min(PITCH_LENGTH, x / self.frame_w * PITCH_LENGTH))
        py = max(0.0, min(PITCH_WIDTH, y / self.frame_h * PITCH_WIDTH))
        return (round(px, 2), round(py, 2))

    def map_frame(self, fd: FrameDetections) -> FrameDetections:
        """Fill ``xy_pitch`` for every detection that doesn't already have one."""
        for d in fd.detections:
            if d.xy_pitch is None:
                d.xy_pitch = self.to_pitch(d.cx, d.cy)
        return fd
