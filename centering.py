"""Turn a detected body tube into a velocity command that centers the drone over it.

Pure functions only (no camera, no MAVLink, no ultralytics), so this can be
unit-tested on a laptop and reused later by the flight-control code.

Conventions
-----------
Image frame: x grows right, y grows down, origin at top-left.
Normalized error: (0, 0) means the target is at the image center. Range [-1, 1].

Body frame (MAVLink MAV_FRAME_BODY_NED): +x forward, +y right.
ASSUMPTION: the camera points straight down with the TOP of the image facing
the drone's FRONT. Under that mounting, a target above image center means
"move forward" and a target right of center means "move right". If the camera
is mounted rotated, change the signs/axes in `compute_velocity` and re-test on
the bench before flying.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


def _norm(name: str) -> str:
    """'Body Tube', 'body_tube' and 'body-tube' all compare equal."""
    return name.lower().replace("_", " ").replace("-", " ").strip()


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def pick_target(
    detections: Iterable[Detection],
    target_class: str = "body tube",
    min_conf: float = 0.4,
) -> Optional[Detection]:
    """Return the highest-confidence detection of `target_class`, or None."""
    wanted = _norm(target_class)
    candidates = [
        d for d in detections if _norm(d.cls) == wanted and d.conf >= min_conf
    ]
    return max(candidates, key=lambda d: d.conf, default=None)


def pixel_error(det: Detection, width: int, height: int) -> Tuple[float, float]:
    """Normalized (ex, ey) offset of the detection center from the image center."""
    ex = (det.cx - width / 2.0) / (width / 2.0)
    ey = (det.cy - height / 2.0) / (height / 2.0)
    return ex, ey


def is_centered(ex: float, ey: float, tol: float = 0.08) -> bool:
    return abs(ex) <= tol and abs(ey) <= tol


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def compute_velocity(
    ex: float,
    ey: float,
    kp: float = 0.8,
    max_speed: float = 0.5,
    deadband: float = 0.05,
) -> Tuple[float, float]:
    """Proportional controller. Returns (vx_forward, vy_right) in m/s, body frame.

    kp         m/s of command per unit of normalized error
    max_speed  hard clamp on each axis (keep low for first flights)
    deadband   errors smaller than this are treated as zero to avoid jitter
    """
    if abs(ex) < deadband:
        ex = 0.0
    if abs(ey) < deadband:
        ey = 0.0

    vx = _clamp(-kp * ey, max_speed)  # target above center -> fly forward
    vy = _clamp(kp * ex, max_speed)   # target right of center -> fly right
    return vx, vy
