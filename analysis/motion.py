from __future__ import annotations

from dataclasses import dataclass, asdict
from math import hypot
from typing import Optional


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class MotionResult:
    state: str
    distance_px: float
    dx: float
    dy: float

    def to_dict(self):
        return asdict(self)


class MotionAnalyzer:
    def __init__(self, threshold_px: float = 18):
        self.threshold_px = threshold_px
        self.previous_centers: dict[str, tuple[float, float]] = {}

    def analyze(self, camera_id: str, track_id: Optional[int], bbox) -> MotionResult:
        cx, cy = _center(bbox)
        key = f'{camera_id}:{track_id if track_id is not None else "no_track"}'
        prev = self.previous_centers.get(key)
        self.previous_centers[key] = (cx, cy)
        if prev is None:
            return MotionResult('new_or_unknown', 0.0, 0.0, 0.0)
        dx = cx - prev[0]
        dy = cy - prev[1]
        dist = hypot(dx, dy)
        state = 'moving' if dist >= self.threshold_px else 'still'
        return MotionResult(state, round(dist, 3), round(dx, 3), round(dy, 3))
