from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    label: str
    cls_id: int
    confidence: float
    bbox_xyxy: list[float]
    track_id: Optional[int] = None

    def to_dict(self):
        return asdict(self)


class YOLODetector:
    def __init__(self, weights: str, conf: float = 0.35, iou: float = 0.45, imgsz: int = 640, tracking: bool = True, tracker: str = 'bytetrack.yaml'):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracking = tracking
        self.tracker = tracker
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        if self.tracking:
            results = self.model.track(frame, persist=True, tracker=self.tracker, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
        else:
            results = self.model(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)

        result = results[0]
        detections: list[Detection] = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = str(self.names.get(cls_id, cls_id)) if isinstance(self.names, dict) else str(self.names[cls_id])
                track_id = int(box.id[0]) if getattr(box, 'id', None) is not None else None
                detections.append(Detection(
                    label=label,
                    cls_id=cls_id,
                    confidence=float(box.conf[0]),
                    bbox_xyxy=[float(v) for v in box.xyxy[0].tolist()],
                    track_id=track_id,
                ))
        return detections, result.plot()
