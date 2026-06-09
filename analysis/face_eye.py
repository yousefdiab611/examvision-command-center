from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import cv2
import numpy as np


@dataclass
class FaceEyeResult:
    face_found: bool
    face_bbox_xyxy: Optional[list[int]] = None
    left_eye_bbox_xyxy: Optional[list[int]] = None
    right_eye_bbox_xyxy: Optional[list[int]] = None
    face_crop_path: Optional[str] = None
    engine: str = 'opencv-haar'

    def to_dict(self):
        return asdict(self)


class FaceEyeAnalyzer:
    """Lightweight face/eye cropper.

    Uses OpenCV Haar cascades for MVP stability. This can later be swapped for
    MediaPipe Tasks / YOLO-face landmarks without changing event schema.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError('OpenCV Haar cascade files could not be loaded')

    @staticmethod
    def _xywh_to_xyxy(rect, pad, w, h):
        x, y, rw, rh = rect
        return [max(0, x - pad), max(0, y - pad), min(w, x + rw + pad), min(h, y + rh + pad)]

    def analyze_person_crop(self, crop_bgr: np.ndarray, save_dir: Optional[Path] = None, prefix: str = 'face') -> FaceEyeResult:
        if crop_bgr.size == 0:
            return FaceEyeResult(face_found=False)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = crop_bgr.shape[:2]
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return FaceEyeResult(face_found=False)

        # Pick the largest face.
        face = max(faces, key=lambda r: r[2] * r[3])
        face_bbox = self._xywh_to_xyxy(face, pad=10, w=w, h=h)
        fx1, fy1, fx2, fy2 = face_bbox
        face_gray = gray[fy1:fy2, fx1:fx2]
        eyes = self.eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=4, minSize=(12, 12))

        eye_boxes = []
        for e in sorted(eyes, key=lambda r: r[0])[:2]:
            ex1, ey1, ex2, ey2 = self._xywh_to_xyxy(e, pad=4, w=fx2-fx1, h=fy2-fy1)
            eye_boxes.append([ex1 + fx1, ey1 + fy1, ex2 + fx1, ey2 + fy1])
        left_eye = eye_boxes[0] if len(eye_boxes) > 0 else None
        right_eye = eye_boxes[1] if len(eye_boxes) > 1 else None

        crop_path = None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            face_crop = crop_bgr[fy1:fy2, fx1:fx2]
            if face_crop.size:
                crop_path = str(save_dir / f'{prefix}_face.jpg')
                cv2.imwrite(crop_path, face_crop)

        return FaceEyeResult(True, face_bbox, left_eye, right_eye, crop_path)
