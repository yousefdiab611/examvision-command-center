from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Union
import time
import cv2
import numpy as np


@dataclass
class FramePacket:
    camera_id: str
    frame_id: int
    timestamp: float
    frame: np.ndarray


def _is_image_path(source: str) -> bool:
    return Path(source).suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def frame_stream(source: Union[str, int], camera_id: str = 'camera_1', fps_sample: float = 5, max_frames: int = 0) -> Generator[FramePacket, None, None]:
    """Yield frames from image/video/webcam/source URL.

    source can be:
    - int webcam index
    - video path
    - image path
    - stream URL
    - 'demo' to generate a synthetic blank frame for smoke tests
    """
    if str(source).lower() == 'demo':
        frame = np.full((480, 640, 3), 240, dtype=np.uint8)
        cv2.putText(frame, 'DEMO FRAME - replace --source with camera/video', (35, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        yield FramePacket(camera_id=camera_id, frame_id=1, timestamp=time.time(), frame=frame)
        return

    if isinstance(source, str) and _is_image_path(source):
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f'Could not read image: {source}')
        yield FramePacket(camera_id=camera_id, frame_id=1, timestamp=time.time(), frame=frame)
        return

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video/camera source: {source}')

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_every = max(1, int(native_fps / fps_sample)) if fps_sample > 0 else 1
    raw_idx = 0
    emitted = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_idx += 1
            if raw_idx % sample_every != 0:
                continue
            emitted += 1
            yield FramePacket(camera_id=camera_id, frame_id=emitted, timestamp=time.time(), frame=frame)
            if max_frames and emitted >= max_frames:
                break
    finally:
        cap.release()
