from __future__ import annotations

from math import hypot
from typing import Iterable

PERSON_LABEL = 'person'


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_diag(bbox):
    x1, y1, x2, y2 = bbox
    return hypot(x2 - x1, y2 - y1)


def nearby_objects_for_person(person_det, all_detections: Iterable, max_center_distance_ratio: float = 1.30):
    """Return non-person detections close to one person bbox.

    Distance is normalized against the person's bbox diagonal so this works
    across resolutions and person sizes.
    """
    pcx, pcy = bbox_center(person_det.bbox_xyxy)
    pdiag = max(1.0, bbox_diag(person_det.bbox_xyxy))
    nearby = []
    for det in all_detections:
        if det is person_det or det.label == PERSON_LABEL:
            continue
        ocx, ocy = bbox_center(det.bbox_xyxy)
        dist_ratio = hypot(ocx - pcx, ocy - pcy) / pdiag
        if dist_ratio <= max_center_distance_ratio:
            nearby.append({
                'label': det.label,
                'confidence': round(float(det.confidence), 4),
                'bbox_xyxy': det.bbox_xyxy,
                'distance_ratio': round(dist_ratio, 4),
                'distance': 'near' if dist_ratio <= 0.75 else 'around',
            })
    return sorted(nearby, key=lambda x: x['distance_ratio'])
