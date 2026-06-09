from __future__ import annotations


def score_anomaly(confidence: float, face_eye: dict | None, motion: dict | None, cfg: dict):
    """Simple explainable anomaly scoring for MVP.

    It is intentionally transparent: scores come with reasons so the dashboard
    can explain why an event looks important instead of saying true/false.
    """
    reasons = []
    score = 0.0
    low_conf = float(cfg.get('low_confidence_threshold', 0.45))
    face_required = bool(cfg.get('face_required', False))

    if confidence < low_conf:
        score += 0.35
        reasons.append('low_detection_confidence')

    if face_required and not ((face_eye or {}).get('face_found')):
        score += 0.30
        reasons.append('face_not_visible')
    elif face_eye is not None and not face_eye.get('face_found'):
        score += 0.10
        reasons.append('face_not_detected_optional')

    if motion and motion.get('state') == 'moving' and float(motion.get('distance_px', 0)) > 80:
        score += 0.25
        reasons.append('sudden_movement')

    return {'anomaly_score': round(min(score, 1.0), 3), 'reasons': reasons}
