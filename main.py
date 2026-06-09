from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any
import yaml
import cv2

from cameras.camera_manager import frame_stream
from detectors.yolo_detector import YOLODetector
from analysis.face_eye import FaceEyeAnalyzer
from analysis.spatial import nearby_objects_for_person
from analysis.motion import MotionAnalyzer
from analysis.anomaly import score_anomaly
from events.event_logger import EventLogger
from events.reporting import read_events, export_csv, write_summary
from utils.config_resolver import resolve_value


def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def clamp_bbox(bbox, w, h):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def parse_source(value: Any):
    value = resolve_value(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def load_camera_profiles(cfg: dict, explicit_source=None, camera_id='camera_1'):
    if explicit_source is not None:
        return [{
            'id': camera_id,
            'name': 'CLI Source',
            'location': 'command line',
            'source': parse_source(explicit_source),
            'fps_sample': float(cfg['camera']['fps_sample']),
            'active': True,
        }]

    profiles_path = cfg.get('camera', {}).get('profiles_path')
    if profiles_path and Path(profiles_path).exists():
        data = load_yaml(profiles_path) or {}
        cams = [c for c in data.get('cameras', []) if c.get('active')]
        if cams:
            for c in cams:
                c['source'] = parse_source(c.get('source', cfg['camera']['default_source']))
                c['fps_sample'] = float(c.get('fps_sample', cfg['camera']['fps_sample']))
            return cams

    return [{
        'id': camera_id,
        'name': 'Default Camera',
        'location': 'default',
        'source': parse_source(cfg['camera']['default_source']),
        'fps_sample': float(cfg['camera']['fps_sample']),
        'active': True,
    }]


def run_pipeline(args):
    cfg = load_yaml(args.config)
    out_dir = Path(cfg['project']['output_dir'])
    persons_dir = out_dir / 'snapshots' / 'persons'
    faces_dir = out_dir / 'snapshots' / 'faces'
    annotated_dir = out_dir / 'snapshots' / 'annotated'
    for d in [persons_dir, faces_dir, annotated_dir, out_dir / 'reports']:
        d.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(
        weights=cfg['model']['yolo_weights'],
        conf=float(args.conf if args.conf is not None else cfg['model']['conf']),
        iou=float(cfg['model']['iou']),
        imgsz=int(cfg['model']['imgsz']),
        tracking=bool(cfg['tracking']['enabled']),
        tracker=cfg['tracking']['tracker'],
    )
    face_eye = FaceEyeAnalyzer(cfg['face_eye']['min_detection_confidence']) if cfg['face_eye']['enabled'] else None
    motion = MotionAnalyzer(cfg['tracking'].get('motion_threshold_px', 18))
    logger = EventLogger(cfg['events']['jsonl_path'])

    cameras = load_camera_profiles(cfg, explicit_source=args.source, camera_id=args.camera_id)
    max_frames = args.max_frames if args.max_frames is not None else int(cfg['camera'].get('max_frames', 0))

    processed = 0
    person_events = 0
    t_start = time.perf_counter()

    for camera in cameras:
        print(f"CAMERA_START id={camera['id']} name={camera.get('name')} source={camera.get('source')}")
        for packet in frame_stream(camera['source'], camera_id=camera['id'], fps_sample=float(camera['fps_sample']), max_frames=max_frames):
            processed += 1
            frame_t0 = time.perf_counter()
            h, w = packet.frame.shape[:2]
            detections, annotated = detector.detect(packet.frame)
            inference_ms = round((time.perf_counter() - frame_t0) * 1000, 2)
            ts_slug = time.strftime('%Y%m%d_%H%M%S', time.localtime(packet.timestamp)) + f'_{packet.frame_id:06d}'

            frame_events = []
            for idx, det in enumerate(detections):
                if det.label != 'person':
                    continue
                person_events += 1
                x1, y1, x2, y2 = clamp_bbox(det.bbox_xyxy, w, h)
                person_crop = packet.frame[y1:y2, x1:x2]
                person_path = None
                face_eye_payload = None

                if cfg['events']['save_snapshots'] and det.confidence >= float(cfg['events']['snapshot_conf_threshold']):
                    person_path = str(persons_dir / f'{packet.camera_id}_{ts_slug}_person{idx}.jpg')
                    cv2.imwrite(person_path, person_crop)

                if face_eye and person_crop.size:
                    res = face_eye.analyze_person_crop(person_crop, faces_dir if cfg['face_eye']['save_crops'] else None, f'{packet.camera_id}_{ts_slug}_person{idx}')
                    face_eye_payload = res.to_dict()

                nearby = nearby_objects_for_person(det, detections, cfg['nearby_objects']['max_center_distance_ratio']) if cfg.get('nearby_objects', {}).get('enabled', True) else []
                motion_payload = motion.analyze(packet.camera_id, det.track_id, det.bbox_xyxy).to_dict()
                anomaly_payload = score_anomaly(det.confidence, face_eye_payload, motion_payload, cfg.get('anomaly', {})) if cfg.get('anomaly', {}).get('enabled', True) else {'anomaly_score': 0, 'reasons': []}

                event = {
                    'event_type': 'person_detected',
                    'camera_id': packet.camera_id,
                    'camera_name': camera.get('name'),
                    'camera_location': camera.get('location'),
                    'frame_id': packet.frame_id,
                    'timestamp': packet.timestamp,
                    'confidence': round(float(det.confidence), 4),
                    'track_id': det.track_id,
                    'bbox_xyxy': det.bbox_xyxy,
                    'person_crop_path': person_path,
                    'face_eye': face_eye_payload,
                    'nearby_objects': nearby,
                    'motion': motion_payload,
                    'anomaly': anomaly_payload,
                    'performance': {'inference_ms': inference_ms},
                }
                logger.log(event)
                frame_events.append(event)

            annotated_path = None
            if cfg.get('runtime', {}).get('save_annotated', True):
                annotated_path = str(annotated_dir / f'{packet.camera_id}_{ts_slug}_annotated.jpg')
                cv2.imwrite(annotated_path, annotated)

            print(f'frame={packet.camera_id}:{packet.frame_id} detections={len(detections)} person_events={len(frame_events)} inference_ms={inference_ms} annotated={annotated_path}')
            if args.show or cfg.get('runtime', {}).get('show_window', False):
                cv2.imshow('CV YOLO Pipeline', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    if args.show:
        cv2.destroyAllWindows()

    events = read_events(cfg['events']['jsonl_path'])
    export_csv(events, cfg['events']['csv_path'])
    summary = write_summary(events, cfg['events']['summary_path'])
    elapsed = time.perf_counter() - t_start
    print(f'DONE processed_frames={processed} person_events={person_events} elapsed_sec={elapsed:.2f} events_file={cfg["events"]["jsonl_path"]}')
    print(f'SUMMARY {summary}')


def main():
    parser = argparse.ArgumentParser(description='YOLO CV Pipeline MVP+')
    parser.add_argument('--source', default=None, help='camera index, video/image path, URL, or demo')
    parser.add_argument('--camera-id', default='camera_1')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--conf', type=float, default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
