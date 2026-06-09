from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_events(jsonl_path: str):
    p = Path(jsonl_path)
    if not p.exists():
        return []
    events = []
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def export_csv(events, csv_path: str):
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = ['timestamp', 'camera_id', 'frame_id', 'event_type', 'track_id', 'confidence', 'motion_state', 'anomaly_score', 'nearby_count', 'person_crop_path']
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ev in events:
            w.writerow({
                'timestamp': ev.get('timestamp'),
                'camera_id': ev.get('camera_id'),
                'frame_id': ev.get('frame_id'),
                'event_type': ev.get('event_type'),
                'track_id': ev.get('track_id'),
                'confidence': ev.get('confidence'),
                'motion_state': (ev.get('motion') or {}).get('state'),
                'anomaly_score': (ev.get('anomaly') or {}).get('anomaly_score'),
                'nearby_count': len(ev.get('nearby_objects') or []),
                'person_crop_path': ev.get('person_crop_path'),
            })
    return str(p)


def write_summary(events, summary_path: str):
    p = Path(summary_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    by_camera = Counter(ev.get('camera_id') for ev in events)
    by_motion = Counter((ev.get('motion') or {}).get('state', 'unknown') for ev in events)
    tracks = defaultdict(int)
    for ev in events:
        if ev.get('track_id') is not None:
            tracks[f"{ev.get('camera_id')}:{ev.get('track_id')}"] += 1
    summary = {
        'total_events': len(events),
        'events_by_camera': dict(by_camera),
        'events_by_motion_state': dict(by_motion),
        'unique_tracks': len(tracks),
        'top_tracks': sorted(tracks.items(), key=lambda kv: kv[1], reverse=True)[:20],
        'avg_confidence': round(sum(float(ev.get('confidence') or 0) for ev in events) / len(events), 4) if events else 0,
    }
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary
