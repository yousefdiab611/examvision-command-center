from __future__ import annotations

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify

ROOT = Path(__file__).resolve().parents[1]
CAMERAS_PATH = ROOT / 'configs/cameras.yaml'
ENV_PATH = ROOT / '.env'

app = Flask(__name__)
_captures: dict[str, 'CameraStream'] = {}
_lock = threading.Lock()


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


def load_env_file() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_PATH.exists():
        return env
    for raw in ENV_PATH.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def resolve_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
        key = value[2:-1]
        env = load_env_file()
        return env.get(key) or value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def load_cameras() -> list[dict[str, Any]]:
    if not CAMERAS_PATH.exists():
        return []
    data = yaml.safe_load(CAMERAS_PATH.read_text(encoding='utf-8')) or {}
    return data.get('cameras', []) or []


def get_camera(camera_id: str) -> dict[str, Any] | None:
    for camera in load_cameras():
        if camera.get('id') == camera_id:
            return camera
    return None


class CameraStream:
    """Low-latency OpenCV reader feeding MJPEG responses.

    This is the productionized version of the uploaded camera.py idea:
    open cv2.VideoCapture(source), keep reading frames continuously, and stream
    them to the dashboard instead of showing a local cv2.imshow window.
    """

    def __init__(self, camera_id: str, source: Any, target_fps: int = 20):
        self.camera_id = camera_id
        self.source = resolve_value(source)
        self.target_fps = max(1, min(int(target_fps or 20), 30))
        self.cap: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.error: str | None = None
        self.running = False
        self.thread: threading.Thread | None = None
        self.last_read = 0.0
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.frame_index = 0
        self.last_analysis: dict[str, Any] = {
            'face_found': False,
            'direction': 'unknown',
            'cheating_alert': False,
            'yaw_score': 0.0,
            'bbox': None,
        }

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def _open(self) -> None:
        if str(self.source).lower() == 'demo':
            return
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

    def _demo_frame(self) -> np.ndarray:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        t = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        cv2.rectangle(frame, (0, 0), (1280, 720), (8, 15, 30), -1)
        cv2.putText(frame, f'DEMO LIVE {t}', (70, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (56, 189, 248), 4, cv2.LINE_AA)
        cv2.putText(frame, f'{self.target_fps} FPS MJPEG stream', (76, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (187, 247, 208), 2, cv2.LINE_AA)
        return frame

    def _analyze_face_pose(self, frame: np.ndarray) -> dict[str, Any]:
        h, w = frame.shape[:2]
        small_w = 640
        scale = w / small_w if w > small_w else 1.0
        small = cv2.resize(frame, (small_w, int(h / scale))) if scale > 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        frontal = self.face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(55, 55))
        direction = 'unknown'
        bbox_small = None
        cheating = True
        face_found = False

        if len(frontal):
            # Largest frontal face = looking forward; require both-eye evidence when possible.
            x, y, fw, fh = max(frontal, key=lambda b: b[2] * b[3])
            roi = gray[y:y + fh, x:x + fw]
            eyes = self.eye_cascade.detectMultiScale(roi, scaleFactor=1.08, minNeighbors=4, minSize=(14, 14))
            bbox_small = (x, y, fw, fh)
            face_found = True
            if len(eyes) >= 1:
                direction = 'front'
                cheating = False
            else:
                # Frontal face without eyes is suspicious but less noisy than profile.
                direction = 'front_uncertain'
                cheating = False
        else:
            profiles = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(55, 55))
            if len(profiles):
                x, y, fw, fh = max(profiles, key=lambda b: b[2] * b[3])
                bbox_small = (x, y, fw, fh)
                direction = 'looking_side'
                cheating = True
                face_found = True
            else:
                flipped = cv2.flip(gray, 1)
                profiles_flipped = self.profile_cascade.detectMultiScale(flipped, scaleFactor=1.08, minNeighbors=4, minSize=(55, 55))
                if len(profiles_flipped):
                    x, y, fw, fh = max(profiles_flipped, key=lambda b: b[2] * b[3])
                    x = gray.shape[1] - x - fw
                    bbox_small = (x, y, fw, fh)
                    direction = 'looking_side'
                    cheating = True
                    face_found = True

        if bbox_small:
            x, y, fw, fh = bbox_small
            x1 = int(x * scale)
            y1 = int(y * scale)
            x2 = int((x + fw) * scale)
            y2 = int((y + fh) * scale)
            bbox = [max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)]
        else:
            bbox = None
            direction = 'no_face'
            cheating = True

        analysis = {
            'face_found': face_found,
            'direction': direction,
            'cheating_alert': cheating,
            'yaw_score': 1.0 if cheating else 0.0,
            'bbox': bbox,
        }
        self.last_analysis = analysis
        return analysis

    def _draw_tracking_overlay(self, frame: np.ndarray) -> np.ndarray:
        self.frame_index += 1
        # Analyze every other frame, draw every frame with the latest state.
        if self.frame_index % 2 == 0 or not self.last_analysis.get('bbox'):
            analysis = self._analyze_face_pose(frame)
        else:
            analysis = self.last_analysis

        h, w = frame.shape[:2]
        color = (0, 0, 255) if analysis.get('cheating_alert') else (0, 220, 0)
        direction = analysis.get('direction', 'unknown')
        label = 'CHEATING ALERT - LOOK FRONT' if analysis.get('cheating_alert') else 'OK - LOOKING FRONT'

        bbox = analysis.get('bbox')
        if bbox:
            x1, y1, x2, y2 = bbox
            face_w = max(1, x2 - x1)
            face_h = max(1, y2 - y1)
            # Expand face box into a rough person/upper-body tracking box.
            px1 = max(0, int(x1 - face_w * 0.75))
            py1 = max(0, int(y1 - face_h * 0.75))
            px2 = min(w - 1, int(x2 + face_w * 0.75))
            py2 = min(h - 1, int(y2 + face_h * 2.6))
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 4)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text_y = max(35, py1 - 12)
        else:
            text_y = 40
            cv2.rectangle(frame, (20, 20), (w - 20, h - 20), color, 4)

        cv2.rectangle(frame, (20, text_y - 34), (min(w - 20, 620), text_y + 12), color, -1)
        cv2.putText(frame, f'{label} | {direction}', (32, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _loop(self) -> None:
        delay = 1.0 / self.target_fps
        if str(self.source).lower() != 'demo':
            self._open()
        while self.running:
            start = time.time()
            try:
                if str(self.source).lower() == 'demo':
                    frame = self._demo_frame()
                    self.error = None
                else:
                    if self.cap is None or not self.cap.isOpened():
                        self.error = 'Camera not open; reconnecting...'
                        self._open()
                        time.sleep(0.2)
                        continue
                    # Grab a few frames and keep the newest one. This reduces stale RTSP/webcam buffer latency.
                    frame = None
                    for _ in range(2):
                        ok, candidate = self.cap.read()
                        if ok and candidate is not None:
                            frame = candidate
                    if frame is None:
                        self.error = 'Frame read failed; reconnecting...'
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = None
                        time.sleep(0.2)
                        continue
                frame = self._draw_tracking_overlay(frame)
                self.frame = frame
                self.last_read = time.time()
                self.error = None
            except Exception as exc:
                self.error = str(exc)
                time.sleep(0.2)
            elapsed = time.time() - start
            if elapsed < delay:
                time.sleep(delay - elapsed)

    def jpeg(self) -> bytes | None:
        frame = self.frame
        if frame is None:
            return None
        ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return None
        return encoded.tobytes()


def get_stream(camera_id: str) -> CameraStream | None:
    camera = get_camera(camera_id)
    if not camera:
        return None
    with _lock:
        stream = _captures.get(camera_id)
        source = resolve_value(camera.get('source'))
        target_fps = int(camera.get('live_fps') or camera.get('fps_sample') or 20)
        if stream is None or stream.source != source or stream.target_fps != target_fps:
            if stream is not None:
                stream.stop()
            stream = CameraStream(camera_id, source, target_fps=target_fps)
            _captures[camera_id] = stream
            stream.start()
        return stream


def placeholder_jpeg(message: str) -> bytes:
    frame = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (854, 480), (8, 15, 30), -1)
    cv2.putText(frame, message[:58], (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 200, 220), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return encoded.tobytes() if ok else b''


def mjpeg_generator(camera_id: str):
    stream = get_stream(camera_id)
    if stream is None:
        yield b''
        return
    min_delay = 1.0 / max(stream.target_fps, 1)
    while True:
        jpg = stream.jpeg() or placeholder_jpeg(stream.error or 'Waiting for live frame...')
        yield b'--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache, no-store, must-revalidate\r\nPragma: no-cache\r\n\r\n' + jpg + b'\r\n'
        time.sleep(min_delay)


@app.get('/health')
def health():
    return jsonify({'ok': True, 'cameras': [c.get('id') for c in load_cameras()]})


@app.get('/status/<camera_id>')
def status(camera_id: str):
    stream = _captures.get(camera_id)
    camera = get_camera(camera_id)
    if not camera:
        return jsonify({'ok': False, 'error': 'camera not found'}), 404
    analysis = {} if not stream else stream.last_analysis
    return jsonify({
        'ok': True,
        'camera_id': camera_id,
        'running': bool(stream and stream.running),
        'last_read_age_sec': None if not stream or not stream.last_read else round(time.time() - stream.last_read, 3),
        'error': None if not stream else stream.error,
        'face_found': bool(analysis.get('face_found')),
        'direction': analysis.get('direction', 'unknown'),
        'cheating_alert': bool(analysis.get('cheating_alert')),
        'yaw_score': analysis.get('yaw_score', 0.0),
    })


@app.get('/video_feed/<camera_id>')
def video_feed(camera_id: str):
    if get_camera(camera_id) is None:
        return jsonify({'ok': False, 'error': 'camera not found'}), 404
    return Response(mjpeg_generator(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.post('/stop/<camera_id>')
def stop_camera(camera_id: str):
    with _lock:
        stream = _captures.pop(camera_id, None)
    if stream:
        stream.stop()
    return jsonify({'ok': True})


@app.post('/stop_all')
def stop_all():
    with _lock:
        streams = list(_captures.values())
        _captures.clear()
    for stream in streams:
        stream.stop()
    return jsonify({'ok': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
