# ExamVision Command Center

AI-assisted exam hall monitoring system built with Python, OpenCV, Ultralytics YOLO, ByteTrack-style tracking, and Streamlit.

The project is designed for a multi-camera examination room setup. It can read from laptop cameras, RTSP/IP cameras such as Tapo C500, images, and video files. It detects people, tracks candidates, checks face visibility, detects nearby objects, scores review risk, saves evidence snapshots, and presents everything in a control-room dashboard.

![Command Center](docs/screenshots/command-center.png)

## Features

- Multi-camera profiles from `configs/cameras.yaml`
- Laptop camera ON/OFF from the dashboard
- RTSP/IP camera support
- Add new cameras directly from the dashboard
- Test camera connection and save a snapshot
- YOLO person/object detection
- ByteTrack tracking through Ultralytics
- Face visibility analysis using OpenCV Haar fallback
- Nearby object analysis around each candidate
- Motion analysis per track
- Explainable anomaly/risk scoring
- Evidence snapshots for people, faces, and annotated frames
- Event logs as JSONL and CSV
- Summary reports
- Professional Streamlit dashboard for invigilators
- Low-latency MJPEG live streaming via `dashboard/live_server.py`
- Live candidate tracking overlay: green while face is forward, red when the face turns sideways or disappears
- Browser sound alert when the live server reports a cheating state

## Dashboard Screenshots

### Command Center

![Command Center](docs/screenshots/command-center.png)

### Camera Wall

![Camera Wall](docs/screenshots/camera-wall.png)

### Control Room and Camera Management

![Control Room](docs/screenshots/control-room.png)

### Alerts Review

![Alerts Review](docs/screenshots/alerts-review.png)

### Reports

![Reports](docs/screenshots/reports.png)

## Architecture

```text
Exam Cameras / RTSP / Laptop Camera / Video Files
        ↓
Frame Sampler
        ↓
YOLO Detection
        ↓
Object Tracking
        ↓
Face Visibility + Nearby Object Scan
        ↓
Motion Analysis + Risk Scoring
        ↓
Evidence Logger
        ↓
Snapshots + CSV/JSON Reports + Streamlit Dashboard
```

## Project Structure

```text
cv-yolo-pipeline/
├── main.py
├── config.yaml
├── requirements.txt
├── .env.example
├── configs/
│   └── cameras.yaml
├── cameras/
│   └── camera_manager.py
├── detectors/
│   └── yolo_detector.py
├── analysis/
│   ├── anomaly.py
│   ├── face_eye.py
│   ├── motion.py
│   └── spatial.py
├── events/
│   ├── event_logger.py
│   └── reporting.py
├── dashboard/
│   ├── app.py
│   └── live_server.py
├── utils/
│   └── config_resolver.py
├── docs/
│   └── screenshots/
└── outputs/
    ├── events/
    ├── reports/
    └── snapshots/
```

## Requirements

- Python 3.9+
- macOS, Linux, or Windows
- Network access to IP/RTSP cameras
- Webcam permission if using laptop camera

## Installation

```bash
cd ~/cv-yolo-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd cv-yolo-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Variables and Camera Secrets

Do not commit camera usernames or passwords to GitHub.

Copy the example file:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
TAPO_C500_RTSP_URL=rtsp://USERNAME:PASSWORD@192.168.1.10:554/stream1
```

The default Tapo profile in `configs/cameras.yaml` uses:

```yaml
source: ${TAPO_C500_RTSP_URL}
```

The app resolves this at runtime from `.env`.

## Running the Dashboard

The dashboard runs on Streamlit port `8501`. The fast camera stream runs on MJPEG port `8765` and is started automatically when you open a backend live preview.

```bash
cd ~/cv-yolo-pipeline
source .venv/bin/activate
streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Optional manual MJPEG server start:

```bash
python dashboard/live_server.py --host 127.0.0.1 --port 8765
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

Open:

```text
http://127.0.0.1:8501
```

To expose it to other devices on the same network:

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Then open from another device:

```text
http://YOUR_LAPTOP_IP:8501
```

Find your Mac IP:

```bash
ipconfig getifaddr en0
```

## Opening Ports

### macOS Firewall

If macOS blocks incoming access:

1. Open System Settings
2. Network or Privacy & Security
3. Firewall
4. Allow Python / Streamlit, or temporarily allow incoming connections for testing

For local-only development, no external port opening is needed because the app runs on `127.0.0.1:8501`.

### Linux UFW

```bash
sudo ufw allow 8501/tcp
sudo ufw allow 8765/tcp  # optional: MJPEG live server, keep private if possible
sudo ufw reload
```

### Cloud VM Security Group

If deployed on a cloud VM, allow inbound TCP port `8501` from your IP only. Keep `8765` private/internal unless you intentionally need remote MJPEG access, because it streams live camera frames. Do not expose the dashboard publicly without authentication.

## Camera Configuration

Cameras live in:

```text
configs/cameras.yaml
```

Example:

```yaml
cameras:
  - id: cam_tapo_c500_exam
    name: Tapo C500
    location: Smart AI-Based Exam Monitoring System
    source: ${TAPO_C500_RTSP_URL}
    fps_sample: 5
    active: true
    type: rtsp
    ip: 192.168.1.10
    model: Tapo C500
    timezone: Africa/Cairo

  - id: cam_webcam
    name: Laptop Camera
    location: Local machine
    source: 0
    fps_sample: 5
    active: false
    type: local
```

## Adding Cameras from the Dashboard

1. Open `http://127.0.0.1:8501`
2. Go to **Control Room**
3. Use **Add camera from dashboard**
4. Choose camera type:
   - RTSP camera
   - Laptop camera
   - Image/video file
   - Demo source
5. Enter camera name, location, camera ID, and source
6. Click **Save camera profile**
7. Click **Test camera connection** to verify frame capture

## Laptop Camera ON/OFF and Live Preview

From **Control Room**:

- Click **Turn laptop camera ON** to activate `cam_webcam` and start the backend live preview.
- Click **Turn laptop camera OFF** to deactivate it and release the backend camera handle.
- Use **Start browser camera** in the Browser laptop camera permission panel if you want the browser itself to request webcam permission and show a true live client-side preview.
- Use **Start backend live preview** to stream any configured source through the low-latency MJPEG server on port `8765`: laptop camera, RTSP/IP camera, image/video source, or demo.
- Use **Camera Wall → Live camera wall** to keep active cameras refreshing as live MJPEG tiles without Streamlit rerun lag.
- The live stream draws a candidate tracking box:
  - Green: candidate is facing forward.
  - Red: candidate turns face left/right or no face is visible.
  - The browser plays a short beep when the server reports a cheating alert after you click **Enable sound** inside the live tile once.

You can also toggle any configured camera in **Configured cameras**.

## Face-Direction Tracking and Cheating Alert

The backend live stream applies a lightweight real-time face-direction check on every camera frame:

- **Green tracking box**: a frontal face is detected and the candidate is looking forward.
- **Red tracking box**: the face turns sideways, looks down, disappears, or both eyes are not clearly visible (`look_away_uncertain`).
- **Sound alert**: click **Enable sound** inside the live tile once. After that, the browser plays a beep whenever `/status/<camera_id>` reports `cheating_alert: true`.

Live status API:

```bash
curl http://127.0.0.1:8765/status/cam_webcam
```

Example response:

```json
{
  "ok": true,
  "camera_id": "cam_webcam",
  "running": true,
  "face_found": true,
  "direction": "front",
  "cheating_alert": false,
  "yaw_score": 0.0
}
```

The uploaded `camera.py` concept was integrated into `dashboard/live_server.py`: it keeps `cv2.VideoCapture(source)` open continuously, reads frames in a background thread, overlays the tracking/alert state, then streams MJPEG to the dashboard instead of using `cv2.imshow()`.

## Running Detection Without Dashboard

Run one pass against active cameras:

```bash
python main.py --max-frames 1
```

Run against laptop camera:

```bash
python main.py --source 0 --max-frames 1
```

Run against a video or image:

```bash
python main.py --source data/bus.jpg --max-frames 1
python main.py --source /path/to/video.mp4 --max-frames 50
```

Run against an RTSP camera directly:

```bash
python main.py --source "$TAPO_C500_RTSP_URL" --max-frames 5
```

## Outputs

```text
outputs/
├── events/
│   ├── events.jsonl
│   └── events.csv
├── reports/
│   └── summary.json
└── snapshots/
    ├── annotated/
    ├── camera_tests/
    ├── faces/
    └── persons/
```

## Event Schema Example

```json
{
  "event_type": "person_detected",
  "camera_id": "cam_tapo_c500_exam",
  "camera_name": "Tapo C500",
  "camera_location": "Smart AI-Based Exam Monitoring System",
  "frame_id": 1,
  "confidence": 0.86,
  "track_id": 2,
  "bbox_xyxy": [10, 20, 300, 500],
  "person_crop_path": "outputs/snapshots/persons/example.jpg",
  "face_eye": {
    "face_found": true,
    "engine": "opencv-haar"
  },
  "nearby_objects": [],
  "motion": {
    "state": "new_or_unknown",
    "distance_px": 0
  },
  "anomaly": {
    "anomaly_score": 0.0,
    "reasons": []
  },
  "performance": {
    "inference_ms": 444.8
  }
}
```

## Verification Commands

Syntax check:

```bash
python -m py_compile main.py dashboard/app.py dashboard/live_server.py analysis/*.py events/*.py cameras/*.py detectors/*.py utils/*.py
```

Smoke test:

```bash
python main.py --source demo --max-frames 1
```

Sample image test:

```bash
python main.py --source data/bus.jpg --max-frames 1
```

Check dashboard HTTP response:

```bash
curl -I http://127.0.0.1:8501
```

Check MJPEG live server:

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/status/cam_webcam
```

Check that the MJPEG feed returns multipart JPEG frames:

```bash
python - <<'PY'
import urllib.request
u = urllib.request.urlopen('http://127.0.0.1:8765/video_feed/cam_webcam', timeout=5)
b = u.read(2048)
print('multipart:', b.startswith(b'--frame'), 'bytes:', len(b))
PY
```

## Security Notes

- Never commit `.env`
- Never commit real RTSP usernames/passwords
- Do not expose port `8501` to the public internet without authentication
- Do not expose MJPEG port `8765` publicly; it streams live camera frames
- Restrict cloud firewall/security-group access to trusted IPs
- Store real camera credentials in environment variables or secrets manager

## Current Supported Camera Types

- RTSP camera: `rtsp://user:password@ip:554/stream1`
- Laptop camera: `0`
- Image file: `data/bus.jpg`
- Video file: `/path/to/video.mp4`
- Demo frame: `demo`

## Notes for Tapo C500

Known device details used by the local profile:

- Device name: Tapo C500
- Model: C500
- IP: `192.168.1.10`
- Timezone: Africa/Cairo
- Location label: Smart AI-Based Exam Monitoring System
- RTSP source is loaded from `.env` as `TAPO_C500_RTSP_URL`

## Future Improvements

- Authentication for dashboard users
- Role-based invigilator review workflow
- SQLite/PostgreSQL event storage
- Better face/landmark model instead of Haar fallback
- PDF evidence export
- Multi-camera identity matching
- Real exam hall seating map import
- Alerts by Telegram/Discord/email
