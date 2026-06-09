
from __future__ import annotations

import json
import subprocess
import sys
import time
import cv2
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import yaml

from utils.config_resolver import resolve_value, safe_source_display

CONFIG_PATH = ROOT / 'config.yaml'
CAMERAS_PATH = ROOT / 'configs/cameras.yaml'
EVENTS_PATH = ROOT / 'outputs/events/events.jsonl'
CSV_PATH = ROOT / 'outputs/events/events.csv'
SUMMARY_PATH = ROOT / 'outputs/reports/summary.json'
SNAP_ROOT = ROOT / 'outputs/snapshots'

st.set_page_config(page_title='ExamVision Command Center', page_icon='▣', layout='wide', initial_sidebar_state='collapsed')

CSS = '''
<style>
:root { --bg:#070a0f; --panel:rgba(16,24,39,.86); --panel-2:rgba(11,18,32,.94); --line:rgba(148,163,184,.20); --line-strong:rgba(148,163,184,.34); --text:#e7eefc; --muted:#8fa2bd; --accent:#38bdf8; --warn:#f59e0b; --danger:#ef4444; --ok:#22c55e; --radius:18px; }
.stApp { background: radial-gradient(circle at 18% 8%, rgba(56,189,248,.12), transparent 34%), radial-gradient(circle at 92% 8%, rgba(34,197,94,.08), transparent 30%), linear-gradient(180deg,#080b12 0%,#05070b 100%); color:var(--text); }
.block-container { padding:1.25rem 1.55rem 2rem; max-width:1600px; }
[data-testid="stHeader"] { background:transparent; } [data-testid="stToolbar"], #MainMenu, footer { display:none; }
.hero { border:1px solid var(--line); border-radius:24px; padding:22px 24px; background:linear-gradient(135deg,rgba(15,23,42,.96),rgba(2,6,23,.88)); box-shadow:0 24px 80px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.05); margin-bottom:18px; }
.hero-title { font-size:34px; line-height:1; font-weight:850; margin:0; color:#f8fbff; }
.hero-sub { color:var(--muted); margin:8px 0 0; font-size:14px; }
.status-pill { display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border:1px solid rgba(34,197,94,.34); border-radius:999px; background:rgba(34,197,94,.10); color:#bbf7d0; font-size:13px; font-weight:700; }
.metric-card { border:1px solid var(--line); border-radius:var(--radius); background:var(--panel); padding:16px; min-height:116px; box-shadow:inset 0 1px 0 rgba(255,255,255,.04); }
.metric-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.12em; }
.metric-value { color:#f8fbff; font-size:33px; font-weight:850; margin-top:8px; line-height:1; }
.metric-note { color:var(--muted); font-size:13px; margin-top:8px; }
.panel { border:1px solid var(--line); border-radius:var(--radius); background:var(--panel-2); padding:16px; box-shadow:inset 0 1px 0 rgba(255,255,255,.04); }
.panel-title { color:#f8fbff; font-weight:780; font-size:16px; margin-bottom:10px; }
.camera-card { border:1px solid var(--line); border-radius:18px; overflow:hidden; background:#050811; min-height:265px; margin-bottom:14px; }
.camera-head { display:flex; justify-content:space-between; align-items:center; padding:10px 12px; border-bottom:1px solid var(--line); color:#dce8fb; }
.camera-feed { aspect-ratio:16/9; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#111827,#020617); position:relative; }
.camera-empty { color:var(--muted); font-size:13px; }
.badge { border:1px solid var(--line-strong); border-radius:999px; padding:4px 9px; font-size:12px; color:#cbd5e1; background:rgba(15,23,42,.72); }
.badge-ok { border-color:rgba(34,197,94,.38); color:#bbf7d0; background:rgba(34,197,94,.10); }
.badge-warn { border-color:rgba(245,158,11,.42); color:#fde68a; background:rgba(245,158,11,.12); }
.badge-danger { border-color:rgba(239,68,68,.42); color:#fecaca; background:rgba(239,68,68,.12); }
.alert-row { border:1px solid var(--line); border-radius:16px; padding:12px; margin-bottom:10px; background:rgba(15,23,42,.70); }
.alert-title { color:#f8fbff; font-weight:760; }
.alert-meta { color:var(--muted); font-size:12px; margin-top:4px; }
.seat-grid { display:grid; grid-template-columns:repeat(8,minmax(0,1fr)); gap:8px; }
.seat { border:1px solid var(--line); border-radius:12px; padding:10px 6px; text-align:center; color:#cbd5e1; background:rgba(15,23,42,.65); font-size:12px; }
.seat-risk { border-color:rgba(239,68,68,.55); background:rgba(239,68,68,.12); color:#fecaca; }
.seat-watch { border-color:rgba(245,158,11,.55); background:rgba(245,158,11,.12); color:#fde68a; }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { background:rgba(15,23,42,.7); border:1px solid var(--line); border-radius:999px; color:#cbd5e1; height:40px; padding:0 16px; }
.stTabs [aria-selected="true"] { border-color:rgba(56,189,248,.55); color:#e0f2fe; background:rgba(56,189,248,.12); }
.stButton>button, .stDownloadButton>button { border-radius:999px; border:1px solid rgba(56,189,248,.38); background:rgba(56,189,248,.13); color:#e0f2fe; font-weight:750; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:16px; overflow:hidden; }
.small-muted { color:var(--muted); font-size:12px; }
hr { border-color:var(--line); }
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding='utf-8')) if path.exists() else {}

def save_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')

def normalize_camera_id(name: str):
    cleaned = ''.join(ch.lower() if ch.isalnum() else '_' for ch in name.strip())
    cleaned = '_'.join([p for p in cleaned.split('_') if p])
    return cleaned or 'camera'

def upsert_camera_profile(profile: dict):
    data = load_yaml(CAMERAS_PATH) or {'cameras': []}
    cams = data.get('cameras', [])
    replaced = False
    for i, cam in enumerate(cams):
        if cam.get('id') == profile.get('id'):
            cams[i] = {**cam, **profile}
            replaced = True
            break
    if not replaced:
        cams.append(profile)
    data['cameras'] = cams
    save_yaml(CAMERAS_PATH, data)
    return data

def set_camera_active(camera_id: str, active: bool):
    data = load_yaml(CAMERAS_PATH) or {'cameras': []}
    for cam in data.get('cameras', []):
        if cam.get('id') == camera_id:
            cam['active'] = bool(active)
    save_yaml(CAMERAS_PATH, data)
    return data

def set_laptop_camera(active: bool):
    profile = {'id': 'cam_webcam', 'name': 'Laptop Camera', 'location': 'Local machine', 'source': 0, 'fps_sample': 5, 'active': bool(active), 'type': 'local'}
    return upsert_camera_profile(profile)

def set_laptop_only():
    data = load_yaml(CAMERAS_PATH) or {'cameras': []}
    found = False
    for cam in data.get('cameras', []):
        if cam.get('id') == 'cam_webcam':
            cam.update({'name': 'Laptop Camera', 'location': 'Local machine', 'source': 0, 'fps_sample': 5, 'active': True, 'type': 'local'})
            found = True
        else:
            cam['active'] = False
    if not found:
        data.setdefault('cameras', []).append({'id': 'cam_webcam', 'name': 'Laptop Camera', 'location': 'Local machine', 'source': 0, 'fps_sample': 5, 'active': True, 'type': 'local'})
    save_yaml(CAMERAS_PATH, data)
    return data

def test_camera_source(source, timeout_sec: float = 6.0):
    cap = cv2.VideoCapture(resolve_value(int(source) if str(source).isdigit() else source))
    start = time.time()
    ok = False
    frame = None
    brightest_frame = None
    brightest_mean = -1.0

    # Laptop webcams often return the first 1-3 frames nearly black while exposure warms up.
    # Do not show the first successful frame blindly; wait for a usable frame.
    while time.time() - start < timeout_sec:
        ok, candidate = cap.read()
        if ok and candidate is not None:
            mean_brightness = float(candidate.mean())
            if mean_brightness > brightest_mean:
                brightest_mean = mean_brightness
                brightest_frame = candidate
            if mean_brightness >= 15.0:
                frame = candidate
                break
        time.sleep(0.15)
    cap.release()

    if frame is None and brightest_frame is not None and brightest_mean >= 5.0:
        frame = brightest_frame
    if frame is None:
        if brightest_frame is not None:
            return {'ok': False, 'message': f'Camera opened but frames are almost black (brightness {brightest_mean:.1f}). Check lens cover, room light, macOS camera permission, or another app using the camera.'}
        return {'ok': False, 'message': 'No frame received from camera'}

    out_dir = ROOT / 'outputs/snapshots/camera_tests'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'test_{int(time.time())}.jpg'
    cv2.imwrite(str(out), frame)
    h, w = frame.shape[:2]
    return {'ok': True, 'message': f'Frame received: {w}x{h} · brightness {float(frame.mean()):.1f}', 'snapshot': str(out)}

def load_events():
    if not EVENTS_PATH.exists():
        return []
    out=[]
    for line in EVENTS_PATH.read_text(encoding='utf-8').splitlines():
        if line.strip():
            try: out.append(json.loads(line))
            except json.JSONDecodeError: pass
    return out

def events_df(events):
    rows=[]
    for i, ev in enumerate(events):
        anomaly=ev.get('anomaly') or {}; motion=ev.get('motion') or {}; face=ev.get('face_eye') or {}
        rows.append({'id':i+1,'time':pd.to_datetime(ev.get('timestamp'),unit='s',errors='coerce'),'camera':ev.get('camera_id'),'hall':ev.get('camera_location') or ev.get('camera_name'),'track_id':ev.get('track_id'),'confidence':ev.get('confidence'),'motion':motion.get('state'),'movement_px':motion.get('distance_px'),'face_visible':bool(face.get('face_found')),'nearby':', '.join([x.get('label','') for x in ev.get('nearby_objects') or []]),'risk':anomaly.get('anomaly_score',0),'reasons':', '.join(anomaly.get('reasons') or []),'snapshot':ev.get('person_crop_path')})
    return pd.DataFrame(rows)

def latest_annotated():
    p=SNAP_ROOT/'annotated'
    return sorted(p.glob('*.jpg')) if p.exists() else []

def risk_label(score: float):
    if score >= .55: return 'Critical','badge-danger'
    if score >= .20: return 'Review','badge-warn'
    return 'Clear','badge-ok'

def metric_card(label, value, note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

cfg=load_yaml(CONFIG_PATH); cams=load_yaml(CAMERAS_PATH).get('cameras',[]); events=load_events(); df=events_df(events); active_cams=[c for c in cams if c.get('active')]
now_label=datetime.now().strftime('%H:%M:%S')
st.markdown(f'<div class="hero"><div style="display:flex;justify-content:space-between;gap:18px;align-items:center;flex-wrap:wrap;"><div><p class="hero-title">ExamVision Command Center</p><p class="hero-sub">AI-assisted exam hall monitoring: cameras, student presence, movement, object proximity, face visibility, and review queue.</p></div><div class="status-pill">System online · Local session · {now_label}</div></div></div>', unsafe_allow_html=True)

c1,c2,c3,c4,c5=st.columns(5)
with c1: metric_card('Active cameras',len(active_cams),f'{len(cams)} configured total')
with c2: metric_card('Person events',len(events),'detections logged')
with c3:
    high_risk=int((df['risk'].fillna(0)>=.20).sum()) if not df.empty else 0
    metric_card('Review queue',high_risk,'events requiring invigilator review')
with c4:
    tracks=df[['camera','track_id']].dropna().drop_duplicates().shape[0] if not df.empty else 0
    metric_card('Unique tracks',tracks,'tracked candidates')
with c5:
    avg_conf=f"{(df['confidence'].dropna().mean()*100):.1f}%" if not df.empty else '0%'
    metric_card('Avg confidence',avg_conf,'YOLO detection confidence')

main_tab,wall_tab,alerts_tab,candidates_tab,control_tab,reports_tab=st.tabs(['Command Center','Camera Wall','Alerts Review','Candidates','Control Room','Reports'])

with main_tab:
    left,right=st.columns([1.35,.65])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Live camera focus</div>', unsafe_allow_html=True)
        imgs=latest_annotated()
        if imgs:
            st.image(str(imgs[-1]), width='stretch')
            st.markdown(f'<div class="small-muted">Latest annotated frame: {imgs[-1].name}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="camera-feed"><div class="camera-empty">No annotated frames yet. Run detection from Control Room.</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Invigilator priority queue</div>', unsafe_allow_html=True)
        if df.empty: st.info('No events yet.')
        else:
            q=df.sort_values(['risk','confidence'],ascending=[False,False]).head(6)
            for _,r in q.iterrows():
                label,klass=risk_label(float(r.get('risk') or 0))
                st.markdown(f'<div class="alert-row"><div style="display:flex;justify-content:space-between;gap:8px;align-items:center;"><div class="alert-title">Candidate track {r.get("track_id")} · {r.get("camera")}</div><span class="badge {klass}">{label}</span></div><div class="alert-meta">Motion: {r.get("motion")} · Face visible: {r.get("face_visible")} · Nearby: {r.get("nearby") or "none"}</div><div class="alert-meta">Reasons: {r.get("reasons") or "clean event"}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.write('')
    h1,h2=st.columns([.9,1.1])
    with h1:
        risk_tracks=set(df[df['risk'].fillna(0)>=.55]['track_id'].dropna().astype(int).tolist()) if not df.empty else set()
        watch_tracks=set(df[(df['risk'].fillna(0)>=.20)&(df['risk'].fillna(0)<.55)]['track_id'].dropna().astype(int).tolist()) if not df.empty else set()
        seats=''.join([f'<div class="seat {"seat-risk" if i in risk_tracks else "seat-watch" if i in watch_tracks else ""}">S{i:02d}</div>' for i in range(1,33)])
        st.markdown(f'<div class="panel"><div class="panel-title">Exam hall seat map</div><div class="seat-grid">{seats}</div></div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="panel"><div class="panel-title">Operational timeline</div>', unsafe_allow_html=True)
        if df.empty: st.caption('No timeline yet.')
        else:
            chart_df=df.copy(); chart_df['minute']=chart_df['time'].dt.floor('min'); timeline=chart_df.groupby('minute').size().rename('events').reset_index()
            st.line_chart(timeline, x='minute', y='events', height=230)
        st.markdown('</div>', unsafe_allow_html=True)

with wall_tab:
    st.markdown('<div class="panel"><div class="panel-title">Multi-camera wall</div>', unsafe_allow_html=True)
    imgs=latest_annotated(); cols=st.columns(3); wall_items=active_cams or cams or [{'id':'camera_1','name':'CLI Source','location':'local'}]
    for i,cam in enumerate(wall_items):
        with cols[i%3]:
            img=imgs[-1] if imgs else None
            st.markdown('<div class="camera-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="camera-head"><strong>{cam.get("id")}</strong><span class="badge badge-ok">online</span></div>', unsafe_allow_html=True)
            if img: st.image(str(img), width='stretch')
            else: st.markdown('<div class="camera-feed"><div class="camera-empty">Waiting for stream</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="padding:10px 12px;color:#8fa2bd;font-size:12px;">{cam.get("name","Camera")} · {cam.get("location","Exam hall")}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with alerts_tab:
    st.markdown('<div class="panel"><div class="panel-title">Alerts review table</div>', unsafe_allow_html=True)
    if df.empty: st.warning('No alerts yet. Run detection first.')
    else:
        fc1,fc2,fc3=st.columns(3)
        with fc1: cam_filter=st.selectbox('Camera',['All']+sorted(df['camera'].dropna().unique().tolist()))
        with fc2: min_risk=st.slider('Minimum risk score',0.0,1.0,0.0,0.05)
        with fc3: face_filter=st.selectbox('Face visibility',['All','Visible','Not visible'])
        fdf=df.copy()
        if cam_filter!='All': fdf=fdf[fdf['camera']==cam_filter]
        fdf=fdf[fdf['risk'].fillna(0)>=min_risk]
        if face_filter=='Visible': fdf=fdf[fdf['face_visible']]
        if face_filter=='Not visible': fdf=fdf[~fdf['face_visible']]
        st.dataframe(fdf.sort_values(['risk','time'],ascending=[False,False]), width='stretch', hide_index=True)
        if len(fdf):
            row_no=st.number_input('Open alert row',min_value=0,max_value=max(0,len(fdf)-1),value=0)
            row=fdf.sort_values(['risk','time'],ascending=[False,False]).iloc[int(row_no)]; sp=row.get('snapshot')
            if sp and Path(sp).exists(): st.image(sp, caption=f'Track {row.get("track_id")} snapshot', width=420)
    st.markdown('</div>', unsafe_allow_html=True)

with candidates_tab:
    st.markdown('<div class="panel"><div class="panel-title">Candidate tracking summary</div>', unsafe_allow_html=True)
    if df.empty: st.info('No candidate tracks yet.')
    else:
        grouped=df.groupby(['camera','track_id'],dropna=True).agg(events=('id','count'),avg_confidence=('confidence','mean'),max_risk=('risk','max'),first_seen=('time','min'),last_seen=('time','max'),motion_states=('motion',lambda s:', '.join(sorted(set(str(x) for x in s.dropna())))),nearby_objects=('nearby',lambda s:', '.join(sorted(set(', '.join(s.dropna()).split(', ')))))).reset_index()
        st.dataframe(grouped.sort_values('max_risk',ascending=False), width='stretch', hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with control_tab:
    st.markdown('<div class="panel"><div class="panel-title">Camera management and detection controls</div>', unsafe_allow_html=True)
    cfg=load_yaml(CONFIG_PATH); cams_data=load_yaml(CAMERAS_PATH) or {'cameras':[]}

    st.subheader('Quick camera controls')
    if st.session_state.get('camera_action_result'):
        result = st.session_state.pop('camera_action_result')
        if result.get('ok'):
            st.success(result.get('message', 'Camera is working'))
            if result.get('snapshot') and Path(result['snapshot']).exists():
                st.image(result['snapshot'], width=420)
        else:
            st.error(result.get('message', 'Camera test failed'))
    q1,q2,q3,q4=st.columns(4)
    with q1:
        if st.button('Turn laptop camera ON'):
            set_laptop_camera(True)
            st.session_state['camera_action_result'] = test_camera_source(0)
            st.rerun()
    with q2:
        if st.button('Turn laptop camera OFF'):
            set_laptop_camera(False)
            st.session_state['camera_action_result'] = {'ok': True, 'message': 'Laptop camera disabled.'}
            st.rerun()
    with q3:
        if st.button('Use laptop only'):
            set_laptop_only()
            st.session_state['camera_action_result'] = test_camera_source(0)
            st.rerun()
    with q4:
        if st.button('Test laptop camera'):
            st.session_state['camera_action_result'] = test_camera_source(0)
            st.rerun()

    t1,t2=st.columns(2)
    with t1:
        if st.button('Enable Tapo C500'):
            upsert_camera_profile({'id':'cam_tapo_c500_exam','name':'Tapo C500','location':'Smart AI-Based Exam Monitoring System','source':'${TAPO_C500_RTSP_URL}','fps_sample':5,'active':True,'type':'rtsp','ip':'192.168.1.10','model':'Tapo C500','timezone':'Africa/Cairo'}); st.success('Tapo C500 enabled.'); st.rerun()
    with t2:
        if st.button('Disable Tapo C500'):
            set_camera_active('cam_tapo_c500_exam', False); st.success('Tapo C500 disabled.'); st.rerun()

    st.divider()
    st.subheader('Add camera from dashboard')
    a1,a2,a3=st.columns([1,1,1])
    with a1:
        cam_name=st.text_input('Camera name', value='Tapo C500')
        cam_location=st.text_input('Location / Hall', value='Smart AI-Based Exam Monitoring System')
    with a2:
        cam_type=st.selectbox('Camera type', ['RTSP camera','Laptop camera','Image/video file','Demo source'])
        cam_id=st.text_input('Camera ID', value=f"cam_{normalize_camera_id(cam_name)}")
    with a3:
        fps_sample=st.number_input('FPS sample', min_value=1, max_value=30, value=5)
        active_new=st.checkbox('Active after save', value=True)

    default_source='${TAPO_C500_RTSP_URL}' if cam_type=='RTSP camera' else '0' if cam_type=='Laptop camera' else 'data/bus.jpg' if cam_type=='Image/video file' else 'demo'
    cam_source=st.text_input('Camera source / RTSP URL / index / file path', value=default_source)

    b1,b2=st.columns(2)
    with b1:
        if st.button('Save camera profile', type='primary'):
            source_value = int(cam_source) if cam_source.isdigit() else cam_source.strip()
            profile={'id':cam_id.strip(),'name':cam_name.strip(),'location':cam_location.strip(),'source':source_value,'fps_sample':int(fps_sample),'active':bool(active_new),'type':cam_type.lower().replace(' ','_')}
            if '192.168.1.10' in str(source_value):
                profile.update({'ip':'192.168.1.10','model':'Tapo C500','timezone':'Africa/Cairo'})
            upsert_camera_profile(profile)
            st.success(f'Saved camera: {profile["id"]}')
            st.rerun()
    with b2:
        if st.button('Test camera connection'):
            with st.spinner('Testing camera source...'):
                result=test_camera_source(cam_source)
            if result.get('ok'):
                st.success(result['message'])
                st.image(result['snapshot'], width=420)
            else:
                st.error(result['message'])

    st.divider()
    st.subheader('Configured cameras')
    cams_data=load_yaml(CAMERAS_PATH) or {'cameras':[]}
    cam_rows=[]
    for c in cams_data.get('cameras',[]):
        safe_source=str(c.get('source'))
        if '@' in safe_source and '://' in safe_source:
            safe_source=safe_source.split('://',1)[0]+'://***@'+safe_source.split('@',1)[1]
        cam_rows.append({**c, 'source': safe_source})
    st.dataframe(pd.DataFrame(cam_rows), width='stretch', hide_index=True)

    toggle_cols=st.columns(3)
    camera_ids=[c.get('id') for c in cams_data.get('cameras',[]) if c.get('id')]
    with toggle_cols[0]:
        selected_cam=st.selectbox('Select camera to toggle', camera_ids if camera_ids else ['none'])
    with toggle_cols[1]:
        if st.button('Activate selected') and selected_cam!='none':
            set_camera_active(selected_cam, True); st.success(f'{selected_cam} activated'); st.rerun()
    with toggle_cols[2]:
        if st.button('Deactivate selected') and selected_cam!='none':
            set_camera_active(selected_cam, False); st.success(f'{selected_cam} deactivated'); st.rerun()

    st.divider()
    st.subheader('Run surveillance pass')
    r1,r2,r3,r4=st.columns(4)
    source_default=''
    with r1: source=st.text_input('Source override for this run',value=source_default,help='Empty = active camera profiles. Use 0 for laptop, RTSP URL, or file path.')
    with r2: max_frames=st.number_input('Max frames',min_value=1,max_value=10000,value=1)
    with r3: conf=st.slider('Confidence threshold',0.05,0.95,float(cfg.get('model',{}).get('conf',.35)),.05)
    with r4: run_mode=st.selectbox('Mode',['single check','short patrol'])
    if st.button('Run exam surveillance pass', type='primary'):
        frames=int(max_frames if run_mode=='single check' else max(max_frames,10)); cmd=[sys.executable,'main.py','--max-frames',str(frames),'--conf',str(conf)]
        if source.strip(): cmd += ['--source',source.strip()]
        with st.spinner('Running YOLO monitoring pass...'):
            proc=subprocess.run(cmd,cwd=str(ROOT),capture_output=True,text=True,timeout=600)
        st.code(proc.stdout + ('\nSTDERR:\n' + proc.stderr if proc.stderr else ''))

    st.divider(); st.subheader('AI modules')
    cfg['tracking']['enabled']=st.checkbox('Tracking enabled',bool(cfg['tracking'].get('enabled',True)))
    cfg['face_eye']['enabled']=st.checkbox('Face visibility analysis',bool(cfg['face_eye'].get('enabled',True)))
    cfg['nearby_objects']['enabled']=st.checkbox('Nearby object analysis',bool(cfg['nearby_objects'].get('enabled',True)))
    cfg['anomaly']['enabled']=st.checkbox('Anomaly scoring',bool(cfg['anomaly'].get('enabled',True)))
    if st.button('Save AI settings'):
        save_yaml(CONFIG_PATH,cfg); st.success('AI settings saved.')
    st.markdown('</div>', unsafe_allow_html=True)

with reports_tab:
    st.markdown('<div class="panel"><div class="panel-title">Reports and evidence export</div>', unsafe_allow_html=True)
    if SUMMARY_PATH.exists(): st.json(json.loads(SUMMARY_PATH.read_text(encoding='utf-8')))
    d1,d2=st.columns(2)
    with d1:
        if CSV_PATH.exists(): st.download_button('Download invigilator CSV',CSV_PATH.read_bytes(),file_name='examvision_events.csv',mime='text/csv')
    with d2:
        if EVENTS_PATH.exists(): st.download_button('Download evidence JSONL',EVENTS_PATH.read_bytes(),file_name='examvision_events.jsonl',mime='application/jsonl')
    st.subheader('System architecture')
    st.code('Exam Cameras\n  -> Frame sampler\n  -> YOLO detection\n  -> ByteTrack identity tracking\n  -> Face visibility check\n  -> Nearby object scan\n  -> Motion and risk scoring\n  -> Evidence log, snapshots, reports\n  -> Invigilator dashboard')
    st.markdown('</div>', unsafe_allow_html=True)
