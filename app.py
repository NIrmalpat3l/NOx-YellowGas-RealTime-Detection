# app.py

import os
import uuid
import time
from threading import Thread, Event
from queue import Queue, Empty
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename

import cv2
from ultralytics import YOLO

from annotation import annotate_frame
from camera_motion_detector import CameraMotionDetector
from annotate_video import SimpleTracker     # your tracker from annotate_video.py
from yellow_event_logger import YellowGasEventLogger
from db_utils import get_db_collection

# ─── Configuration ─────────────────────────────────────────────────

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT   = {"mp4"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# session_id → stop flag
processors   = {}
# session_id → frame queue
frame_queues = {}

# one global logger for all sessions (uses chimney IDs internally)
logger = YellowGasEventLogger()

# ─── Helpers ────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXT

def process_video(session_id, filepath):
    """
    Background worker: reads video at 1 FPS, runs motion→detect→track→log→annotate,
    and pushes JPEG bytes into a per-session queue.
    """
    model_path = os.getenv("YOLO_MODEL_PATH")
    model = YOLO(model_path)
    tracker = SimpleTracker()
    motion  = CameraMotionDetector()
    cap     = cv2.VideoCapture(filepath)

    q = frame_queues[session_id]
    last_ms    = -1000
    pipeline_start = time.time()

    while not processors[session_id].is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # 1) check camera motion
        if motion.is_camera_moved(frame):
            # reset tracker & logger on camera shift
            tracker = SimpleTracker()
            logger.close_all(timestamp=time.time() - pipeline_start)
            continue

        # 2) throttle to ~1 FPS
        now_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if now_ms - last_ms < 1000:
            continue
        last_ms = now_ms

        # 3) detect + ROI + yellow
        boxes, rois, yellow_flags = annotate_frame(frame, model_path)

        # 4) track to assign persistent IDs
        tracks = tracker.update(boxes)
        det_to_tid = { tuple(v["box"]): tid for tid, v in tracks.items() }

        # 5) log to MongoDB
        yellow_map = {}
        elapsed = time.time() - pipeline_start
        for idx, flag in enumerate(yellow_flags):
            tid = det_to_tid.get(tuple(boxes[idx]), idx+1)
            yellow_map[tid] = flag
        # logger.update(yellow_map, timestamp=elapsed)

        try:
            logger.update(yellow_map, timestamp=elapsed)
        except Exception as e:
            # ensure a DB error doesn’t kill the streaming thread
            print(f"[LOGGER ERROR] {e}")

        # 6) draw annotations
        canvas = frame.copy()
        for idx, box in enumerate(boxes):
            x1,y1,x2,y2 = box
            sx1,sy1,sx2,sy2 = rois[idx]
            tid = det_to_tid.get(tuple(box), idx+1)
            is_y = yellow_flags[idx]

            box_col = (255,0,0) if is_y else (0,255,255)
            cv2.rectangle(canvas, (x1,y1),(x2,y2), box_col, 2)
            cv2.putText(canvas, f"Chimney {tid}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_col, 2)

            roi_col = (0,0,255) if is_y else (0,255,0)
            cv2.rectangle(canvas, (sx1,sy1),(sx2,sy2), roi_col, 2)
            cv2.putText(canvas, f"SmokeROI {tid}", (sx1,sy1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_col, 2)

        # 7) encode & queue for MJPEG stream
        success, jpg = cv2.imencode(".jpg", canvas)
        if success:
            b = jpg.tobytes()
            if q.full():
                try: q.get_nowait()
                except: pass
            q.put(b)

    # cleanup when video ends or stop flag set
    cap.release()
    logger.close_all(timestamp=time.time() - pipeline_start)
    processors.pop(session_id, None)
    frame_queues.pop(session_id, None)
    try: os.remove(filepath)
    except: pass

# ─── Endpoints ──────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or not allowed_file(file.filename):
        return jsonify(error="Invalid file"), 400

    session_id = uuid.uuid4().hex
    filename   = secure_filename(f"{session_id}.mp4")
    path       = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # spawn processing thread
    stop_evt = Event()
    processors[session_id]   = stop_evt
    frame_queues[session_id] = Queue(maxsize=2)
    Thread(target=process_video, args=(session_id, path), daemon=True).start()

    return jsonify(session_id=session_id)

@app.route('/video_feed/<session_id>')
def video_feed(session_id):
    """
    MJPEG stream of annotated frames for this session.
    """
    if session_id not in frame_queues:
        return "Session not found", 404

    def gen():
        q = frame_queues[session_id]
        while True:
            try:
                jpg = q.get(timeout=5)
            except Empty:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/summary")
def summary():
    days = int(request.args.get("last_days", 30))
    coll = get_db_collection().database["yellow_gas_summary"]
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    docs = coll.find({"day": {"$gte": cutoff}}).sort([("day", -1), ("chimney_number", 1)])
    out = [
        {"chimney_number": d["chimney_number"], "day": d["day"], "total_duration": d["total_duration"]}
        for d in docs
    ]
    return jsonify(out)

# serve your three static HTML pages
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path and os.path.exists(os.path.join("static", path)):
        return send_from_directory("static", path)
    return send_from_directory("static", "index.html")

# ─── Run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"▶️  Starting NOx Flask server on http://0.0.0.0:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)