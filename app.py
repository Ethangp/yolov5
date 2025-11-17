"""
Cat Tracker - Real-time YOLO Object Detection for Cat Monitoring

A Flask web application that:
- Streams live video from an RTSP camera
- Detects cats using YOLOv5 object detection
- Automatically captures snapshots when cats are detected
- Provides a web dashboard for viewing live feed, snapshots, and statistics
- Offers REST API endpoints for programmatic access

Routes:
    /               - Main dashboard (live stream + stats + snapshots)
    /video          - Live video stream (MJPEG)
    /gallery        - Gallery page for managing snapshots
    /stats          - JSON API: statistics and recent events
    /api/events     - JSON API: all cat detection events
    /api/events/<id> - DELETE: remove a specific event
    /captures/<file> - Serve snapshot images
"""
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np

import torch
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    Response,
    jsonify,
    url_for,
    send_from_directory,
)
from datetime import datetime

app = Flask(__name__)

# Global state
cat_count = 0
cat_events = []  # List of detection events: {timestamp, filename, path, count}

# Configuration
CAPTURES_DIR = os.path.join(os.path.dirname(__file__), "captures")
RTSP_URL = "rtsp://hyztigrozc:hyztigrozc@192.168.0.182/live"
MAX_EVENTS = 1000  # Maximum events to keep in memory
TRIM_TO = 500      # Trim to this many when limit reached

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True)
model.eval()
model.conf = 0.6  # Confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
print("Model loaded successfully!") 

def gen():
    """
    Video frame generator for live streaming.
    Processes RTSP stream, runs YOLO detection, and saves snapshots when cats are detected.
    """
    global cat_count, cat_events
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print(f"Error: Unable to open RTSP stream: {RTSP_URL}")
        return
    
    print(f"Connected to RTSP stream: {RTSP_URL}")

    # Read frames in a loop
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from RTSP stream")
            break

        # Encode frame as JPEG bytes
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            print("Error: Failed to encode frame")
            continue

        frame_bytes = buffer.tobytes()

        # Run YOLO on the frame
        img = Image.open(io.BytesIO(frame_bytes))
        results = model(img, size=640)
        results.print()

        # Use pandas output to find cats
        try:
            df = results.pandas().xyxy[0]
            cat_rows = df[df["name"] == "cat"]
        except Exception as e:
            cat_rows = []

        # Render detections (RGB)
        img_rgb = np.squeeze(results.render())

        # Convert back to BGR for OpenCV encoding
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # If any cats were detected, increment counter and save a snapshot
        if len(cat_rows) > 0:
            cat_count += int(len(cat_rows))

            # Ensure captures directory exists
            os.makedirs(CAPTURES_DIR, exist_ok=True)

            now = datetime.now()
            ts_file = now.strftime("%Y%m%d_%H%M%S_%f")  # for filename
            ts_human = now.strftime("%Y-%m-%d %H:%M:%S")  # human readable
            
            snap_filename = f"cat_{ts_file}.jpg"
            snap_path = os.path.join(CAPTURES_DIR, snap_filename)

            # Save the annotated frame
            cv2.imwrite(snap_path, img_bgr)

            cat_events.append({
                "timestamp": ts_human,
                "filename": snap_filename,
                "path": snap_path,
                "count": int(len(cat_rows)),
            })

            # Limit history size to avoid unbounded growth
            if len(cat_events) > MAX_EVENTS:
                cat_events = cat_events[-TRIM_TO:]
                print(f"Trimmed events list to {TRIM_TO} most recent")

        # Encode the rendered frame to JPEG
        ret, buffer = cv2.imencode(".jpg", img_bgr)
        if not ret:
            print("Error: Failed to encode rendered frame")
            continue

        output_bytes = buffer.tobytes()

        # Yield frame to the browser
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + output_bytes + b"\r\n")

    cap.release()
    print("RTSP stream closed")


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard - live stream, stats, and snapshots."""
    return render_template('index.html')


@app.route('/video')
def video():
    """Live video stream endpoint (MJPEG format)."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/captures/<path:filename>")
def captures_file(filename):
    """Serve snapshot images from the captures directory."""
    return send_from_directory(CAPTURES_DIR, filename)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route("/stats")
def stats():
    """Get statistics and recent events (JSON)."""
    last_event = cat_events[-1] if cat_events else None
    return jsonify({
        "cat_count": cat_count,
        "last_event": last_event,
        "events": cat_events[-50:],  # Last 50 events
    })


@app.route("/api/events")
def api_events():
    """Get all cat detection events with full details (JSON)."""
    events = []
    for idx, e in enumerate(cat_events):
        events.append({
            "id": idx,
            "timestamp": e["timestamp"],
            "filename": e["filename"],
            "count": e["count"],
            "url": url_for("captures_file", filename=e["filename"], _external=False),
        })
    return jsonify({
        "cat_count": cat_count,
        "events": events,
    })


@app.route("/api/events/<int:event_id>", methods=["DELETE"])
def delete_event(event_id):
    """Delete a specific event and its snapshot file."""
    global cat_events, cat_count

    if event_id < 0 or event_id >= len(cat_events):
        return jsonify({"error": "invalid event id"}), 404

    ev = cat_events[event_id]

    # Try to remove the file
    try:
        if os.path.exists(ev["path"]):
            os.remove(ev["path"])
            print(f"Deleted snapshot: {ev['filename']}")
    except Exception as e:
        print(f"Failed to delete file {ev['path']}: {e}")

    # Adjust cat_count
    cat_count = max(0, cat_count - ev.get("count", 0))

    # Remove from list
    del cat_events[event_id]

    return jsonify({"status": "ok"})


# ============================================================================
# GALLERY & MANAGEMENT PAGES
# ============================================================================

@app.route("/gallery", methods=["GET"])
def gallery():
    """Gallery page for viewing and managing snapshots."""
    events = list(reversed(cat_events[-200:]))  # Show newest first
    return render_template("gallery.html", events=events, total_cats=cat_count)


@app.route("/delete_snapshot", methods=["POST"])
def delete_snapshot():
    """Delete a snapshot by filename (form-based, redirects to gallery)."""
    global cat_events, cat_count

    filename = request.form.get("filename")
    if not filename:
        return redirect(url_for("gallery"))

    new_events = []
    for event in cat_events:
        if event.get("filename") == filename:
            # Try to delete the file on disk
            path = event.get("path")
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    print(f"Deleted snapshot: {filename}")
            except OSError as e:
                print(f"Error deleting file {path}: {e}")

            # Decrement the count
            cat_count = max(0, cat_count - event.get("count", 0))
        else:
            new_events.append(event)

    cat_events = new_events
    return redirect(url_for("gallery"))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cat Tracker - Real-time YOLO object detection for cat monitoring"
    )
    parser.add_argument(
        "--port", 
        default=5001, 
        type=int, 
        help="Port number to run the server on (default: 5001)"
    )
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Cat Tracker - Starting Server")
    print(f"{'='*60}")
    print(f"Server will be available at: http://0.0.0.0:{args.port}")
    print(f"Dashboard: http://localhost:{args.port}/")
    print(f"Gallery: http://localhost:{args.port}/gallery")
    print(f"API Stats: http://localhost:{args.port}/stats")
    print(f"{'='*60}\n")
    
    app.run(host="0.0.0.0", port=args.port, threaded=True)
