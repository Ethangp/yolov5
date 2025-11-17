# Cat Tracker 🐱

A real-time cat detection and monitoring system using YOLOv5 object detection, RTSP camera streaming, and a Flask web dashboard.

## Features

- **Live Video Stream**: Real-time RTSP camera feed with YOLO object detection overlay
- **Automatic Snapshots**: Captures annotated images when cats are detected
- **Web Dashboard**: Single-page interface with live stream, statistics, and snapshot gallery
- **REST API**: JSON endpoints for programmatic access to detection data
- **Gallery Management**: View and delete snapshots through a clean web interface

## Quick Start

### Prerequisites

- Python 3.8+
- RTSP camera (or modify `RTSP_URL` in `app.py` for your camera)
- Required packages (see `requirements.txt`)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd Yolov5-Real-Time-Object-Detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your RTSP camera URL:**
   
   Edit `app.py` and update the `RTSP_URL` variable:
   ```python
   RTSP_URL = "rtsp://username:password@your-camera-ip/live"
   ```

5. **Run the application:**
   ```bash
   python app.py --port 5001
   ```

6. **Access the dashboard:**
   - Local: http://localhost:5001/
   - Network: http://YOUR-IP:5001/ (see "Network Access" below)

## Usage

### Web Interface

- **Dashboard (`/`)**: Main page with live stream, statistics, and snapshots
- **Gallery (`/gallery`)**: View and manage all captured snapshots
- **Live Stream (`/video`)**: Direct MJPEG video stream endpoint

### API Endpoints

- **`GET /stats`**: Get statistics and last 50 events
  ```json
  {
    "cat_count": 42,
    "last_event": {...},
    "events": [...]
  }
  ```

- **`GET /api/events`**: Get all detection events with full details
  ```json
  {
    "cat_count": 42,
    "events": [
      {
        "id": 0,
        "timestamp": "2025-01-17 14:30:25",
        "filename": "cat_20250117_143025_123456.jpg",
        "count": 1,
        "url": "/captures/cat_20250117_143025_123456.jpg"
      }
    ]
  }
  ```

- **`DELETE /api/events/<id>`**: Delete a specific event and its snapshot
  ```bash
  curl -X DELETE http://localhost:5001/api/events/0
  ```

### Network Access (iPhone/Mobile)

To access from your phone or other devices on the same network:

1. **Find your computer's IP address:**
   ```bash
   # macOS/Linux
   ipconfig getifaddr en0
   
   # Windows
   ipconfig
   ```

2. **Ensure the server is running with `host="0.0.0.0"`** (already configured)

3. **Access from your phone:**
   ```
   http://YOUR-COMPUTER-IP:5001/
   ```

4. **If it doesn't work:**
   - Check firewall settings (allow Python/Terminal incoming connections)
   - Ensure both devices are on the same Wi-Fi network
   - Verify the IP address is correct

## Configuration

Edit these variables in `app.py`:

- **`RTSP_URL`**: Your RTSP camera stream URL
- **`CAPTURES_DIR`**: Directory where snapshots are saved (default: `./captures`)
- **`MAX_EVENTS`**: Maximum events to keep in memory (default: 1000)
- **`model.conf`**: Detection confidence threshold (default: 0.6)
- **`model.iou`**: NMS IoU threshold (default: 0.45)

## Project Structure

```
.
├── app.py                 # Main Flask application
├── templates/
│   ├── index.html        # Main dashboard
│   └── gallery.html      # Snapshot gallery page
├── captures/             # Saved snapshots (auto-created)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## How It Works

1. **Video Capture**: Connects to RTSP stream and reads frames
2. **Object Detection**: Runs YOLOv5 on each frame to detect objects
3. **Cat Detection**: Filters detections for "cat" class
4. **Snapshot Capture**: When cats are detected:
   - Saves annotated frame to `captures/` directory
   - Records event with timestamp and metadata
   - Increments total cat count
5. **Web Interface**: Dashboard displays live stream and fetches stats via API

## Troubleshooting

**RTSP stream won't connect:**
- Verify the RTSP URL is correct
- Check camera credentials
- Ensure camera is accessible on your network

**No cats detected:**
- Lower the confidence threshold (`model.conf`)
- Check that YOLO is actually detecting objects (check terminal output)
- Verify camera view includes areas where cats appear

**Snapshots not saving:**
- Check write permissions for the `captures/` directory
- Verify disk space is available

**Can't access from phone:**
- Check firewall settings
- Verify both devices on same network
- Try accessing from computer's browser first to verify server is running

## License

This project uses YOLOv5 from Ultralytics. See their repository for license information.

## Credits

- YOLOv5: https://github.com/ultralytics/yolov5
- Flask: https://flask.palletsprojects.com/
