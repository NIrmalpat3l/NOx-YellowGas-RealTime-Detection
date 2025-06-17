# ChimneyVision

ChimneyVision is a real-time video processing application designed to detect yellow gas emissions from chimneys using Ultralytics YOLOv8 models. It processes video feeds from CCTV cameras or uploaded videos, annotates frames with detected chimneys and emissions, logs events to a MongoDB database, and exposes a web interface and REST endpoints for live viewing and data retrieval.

## Features

- Real-time detection of chimneys and yellow gas emissions using YOLOv8
- Live video annotation with bounding boxes and ROI overlays for each chimney
- Asynchronous event logging to MongoDB immediately upon detecting yellow smoke
- Threaded video processing for smooth, non-blocking performance
- REST API endpoints for video streaming and event data retrieval
- Web interface for uploading videos, viewing the live annotated stream, and browsing logged events

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nirmalpat3l/NOx-YellowGas-RealTime-Detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd NOx-YellowGas-RealTime-Detection
   ```
3. (Optional) Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
4. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Create a `.env` file in the project root and configure your MongoDB connection:
   ```ini
   MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net
   DB_NAME=ChimneyVisionDB
   EVENTS_COLLECTION=emissionEvents
   ```

## Configuration

- **YOLO model path**: Modify the `MODEL_PATH` constant in `app.py` (or set via environment variable) to point to your `bestYolo12Mixedupdated.pt` file.
- **Video source**: By default, use file uploads via the web UI. To connect a live camera feed, update the `VIDEO_SOURCE` in `app.py` or use the `LIVE_FEED_URL` environment variable.
- **Color thresholds**: Adjust the HSV limits in `annotation.py` if you need to detect different shades of emissions (e.g., light orange-yellow). The function `get_limits(color_bgr)` can be customized.

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:5000`.
3. Upload a video file or configure a live feed URL to begin processing.
4. Watch the live annotated video stream and check the Events page for logged emission records.

## API Endpoints

- `GET /video_feed`\
  Returns an MJPEG stream of annotated frames in real time.
- `GET /events`\
  Returns a JSON list of all emission events stored in MongoDB.
- `POST /upload`\
  Accepts video file uploads via the web interface to start processing.

## File Structure

```
ChimneyVision/
├── annotation.py            # Detects chimneys, computes ROI and identifies yellow smoke
├── app.py                   # Flask app defining routes for upload, streaming, and events
├── video_process_threaded.py# Background thread handling video capture, processing, and streaming
├── data_retrieval.py        # Functions to query MongoDB and fetch logged emission events
├── db_utils.py              # MongoDB connection and insert/fetch helper functions
├── requirements.txt         # Python dependencies (Flask, ultralytics, opencv-python, pymongo, python-dotenv, etc.)
├── .env                     # Environment variables (not committed to Git)
└── LICENSE                  # MIT License file
```

## License

This project is licensed under the MIT License.
