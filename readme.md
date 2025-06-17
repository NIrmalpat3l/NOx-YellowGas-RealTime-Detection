# ChimneyVision
ChimneyVision is a real-time video processing application designed to detect yellow gas emissions from chimneys using YOLO models. It processes video feeds from CCTV cameras, annotates the frames with detected chimneys and emissions, and logs events to a MongoDB database.

## Features
- Real-time detection of chimneys and yellow gas emissions
- Annotation of video frames with bounding boxes
- Integration with MongoDB to log emission events
- Web interface for uploading videos and viewing results

## Installation
1. Clone the repository:   

```
git clone https://github.   com/yourusername/   ChimneyVision.git
```


2. Navigate to the project directory:

```
cd ChimneyVision
Install the required packages:
```

3. Install the required packages:

```
pip install -r requirements.txt
Set up your MongoDB connection in a .env file:
```

## Usage

1. Start the Flask application:

```
python app.py
```
2. Access the web interface at http://localhost:5000.
3. Upload a video file to start processing.

## File Descriptions
- *annotation.py* : Contains the annotate_frame function that processes each video frame to detect chimneys and yellow gas emissions.
- *app.py* : The main Flask application that handles video uploads and displays results.
- *data_retrieval.py* : Provides functions to fetch event data from MongoDB.
- *db_utils.py* : Contains utility functions for interacting with the MongoDB database.
- *video_process_threaded.py* : Manages video capture, processing, and display using threading.

## License
This project is licensed under the MIT License.