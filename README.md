# AI Exam Invigilator

An AI-powered proctoring system that uses a webcam or an IP camera stream to monitor students during exams. The system detects suspicious activity such as mobile phone usage, looking away from the screen for extended periods (inattention), and abnormal gaze deviation.

Features a live HUD (Heads-Up Display) and a local Flask-based admin dashboard for invigilators to review logged violations along with recorded video clips.

## Features

- **Phone & Object Detection**: Uses YOLO11 (via `ultralytics`) to identify cell phones, laptops, and books with configurable duration thresholds.
- **Head Pose & Attention Tracking**: Uses MediaPipe to track face landmarks and estimate yaw/pitch to ensure the student is looking at the screen.
- **Eye Gaze Tracking**: Monitors the direction of the student's gaze to identify if they are constantly looking away.
- **Auto Video Clip Recording**: Automatically records a 3-second video clip when a violation is detected.
- **Admin Web Dashboard**: A Flask server running concurrently allows an admin to:
  - View live statistics (FPS, faces tracked, active violations).
  - Review the violation history along with the recorded video evidence.
  - Tweak sensitivity and thresholds dynamically.

## Prerequisites

- Python 3.9+ recommended.
- A webcam (or an RTSP stream).
- You will need the MediaPipe Face Landmarker model. Download the `face_landmarker.task` file from the [official MediaPipe models page](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) and place it in the root directory.
- The YOLO weights (`yolo11m.pt`) will automatically download upon the first run.

## Installation

1. **Clone the repository** (or copy the files):
   ```bash
   git clone <repository_url>
   cd "SEIS 2"
   ```

2. **Install the dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Ensure the MediaPipe model is present**:
   Make sure `face_landmarker.task` is in the same directory as `main.py`.

## Usage

Start the invigilator program using `main.py`.

```bash
# Default (uses webcam 0)
python main.py

# Use a specific webcam index
python main.py --source 1

# Use a pre-recorded video file
python main.py --source path/to/video.mp4

# Use an IP Camera RTSP stream
python main.py --source rtsp://username:password@192.168.1.100/stream

# Use a faster (nano) YOLO model instead of medium
python main.py --model n

# Disable head pose tracking (only phone detection)
python main.py --no-headpose
```

### In-App Controls
While the application is running, select the video window and use the following keys:
- **`[Q]` or `[ESC]`**: Quit the application
- **`[S]`**: Save a screenshot of the current frame
- **`[P]`**: Pause or resume the feed
- **`[C]`**: Clear the on-screen alert log

## Admin Dashboard

Once `main.py` is running, the admin dashboard server starts automatically in the background. 
Open your web browser and navigate to:

```
http://localhost:8080
```

From the dashboard, you can view all registered violations, watch the associated video clips, and adjust the violation thresholds (e.g., confidence threshold, duration before firing an alert).

## Structure

- `main.py`: The entry point script that connects the detectors and runs the video loop.
- `phone_detector.py`: YOLO-based detection logic.
- `head_pose_detector.py`: MediaPipe-based facial tracking and gaze estimation.
- `video_recorder.py`: Handles saving video clips of violations.
- `admin_server.py`: Flask application for the dashboard.
- `config.py`: Centralized configuration shared between the detection loop and the web server.

## License

This project is for educational and evaluation purposes.
