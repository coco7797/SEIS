# AI Exam Invigilator

An AI-powered proctoring system that uses a webcam or an IP camera stream to monitor students during exams. The system detects suspicious activity such as mobile phone usage, looking away from the screen for extended periods (inattention), and abnormal gaze deviation.

Features a live HUD (Heads-Up Display), a local Flask-based admin dashboard for invigilators to review logged violations along with recorded video clips, and a **post-exam AI review** powered by Qwen2.5-VL to verify detected violations and filter out false positives.

## Features

- **Phone & Object Detection**: Uses RF-DETR (Real-time Detection Transformer) with supervision's ByteTrack to identify cell phones, laptops, and books with configurable duration thresholds.
- **Head Pose & Attention Tracking**: Uses MediaPipe to track face landmarks and estimate yaw/pitch to ensure the student is looking at the screen.
- **Eye Gaze Tracking**: Monitors the direction of the student's gaze to identify if they are constantly looking away.
- **Auto Video Clip Recording**: Automatically records a 3-second video clip when a violation is detected.
- **Qwen2.5-VL Post-Exam Review**: After the exam session ends, an AI reviewer powered by the Qwen2.5-VL vision-language model (running locally via Ollama) analyzes each violation's recorded video clip or image. It classifies each as **CONFIRMED**, **FALSE_POSITIVE**, or **INCONCLUSIVE** with written reasoning, significantly reducing false alarms before results reach the invigilator.
- **Admin Web Dashboard**: A Flask server running concurrently allows an admin to:
  - View live statistics (FPS, faces tracked, active violations).
  - Review the violation history along with the recorded video evidence.
  - See the AI reviewer's verdict and reasoning for each violation.
  - Tweak sensitivity and thresholds dynamically.

## Prerequisites

- Python 3.9+ recommended.
- A webcam (or an RTSP stream).
- You will need the MediaPipe Face Landmarker model. Download the `face_landmarker.task` file from the [official MediaPipe models page](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) and place it in the root directory.
- The RF-DETR model weights will automatically download upon the first run.
- **(Optional — for AI Review)** [Ollama](https://ollama.com/) installed and running, with the Qwen2.5-VL model pulled:
  ```bash
  ollama pull qwen2.5vl:7b
  ```

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

4. **(Optional) Set up Ollama for AI Review**:
   Install [Ollama](https://ollama.com/), start the server, and pull the model:
   ```bash
   ollama serve          # Start the Ollama server (runs on localhost:11434)
   ollama pull qwen2.5vl:7b   # Download the Qwen2.5-VL 7B model (~4.7 GB)
   ```

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

## Qwen2.5-VL AI Reviewer

The **Qwen Reviewer** (`qwen_reviewer.py`) is a post-exam verification layer that uses the Qwen2.5-VL vision-language model running locally through Ollama to double-check every detected violation.

### How It Works

1. After the exam session ends, the reviewer iterates through all violations that have recorded media (video clips or images) but no LLM verdict yet.
2. For each violation, it extracts key frames from the video clip (or uses the image directly), resizes them to fit within the model's context window, and sends them along with a violation-type-specific prompt.
3. The model analyzes the visual evidence and returns a structured verdict:
   - **CONFIRMED** — The violation appears genuine.
   - **FALSE_POSITIVE** — The detection was likely a false alarm (e.g., a pencil case misidentified as a phone).
   - **INCONCLUSIVE** — Not enough evidence to decide.
4. The verdict and reasoning are written back to the violations database and displayed in the admin dashboard.

### Key Design Decisions

- **Post-exam only**: The reviewer runs after the exam ends, so it never competes with the real-time detectors (RF-DETR / MediaPipe) for GPU time or FPS.
- **Fully local**: All inference happens on-device via Ollama — no data leaves the machine.
- **Graceful fallback**: If Ollama is not running or the model is not available, the system still works normally; violations simply won't have an AI verdict attached.

### Enabling the Reviewer

The AI review can be toggled via the `qwen_review_enabled` setting in the admin dashboard or in `settings.json`. It is disabled by default.

## Structure

- `main.py`: The entry point script that connects the detectors and runs the video loop.
- `phone_detector.py`: RF-DETR-based detection logic with ByteTrack tracking.
- `head_pose_detector.py`: MediaPipe-based facial tracking and gaze estimation.
- `video_recorder.py`: Handles saving video clips of violations.
- `qwen_reviewer.py`: Post-exam AI verification of violations using Qwen2.5-VL via Ollama.
- `admin_server.py`: Flask application for the dashboard.
- `config.py`: Centralized configuration and SQLite-backed violation storage.

## License

This project is for educational and evaluation purposes.
