"""
main.py
=======
Entry point for the AI Exam Invigilator system.
Runs Phone Detection (YOLO) and Head Pose / Gaze Estimation (MediaPipe)
on a live webcam, video file, or RTSP stream.

Controls while running:
  [Q] or [ESC]  → Quit
  [S]           → Save screenshot of current frame
  [C]           → Clear the on-screen violation log
  [P]           → Pause / Resume

Usage:
  python main.py                                  # webcam (default)
  python main.py --source 1                       # second webcam
  python main.py --source video.mp4              # video file
  python main.py --source rtsp://192.168.1.5/cam  # IP camera RTSP stream
  python main.py --model n                        # use faster nano model
  python main.py --model l                        # use more accurate large model
  python main.py --no-headpose                    # disable head pose detection
"""

import cv2
import time
import argparse
import sys
import os
import threading
from pathlib import Path
from collections import deque

# Allow importing from the same folder
sys.path.insert(0, str(Path(__file__).parent))

from phone_detector import (
    PhoneDetector,
    ViolationEvent,
    VIOLATION_DURATION_SECONDS,
    CONFIDENCE_THRESHOLD,
    COLOUR_HIGH,
    COLOUR_MEDIUM,
)

from head_pose_detector import (
    HeadPoseDetector,
    HeadPoseViolation,
    EyeTrackingViolation,
    ATTENTION_VIOLATION_SECONDS,
    GAZE_VIOLATION_SECONDS,
    COLOUR_ATTENTIVE,
    COLOUR_WARNING,
    COLOUR_VIOLATION,
)

from config import SharedConfig, ViolationStore
from video_recorder import FrameBuffer, ViolationVideoWriter
import admin_server


# ─────────────────────────────────────────────────────────
#  Argument Parser
# ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Exam Invigilator — Phone & Object Detection"
    )
    parser.add_argument(
        "--source", default=0,
        help="Camera index (0, 1...), video file path, or RTSP URL. Default: 0"
    )
    parser.add_argument(
        "--model", default="m", choices=["n", "s", "m", "l", "x"],
        help="YOLO11 model size. n=fastest, x=most accurate. Default: m"
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Capture width. Default: 1280"
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Capture height. Default: 720"
    )
    parser.add_argument(
        "--no-headpose", action="store_true",
        help="Disable head pose & gaze estimation (only run phone detection)"
    )
    parser.add_argument(
        "--attention-seconds", type=float, default=ATTENTION_VIOLATION_SECONDS,
        help=f"Seconds of inattention before a head-pose violation fires. Default: {ATTENTION_VIOLATION_SECONDS}"
    )
    parser.add_argument(
        "--gaze-seconds", type=float, default=GAZE_VIOLATION_SECONDS,
        help=f"Seconds of gaze deviation before an eye tracking violation fires. Default: {GAZE_VIOLATION_SECONDS}"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
#  FPS Counter (rolling average)
# ─────────────────────────────────────────────────────────

class FPSCounter:
    """
    Computes a smooth rolling-average FPS using the last N frame times.
    A rolling average is more stable than measuring individual frames
    because individual frames vary (some have many detections, some have few).
    """
    def __init__(self, window: int = 30):
        # deque with maxlen automatically discards old entries
        self._times = deque(maxlen=window)
        self._last  = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        avg_frame_time = sum(self._times) / len(self._times)
        return 1.0 / avg_frame_time  # FPS = 1 / seconds_per_frame


# ─────────────────────────────────────────────────────────
#  HUD (Heads-Up Display) Overlay
# ─────────────────────────────────────────────────────────

def draw_hud(
    frame,
    fps: float,
    total_detections: int,
    active_violations: int,
    all_events: list,
    paused: bool,
    num_faces: int = 0,
    num_inattentive: int = 0,
    headpose_enabled: bool = True,
):
    """
    Draws a semi-transparent information panel in the top-left corner.

    Shows: FPS, detection count, violation count, face/attention stats,
    and the recent alert log (merged phone + head-pose alerts).
    Uses cv2.addWeighted() to blend the panel with the frame, creating
    the semi-transparent "glass" effect.
    """
    # ── Panel size ──
    log_entries = min(len(all_events), 5)       # Show last 5 alerts max
    extra_lines = 2 if headpose_enabled else 0  # faces + inattentive rows
    panel_h = 175 + extra_lines * 18 + log_entries * 22
    panel_w = 420

    # ── Draw semi-transparent dark background ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
    # alpha=0.65 means 65% overlay, 35% original frame
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # ── Title ──
    cv2.putText(frame, "AI EXAM INVIGILATOR", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
    subtitle = "Phone & Attention Detection" if headpose_enabled else "Phone & Object Detection"
    cv2.putText(frame, subtitle, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.line(frame, (10, 50), (panel_w - 10, 50), (0, 200, 255), 1)

    # ── Stats ──
    fps_col = (0, 220, 80) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 60, 220)
    cv2.putText(frame, f"FPS:         {fps:5.1f}", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, fps_col, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Detections:  {total_detections}", (10, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    vc = (0, 50, 220) if active_violations > 0 else (0, 200, 80)
    cv2.putText(frame, f"Violations:  {active_violations}", (10, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, vc, 1, cv2.LINE_AA)

    y_cursor = 122

    # ── Head-pose stats (if enabled) ──
    if headpose_enabled:
        fc = (220, 220, 220)
        cv2.putText(frame, f"Faces:       {num_faces}", (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, fc, 1, cv2.LINE_AA)
        y_cursor += 18

        ic = (0, 50, 220) if num_inattentive > 0 else (0, 200, 80)
        cv2.putText(frame, f"Inattentive: {num_inattentive}", (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, ic, 1, cv2.LINE_AA)
        y_cursor += 18

    cv2.putText(frame,
                f"Conf thresh: {CONFIDENCE_THRESHOLD:.0%}   "
                f"Dur thresh: {VIOLATION_DURATION_SECONDS:.1f}s",
                (10, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
    y_cursor += 18

    cv2.putText(frame, "[Q] Quit  [S] Screenshot  [P] Pause  [C] Clear log",
                (10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)
    y_cursor += 18

    if paused:
        cv2.putText(frame, "  PAUSED  ", (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        y_cursor += 20

    # ── Alert log (merged phone + head-pose events) ──
    if all_events:
        cv2.line(frame, (10, y_cursor - 5), (panel_w - 10, y_cursor - 5), (60, 60, 60), 1)
        y_cursor += 10
        cv2.putText(frame, "RECENT ALERTS:", (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1, cv2.LINE_AA)
        y_cursor += 18

        # Show most recent alerts at the top (reversed list, last 5)
        for i, ev in enumerate(reversed(all_events[-5:])):
            t = time.strftime('%H:%M:%S', time.localtime(ev.timestamp))

            # Handle ViolationEvent (phone), HeadPoseViolation, and EyeTrackingViolation
            if isinstance(ev, HeadPoseViolation):
                line = f"  [{t}] {ev.label} HEAD:{ev.head_pose_status} ({ev.duration:.1f}s)"
                col = COLOUR_VIOLATION
            elif isinstance(ev, EyeTrackingViolation):
                line = f"  [{t}] {ev.label} GAZE:{ev.gaze_direction} ({ev.duration:.1f}s)"
                col = COLOUR_VIOLATION
            else:
                line = f"  [{t}] {ev.class_name} #{ev.track_id}  ({ev.duration:.1f}s)"
                col = COLOUR_HIGH if ev.severity == "HIGH" else COLOUR_MEDIUM

            cv2.putText(frame, line, (10, y_cursor + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Open video source ──
    # Convert "0", "1", etc. to integers for webcam indices
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # Keep as string if it's a file path or URL

    print(f"[INFO] Opening source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    # Request resolution (works for webcams; video files use their native size)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolution: {actual_w}x{actual_h}")

    # ── Shared configuration & violation store ──
    shared_config = SharedConfig()
    violation_store = ViolationStore()

    # ── Initialise detectors (with shared config for live threshold updates) ──
    detector    = PhoneDetector(model_size=args.model, config=shared_config)
    fps_counter = FPSCounter()

    # Head pose detector (optional — disabled with --no-headpose)
    headpose_enabled = not args.no_headpose
    head_detector = None
    if headpose_enabled:
        head_detector = HeadPoseDetector(
            violation_seconds=args.attention_seconds,
            gaze_violation_seconds=args.gaze_seconds,
            config=shared_config,
        )
    else:
        print("[INFO] Head pose detection DISABLED (--no-headpose flag).")

    # ── Start admin dashboard server in background ──
    admin_server.init_app(shared_config, violation_store)
    flask_thread = threading.Thread(
        target=admin_server.run_server,
        kwargs={"host": "0.0.0.0", "port": 8080},
        daemon=True,
    )
    flask_thread.start()

    all_events:    list = []   # Growing log of all violations (phone + head pose)
    screenshot_n:  int = 0
    paused:        bool = False

    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("violation_clips", exist_ok=True)
    os.makedirs("violation_crops", exist_ok=True)

    # ── Video recorder for violation clips ──
    frame_buffer = FrameBuffer(max_seconds=3.0, fps_estimate=30.0)
    video_writer = ViolationVideoWriter(frame_buffer, clips_dir="violation_clips")

    print("[INFO] Running. Press Q or ESC to quit.")
    print(f"[INFO] Watching for: Cell Phone, Laptop, Book")
    print(f"[INFO] Phone violation fires after {VIOLATION_DURATION_SECONDS}s continuous detection")
    if headpose_enabled:
        print(f"[INFO] Head pose violation fires after {args.attention_seconds}s continuous inattention")
        print(f"[INFO] Eye tracking violation fires after {args.gaze_seconds}s continuous gaze deviation")
    print()

    while True:

        # ── Pause handling ──
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('p'):
                paused = False
            elif key in (ord('q'), 27):
                break
            continue

        # ── Read frame ──
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended or frame read failed.")
            break

        # ── Run YOLO phone detection ──
        detections, new_events = detector.process(frame)

        # ── Run head pose & gaze estimation ──
        face_attentions = []
        head_events = []
        gaze_events = []
        if head_detector is not None:
            face_attentions, head_events, gaze_events = head_detector.process(frame)

        # ── Handle new phone violation events ──
        for ev in new_events:
            all_events.append(ev)

            # Print to terminal with clear formatting
            print(
                f"\n{'='*55}\n"
                f"  ⚠  PHONE VIOLATION DETECTED\n"
                f"{'='*55}\n"
                f"  Object   : {ev.class_name}\n"
                f"  Track ID : #{ev.track_id}\n"
                f"  Severity : {ev.severity}\n"
                f"  Confidence: {ev.confidence:.0%}\n"
                f"  Duration : {ev.duration:.1f} seconds\n"
                f"  Time     : {time.strftime('%H:%M:%S', time.localtime(ev.timestamp))}\n"
                f"  Location : box={ev.bbox}\n"
                f"{'='*55}"
            )

            # Save a screenshot of the violation
            label = f"{ev.class_name.replace(' ','_')}_track{ev.track_id}"
            image_path = f"violation_crops/phone_{label}_{int(ev.timestamp)}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"  Recording image: {image_path}")
            violation_store.add_phone_violation(ev, media_path=image_path)

        # ── Handle new head pose violation events ──
        for ev in head_events:
            all_events.append(ev)

            print(
                f"\n{'='*55}\n"
                f"  ⚠  HEAD POSE VIOLATION DETECTED\n"
                f"{'='*55}\n"
                f"  Student  : {ev.label}\n"
                f"  Status   : {ev.head_pose_status}\n"
                f"  Duration : {ev.duration:.1f} seconds\n"
                f"  Yaw      : {ev.yaw:.1f}°\n"
                f"  Pitch    : {ev.pitch:.1f}°\n"
                f"  Time     : {time.strftime('%H:%M:%S', time.localtime(ev.timestamp))}\n"
                f"{'='*55}"
            )

            # Record a short video clip of the violation
            video_path, post_recorder = video_writer.record_violation(
                violation_type="headpose",
                label=ev.label.replace(' ', '_'),
                timestamp=ev.timestamp,
            )
            print(f"  Recording clip: {video_path}")
            violation_store.add_head_pose_violation(ev, media_path=video_path)

        # ── Handle new eye tracking violation events ──
        for ev in gaze_events:
            all_events.append(ev)

            print(
                f"\n{'='*55}\n"
                f"  ⚠  EYE TRACKING VIOLATION DETECTED\n"
                f"{'='*55}\n"
                f"  Student  : {ev.label}\n"
                f"  Gaze     : {ev.gaze_direction}\n"
                f"  Duration : {ev.duration:.1f} seconds\n"
                f"  H Ratio  : {ev.gaze_h_ratio:.3f}\n"
                f"  V Ratio  : {ev.gaze_v_ratio:.3f}\n"
                f"  Time     : {time.strftime('%H:%M:%S', time.localtime(ev.timestamp))}\n"
                f"{'='*55}"
            )

            # Record a short video clip of the violation
            video_path, post_recorder = video_writer.record_violation(
                violation_type="gaze",
                label=ev.label.replace(' ', '_'),
                timestamp=ev.timestamp,
            )
            print(f"  Recording clip: {video_path}")
            violation_store.add_eye_tracking_violation(ev, media_path=video_path)

        # ── Push clean frame to the rolling buffer (before annotations) ──
        frame_buffer.push(frame)

        # ── Feed clean frames to any active post-recorders ──
        for recorder in video_writer.get_active_recorders():
            recorder.add_frame(frame)

        # ── Annotate frame (for the live display window only) ──
        frame = detector.draw(frame, detections)
        if head_detector is not None:
            frame = head_detector.draw(frame, face_attentions)

        fps = fps_counter.tick()
        active_violations = sum(1 for d in detections if d.is_violation)
        num_faces = len(face_attentions)
        num_inattentive = sum(1 for fa in face_attentions
                              if fa.attention_status != "ATTENTIVE")

        # Update admin dashboard live stats
        admin_server.update_stats(
            fps=fps,
            faces=num_faces,
            active_violations=active_violations + num_inattentive,
        )

        draw_hud(
            frame, fps, len(detections), active_violations, all_events, paused,
            num_faces=num_faces,
            num_inattentive=num_inattentive,
            headpose_enabled=headpose_enabled,
        )

        # ── Display ──
        window_title = ("AI Exam Invigilator — Phone & Attention Detection"
                        if headpose_enabled
                        else "AI Exam Invigilator — Phone Detection")
        cv2.imshow(window_title, frame)

        # ── Key handling ──
        # cv2.waitKey(1) pauses 1ms and returns the key pressed (-1 if none)
        # & 0xFF masks to the lowest 8 bits (needed on some systems)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):          # Q or ESC → quit
            break
        elif key == ord('s'):              # S → screenshot
            path = f"screenshots/frame_{screenshot_n:04d}.jpg"
            cv2.imwrite(path, frame)
            print(f"[INFO] Screenshot saved: {path}")
            screenshot_n += 1
        elif key == ord('c'):              # C → clear alert log on screen
            all_events.clear()
            print("[INFO] Alert log cleared.")
        elif key == ord('p'):              # P → pause
            paused = True
            print("[INFO] Paused. Press P to resume.")

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()

    # ── End-of-session summary ──
    phone_events = [e for e in all_events if isinstance(e, ViolationEvent)]
    head_events_all = [e for e in all_events if isinstance(e, HeadPoseViolation)]
    gaze_events_all = [e for e in all_events if isinstance(e, EyeTrackingViolation)]

    print(f"\n{'='*55}")
    print(f"  SESSION SUMMARY")
    print(f"{'='*55}")
    print(f"  Total violations: {len(all_events)}")
    print(f"    Phone/Object:   {len(phone_events)}")
    if headpose_enabled:
        print(f"    Head Pose:      {len(head_events_all)}")
        print(f"    Eye Tracking:   {len(gaze_events_all)}")

    if phone_events:
        # Count phone violations by object type
        counts = {}
        for ev in phone_events:
            counts[ev.class_name] = counts.get(ev.class_name, 0) + 1

        print("\n  Phone violations by type:")
        for name, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {name}: {count}")

        print("\n  Phone violation log:")
        for ev in phone_events:
            t = time.strftime('%H:%M:%S', time.localtime(ev.timestamp))
            print(f"    [{t}]  {ev.class_name} #{ev.track_id}  "
                  f"{ev.severity}  conf={ev.confidence:.0%}  "
                  f"duration={ev.duration:.1f}s")

    if head_events_all:
        # Count head-pose violations by status
        attn_counts = {}
        for ev in head_events_all:
            attn_counts[ev.head_pose_status] = attn_counts.get(ev.head_pose_status, 0) + 1

        print("\n  Head pose violations by status:")
        for status, count in sorted(attn_counts.items(), key=lambda x: -x[1]):
            print(f"    {status}: {count}")

        print("\n  Head pose violation log:")
        for ev in head_events_all:
            t = time.strftime('%H:%M:%S', time.localtime(ev.timestamp))
            print(f"    [{t}]  {ev.label}  {ev.head_pose_status}  "
                  f"yaw={ev.yaw:.0f}° pitch={ev.pitch:.0f}°  "
                  f"duration={ev.duration:.1f}s")

    if gaze_events_all:
        # Count eye tracking violations by gaze direction
        gaze_counts = {}
        for ev in gaze_events_all:
            gaze_counts[ev.gaze_direction] = gaze_counts.get(ev.gaze_direction, 0) + 1

        print("\n  Eye tracking violations by gaze direction:")
        for direction, count in sorted(gaze_counts.items(), key=lambda x: -x[1]):
            print(f"    {direction}: {count}")

        print("\n  Eye tracking violation log:")
        for ev in gaze_events_all:
            t = time.strftime('%H:%M:%S', time.localtime(ev.timestamp))
            print(f"    [{t}]  {ev.label}  GAZE:{ev.gaze_direction}  "
                  f"h_ratio={ev.gaze_h_ratio:.3f} v_ratio={ev.gaze_v_ratio:.3f}  "
                  f"duration={ev.duration:.1f}s")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
