"""
video_recorder.py
=================
Rolling frame buffer and violation video clip writer for the AI Exam Invigilator.

How it works:
  1. FrameBuffer stores the last N seconds of full annotated frames in a
     thread-safe circular buffer (collections.deque with maxlen).  The main
     loop pushes every annotated frame into this buffer.

  2. When a violation fires, ViolationVideoWriter:
       a) Snapshots the pre-violation frames from the buffer.
       b) Starts a "post-recording" window — the main loop continues pushing
          frames, and after ~2 seconds the writer collects them.
       c) A background thread encodes all collected frames into an .mp4 file
          using OpenCV's VideoWriter.

  The resulting clips show ~3 seconds before and ~2 seconds after the
  violation, giving invigilators full context of what happened.

Dependencies: opencv-python, numpy
"""

import cv2
import os
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


# ─────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────

# How many seconds of footage to keep before the violation fires.
PRE_VIOLATION_SECONDS = 5.0

# How many seconds to continue recording after the violation fires.
POST_VIOLATION_SECONDS = 2.0

# Output directory for violation clips.
CLIPS_DIR = "violation_clips"

# Video encoding settings.
# We try multiple codecs in order of browser compatibility.
# H.264 (avc1/H264) plays in all browsers; mp4v does NOT.
VIDEO_CODECS_TO_TRY = [
    ("avc1", ".mp4"),   # H.264 — best browser support
    ("H264", ".mp4"),   # H.264 — alternative fourcc
    ("mp4v", ".mp4"),   # MPEG-4 Part 2 — fallback (may not play in browser)
]
VIDEO_FPS = 15.0  # Target FPS for saved clips (lower than live to save space)

# Cache the working codec so we only probe once
_working_codec = None


def _find_working_codec(width: int, height: int) -> tuple[str, str]:
    """
    Probe for a working H.264 codec by trying to create a test VideoWriter.
    Returns (fourcc_string, file_extension) for the first codec that works.
    Caches the result so subsequent calls are instant.
    """
    global _working_codec
    if _working_codec is not None:
        return _working_codec

    import tempfile

    for codec, ext in VIDEO_CODECS_TO_TRY:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            # Write a tiny test file to see if the codec actually works
            test_path = os.path.join(tempfile.gettempdir(), f"_codec_test{ext}")
            writer = cv2.VideoWriter(test_path, fourcc, 15.0, (width, height))
            if writer.isOpened():
                # Write a test frame to make sure it doesn't fail on write
                test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                writer.write(test_frame)
                writer.release()
                # Check the file was actually written with content
                if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                    os.remove(test_path)
                    _working_codec = (codec, ext)
                    print(f"[VideoRecorder] Using codec: {codec} ({ext})")
                    return _working_codec
                if os.path.exists(test_path):
                    os.remove(test_path)
            else:
                writer.release()
        except Exception:
            pass

    # Ultimate fallback — mp4v should always work even if browser can't play it
    _working_codec = ("mp4v", ".mp4")
    print(f"[VideoRecorder] WARNING: No H.264 codec found, using mp4v (videos may not play in browser)")
    return _working_codec


# ─────────────────────────────────────────────────────────
#  Frame Buffer Entry
# ─────────────────────────────────────────────────────────

@dataclass
class BufferedFrame:
    """A single frame stored in the rolling buffer."""
    frame: np.ndarray   # Full annotated BGR frame
    timestamp: float     # time.time() when the frame was captured


# ─────────────────────────────────────────────────────────
#  FrameBuffer — Rolling circular buffer
# ─────────────────────────────────────────────────────────

class FrameBuffer:
    """
    Thread-safe rolling buffer that stores the last N seconds of frames.

    The main detection loop calls push() every frame.  When a violation
    fires, snapshot() returns all buffered frames (the "pre" footage).

    Usage:
        buf = FrameBuffer(max_seconds=3.0, fps_estimate=30)
        buf.push(frame)                 # every frame
        pre_frames = buf.snapshot()     # when violation fires
    """

    def __init__(self, max_seconds: float = PRE_VIOLATION_SECONDS,
                 fps_estimate: float = 30.0):
        # Estimate how many frames to buffer.
        # We over-estimate slightly to account for FPS fluctuations.
        max_frames = int(max_seconds * fps_estimate * 1.2)
        self._buffer: deque[BufferedFrame] = deque(maxlen=max(max_frames, 30))
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray):
        """
        Add a frame to the buffer.  Old frames are automatically
        discarded when the buffer is full (deque maxlen handles this).

        We store a .copy() so the buffer owns its own memory and
        the caller can safely overwrite the original frame.
        """
        entry = BufferedFrame(
            frame=frame.copy(),
            timestamp=time.time(),
        )
        with self._lock:
            self._buffer.append(entry)

    def snapshot(self) -> list[BufferedFrame]:
        """
        Return a copy of all currently buffered frames.
        Called when a violation fires to capture the "pre" footage.
        """
        with self._lock:
            return list(self._buffer)

    def clear(self):
        """Clear the buffer (e.g. when pausing)."""
        with self._lock:
            self._buffer.clear()


# ─────────────────────────────────────────────────────────
#  ViolationVideoWriter
# ─────────────────────────────────────────────────────────

class ViolationVideoWriter:
    """
    Records short video clips of violations.

    When a violation fires, call record_violation().  This:
      1) Snapshots pre-violation frames from the FrameBuffer.
      2) Returns a PostRecorder object that the main loop feeds
         additional frames to for POST_VIOLATION_SECONDS.
      3) When the post-recording window ends, a background thread
         encodes the full clip to an .mp4 file.

    The video path is returned synchronously so it can be stored
    in the database immediately (the file is written asynchronously).

    Usage:
        writer = ViolationVideoWriter(frame_buffer)
        video_path, post_recorder = writer.record_violation("phone", "track5")

        # In the main loop, for the next ~2 seconds:
        if post_recorder and post_recorder.is_recording():
            post_recorder.add_frame(annotated_frame)
    """

    def __init__(self, frame_buffer: FrameBuffer,
                 clips_dir: str = CLIPS_DIR):
        self.frame_buffer = frame_buffer
        self.clips_dir = clips_dir
        os.makedirs(clips_dir, exist_ok=True)

        # Track active post-recorders to avoid duplicate recordings
        # for the same violation within a short time window.
        self._active_recorders: list['PostRecorder'] = []
        self._lock = threading.Lock()

    def record_violation(
        self,
        violation_type: str,
        label: str,
        timestamp: float = None,
    ) -> tuple[str, 'PostRecorder']:
        """
        Start recording a violation clip.

        Args:
            violation_type: "phone", "headpose", "gaze" etc.
            label:          Identifier (e.g. "Cell_Phone_track5", "Person_1")
            timestamp:      Violation timestamp (defaults to now).

        Returns:
            (video_path, post_recorder):
                video_path    — The path where the .mp4 will be written.
                post_recorder — Feed this additional frames for post-violation footage.
        """
        if timestamp is None:
            timestamp = time.time()

        # Generate filename
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        safe_label = label.replace(" ", "_").replace("#", "")
        filename = f"{violation_type}_{safe_label}_{time_str}.mp4"
        video_path = os.path.join(self.clips_dir, filename)

        # Snapshot pre-violation frames
        pre_frames = self.frame_buffer.snapshot()

        # Create post-recorder
        post_recorder = PostRecorder(
            pre_frames=pre_frames,
            video_path=video_path,
            post_seconds=POST_VIOLATION_SECONDS,
        )

        with self._lock:
            # Clean up finished recorders
            self._active_recorders = [
                r for r in self._active_recorders if r.is_recording()
            ]
            self._active_recorders.append(post_recorder)

        return video_path, post_recorder

    def get_active_recorders(self) -> list['PostRecorder']:
        """Return all post-recorders that are still collecting frames."""
        with self._lock:
            self._active_recorders = [
                r for r in self._active_recorders if r.is_recording()
            ]
            return list(self._active_recorders)


# ─────────────────────────────────────────────────────────
#  PostRecorder — Collects post-violation frames then writes video
# ─────────────────────────────────────────────────────────

class PostRecorder:
    """
    Collects frames for a short window after a violation fires,
    then writes the complete clip (pre + post frames) to disk
    in a background thread.

    The main loop should call add_frame() every frame while
    is_recording() returns True.  Once the post window expires,
    the recorder automatically starts encoding.
    """

    def __init__(
        self,
        pre_frames: list[BufferedFrame],
        video_path: str,
        post_seconds: float = POST_VIOLATION_SECONDS,
    ):
        self._pre_frames = pre_frames
        self._post_frames: list[BufferedFrame] = []
        self._video_path = video_path
        self._post_seconds = post_seconds
        self._start_time = time.time()
        self._recording = True
        self._encoding = False
        self._lock = threading.Lock()

    def is_recording(self) -> bool:
        """True while still collecting post-violation frames."""
        return self._recording

    def add_frame(self, frame: np.ndarray):
        """
        Add a post-violation frame.  Automatically stops recording
        and triggers encoding when the post window expires.
        """
        if not self._recording:
            return

        elapsed = time.time() - self._start_time

        with self._lock:
            self._post_frames.append(BufferedFrame(
                frame=frame.copy(),
                timestamp=time.time(),
            ))

        if elapsed >= self._post_seconds:
            self._recording = False
            self._start_encoding()

    def _start_encoding(self):
        """Spawn a background thread to encode the video."""
        if self._encoding:
            return
        self._encoding = True

        thread = threading.Thread(
            target=self._encode_video,
            daemon=True,
        )
        thread.start()

    def _encode_video(self):
        """
        Encode all collected frames (pre + post) into a video file.
        Tries H.264 first for browser compatibility, with fallbacks.
        Runs in a background thread so it doesn't block the main loop.
        """
        all_frames = self._pre_frames + self._post_frames

        if not all_frames:
            print(f"[VideoRecorder] No frames to encode for {self._video_path}")
            return

        # Determine frame dimensions from the first frame
        h, w = all_frames[0].frame.shape[:2]

        # Calculate actual FPS from frame timestamps
        if len(all_frames) >= 2:
            total_time = all_frames[-1].timestamp - all_frames[0].timestamp
            if total_time > 0:
                actual_fps = len(all_frames) / total_time
            else:
                actual_fps = VIDEO_FPS
        else:
            actual_fps = VIDEO_FPS

        # Cap FPS to a reasonable range for the output file
        output_fps = min(max(actual_fps, 10.0), 30.0)

        # Find a working codec (probes once, then caches)
        codec_str, ext = _find_working_codec(w, h)

        # Update file extension if needed (e.g. if codec probe chose .avi)
        if not self._video_path.endswith(ext):
            self._video_path = os.path.splitext(self._video_path)[0] + ext

        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        writer = cv2.VideoWriter(self._video_path, fourcc, output_fps, (w, h))

        if not writer.isOpened():
            print(f"[VideoRecorder] ERROR: Cannot create video writer for {self._video_path}")
            return

        try:
            for bf in all_frames:
                # Ensure frame matches expected dimensions
                fh, fw = bf.frame.shape[:2]
                if fh != h or fw != w:
                    resized = cv2.resize(bf.frame, (w, h))
                    writer.write(resized)
                else:
                    writer.write(bf.frame)
        finally:
            writer.release()

        duration = len(all_frames) / output_fps
        size_mb = os.path.getsize(self._video_path) / (1024 * 1024)
        print(f"[VideoRecorder] Clip saved: {self._video_path} "
              f"({len(all_frames)} frames, {duration:.1f}s, "
              f"{output_fps:.0f}fps, {size_mb:.1f}MB, codec={codec_str})")

