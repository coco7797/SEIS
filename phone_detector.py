"""
phone_detector.py
=================
Phone (and prohibited object) detection module for the AI Exam Invigilator System.

How it works:
  1. RF-DETR (Real-time Detection Transformer) is a deep learning model
     pre-trained on the COCO dataset — a massive collection of 330,000
     real-world images covering 80 object classes.
     "Cell phone" (COCO category ID 77) is one of those 80 classes, so
     RF-DETR already knows what phones look like WITHOUT any training.

  2. Every frame from the camera is passed through RF-DETR. It analyses the
     entire image and returns bounding boxes around every object it finds,
     along with a confidence score (0.0 to 1.0) saying how sure it is.

  3. ByteTrack (via the supervision library) is a tracking algorithm that
     runs on top of RF-DETR's detections. RF-DETR by itself re-detects
     everything from scratch each frame — it has no memory. ByteTrack gives
     each detected object a persistent ID (tracker_id) that stays consistent
     across frames. So if a student picks up a phone, that phone gets ID #5
     and keeps that ID until it disappears.

  4. A ViolationTracker watches each tracked phone. If a phone stays visible
     for more than N seconds, it fires a ViolationEvent. This avoids false
     positives from momentary glints or brief reflections being detected
     as phones.

COCO classes we detect (original COCO category IDs, NOT sequential):
  - 77: cell phone
  - 84: book        (unauthorised notes)
  - 73: laptop      (unauthorised device)

Dependencies: rfdetr, supervision, opencv-python, numpy
"""

import cv2
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

import supervision as sv
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES


# ─────────────────────────────────────────────────────────
#  COCO Class Configuration
# ─────────────────────────────────────────────────────────

# COCO dataset class IDs that we care about in an exam hall.
# IMPORTANT: RF-DETR uses ORIGINAL COCO category IDs (1-90, with gaps),
# NOT the sequential 0-79 indices that YOLO uses.
# Full mapping: rfdetr.assets.coco_classes.COCO_CLASSES
# Each entry is:  class_id : (display_name, severity)
# severity: "HIGH" = alert invigilator immediately
#           "MEDIUM" = log and flag for review
WATCHED_CLASSES = {
    77: ("Cell Phone",  "HIGH"),
    84: ("Book",        "MEDIUM"),
    73: ("Laptop",      "HIGH"),
}

# How many seconds a detected object must be CONTINUOUSLY visible
# before we fire a violation. Prevents false positives.
VIOLATION_DURATION_SECONDS = 2.0

# Minimum confidence to consider a detection real.
# 0.5 = must be at least 50% sure. Lower = more detections but more noise.
CONFIDENCE_THRESHOLD = 0.50

# Colours for drawing — BGR format (Blue, Green, Red)
COLOUR_HIGH   = (0,   50,  220)  # Red   — HIGH severity
COLOUR_MEDIUM = (0,  165,  255)  # Orange — MEDIUM severity
COLOUR_OK     = (0,  200,   80)  # Green  — no violations



# ─────────────────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────────────────

@dataclass
class Detection:
    """
    Represents one detected object in one frame.
    Think of this as a single row in a detection results table.
    """
    track_id:    int       # Persistent ID assigned by ByteTrack (stays same across frames)
    class_id:    int       # COCO class number (e.g. 67 for phone)
    class_name:  str       # Human-readable name (e.g. "Cell Phone")
    confidence:  float     # How sure the model is, 0.0–1.0
    severity:    str       # "HIGH" or "MEDIUM"
    bbox:        tuple     # (x1, y1, x2, y2) pixel coordinates of the box
    is_violation: bool     # Has this object been visible long enough to be a violation?


@dataclass
class ViolationEvent:
    """
    Fired when a prohibited object has been visible for VIOLATION_DURATION_SECONDS.
    This is what gets sent to the dashboard / invigilator.
    """
    track_id:   int
    class_name: str
    severity:   str
    confidence: float
    duration:   float    # How long it has been visible (seconds)
    bbox:       tuple    # Where it was in the frame when violation fired
    timestamp:  float    # Unix timestamp (time.time())
    frame_crop: Optional[np.ndarray] = None  # Cropped image of the object


# ─────────────────────────────────────────────────────────
#  Violation Tracker
# ─────────────────────────────────────────────────────────

class ViolationTracker:
    """
    Keeps a timer for every tracked object.

    Each object gets a track_id from ByteTrack. This class watches how long
    each track_id has been continuously visible. Once it exceeds
    VIOLATION_DURATION_SECONDS, a ViolationEvent is returned.

    The timer resets if the object disappears from the frame (track_id gone).
    This is how we handle someone briefly raising a phone then putting it down.
    """

    def __init__(self):
        # Maps  track_id → time when we first saw this object
        self._first_seen:  dict[int, float] = {}

        # Maps  track_id → True/False whether we already fired a violation for it.
        # We only fire ONCE per continuous episode, not every frame.
        self._fired: dict[int, bool] = {}

        # Set of track_ids we saw in the PREVIOUS frame.
        # If a track_id disappears between frames, we reset its timer.
        self._previous_track_ids: set[int] = set()

    def update(
        self,
        current_detections: list[Detection],
        violation_threshold: float = VIOLATION_DURATION_SECONDS,
    ) -> list[ViolationEvent]:
        """
        Call this every frame with the list of current detections.

        Returns a list of ViolationEvents for any objects that just
        crossed the duration threshold. Usually returns an empty list.

        Args:
            current_detections: All Detection objects found this frame.

        Returns:
            List of new ViolationEvent objects (can be empty).
        """
        now = time.time()
        current_track_ids = {d.track_id for d in current_detections}

        # ── Find disappeared objects and reset their timers ──
        # Set difference: IDs that were there before but aren't now
        disappeared = self._previous_track_ids - current_track_ids
        for gone_id in disappeared:
            # Object left the frame — reset everything for this ID
            self._first_seen.pop(gone_id, None)
            self._fired.pop(gone_id, None)

        # ── Check each current detection ──
        new_violations = []
        for det in current_detections:
            tid = det.track_id

            # First time we're seeing this track_id? Record when.
            if tid not in self._first_seen:
                self._first_seen[tid] = now
                self._fired[tid]      = False

            elapsed = now - self._first_seen[tid]

            # Mark the detection with whether it's currently a violation
            det.is_violation = elapsed >= violation_threshold

            # Has it crossed the threshold AND we haven't fired for it yet?
            if elapsed >= violation_threshold and not self._fired[tid]:
                self._fired[tid] = True   # Only fire once per episode

                event = ViolationEvent(
                    track_id=tid,
                    class_name=det.class_name,
                    severity=det.severity,
                    confidence=round(det.confidence, 2),
                    duration=round(elapsed, 1),
                    bbox=det.bbox,
                    timestamp=now,
                    frame_crop=None,  # filled in by PhoneDetector.process()
                )
                new_violations.append(event)

        # Remember this frame's IDs for next frame's comparison
        self._previous_track_ids = current_track_ids
        return new_violations

    def get_duration(self, track_id: int) -> float:
        """How long (seconds) has this track_id been continuously visible?"""
        if track_id not in self._first_seen:
            return 0.0
        return time.time() - self._first_seen[track_id]


# ─────────────────────────────────────────────────────────
#  Main Detector Class
# ─────────────────────────────────────────────────────────

class PhoneDetector:
    """
    Detects phones and prohibited objects in exam hall camera frames.

    Uses RF-DETR (Real-time Detection Transformer) for detection and
    supervision's ByteTrack for persistent object tracking.

    Usage:
        detector = PhoneDetector()
        detections, events = detector.process(frame)
        annotated_frame     = detector.draw(frame, detections)
    """

    def __init__(self, config=None):
        """
        Args:
            config: Optional SharedConfig for live threshold updates.
                    If None, uses module-level constants.

        On first run, rfdetr automatically downloads the model weights
        from the internet. Subsequent runs use the cached file.
        """
        self.config = config
        print("[PhoneDetector] Loading RF-DETR model ...")

        # Load the RF-DETR model (pretrained on COCO).
        # Load the smallest RF-DETR variant (Nano) for maximum FPS.
        self.model = RFDETRNano()

        # Optimise the model for inference using JIT compilation.
        # This significantly improves FPS by fusing operations.
        print("[PhoneDetector] Optimizing model for inference ...")
        self.model.optimize_for_inference()

        print(f"[PhoneDetector] Model loaded and optimized. "
              f"Watching classes: { {v[0] for v in WATCHED_CLASSES.values()} }")

        # Supervision ByteTrack for persistent object tracking across frames.
        # This replaces the YOLO built-in tracker — same algorithm, used via
        # the supervision library which pairs naturally with RF-DETR.
        self.sv_tracker = sv.ByteTrack()

        self.tracker = ViolationTracker()

        # COCO class names lookup (original COCO category IDs → names)
        self.class_names = COCO_CLASSES

    # ──────────────────────────────────────────
    #  Private Helpers
    # ──────────────────────────────────────────

    def _extract_detections(
        self, sv_detections: sv.Detections, frame: np.ndarray
    ) -> list[Detection]:
        """
        Converts supervision Detections into our clean Detection dataclass list.

        RF-DETR returns a supervision.Detections object. After passing it
        through ByteTrack, each detection gets a tracker_id. This method
        unpacks the parts we need into a simple, readable format.

        Args:
            sv_detections: supervision.Detections after ByteTrack update.
            frame:         The original frame (used to clamp coordinates).

        Returns:
            List of Detection objects, filtered to only WATCHED_CLASSES.
        """
        detections = []

        # If no detections or no tracker IDs, return empty
        if sv_detections is None or len(sv_detections) == 0:
            return detections
        if sv_detections.tracker_id is None:
            return detections

        h, w = frame.shape[:2]

        # ── Loop through every detected object ──
        for i in range(len(sv_detections)):
            cid = int(sv_detections.class_id[i])

            # Skip anything not in our WATCHED_CLASSES
            if cid not in WATCHED_CLASSES:
                continue

            class_name, severity = WATCHED_CLASSES[cid]
            x1, y1, x2, y2 = sv_detections.xyxy[i].astype(int)

            # Clamp coordinates to frame boundaries.
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(w, x2);  y2 = min(h, y2)

            detections.append(Detection(
                track_id=int(sv_detections.tracker_id[i]),
                class_id=cid,
                class_name=class_name,
                confidence=float(sv_detections.confidence[i]),
                severity=severity,
                bbox=(x1, y1, x2, y2),
                is_violation=False,  # ViolationTracker fills this in
            ))

        return detections

    def _crop_object(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Crops the detected object out of the frame.
        This cropped image is saved with each ViolationEvent so that:
          - The invigilator can see exactly what was detected
          - Later, Gemma can analyse it for the second-opinion review

        Args:
            frame: Full camera frame
            bbox:  (x1, y1, x2, y2) of the object

        Returns:
            Cropped BGR image of the object. Returns empty array if invalid.
        """
        x1, y1, x2, y2 = bbox

        # Make sure the crop region is valid (has area > 0)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((10, 10, 3), dtype=np.uint8)

        # NumPy array slicing: frame[y1:y2, x1:x2]
        # Remember: images are stored as [rows, columns] = [y, x]
        crop = frame[y1:y2, x1:x2].copy()
        return crop

    # ──────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────

    def process(
        self, frame: np.ndarray
    ) -> tuple[list[Detection], list[ViolationEvent]]:
        """
        Run RF-DETR + ByteTrack on one camera frame.

        This is the main function you call every frame in your camera loop.

        Args:
            frame: OpenCV BGR image from cv2.VideoCapture.read()

        Returns:
            detections: Every prohibited object found this frame (even below threshold).
            events:     ViolationEvents for objects that have been visible long enough.
                        This is what gets sent to the invigilator dashboard.
        """

        # ── Convert BGR (OpenCV) → RGB (RF-DETR expects RGB) ──
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Run RF-DETR detection ──
        #
        # model.predict() does the detection in one call.
        # It returns a supervision.Detections object with:
        #   .xyxy        → bounding boxes [x1, y1, x2, y2]
        #   .confidence  → confidence scores (0.0 to 1.0)
        #   .class_id    → COCO class IDs
        #
        # threshold parameter filters out low-confidence detections.

        conf_threshold = (self.config.get("phone_confidence_threshold")
                          if self.config else CONFIDENCE_THRESHOLD)

        sv_detections = self.model.predict(rgb_frame, threshold=conf_threshold)

        # ── Run ByteTrack to assign persistent tracker IDs ──
        #
        # ByteTrack gives each detected object a persistent ID that stays
        # consistent across frames. Without this, IDs reset every frame.
        sv_detections = self.sv_tracker.update_with_detections(sv_detections)

        # ── Convert supervision detections → our Detection objects ──
        detections = self._extract_detections(sv_detections, frame)

        # ── Update violation timers and get any new events ──
        violation_thresh = (self.config.get("phone_violation_seconds")
                           if self.config else VIOLATION_DURATION_SECONDS)
        events = self.tracker.update(detections, violation_threshold=violation_thresh)

        # ── Attach cropped images to each violation event ──
        # This crop is saved for the invigilator and for Gemma later
        for event in events:
            event.frame_crop = self._crop_object(frame, event.bbox)

        return detections, events

    # ──────────────────────────────────────────
    #  Drawing / Visualisation
    # ──────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """
        Draws bounding boxes and labels on the frame for every detection.

        Box colour:
          RED    = HIGH severity (phone, laptop) — violation confirmed
          ORANGE = HIGH severity — timer running, not yet a confirmed violation
          YELLOW = MEDIUM severity (book) — violation confirmed
          GREEN  = MEDIUM severity — timer running

        Args:
            frame:      Original BGR frame (we copy it, never modify original)
            detections: Output from self.process()

        Returns:
            Annotated BGR frame.
        """
        out = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            duration = self.tracker.get_duration(det.track_id)

            # ── Choose colour based on severity and violation status ──
            if det.severity == "HIGH":
                # Confirmed violation = solid red. Timer running = orange.
                colour = (0, 50, 220) if det.is_violation else (0, 130, 255)
            else:
                # MEDIUM: confirmed = yellow, timer = light green
                colour = (0, 220, 220) if det.is_violation else (80, 200, 80)

            # ── Bounding box ──
            # Thicker border (3px) for confirmed violations, thinner (2px) otherwise
            thickness = 3 if det.is_violation else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)

            # ── Label background (filled rectangle for readability) ──
            label = (f"#{det.track_id} {det.class_name} "
                     f"{det.confidence:.0%}")
            # cv2.getTextSize returns (width, height) of the text in pixels
            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_y = max(y1 - 5, lh + 5)   # don't go above frame top
            # Draw filled rectangle behind text
            cv2.rectangle(
                out,
                (x1, label_y - lh - baseline),
                (x1 + lw, label_y + baseline),
                colour, -1   # -1 = filled
            )
            # Draw the label text in white (readable on any background)
            cv2.putText(
                out, label, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
            )

            # ── VIOLATION stamp ──
            if det.is_violation:
                stamp = f"!! VIOLATION ({duration:.1f}s)"
                cv2.putText(
                    out, stamp, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA
                )

            # ── Duration progress bar ──
            # Shows how close this detection is to triggering a violation
            bar_w = x2 - x1
            max_d = VIOLATION_DURATION_SECONDS
            # min(..., 1.0) caps the bar at 100% even after violation fires
            fill_ratio = min(duration / max_d, 1.0)
            fill_w = int(bar_w * fill_ratio)

            # Background (dark grey bar)
            cv2.rectangle(out, (x1, y2 + 2), (x2, y2 + 8), (60, 60, 60), -1)
            # Fill (colour matches box)
            if fill_w > 0:
                cv2.rectangle(out, (x1, y2 + 2), (x1 + fill_w, y2 + 8), colour, -1)

        return out
