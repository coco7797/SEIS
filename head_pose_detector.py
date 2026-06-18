"""
head_pose_detector.py
=====================
Head Pose & Gaze Estimation module for the AI Exam Invigilator System.

How it works:
  1. MediaPipe Face Mesh detects 468 facial landmarks (+ 10 iris landmarks
     with refine_landmarks=True) for every visible face in the frame.
     It runs entirely on the CPU and is very lightweight (~2–5 ms/frame).

  2. For EACH detected face, we estimate HEAD POSE using OpenCV's solvePnP:
     - Pick 6 key facial landmarks (nose tip, chin, left/right eye corners,
       left/right mouth corners).
     - Define their corresponding positions on a generic 3D face model.
     - solvePnP solves the Perspective-n-Point problem to find the rotation
       and translation that map the 3D model onto the 2D image landmarks.
     - From the rotation vector we extract yaw, pitch, and roll in degrees.

  3. GAZE DIRECTION is estimated using iris landmark analysis:
     - MediaPipe provides refined iris landmarks (indices 468–477).
     - We compute the horizontal position of the iris centre relative to
       the eye corners to get a left/right gaze ratio.
     - We compute the vertical position for up/down gaze.
     - The Eye Aspect Ratio (EAR) is used to detect blinks.

  4. ATTENTION is classified by combining head pose + gaze:
     - ATTENTIVE     — head forward, gaze centred
     - LOOKING LEFT  — head turned left OR gaze shifted left
     - LOOKING RIGHT — head turned right OR gaze shifted right
     - LOOKING DOWN  — head tilted down (possible hidden notes)
     - LOOKING UP    — head tilted up

  5. A ViolationTracker (same pattern as phone_detector.py) fires a
     HeadPoseViolation if a student is inattentive for too long.

Dependencies: mediapipe, opencv-python, numpy
"""

import cv2
import math
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────
#  Tunable Constants
# ─────────────────────────────────────────────────────────

# Head-pose angle thresholds (degrees).
# If |yaw| > YAW_THRESHOLD the student is turning their head left/right.
YAW_THRESHOLD = 30.0

# If pitch > PITCH_DOWN_THRESHOLD the student is looking down.
# If pitch < -PITCH_UP_THRESHOLD the student is looking up.
PITCH_DOWN_THRESHOLD = 25.0
PITCH_UP_THRESHOLD   = 50.0

# Gaze-ratio thresholds for iris position.
# The ratio goes from 0.0 (far left) to 1.0 (far right).
# Centre is ~0.5.  These define the "looking left" / "looking right" bands.
GAZE_LEFT_THRESHOLD  = 0.35   # ratio < this → looking left
GAZE_RIGHT_THRESHOLD = 0.65   # ratio > this → looking right

# Vertical gaze thresholds (0.0 = top, 1.0 = bottom of the eye).
GAZE_UP_THRESHOLD   = 0.35
GAZE_DOWN_THRESHOLD = 0.65

# Eye Aspect Ratio below this → blink (ignore frame, don't mark as "looking down").
EAR_BLINK_THRESHOLD = 0.20

# Seconds of continuous inattention before a violation is fired.
# Change this value to adjust sensitivity.
ATTENTION_VIOLATION_SECONDS = 5.0
GAZE_VIOLATION_SECONDS = 5.0

# Maximum number of faces to process per frame.
MAX_FACES = 10

# Colours — BGR format
COLOUR_ATTENTIVE   = (0, 200, 80)    # Green
COLOUR_WARNING     = (0, 220, 220)   # Yellow
COLOUR_VIOLATION   = (0, 50, 220)    # Red
COLOUR_AXIS_X      = (0, 0, 255)     # Red   — X axis (pitch)
COLOUR_AXIS_Y      = (0, 255, 0)     # Green — Y axis (yaw)
COLOUR_AXIS_Z      = (255, 0, 0)     # Blue  — Z axis (roll)

# Path to the FaceLandmarker model bundle.
# Downloaded from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
FACE_LANDMARKER_MODEL_PATH = str(Path(__file__).parent / "face_landmarker.task")
COLOUR_MESH        = (200, 200, 200) # Light grey — face mesh wireframe


# ─────────────────────────────────────────────────────────
#  3D Face Model Landmarks (for solvePnP)
# ─────────────────────────────────────────────────────────

# These are canonical 3D coordinates of key facial points on a generic
# face model.  Units are arbitrary (we use a normalised face where the
# nose tip is at the origin).  The Z values represent approximate depth.
#
# The order here MUST match the order of the 2D landmark indices in
# _POSE_LANDMARK_INDICES below.

_MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),     # Nose tip
    (0.0,    63.6,  -12.5),    # Chin          (below nose → positive Y in camera coords)
    (-43.3, -32.7,  -26.0),    # Left eye left corner  (above nose → negative Y)
    (43.3,  -32.7,  -26.0),    # Right eye right corner (above nose → negative Y)
    (-28.9,  28.9,  -24.1),    # Left mouth corner  (below nose → positive Y)
    (28.9,   28.9,  -24.1),    # Right mouth corner (below nose → positive Y)
], dtype=np.float64)

# MediaPipe Face Mesh landmark indices corresponding to the 3D model
# points above.  These indices are from the 468-landmark topology.
_POSE_LANDMARK_INDICES = [
    1,     # Nose tip
    152,   # Chin
    33,    # Left eye left corner
    263,   # Right eye right corner
    61,    # Left mouth corner
    291,   # Right mouth corner
]

# MediaPipe landmark indices for EAR (Eye Aspect Ratio) calculation.
# Each eye has 6 key landmarks forming the eye outline.
_LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Iris landmark indices (available when refine_landmarks=True).
# Left iris: 468–472,  Right iris: 473–477
# Index 468 = left iris centre,  473 = right iris centre
_LEFT_IRIS_INDICES  = [468, 469, 470, 471, 472]
_RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# Face oval landmarks — a subset used for lightweight wireframe drawing.
_FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]

# Minimal mesh connections for lightweight face wireframe.
_MESH_CONNECTIONS_MINIMAL = [
    # Eyebrows
    (70, 63), (63, 105), (105, 66), (66, 107),
    (336, 296), (296, 334), (334, 293), (293, 300),
    # Left eye
    (33, 160), (160, 158), (158, 133), (133, 153), (153, 144), (144, 33),
    # Right eye
    (362, 385), (385, 387), (387, 263), (263, 373), (373, 380), (380, 362),
    # Nose
    (1, 2), (2, 98), (98, 327), (327, 2),
    # Mouth outer
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269),
    (269, 270), (270, 409), (409, 291), (291, 375), (375, 321), (321, 405),
    (405, 314), (314, 17), (17, 84), (84, 181), (181, 91), (91, 146), (146, 61),
]


# ─────────────────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────────────────

@dataclass
class FaceAttention:
    """
    Per-face, per-frame result.

    Attributes:
        face_id:           Persistent label for this face (1-indexed, e.g. 1 → "Person 1").
        label:             Human-readable label (e.g. "Person 1").
        yaw:               Head rotation left/right in degrees. Positive = right.
        pitch:             Head rotation up/down in degrees. Positive = down.
        roll:              Head tilt in degrees. Positive = clockwise.
        gaze_h_ratio:      Horizontal iris position (0.0=left, 1.0=right).
        gaze_v_ratio:      Vertical iris position (0.0=top, 1.0=bottom).
        gaze_direction:    "CENTER", "LEFT", "RIGHT", "UP", "DOWN"
        head_pose_status:  "FORWARD", "LOOKING LEFT", "LOOKING RIGHT",
                           "LOOKING DOWN", "LOOKING UP"  — based on head yaw/pitch only.
        attention_status:  Combined status from head pose + gaze.
                           "ATTENTIVE", "LOOKING LEFT", "LOOKING RIGHT",
                           "LOOKING DOWN", "LOOKING UP"
        is_violation:      True if ANY violation (head or gaze) is active.
        is_head_violation: True if head pose inattention exceeded threshold.
        is_gaze_violation: True if gaze deviation exceeded threshold.
        bbox:              (x1, y1, x2, y2) bounding box around the face.
        nose_2d:           (x, y) position of the nose tip in the image — used as
                           the origin for drawing the 3D orientation axes.
        rotation_vec:      Rotation vector from solvePnP (for drawing axes).
        translation_vec:   Translation vector from solvePnP.
        ear_left:          Eye Aspect Ratio for left eye.
        ear_right:         Eye Aspect Ratio for right eye.
        landmarks_px:      All 468 landmark positions in pixel coordinates [(x,y), ...].
    """
    face_id:           int
    label:             str
    yaw:               float
    pitch:             float
    roll:              float
    gaze_h_ratio:      float
    gaze_v_ratio:      float
    gaze_direction:    str
    head_pose_status:  str
    attention_status:  str
    is_violation:      bool
    is_head_violation: bool
    is_gaze_violation: bool
    bbox:              tuple
    nose_2d:           tuple
    rotation_vec:      np.ndarray
    translation_vec:   np.ndarray
    ear_left:          float
    ear_right:         float
    landmarks_px:      list = field(default_factory=list, repr=False)


@dataclass
class HeadPoseViolation:
    """
    Fired when a student's head has been turned away for
    ATTENTION_VIOLATION_SECONDS.
    """
    face_id:          int
    label:            str
    head_pose_status: str
    duration:         float
    timestamp:        float
    yaw:              float
    pitch:            float
    roll:             float
    bbox:             tuple
    frame_crop:       Optional[np.ndarray] = None


@dataclass
class EyeTrackingViolation:
    """
    Fired when a student's gaze has deviated for
    GAZE_VIOLATION_SECONDS (iris tracking).
    """
    face_id:          int
    label:            str
    gaze_direction:   str
    duration:         float
    timestamp:        float
    gaze_h_ratio:     float
    gaze_v_ratio:     float
    bbox:             tuple
    frame_crop:       Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────
#  Violation Trackers (Head Pose + Gaze — independent)
# ─────────────────────────────────────────────────────────

class HeadPoseViolationTracker:
    """
    Tracks head pose inattention duration per face_id.

    A violation fires when a student's head is continuously turned away
    for threshold_seconds.  The timer resets when the head returns to
    FORWARD or the face disappears from frame.
    """

    def __init__(self, threshold_seconds: float = ATTENTION_VIOLATION_SECONDS):
        self.threshold = threshold_seconds
        self._inattn_start: dict[int, float] = {}
        self._fired: dict[int, bool] = {}
        self._prev_ids: set[int] = set()

    def update(
        self,
        face_attentions: list[FaceAttention],
    ) -> list[HeadPoseViolation]:
        """
        Call every frame.  Returns new head pose violations.
        Sets is_head_violation on each FaceAttention in-place.
        """
        now = time.time()
        current_ids = {fa.face_id for fa in face_attentions}

        for gone_id in (self._prev_ids - current_ids):
            self._inattn_start.pop(gone_id, None)
            self._fired.pop(gone_id, None)

        new_violations: list[HeadPoseViolation] = []

        for fa in face_attentions:
            fid = fa.face_id

            if fa.head_pose_status == "FORWARD":
                self._inattn_start.pop(fid, None)
                self._fired[fid] = False
                fa.is_head_violation = False
                continue

            if fid not in self._inattn_start:
                self._inattn_start[fid] = now
                self._fired[fid] = False

            elapsed = now - self._inattn_start[fid]
            fa.is_head_violation = elapsed >= self.threshold

            if elapsed >= self.threshold and not self._fired.get(fid, False):
                self._fired[fid] = True
                new_violations.append(HeadPoseViolation(
                    face_id=fid,
                    label=fa.label,
                    head_pose_status=fa.head_pose_status,
                    duration=round(elapsed, 1),
                    timestamp=now,
                    yaw=round(fa.yaw, 1),
                    pitch=round(fa.pitch, 1),
                    roll=round(fa.roll, 1),
                    bbox=fa.bbox,
                ))

        self._prev_ids = current_ids
        return new_violations

    def get_duration(self, face_id: int) -> float:
        """Seconds of continuous head pose inattention for this face."""
        if face_id not in self._inattn_start:
            return 0.0
        return time.time() - self._inattn_start[face_id]


class GazeViolationTracker:
    """
    Tracks gaze / eye tracking deviation duration per face_id.

    A violation fires when a student's gaze is continuously off-centre
    for threshold_seconds.  The timer resets when gaze returns to CENTER
    or the face disappears from frame.
    """

    def __init__(self, threshold_seconds: float = GAZE_VIOLATION_SECONDS):
        self.threshold = threshold_seconds
        self._inattn_start: dict[int, float] = {}
        self._fired: dict[int, bool] = {}
        self._prev_ids: set[int] = set()

    def update(
        self,
        face_attentions: list[FaceAttention],
    ) -> list[EyeTrackingViolation]:
        """
        Call every frame.  Returns new eye tracking violations.
        Sets is_gaze_violation on each FaceAttention in-place.
        """
        now = time.time()
        current_ids = {fa.face_id for fa in face_attentions}

        for gone_id in (self._prev_ids - current_ids):
            self._inattn_start.pop(gone_id, None)
            self._fired.pop(gone_id, None)

        new_violations: list[EyeTrackingViolation] = []

        for fa in face_attentions:
            fid = fa.face_id

            if fa.gaze_direction == "CENTER":
                self._inattn_start.pop(fid, None)
                self._fired[fid] = False
                fa.is_gaze_violation = False
                continue

            if fid not in self._inattn_start:
                self._inattn_start[fid] = now
                self._fired[fid] = False

            elapsed = now - self._inattn_start[fid]
            fa.is_gaze_violation = elapsed >= self.threshold

            if elapsed >= self.threshold and not self._fired.get(fid, False):
                self._fired[fid] = True
                new_violations.append(EyeTrackingViolation(
                    face_id=fid,
                    label=fa.label,
                    gaze_direction=fa.gaze_direction,
                    duration=round(elapsed, 1),
                    timestamp=now,
                    gaze_h_ratio=fa.gaze_h_ratio,
                    gaze_v_ratio=fa.gaze_v_ratio,
                    bbox=fa.bbox,
                ))

        self._prev_ids = current_ids
        return new_violations

    def get_duration(self, face_id: int) -> float:
        """Seconds of continuous gaze deviation for this face."""
        if face_id not in self._inattn_start:
            return 0.0
        return time.time() - self._inattn_start[face_id]


# ─────────────────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────────────────

def _eye_aspect_ratio(landmarks_px: list, eye_indices: list) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    When the eye is open, EAR ≈ 0.25–0.30.
    When the eye is closed (blink), EAR drops below ~0.20.

    The 6 points are ordered:  p1=corner, p2=upper1, p3=upper2,
    p4=corner, p5=lower2, p6=lower1  (going clockwise).
    """
    pts = [np.array(landmarks_px[i]) for i in eye_indices]
    # Vertical distances (two pairs across the eye opening)
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance (corner to corner)
    h  = np.linalg.norm(pts[0] - pts[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def _iris_gaze_ratios(
    landmarks_px: list,
) -> tuple[float, float]:
    """
    Compute horizontal and vertical gaze ratios from iris landmarks.

    Horizontal ratio:
      0.0 = iris at the left corner of the eye
      1.0 = iris at the right corner
      0.5 = centred

    Vertical ratio:
      0.0 = iris at the top of the eye
      1.0 = iris at the bottom
      0.5 = centred

    We average both eyes for a more stable result.
    """
    def _one_eye_ratios(iris_indices, eye_corner_left_idx, eye_corner_right_idx,
                        eye_top_idx, eye_bottom_idx):
        # Iris centre = average of the 5 iris landmarks
        iris_pts = [np.array(landmarks_px[i], dtype=np.float64) for i in iris_indices]
        iris_centre = np.mean(iris_pts, axis=0)

        left_corner  = np.array(landmarks_px[eye_corner_left_idx], dtype=np.float64)
        right_corner = np.array(landmarks_px[eye_corner_right_idx], dtype=np.float64)
        top_pt       = np.array(landmarks_px[eye_top_idx], dtype=np.float64)
        bottom_pt    = np.array(landmarks_px[eye_bottom_idx], dtype=np.float64)

        h_range = np.linalg.norm(right_corner - left_corner)
        v_range = np.linalg.norm(bottom_pt - top_pt)

        if h_range < 1e-6 or v_range < 1e-6:
            return 0.5, 0.5

        # Project iris centre onto the line from left→right corner
        h_ratio = np.dot(iris_centre - left_corner,
                         right_corner - left_corner) / (h_range ** 2)
        v_ratio = np.dot(iris_centre - top_pt,
                         bottom_pt - top_pt) / (v_range ** 2)

        return float(np.clip(h_ratio, 0.0, 1.0)), float(np.clip(v_ratio, 0.0, 1.0))

    # Left eye:  corner_left=33, corner_right=133, top=159, bottom=145
    lh, lv = _one_eye_ratios(_LEFT_IRIS_INDICES, 33, 133, 159, 145)
    # Right eye: corner_left=362, corner_right=263, top=386, bottom=374
    rh, rv = _one_eye_ratios(_RIGHT_IRIS_INDICES, 362, 263, 386, 374)

    # Average both eyes
    return (lh + rh) / 2.0, (lv + rv) / 2.0


def _classify_gaze(
    h_ratio: float, v_ratio: float,
    left_thresh: float = GAZE_LEFT_THRESHOLD,
    right_thresh: float = GAZE_RIGHT_THRESHOLD,
    up_thresh: float = GAZE_UP_THRESHOLD,
    down_thresh: float = GAZE_DOWN_THRESHOLD,
) -> str:
    """
    Classify gaze direction from iris ratios.

    Returns one of: "CENTER", "LEFT", "RIGHT", "UP", "DOWN"
    Horizontal takes priority over vertical when both are off-centre.
    """
    if h_ratio < left_thresh:
        return "LEFT"
    if h_ratio > right_thresh:
        return "RIGHT"
    if v_ratio < up_thresh:
        return "UP"
    if v_ratio > down_thresh:
        return "DOWN"
    return "CENTER"


def _classify_head_pose(
    yaw: float, pitch: float,
    yaw_thresh: float = YAW_THRESHOLD,
    pitch_down: float = PITCH_DOWN_THRESHOLD,
    pitch_up: float = PITCH_UP_THRESHOLD,
) -> str:
    """
    Classify head pose from yaw and pitch angles ONLY (no gaze).

    Returns: "FORWARD", "LOOKING LEFT", "LOOKING RIGHT",
             "LOOKING DOWN", "LOOKING UP"
    """
    if yaw < -yaw_thresh:
        return "LOOKING LEFT"
    if yaw > yaw_thresh:
        return "LOOKING RIGHT"
    if pitch > pitch_down:
        return "LOOKING DOWN"
    if pitch < -pitch_up:
        return "LOOKING UP"
    return "FORWARD"


def _classify_attention(head_pose_status: str, gaze_dir: str) -> str:
    """
    Combine head pose status and gaze direction into an overall
    attention status.  Used for the combined view in the HUD.

    Returns: "ATTENTIVE", "LOOKING LEFT", "LOOKING RIGHT",
             "LOOKING DOWN", "LOOKING UP"
    """
    # ── Head pose check (takes priority — larger signal) ──
    if head_pose_status != "FORWARD":
        return head_pose_status

    # ── Gaze check (smaller signal, used when head is roughly forward) ──
    if gaze_dir == "LEFT":
        return "LOOKING LEFT"
    if gaze_dir == "RIGHT":
        return "LOOKING RIGHT"
    if gaze_dir == "DOWN":
        return "LOOKING DOWN"
    if gaze_dir == "UP":
        return "LOOKING UP"

    return "ATTENTIVE"


# ─────────────────────────────────────────────────────────
#  Main Detector Class
# ─────────────────────────────────────────────────────────

class HeadPoseDetector:
    """
    Detects faces and estimates head pose + gaze direction for each.

    Usage:
        detector = HeadPoseDetector()
        attentions, events = detector.process(frame)
        annotated = detector.draw(frame, attentions)
    """

    def __init__(
        self,
        max_faces: int = MAX_FACES,
        violation_seconds: float = ATTENTION_VIOLATION_SECONDS,
        gaze_violation_seconds: float = GAZE_VIOLATION_SECONDS,
        model_path: str = FACE_LANDMARKER_MODEL_PATH,
        config=None,
    ):
        """
        Args:
            max_faces:              Maximum number of faces to process (default 10).
            violation_seconds:      Seconds of continuous head pose inattention
                                    before a violation is fired (default 5.0).
            gaze_violation_seconds: Seconds of continuous gaze deviation before
                                    an eye tracking violation is fired (default 5.0).
            model_path:             Path to the face_landmarker.task model bundle.
            config:                 Optional SharedConfig for live threshold updates.
                                    If None, uses module-level constants.
        """
        self.config = config
        print(f"[HeadPoseDetector] Initialising MediaPipe FaceLandmarker "
              f"(max_faces={max_faces}, head_pose_threshold={violation_seconds}s, "
              f"gaze_threshold={gaze_violation_seconds}s) ...")

        self.max_faces = max_faces

        # ── MediaPipe FaceLandmarker (Tasks API) ──
        # This replaces the legacy mp.solutions.face_mesh API.
        # The Tasks API requires a .task model bundle file.
        #
        # running_mode=VIDEO  → optimised for video (uses temporal smoothing)
        # num_faces           → cap on how many faces to detect
        # min_face_detection_confidence → minimum confidence to accept a face
        # min_face_presence_confidence  → minimum confidence face is still there
        # min_tracking_confidence       → minimum confidence to keep tracking
        # output_face_blendshapes=False → we don't need blendshapes
        # output_facial_transformation_matrixes=False → we compute our own via solvePnP
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # Frame counter for VIDEO running_mode (needs monotonically increasing timestamps)
        self._frame_timestamp_ms = 0

        self.head_tracker = HeadPoseViolationTracker(threshold_seconds=violation_seconds)
        self.gaze_tracker = GazeViolationTracker(threshold_seconds=gaze_violation_seconds)

        # Camera matrix placeholder — updated on first frame based on resolution.
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # assume no distortion

        print("[HeadPoseDetector] Ready.")

    def _get_camera_matrix(self, h: int, w: int) -> np.ndarray:
        """
        Build an approximate camera intrinsic matrix from frame dimensions.

        We assume a pinhole camera with focal length ≈ frame width (a
        reasonable approximation for typical webcams with ~60° FoV).
        The principal point is at the frame centre.

        This is recalculated only when the resolution changes.
        """
        if self._camera_matrix is not None:
            return self._camera_matrix

        focal_length = w  # approximation
        cx, cy = w / 2.0, h / 2.0

        self._camera_matrix = np.array([
            [focal_length, 0,            cx],
            [0,            focal_length, cy],
            [0,            0,            1.0],
        ], dtype=np.float64)

        return self._camera_matrix

    # ──────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────

    def process(
        self, frame: np.ndarray,
    ) -> tuple[list[FaceAttention], list[HeadPoseViolation], list[EyeTrackingViolation]]:
        """
        Run face mesh + head pose + gaze estimation on one frame.

        Args:
            frame: BGR image from OpenCV (cv2.VideoCapture).

        Returns:
            face_attentions:  List of FaceAttention for each detected face.
            head_events:      List of new HeadPoseViolation events.
            gaze_events:      List of new EyeTrackingViolation events.
        """
        h, w = frame.shape[:2]
        cam_matrix = self._get_camera_matrix(h, w)

        # ── Convert frame to MediaPipe Image ──
        # The Tasks API requires an mp.Image object, not a raw NumPy array.
        # We convert BGR→RGB first, then wrap it.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── Run FaceLandmarker ──
        # VIDEO mode requires a monotonically increasing timestamp in ms.
        self._frame_timestamp_ms += 33  # ~30 FPS increment
        results = self.face_landmarker.detect_for_video(
            mp_image, self._frame_timestamp_ms
        )

        face_attentions: list[FaceAttention] = []

        if not results.face_landmarks:
            # No faces detected — still update trackers to reset timers
            self.head_tracker.update([])
            self.gaze_tracker.update([])
            return [], [], []

        for face_idx, face_landmarks in enumerate(results.face_landmarks):
            if face_idx >= self.max_faces:
                break

            # ── Convert normalised landmarks → pixel coordinates ──
            # The Tasks API returns landmarks as NormalizedLandmark objects
            # with .x, .y, .z attributes (normalised to [0, 1]).
            landmarks_px = []
            for lm in face_landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                landmarks_px.append((px, py))

            # ── Compute bounding box from face oval landmarks ──
            oval_pts = [landmarks_px[i] for i in _FACE_OVAL_INDICES
                        if i < len(landmarks_px)]
            xs = [p[0] for p in oval_pts]
            ys = [p[1] for p in oval_pts]
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(w, max(xs)), min(h, max(ys))
            bbox = (x1, y1, x2, y2)

            # ── Head pose via solvePnP ──
            image_points = np.array(
                [landmarks_px[i] for i in _POSE_LANDMARK_INDICES],
                dtype=np.float64,
            )

            success, rotation_vec, translation_vec = cv2.solvePnP(
                _MODEL_POINTS_3D,
                image_points,
                cam_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                continue

            # ── Convert rotation vector → rotation matrix ──
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)

            # ── Extract Euler angles directly from the rotation matrix ──
            # decomposeProjectionMatrix can return ambiguous angles (e.g.
            # -175° when facing forward).  Computing from the rotation
            # matrix via atan2 gives stable values in a ±90° range.
            #
            # Convention: R = Rz · Ry · Rx  (yaw · pitch · roll)
            #   pitch (x) = rotation about the camera's X axis (nod up/down)
            #   yaw   (y) = rotation about the camera's Y axis (turn left/right)
            #   roll  (z) = rotation about the camera's Z axis (head tilt)

            sy = math.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)

            if sy > 1e-6:  # not at gimbal lock
                pitch = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2])
                yaw   = math.atan2(-rotation_mat[2, 0], sy)
                roll  = math.atan2(rotation_mat[1, 0], rotation_mat[0, 0])
            else:
                pitch = math.atan2(-rotation_mat[1, 2], rotation_mat[1, 1])
                yaw   = math.atan2(-rotation_mat[2, 0], sy)
                roll  = 0.0

            pitch = math.degrees(pitch)
            yaw   = math.degrees(yaw)
            roll  = math.degrees(roll)

            # Webcams produce a horizontally mirrored image, which inverts
            # the yaw and pitch signs.  Negate to correct:
            #   positive yaw   = student looking RIGHT  (from their perspective)
            #   positive pitch = student looking DOWN
            yaw   = -yaw
            pitch = -pitch

            # ── Nose tip position for axis drawing ──
            nose_2d = landmarks_px[1]

            # ── Eye Aspect Ratio ──
            ear_left  = _eye_aspect_ratio(landmarks_px, _LEFT_EYE_INDICES)
            ear_right = _eye_aspect_ratio(landmarks_px, _RIGHT_EYE_INDICES)
            avg_ear   = (ear_left + ear_right) / 2.0

            # ── Gaze estimation (iris) ──
            # Only compute gaze if eyes are open (not blinking)
            if avg_ear > EAR_BLINK_THRESHOLD and len(landmarks_px) > 477:
                gaze_h, gaze_v = _iris_gaze_ratios(landmarks_px)
                # Webcams produce a mirrored image — flip horizontal ratio
                # so LEFT/RIGHT matches the student's perspective (same
                # correction as yaw = -yaw above).
                gaze_h = 1.0 - gaze_h
                gaze_dir = _classify_gaze(
                    gaze_h, gaze_v,
                    left_thresh=(self.config.get("gaze_left_threshold")
                                 if self.config else GAZE_LEFT_THRESHOLD),
                    right_thresh=(self.config.get("gaze_right_threshold")
                                  if self.config else GAZE_RIGHT_THRESHOLD),
                    up_thresh=(self.config.get("gaze_up_threshold")
                               if self.config else GAZE_UP_THRESHOLD),
                    down_thresh=(self.config.get("gaze_down_threshold")
                                 if self.config else GAZE_DOWN_THRESHOLD),
                )
            else:
                gaze_h, gaze_v = 0.5, 0.5
                gaze_dir = "CENTER"  # ignore gaze during blinks

            # ── Classify head pose and attention ──
            head_pose_status = _classify_head_pose(
                yaw, pitch,
                yaw_thresh=(self.config.get("yaw_threshold")
                            if self.config else YAW_THRESHOLD),
                pitch_down=(self.config.get("pitch_down_threshold")
                            if self.config else PITCH_DOWN_THRESHOLD),
                pitch_up=(self.config.get("pitch_up_threshold")
                          if self.config else PITCH_UP_THRESHOLD),
            )
            attention = _classify_attention(head_pose_status, gaze_dir)

            # ── Label: "Person 1", "Person 2", etc. ──
            face_id = face_idx + 1
            label = f"Person {face_id}"

            fa = FaceAttention(
                face_id=face_id,
                label=label,
                yaw=round(yaw, 1),
                pitch=round(pitch, 1),
                roll=round(roll, 1),
                gaze_h_ratio=round(gaze_h, 3),
                gaze_v_ratio=round(gaze_v, 3),
                gaze_direction=gaze_dir,
                head_pose_status=head_pose_status,
                attention_status=attention,
                is_violation=False,       # set below after both trackers run
                is_head_violation=False,  # head_tracker fills this in
                is_gaze_violation=False,  # gaze_tracker fills this in
                bbox=bbox,
                nose_2d=nose_2d,
                rotation_vec=rotation_vec,
                translation_vec=translation_vec,
                ear_left=round(ear_left, 3),
                ear_right=round(ear_right, 3),
                landmarks_px=landmarks_px,
            )
            face_attentions.append(fa)

        # ── Update violation trackers (independent) ──
        # Update tracker thresholds from live config if available
        if self.config:
            self.head_tracker.threshold = self.config.get("head_violation_seconds")
            self.gaze_tracker.threshold = self.config.get("gaze_violation_seconds")

        head_events = self.head_tracker.update(face_attentions)
        gaze_events = self.gaze_tracker.update(face_attentions)

        # ── Set combined is_violation flag ──
        for fa in face_attentions:
            fa.is_violation = fa.is_head_violation or fa.is_gaze_violation

        # ── Attach frame crops to violations ──
        for ev in head_events + gaze_events:
            x1, y1, x2, y2 = ev.bbox
            if x2 > x1 and y2 > y1:
                ev.frame_crop = frame[y1:y2, x1:x2].copy()

        return face_attentions, head_events, gaze_events

    # ──────────────────────────────────────────
    #  Drawing / Visualisation
    # ──────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        face_attentions: list[FaceAttention],
    ) -> np.ndarray:
        """
        Draw head pose annotations on the frame.

        For each face:
          - Lightweight wireframe mesh overlay
          - 3D orientation axes (RGB = XYZ) projected from the nose tip
          - Bounding box coloured by attention status
          - Label showing "Person N — STATUS (yaw/pitch)"
          - Duration progress bar (same style as phone_detector)

        Args:
            frame:           Original BGR frame.
            face_attentions: Output from self.process().

        Returns:
            Annotated BGR frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        cam_matrix = self._get_camera_matrix(h, w)

        for fa in face_attentions:
            # ── Choose colour ──
            if fa.is_violation:
                colour = COLOUR_VIOLATION
            elif fa.attention_status != "ATTENTIVE":
                colour = COLOUR_WARNING
            else:
                colour = COLOUR_ATTENTIVE

            # ── Draw face wireframe mesh ──
            mesh_colour = colour
            for (start_idx, end_idx) in _MESH_CONNECTIONS_MINIMAL:
                if start_idx < len(fa.landmarks_px) and end_idx < len(fa.landmarks_px):
                    pt1 = fa.landmarks_px[start_idx]
                    pt2 = fa.landmarks_px[end_idx]
                    cv2.line(out, pt1, pt2, mesh_colour, 1, cv2.LINE_AA)

            # ── Draw face oval ──
            for i in range(len(_FACE_OVAL_INDICES) - 1):
                idx1 = _FACE_OVAL_INDICES[i]
                idx2 = _FACE_OVAL_INDICES[i + 1]
                if idx1 < len(fa.landmarks_px) and idx2 < len(fa.landmarks_px):
                    cv2.line(out, fa.landmarks_px[idx1], fa.landmarks_px[idx2],
                             mesh_colour, 1, cv2.LINE_AA)

            # ── Draw iris circles ──
            if len(fa.landmarks_px) > 477:
                for iris_indices, eye_colour in [(_LEFT_IRIS_INDICES, colour),
                                                  (_RIGHT_IRIS_INDICES, colour)]:
                    iris_pts = [fa.landmarks_px[i] for i in iris_indices]
                    centre = (int(np.mean([p[0] for p in iris_pts])),
                              int(np.mean([p[1] for p in iris_pts])))
                    # Radius from iris landmark spread
                    radius = int(np.mean([
                        np.linalg.norm(np.array(iris_pts[0]) - np.array(p))
                        for p in iris_pts[1:]
                    ]))
                    radius = max(radius, 2)
                    cv2.circle(out, centre, radius, eye_colour, 1, cv2.LINE_AA)

            # ── Draw 3D axes from nose tip ──
            axis_length = 60.0
            axis_points_3d = np.array([
                [0, 0, 0],                  # Origin (nose)
                [axis_length, 0, 0],        # X axis (red)
                [0, axis_length, 0],        # Y axis (green)
                [0, 0, -axis_length],       # Z axis (blue, pointing forward)
            ], dtype=np.float64)

            projected, _ = cv2.projectPoints(
                axis_points_3d,
                fa.rotation_vec,
                fa.translation_vec,
                cam_matrix,
                self._dist_coeffs,
            )

            origin = (int(projected[0][0][0]), int(projected[0][0][1]))
            pt_x   = (int(projected[1][0][0]), int(projected[1][0][1]))
            pt_y   = (int(projected[2][0][0]), int(projected[2][0][1]))
            pt_z   = (int(projected[3][0][0]), int(projected[3][0][1]))

            cv2.line(out, origin, pt_x, COLOUR_AXIS_X, 2, cv2.LINE_AA)
            cv2.line(out, origin, pt_y, COLOUR_AXIS_Y, 2, cv2.LINE_AA)
            cv2.line(out, origin, pt_z, COLOUR_AXIS_Z, 2, cv2.LINE_AA)

            # ── Bounding box ──
            x1, y1, x2, y2 = fa.bbox
            thickness = 3 if fa.is_violation else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)

            # ── Label ──
            status_text = fa.attention_status
            label = f"{fa.label} - {status_text}"
            angle_info = f"Y:{fa.yaw:.0f} P:{fa.pitch:.0f} G:{fa.gaze_direction}"

            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2
            )
            label_y = max(y1 - 8, lh + 5)

            # Background rectangle
            cv2.rectangle(
                out,
                (x1, label_y - lh - baseline - 2),
                (x1 + lw + 4, label_y + baseline + 2),
                colour, -1,
            )
            cv2.putText(
                out, label, (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA,
            )

            # Angle info below the bounding box
            cv2.putText(
                out, angle_info, (x1, y2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA,
            )

            # ── Duration progress bars (head pose + gaze, independent) ──
            bar_y = y2 + 22

            # Head pose progress bar
            if fa.head_pose_status != "FORWARD":
                duration = self.head_tracker.get_duration(fa.face_id)
                bar_w = x2 - x1
                fill_ratio = min(duration / self.head_tracker.threshold, 1.0)
                fill_w = int(bar_w * fill_ratio)

                cv2.rectangle(out, (x1, bar_y), (x2, bar_y + 4), (60, 60, 60), -1)
                if fill_w > 0:
                    bar_col = COLOUR_VIOLATION if fa.is_head_violation else COLOUR_WARNING
                    cv2.rectangle(out, (x1, bar_y), (x1 + fill_w, bar_y + 4),
                                  bar_col, -1)
                cv2.putText(out, "HEAD", (x2 + 4, bar_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1, cv2.LINE_AA)
                bar_y += 8

            # Gaze progress bar
            if fa.gaze_direction != "CENTER":
                duration = self.gaze_tracker.get_duration(fa.face_id)
                bar_w = x2 - x1
                fill_ratio = min(duration / self.gaze_tracker.threshold, 1.0)
                fill_w = int(bar_w * fill_ratio)

                cv2.rectangle(out, (x1, bar_y), (x2, bar_y + 4), (60, 60, 60), -1)
                if fill_w > 0:
                    bar_col = COLOUR_VIOLATION if fa.is_gaze_violation else COLOUR_WARNING
                    cv2.rectangle(out, (x1, bar_y), (x1 + fill_w, bar_y + 4),
                                  bar_col, -1)
                cv2.putText(out, "GAZE", (x2 + 4, bar_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1, cv2.LINE_AA)
                bar_y += 8

            # ── VIOLATION stamps ──
            stamp_y = bar_y + 12
            if fa.is_head_violation:
                duration = self.head_tracker.get_duration(fa.face_id)
                stamp = f"!! HEAD POSE ({duration:.1f}s)"
                cv2.putText(
                    out, stamp, (x1, stamp_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOUR_VIOLATION, 2, cv2.LINE_AA,
                )
                stamp_y += 20
            if fa.is_gaze_violation:
                duration = self.gaze_tracker.get_duration(fa.face_id)
                stamp = f"!! GAZE ({duration:.1f}s)"
                cv2.putText(
                    out, stamp, (x1, stamp_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOUR_VIOLATION, 2, cv2.LINE_AA,
                )

        return out
