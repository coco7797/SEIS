"""
qwen_reviewer.py
================
Post-examination violation verification using Qwen2.5-VL via Ollama.

This module sends violation clips/images to the locally-running Qwen2.5-VL
model (through Ollama's REST API) to determine whether a detected violation
was genuine.  It runs ONLY after the exam session ends, so it never competes
with RF-DETR / MediaPipe for GPU time or FPS.

The reviewer:
  1. Extracts key frames from video clips (or uses images directly).
  2. Sends them to Qwen2.5-VL with a violation-type-specific prompt.
  3. Parses the model's response into a structured verdict.
  4. Writes CONFIRMED / FALSE_POSITIVE / INCONCLUSIVE + reasoning
     back to the violations database.

Dependencies: requests, opencv-python (already in the project)
External:     Ollama with qwen2.5vl:7b pulled
"""

import base64
import json
import os
import re
import time

import cv2
import requests


# ─────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5vl:7b"

# Number of frames to extract from a video clip for analysis.
# More frames = better context but slower inference.
# Keep low to avoid exceeding model context window.
MAX_VIDEO_FRAMES = 3

# Max dimension for frames sent to the model.
# Resizing reduces token count and avoids context overflow.
MAX_FRAME_DIMENSION = 512

# Valid verdicts the LLM should return
VALID_VERDICTS = {"CONFIRMED", "FALSE_POSITIVE", "INCONCLUSIVE"}

# Prompts tailored to each violation type
VIOLATION_PROMPTS = {
    "phone": (
        "You are an AI exam invigilator assistant. Analyze the following image(s) "
        "from an exam monitoring camera.\n\n"
        "TASK: Determine whether this shows a student holding, using, or having "
        "a mobile phone, laptop, tablet, or other prohibited electronic device "
        "during an examination.\n\n"
        "Consider:\n"
        "- Is there actually a phone/device visible, or is it a false detection "
        "(e.g., a pencil case, calculator, water bottle, hand gesture)?\n"
        "- Is the student actively using or holding the device?\n"
        "- Could this be an innocent object misidentified by the detection model?\n\n"
        "Respond with EXACTLY this format:\n"
        "VERDICT: [CONFIRMED or FALSE_POSITIVE or INCONCLUSIVE]\n"
        "REASONING: [One or two sentences explaining your decision]"
    ),
    "head_pose": (
        "You are an AI exam invigilator assistant. Analyze the following image(s) "
        "from an exam monitoring camera.\n\n"
        "TASK: Determine whether this shows a student turning their head "
        "significantly away from their exam paper in a manner that suggests "
        "they may be looking at another student's paper or cheating.\n\n"
        "Consider:\n"
        "- Is the student clearly looking away from their own desk/paper?\n"
        "- Could they simply be stretching, thinking, or looking at the clock?\n"
        "- Is the head turn sustained and directed toward another student?\n"
        "- Natural brief glances are normal and should not be flagged.\n\n"
        "Respond with EXACTLY this format:\n"
        "VERDICT: [CONFIRMED or FALSE_POSITIVE or INCONCLUSIVE]\n"
        "REASONING: [One or two sentences explaining your decision]"
    ),
    "eye_tracking": (
        "You are an AI exam invigilator assistant. Analyze the following image(s) "
        "from an exam monitoring camera.\n\n"
        "TASK: Determine whether this shows a student whose eyes are clearly "
        "looking away from their own exam paper toward another student's work "
        "or prohibited materials.\n\n"
        "Consider:\n"
        "- Are the student's eyes clearly directed away from their paper?\n"
        "- Could they be looking at the question paper, thinking, or blinking?\n"
        "- Is the gaze direction consistent with attempting to view someone "
        "else's answers?\n"
        "- Brief natural eye movements should not be flagged.\n\n"
        "Respond with EXACTLY this format:\n"
        "VERDICT: [CONFIRMED or FALSE_POSITIVE or INCONCLUSIVE]\n"
        "REASONING: [One or two sentences explaining your decision]"
    ),
}


# ─────────────────────────────────────────────────────────
#  Helper: Encode image to base64
# ─────────────────────────────────────────────────────────

def _encode_image_to_base64(image_path: str) -> str:
    """Read an image file, resize if needed, and return its base64 encoding."""
    img = cv2.imread(image_path)
    if img is not None:
        img = _resize_frame(img)
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")
    # Fallback: read raw bytes if cv2 can't decode
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _resize_frame(frame, max_dim: int = MAX_FRAME_DIMENSION):
    """Resize a frame so its largest dimension is at most max_dim."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _encode_frame_to_base64(frame) -> str:
    """Encode an OpenCV BGR frame to base64 JPEG (resized for the model)."""
    frame = _resize_frame(frame)
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


# ─────────────────────────────────────────────────────────
#  Helper: Extract key frames from video
# ─────────────────────────────────────────────────────────

def _extract_key_frames(video_path: str, max_frames: int = MAX_VIDEO_FRAMES) -> list[str]:
    """
    Extract evenly-spaced key frames from a video file.

    Returns a list of base64-encoded JPEG strings.
    If the video has fewer frames than max_frames, returns all frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[QwenReviewer] Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Calculate which frame indices to grab
    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        # Evenly space, skipping first and last few frames (often black)
        start = max(1, total_frames // 10)
        end = total_frames - max(1, total_frames // 10)
        step = max(1, (end - start) // (max_frames - 1))
        indices = list(range(start, end + 1, step))[:max_frames]

    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames_b64.append(_encode_frame_to_base64(frame))

    cap.release()
    return frames_b64


# ─────────────────────────────────────────────────────────
#  Helper: Parse LLM response
# ─────────────────────────────────────────────────────────

def _parse_verdict(response_text: str) -> tuple[str, str]:
    """
    Parse the LLM's response to extract verdict and reasoning.

    Returns (verdict, reasoning).  Falls back to INCONCLUSIVE if
    the response doesn't match the expected format.
    """
    text = response_text.strip()

    # Try to extract VERDICT: ... pattern
    verdict_match = re.search(
        r"VERDICT\s*:\s*(CONFIRMED|FALSE_POSITIVE|INCONCLUSIVE)",
        text,
        re.IGNORECASE,
    )

    # Try to extract REASONING: ... pattern
    reasoning_match = re.search(
        r"REASONING\s*:\s*(.+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if verdict_match:
        verdict = verdict_match.group(1).upper()
        # Normalize slight variations
        if "FALSE" in verdict:
            verdict = "FALSE_POSITIVE"
    else:
        # Fallback: look for keywords in the response
        upper = text.upper()
        if "FALSE POSITIVE" in upper or "FALSE_POSITIVE" in upper or "NOT A VIOLATION" in upper:
            verdict = "FALSE_POSITIVE"
        elif "CONFIRMED" in upper or "GENUINE" in upper or "VIOLATION IS REAL" in upper:
            verdict = "CONFIRMED"
        else:
            verdict = "INCONCLUSIVE"

    reasoning = reasoning_match.group(1).strip() if reasoning_match else text[:200]

    return verdict, reasoning


# ─────────────────────────────────────────────────────────
#  QwenReviewer — Main class
# ─────────────────────────────────────────────────────────

class QwenReviewer:
    """
    Reviews violation clips/images using Qwen2.5-VL via Ollama.

    Usage:
        reviewer = QwenReviewer()
        if reviewer.is_available():
            verdict, reasoning = reviewer.review_violation(
                "violation_clips/phone_cell_phone_track5_20250901.mp4",
                "phone",
            )
            print(f"{verdict}: {reasoning}")
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = MODEL_NAME):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._available = None

    def is_available(self) -> bool:
        """
        Check if Ollama is running and the Qwen model is available.
        Caches the result after the first check.
        """
        if self._available is not None:
            return self._available

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check if our model (or a variant) is available
                self._available = any(
                    self.model.split(":")[0] in name for name in model_names
                )
                if self._available:
                    print(f"[QwenReviewer] Model '{self.model}' found in Ollama")
                else:
                    available_str = ", ".join(model_names) if model_names else "none"
                    print(f"[QwenReviewer] Model '{self.model}' not found. "
                          f"Available: {available_str}")
            else:
                self._available = False
                print(f"[QwenReviewer] Ollama returned status {resp.status_code}")
        except requests.ConnectionError:
            self._available = False
            print("[QwenReviewer] Cannot connect to Ollama. "
                  "Make sure it's running (ollama serve)")
        except Exception as e:
            self._available = False
            print(f"[QwenReviewer] Error checking Ollama: {e}")

        return self._available

    def review_violation(
        self,
        media_path: str,
        violation_type: str,
        metadata: dict = None,
    ) -> tuple[str, str]:
        """
        Send a violation clip/image to Qwen2.5-VL for verification.

        Args:
            media_path:     Path to the video clip (.mp4) or image (.jpg/.png).
            violation_type: "phone", "head_pose", or "eye_tracking".
            metadata:       Optional extra context (duration, confidence, etc.).

        Returns:
            (verdict, reasoning) where verdict is one of:
                "CONFIRMED"      — The violation appears genuine.
                "FALSE_POSITIVE" — The detection was likely a false alarm.
                "INCONCLUSIVE"   — Not enough evidence to decide.
        """
        if not os.path.exists(media_path):
            return "INCONCLUSIVE", f"Media file not found: {media_path}"

        # Get the appropriate prompt
        prompt = VIOLATION_PROMPTS.get(
            violation_type,
            VIOLATION_PROMPTS["phone"],  # fallback
        )

        # Add metadata context if available
        if metadata:
            context_parts = []
            if metadata.get("duration"):
                context_parts.append(f"Detection duration: {metadata['duration']:.1f}s")
            if metadata.get("confidence"):
                context_parts.append(f"Detection confidence: {metadata['confidence']:.0%}")
            if metadata.get("class_name"):
                context_parts.append(f"Detected object: {metadata['class_name']}")
            if metadata.get("head_pose_status"):
                context_parts.append(f"Head pose: {metadata['head_pose_status']}")
            if metadata.get("gaze_direction"):
                context_parts.append(f"Gaze direction: {metadata['gaze_direction']}")

            if context_parts:
                prompt += "\n\nAdditional context from the detection system:\n"
                prompt += "\n".join(f"- {p}" for p in context_parts)

        # Prepare images
        ext = os.path.splitext(media_path)[1].lower()
        if ext in (".mp4", ".webm", ".avi"):
            images_b64 = _extract_key_frames(media_path)
            if not images_b64:
                return "INCONCLUSIVE", "Could not extract frames from video"
        elif ext in (".jpg", ".jpeg", ".png", ".bmp"):
            images_b64 = [_encode_image_to_base64(media_path)]
        else:
            return "INCONCLUSIVE", f"Unsupported media format: {ext}"

        # Call Ollama API
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": images_b64,
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent verdicts
                    "num_predict": 200,  # We only need a short response
                    "num_ctx": 8192,     # Larger context for multi-image input
                },
            }

            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,  # VLMs can be slow
            )

            if resp.status_code != 200:
                return "INCONCLUSIVE", f"Ollama API error (status {resp.status_code})"

            result = resp.json()
            response_text = result.get("message", {}).get("content", "")

            if not response_text:
                return "INCONCLUSIVE", "Empty response from model"

            return _parse_verdict(response_text)

        except requests.Timeout:
            return "INCONCLUSIVE", "Model inference timed out"
        except requests.ConnectionError:
            return "INCONCLUSIVE", "Lost connection to Ollama during inference"
        except Exception as e:
            return "INCONCLUSIVE", f"Error during review: {str(e)}"

    def review_all(self, violation_store) -> dict:
        """
        Batch-review all violations that have media but no LLM verdict.

        Args:
            violation_store: ViolationStore instance from config.py.

        Returns:
            Summary dict with counts per verdict type.
        """
        pending = violation_store.get_pending_reviews()

        if not pending:
            print("[QwenReviewer] No violations to review.")
            return {"total": 0, "CONFIRMED": 0, "FALSE_POSITIVE": 0, "INCONCLUSIVE": 0}

        print(f"\n{'='*55}")
        print(f"   QWEN2.5-VL VIOLATION REVIEW")
        print(f"{'='*55}")
        print(f"  Reviewing {len(pending)} violation(s)...\n")

        counts = {"CONFIRMED": 0, "FALSE_POSITIVE": 0, "INCONCLUSIVE": 0}
        total = len(pending)

        for i, violation in enumerate(pending, 1):
            v_id = violation["id"]
            v_type = violation["type"]
            media_path = violation.get("media_path", "")
            v_time = time.strftime(
                "%H:%M:%S", time.localtime(violation["timestamp"])
            )

            # Build metadata dict for extra context
            metadata = {
                k: violation[k]
                for k in (
                    "duration", "confidence", "class_name",
                    "head_pose_status", "gaze_direction",
                )
                if violation.get(k) is not None
            }

            print(f"  [{i}/{total}] Reviewing #{v_id} ({v_type} @ {v_time})...",
                  end=" ", flush=True)

            start_time = time.time()
            verdict, reasoning = self.review_violation(media_path, v_type, metadata)
            elapsed = time.time() - start_time

            # Save to database
            violation_store.update_llm_verdict(v_id, verdict, reasoning)
            counts[verdict] = counts.get(verdict, 0) + 1

            # Status symbol
            symbol = {"CONFIRMED": "⚠️", "FALSE_POSITIVE": "✅", "INCONCLUSIVE": "❓"}
            print(f"{symbol.get(verdict, '?')} {verdict} ({elapsed:.1f}s)")
            print(f"          → {reasoning[:80]}")

        # Print summary
        print(f"\n{'─'*55}")
        print(f"  REVIEW COMPLETE")
        print(f"{'─'*55}")
        print(f"  Total reviewed:   {total}")
        print(f"   Confirmed:     {counts['CONFIRMED']}")
        print(f"   False Positive: {counts['FALSE_POSITIVE']}")
        print(f"   Inconclusive:   {counts['INCONCLUSIVE']}")
        print(f"{'='*55}\n")

        return {"total": total, **counts}

    def review_single(self, violation_store, violation_id: int) -> tuple[str, str]:
        """
        Review a single violation by ID.

        Args:
            violation_store: ViolationStore instance.
            violation_id:    The violation ID to review.

        Returns:
            (verdict, reasoning)
        """
        conn = violation_store._get_conn()
        row = conn.execute(
            "SELECT * FROM violations WHERE id = ?",
            (violation_id,),
        ).fetchone()
        conn.close()

        if not row:
            return "INCONCLUSIVE", "Violation not found"

        violation = dict(row)
        media_path = violation.get("media_path", "")
        if not media_path:
            return "INCONCLUSIVE", "No media file for this violation"

        metadata = {
            k: violation[k]
            for k in (
                "duration", "confidence", "class_name",
                "head_pose_status", "gaze_direction",
            )
            if violation.get(k) is not None
        }

        verdict, reasoning = self.review_violation(
            media_path, violation["type"], metadata
        )
        violation_store.update_llm_verdict(violation_id, verdict, reasoning)
        return verdict, reasoning
