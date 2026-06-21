"""
config.py
=========
Shared configuration and violation storage for the AI Exam Invigilator.

Provides:
  - SharedConfig: Thread-safe configuration that can be updated live
    from the admin dashboard while the detection loop is running.
  - ViolationStore: SQLite-backed persistent storage for all violations
    with video clips (or legacy crop images).

Both classes are designed to be shared between the detection thread
and the Flask admin server thread.
"""

import json
import os
import sqlite3
import threading
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────
#  Default Configuration Values
# ─────────────────────────────────────────────────────────

DEFAULTS = {
    # Feature toggles
    "phone_detection_enabled": True,
    "head_pose_enabled": True,
    "eye_tracking_enabled": True,

    # Phone detection
    "phone_violation_seconds": 2.0,
    "phone_confidence_threshold": 0.50,

    # Head pose detection
    "head_violation_seconds": 5.0,
    "yaw_threshold": 30.0,
    "pitch_down_threshold": 25.0,
    "pitch_up_threshold": 50.0,

    # Eye tracking / gaze detection
    "gaze_violation_seconds": 5.0,
    "gaze_left_threshold": 0.35,
    "gaze_right_threshold": 0.65,
    "gaze_up_threshold": 0.35,
    "gaze_down_threshold": 0.65,

    # Video recording
    "video_pre_seconds": 5.0,
    "video_post_seconds": 2.0,
}


# ─────────────────────────────────────────────────────────
#  SharedConfig — Thread-safe live configuration
# ─────────────────────────────────────────────────────────

class SharedConfig:
    """
    Thread-safe configuration container.

    The detection thread reads values every frame, and the Flask
    server writes values when the admin updates settings.  A
    threading.Lock ensures no torn reads/writes.

    Usage:
        config = SharedConfig()
        val = config.get("yaw_threshold")   # read
        config.update({"yaw_threshold": 25.0})  # write
    """

    def __init__(self, config_file="settings.json"):
        self._lock = threading.Lock()
        self._config_file = config_file
        self._config = dict(DEFAULTS)
        self._load()

    def _load(self):
        """Load settings from JSON file if it exists."""
        if os.path.exists(self._config_file):
            try:
                with open(self._config_file, "r") as f:
                    saved_config = json.load(f)
                    # Only load keys that exist in DEFAULTS and cast appropriately
                    for key, value in saved_config.items():
                        if key in self._config:
                            self._config[key] = type(DEFAULTS[key])(value)
            except Exception as e:
                print(f"[SharedConfig] Failed to load {self._config_file}: {e}")

    def _save(self):
        """Save settings to JSON file."""
        try:
            with open(self._config_file, "w") as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            print(f"[SharedConfig] Failed to save {self._config_file}: {e}")

    def get(self, key: str):
        """Read a single config value (thread-safe)."""
        with self._lock:
            return self._config[key]

    def get_all(self) -> dict:
        """Return a snapshot of all config values (thread-safe)."""
        with self._lock:
            return dict(self._config)

    def update(self, updates: dict):
        """
        Apply partial updates.  Only known keys are accepted;
        values are cast to the correct type automatically.
        """
        with self._lock:
            for key, value in updates.items():
                if key in self._config:
                    # Cast to the same type as the default
                    self._config[key] = type(DEFAULTS[key])(value)
            self._save()

    def reset(self):
        """Reset all settings to defaults."""
        with self._lock:
            self._config = dict(DEFAULTS)
            self._save()


# ─────────────────────────────────────────────────────────
#  ViolationStore — SQLite-backed persistent storage
# ─────────────────────────────────────────────────────────

class ViolationStore:
    """
    Stores all violations in a SQLite database for persistence
    across sessions.  Crop images are saved as files on disk;
    the database stores the file paths.

    Thread-safe: each method opens its own database connection.

    Usage:
        store = ViolationStore()
        store.add_phone_violation(event, image_path="violation_crops/...")
        violations = store.get_all()
        store.clear_all()
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent / "violations.db")
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new connection (safe for multi-threaded use)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create the violations table if it doesn't exist, and migrate old schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                type        TEXT    NOT NULL,
                timestamp   REAL    NOT NULL,
                duration    REAL    NOT NULL,
                class_name  TEXT,
                track_id    INTEGER,
                severity    TEXT,
                confidence  REAL,
                label       TEXT,
                head_pose_status TEXT,
                yaw         REAL,
                pitch       REAL,
                roll        REAL,
                gaze_direction TEXT,
                gaze_h_ratio   REAL,
                gaze_v_ratio   REAL,
                bbox        TEXT,
                media_path  TEXT
            )
        """)
        conn.commit()

        # ── Migrate old schema: rename image_path → media_path ──
        # If the table was created with the old schema, it will have
        # 'image_path' but not 'media_path'.  Add the new column and
        # copy values over.
        try:
            columns = [row[1] for row in
                       conn.execute("PRAGMA table_info(violations)").fetchall()]
            if "image_path" in columns and "media_path" not in columns:
                conn.execute("ALTER TABLE violations ADD COLUMN media_path TEXT")
                conn.execute("UPDATE violations SET media_path = image_path")
                conn.commit()
                print("[ViolationStore] Migrated image_path → media_path")
        except Exception as e:
            print(f"[ViolationStore] Migration note: {e}")

        conn.close()

    @staticmethod
    def _safe_bbox(bbox) -> str:
        """Convert bbox tuple (may contain numpy ints) to JSON string."""
        if bbox is None:
            return "[]"
        return json.dumps([int(x) for x in bbox])

    # ── Add violations ──

    def add_phone_violation(self, event, media_path: str = None):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO violations
            (type, timestamp, duration, class_name, track_id, severity,
             confidence, bbox, media_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "phone", event.timestamp, event.duration,
            event.class_name, event.track_id, event.severity,
            float(event.confidence), self._safe_bbox(event.bbox),
            media_path,
        ))
        conn.commit()
        conn.close()

    def add_head_pose_violation(self, event, media_path: str = None):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO violations
            (type, timestamp, duration, label, head_pose_status,
             yaw, pitch, roll, bbox, media_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "head_pose", event.timestamp, event.duration,
            event.label, event.head_pose_status,
            float(event.yaw), float(event.pitch), float(event.roll),
            self._safe_bbox(event.bbox), media_path,
        ))
        conn.commit()
        conn.close()

    def add_eye_tracking_violation(self, event, media_path: str = None):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO violations
            (type, timestamp, duration, label, gaze_direction,
             gaze_h_ratio, gaze_v_ratio, bbox, media_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "eye_tracking", event.timestamp, event.duration,
            event.label, event.gaze_direction,
            float(event.gaze_h_ratio), float(event.gaze_v_ratio),
            self._safe_bbox(event.bbox), media_path,
        ))
        conn.commit()
        conn.close()

    # ── Query violations ──

    def get_all(self, violation_type: str = None) -> list[dict]:
        """Return all violations, newest first.  Optionally filter by type."""
        conn = self._get_conn()
        if violation_type and violation_type != "all":
            rows = conn.execute(
                "SELECT * FROM violations WHERE type = ? ORDER BY timestamp DESC",
                (violation_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM violations ORDER BY timestamp DESC"
            ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_media_path(self, violation_id: int) -> str | None:
        """Return the media_path for a specific violation, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT media_path FROM violations WHERE id = ?",
            (violation_id,),
        ).fetchone()
        conn.close()
        if row and row["media_path"]:
            return row["media_path"]
        return None

    def get_counts(self) -> dict:
        """Return violation counts by type."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        phone = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE type='phone'"
        ).fetchone()[0]
        head = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE type='head_pose'"
        ).fetchone()[0]
        gaze = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE type='eye_tracking'"
        ).fetchone()[0]
        conn.close()
        return {
            "total": total,
            "phone": phone,
            "head_pose": head,
            "eye_tracking": gaze,
        }

    def clear_all(self):
        """Delete all violations from the database and remove media files."""
        # Get all media paths before deleting
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT media_path FROM violations WHERE media_path IS NOT NULL"
        ).fetchall()

        # Delete from database
        conn.execute("DELETE FROM violations")
        conn.commit()
        conn.close()

        # Delete media files (video clips and legacy crop images) from disk
        for row in rows:
            path = row["media_path"]
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
