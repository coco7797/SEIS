"""
admin_server.py
===============
Flask web server for the AI Exam Invigilator admin dashboard.

Provides a browser-based UI at http://localhost:5000 where
invigilators can:
  - View all violations with video clips
  - Adjust detection thresholds in real-time
  - Monitor live stats (FPS, faces, violations)
  - Clear the violation log

Started as a daemon thread from main.py — runs alongside
the detection loop.
"""

import os
import time
from flask import Flask, render_template, jsonify, request, send_file, abort

from config import SharedConfig, ViolationStore


# ─────────────────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")

# Shared state — injected by main.py via init_app()
_config: SharedConfig = None
_store: ViolationStore = None

# Live stats updated by main.py each frame
live_stats = {
    "fps": 0.0,
    "faces": 0,
    "active_violations": 0,
    "start_time": time.time(),
}


def init_app(config: SharedConfig, store: ViolationStore):
    """Called by main.py to inject shared config and store."""
    global _config, _store
    _config = config
    _store = store
    live_stats["start_time"] = time.time()


def update_stats(fps: float, faces: int, active_violations: int):
    """Called by main.py each frame to update live stats."""
    live_stats["fps"] = round(fps, 1)
    live_stats["faces"] = faces
    live_stats["active_violations"] = active_violations


# ─────────────────────────────────────────────────────────
#  Routes — Pages
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the admin dashboard."""
    return render_template("admin.html")


# ─────────────────────────────────────────────────────────
#  Routes — API
# ─────────────────────────────────────────────────────────

@app.route("/api/violations")
def api_get_violations():
    """Return all violations as JSON.  ?type=phone|head_pose|eye_tracking"""
    v_type = request.args.get("type")
    violations = _store.get_all(v_type)
    return jsonify(violations)


@app.route("/api/violations/<int:vid>/image")
def api_get_violation_image(vid):
    """Serve the media file (video clip or legacy crop image) for a violation."""
    media_path = _store.get_media_path(vid)
    if media_path and os.path.exists(media_path):
        # Detect mimetype from file extension
        ext = os.path.splitext(media_path)[1].lower()
        if ext == ".mp4":
            mimetype = "video/mp4"
        elif ext == ".webm":
            mimetype = "video/webm"
        elif ext in (".jpg", ".jpeg"):
            mimetype = "image/jpeg"
        elif ext == ".png":
            mimetype = "image/png"
        else:
            mimetype = "application/octet-stream"
        return send_file(
            os.path.abspath(media_path),
            mimetype=mimetype,
        )
    abort(404)


@app.route("/api/violations/<int:vid>/video")
def api_get_violation_video(vid):
    """Serve the video clip for a specific violation (alias for /image)."""
    return api_get_violation_image(vid)


@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Return current configuration."""
    return jsonify(_config.get_all())


@app.route("/api/config", methods=["POST"])
def api_update_config():
    """Update configuration (partial updates allowed)."""
    updates = request.get_json()
    if not updates:
        return jsonify({"error": "No JSON body"}), 400
    _config.update(updates)
    return jsonify({"status": "ok", "config": _config.get_all()})


@app.route("/api/config/reset", methods=["POST"])
def api_reset_config():
    """Reset all settings to defaults."""
    _config.reset()
    return jsonify({"status": "ok", "config": _config.get_all()})


@app.route("/api/stats")
def api_get_stats():
    """Return live stats + violation counts."""
    counts = _store.get_counts()
    return jsonify({
        "fps": live_stats["fps"],
        "faces": live_stats["faces"],
        "active_violations": live_stats["active_violations"],
        "uptime": round(time.time() - live_stats["start_time"], 1),
        "violations": counts,
    })


@app.route("/api/clear", methods=["POST"])
def api_clear_violations():
    """Clear all violations from the database and disk."""
    _store.clear_all()
    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────────────────
#  Server Launcher
# ─────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8080):
    """
    Start the Flask server.  Called from a daemon thread in main.py.

    Args:
        host: Bind address. "0.0.0.0" = accessible from other devices.
        port: Port number. Default 5000.
    """
    # Suppress Flask's startup banner and request logs
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    print(f"[AdminServer] Dashboard running at http://localhost:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
