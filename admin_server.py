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
import threading
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
    """Return all violations as JSON.  ?type=phone|head_pose|eye_tracking  ?verdict=CONFIRMED|FALSE_POSITIVE|INCONCLUSIVE|PENDING"""
    v_type = request.args.get("type")
    verdict = request.args.get("verdict")
    violations = _store.get_all(v_type, verdict)
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


@app.route("/api/violations/<int:vid>", methods=["DELETE", "POST"])
@app.route("/api/violations/<int:vid>/delete", methods=["POST"])
def api_delete_single_violation(vid):
    """Delete an individual violation by ID."""
    success = _store.delete_one(vid)
    if success:
        return jsonify({"status": "ok", "deleted_id": vid})
    return jsonify({"error": "Violation not found"}), 404


@app.route("/api/violations/delete-batch", methods=["POST"])
def api_delete_violations_batch():
    """Delete multiple selected violations by ID list."""
    data = request.get_json() or {}
    ids = data.get("ids", [])
    if not isinstance(ids, list):
        return jsonify({"error": "ids must be a list"}), 400
    valid_ids = [int(x) for x in ids if str(x).isdigit()]
    deleted_count = _store.delete_many(valid_ids)
    return jsonify({"status": "ok", "deleted_count": deleted_count})


# ─────────────────────────────────────────────────────────
#  Routes — Qwen2.5-VL AI Review
# ─────────────────────────────────────────────────────────

# Track background review state
_review_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_id": None,
}


@app.route("/api/violations/<int:vid>/review", methods=["POST"])
def api_review_single(vid):
    """Review a single violation with Qwen2.5-VL (runs in background thread)."""
    try:
        from qwen_reviewer import QwenReviewer
    except ImportError:
        return jsonify({"error": "qwen_reviewer module not available"}), 500

    reviewer = QwenReviewer()
    if not reviewer.is_available():
        return jsonify({"error": "Ollama / Qwen2.5-VL not available"}), 503

    def _do_review():
        verdict, reasoning = reviewer.review_single(_store, vid)
        print(f"[AdminServer] Single review #{vid}: {verdict}")

    thread = threading.Thread(target=_do_review, daemon=True)
    thread.start()

    return jsonify({"status": "ok", "message": f"Reviewing violation #{vid}..."})


@app.route("/api/review-all", methods=["POST"])
def api_review_all():
    """Batch-review all unreviewed violations (runs in background thread)."""
    if _review_state["running"]:
        return jsonify({"error": "A review is already in progress"}), 409

    try:
        from qwen_reviewer import QwenReviewer
    except ImportError:
        return jsonify({"error": "qwen_reviewer module not available"}), 500

    reviewer = QwenReviewer()
    if not reviewer.is_available():
        return jsonify({"error": "Ollama / Qwen2.5-VL not available"}), 503

    pending = _store.get_pending_reviews()
    if not pending:
        return jsonify({"status": "ok", "message": "No violations to review", "total": 0})

    def _do_batch():
        _review_state["running"] = True
        _review_state["total"] = len(pending)
        _review_state["progress"] = 0
        try:
            for i, v in enumerate(pending, 1):
                _review_state["progress"] = i
                _review_state["current_id"] = v["id"]
                reviewer.review_single(_store, v["id"])
        finally:
            _review_state["running"] = False
            _review_state["current_id"] = None

    thread = threading.Thread(target=_do_batch, daemon=True)
    thread.start()

    return jsonify({
        "status": "ok",
        "message": f"Reviewing {len(pending)} violation(s)...",
        "total": len(pending),
    })


@app.route("/api/review-status")
def api_review_status():
    """Return the current status of a batch review."""
    return jsonify(_review_state)


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
    global _config, _store
    if _config is None or _store is None:
        from config import SharedConfig, ViolationStore
        init_app(SharedConfig(), ViolationStore())

    # Suppress Flask's startup banner and request logs
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    print(f"[AdminServer] Dashboard running at http://localhost:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)



if __name__ == "__main__":
    run_server()

