"""Web application backend server connecting frontend controls to InSAR & USGS processing.

Endpoints:
- POST /api/trigger-search: Automated USGS trigger to ASF search
- POST /api/search-nisar: Search NASA ASF for NISAR GUNW granules
- POST /api/clip-faults: Dynamic fault clipping to bounding box
- POST /api/save-token: Save Earthdata token locally
- GET  /api/token-status: Check if Earthdata token is configured
- POST /api/remote-process: Stream-process granules without full download (range requests)
- POST /api/download-and-process: Legacy alias of /api/remote-process
- GET  /api/job-progress/<job_id>: SSE stream of live job progress
- GET  /api/job-status/<job_id>: Polling fallback for job progress
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
import requests
from werkzeug.exceptions import HTTPException

from flask import Flask, jsonify, redirect, request, send_from_directory, Response, stream_with_context

ROOT_DIR = Path(__file__).resolve().parent
SAT_DATA_DIR = ROOT_DIR / "sat-data"
MAP_DIR = ROOT_DIR / "map"
MAP_DATA_DIR = MAP_DIR / "data"
MANIPULATE_DIR = ROOT_DIR / "manipulate"
TOKEN_FILE = SAT_DATA_DIR / "earthdata_token.txt"

sys.path.insert(0, str(SAT_DATA_DIR))
sys.path.insert(0, str(MANIPULATE_DIR))

from fault_clipper import clip_faults_to_bbox
from trigger_search import (
    calculate_spatial_bbox,
    calculate_temporal_window,
    fetch_usgs_events,
    process_event_trigger,
)
from download_nisar import (
    DEFAULT_OUTPUT_DIR,
    search_products,
    latest_acquisition_pair,
    filter_orbit_direction,
    _earthdata_token,
    ensure_output_maturity,
    remote_process_products,
    sort_products_by_proximity,
)

app = Flask(__name__, static_folder=str(ROOT_DIR))

# ---------------------------------------------------------------------------
# In-memory job progress store
# job_id -> {"status": str, "pct": int, "log": [str], "done": bool, "error": str|None}
# Each job also has a queue used by SSE subscribers.
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_job_queues: dict[str, queue.Queue] = {}
_jobs_lock = threading.Lock()
JOB_RETENTION_SECONDS = 3600   # finished jobs stay pollable this long
MAX_FINISHED_JOBS = 50
MAX_JOB_LOG_LINES = 500


def _prune_jobs_locked() -> None:
    """Drop finished jobs that are old or over the cap. Caller holds ``_jobs_lock``."""
    now = time.time()
    finished = sorted(
        (job["finished_at"], job_id) for job_id, job in _jobs.items() if job["done"]
    )
    expired = [job_id for at, job_id in finished if now - at > JOB_RETENTION_SECONDS]
    overflow = [job_id for _, job_id in finished[:max(0, len(finished) - MAX_FINISHED_JOBS)]]
    for job_id in set(expired + overflow):
        _jobs.pop(job_id, None)
        _job_queues.pop(job_id, None)


def _new_job() -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _prune_jobs_locked()
        _jobs[job_id] = {
            "status": "queued", "pct": 0, "log": [], "done": False, "error": None, "finished_at": None,
        }
        _job_queues[job_id] = queue.Queue()
    return job_id


def _push_progress(job_id: str, step: str, pct: int, detail: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        q = _job_queues.get(job_id)
        if job is None:
            return
        job["status"] = step
        job["pct"] = pct
        job["log"].append(detail)
        del job["log"][:-MAX_JOB_LOG_LINES]
    if q:
        q.put({"step": step, "pct": pct, "detail": detail})


def _finish_job(job_id: str, step: str, pct: int, detail: str, error: str | None = None) -> None:
    """Record a terminal state ("done" or "error") and notify SSE subscribers once."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        q = _job_queues.get(job_id)
        if job is None:
            return
        job.update(status=step, pct=pct, done=True, error=error, finished_at=time.time())
        job["log"].append(detail)
    if q:
        q.put({"step": step, "pct": pct, "detail": detail})


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def handle_options(path=""):
    return ("", 204)


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    if request.path.startswith("/api/"):
        response = jsonify({"success": False, "error": e.description})
        response.status_code = e.code or 500
        return response
    return e


@app.errorhandler(Exception)
def handle_general_exception(e):
    if request.path.startswith("/api/"):
        response = jsonify({"success": False, "error": str(e)})
        response.status_code = 500
        return response
    raise e


# ---------------------------------------------------------------------------
# SSE job-progress stream
# ---------------------------------------------------------------------------

@app.route("/api/job-progress/<job_id>")
def job_progress_stream(job_id: str):
    """Server-Sent Events stream for a background processing job."""
    with _jobs_lock:
        q = _job_queues.get(job_id)
        job = _jobs.get(job_id)
    if q is None or job is None:
        return jsonify({"error": "Unknown job ID"}), 404

    def _generate():
        # Send current state immediately on connect
        yield f"data: {json.dumps({'step': job['status'], 'pct': job['pct'], 'detail': job['log'][-1] if job['log'] else ''})}\n\n"
        while True:
            try:
                msg = q.get(timeout=25)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("step") in ("done", "error"):
                    break
            except queue.Empty:
                # Heartbeat to keep connection alive
                yield ": heartbeat\n\n"
                with _jobs_lock:
                    j = _jobs.get(job_id, {})
                if j.get("done"):
                    break

    return Response(
        stream_with_context(_generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/job-status/<job_id>")
def job_status(job_id: str):
    """Polling fallback: returns latest job state as JSON."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        snapshot = {**job, "log": list(job["log"])} if job is not None else None
    if snapshot is None:
        return jsonify({"success": False, "error": "Unknown job ID"}), 404
    return jsonify(snapshot)


# ---------------------------------------------------------------------------
# Remote-process endpoint: stream only needed HDF5 datasets (no full download)
# ---------------------------------------------------------------------------

def _crop_bbox(data: dict):
    """Crop to the search bbox only when the client explicitly asks (crop_to_bbox: true)."""
    if not data.get("crop_to_bbox"):
        return None
    return (float(data["west"]), float(data["south"]), float(data["east"]), float(data["north"]))


def _select_products(data: dict) -> tuple[list[dict], int]:
    """Resolve the granules a job should process.

    Returns ``(products, available)`` where ``available`` is how many granules
    matched before any ``max_granules`` cap, so callers can report truncation.
    Raises ``LookupError`` when the search finds nothing.
    """
    products = data.get("products") or []
    if products:
        return products, len(products)

    bbox = (float(data["west"]), float(data["south"]), float(data["east"]), float(data["north"]))
    found = search_products(
        bbox,
        data.get("start_date", "2026-10-01"),
        data.get("end_date", "2026-11-14"),
        data.get("maturity", "PROVISIONAL"),
    )
    orbit_direction = data.get("orbit_direction")
    if found and orbit_direction:
        found = filter_orbit_direction(found, orbit_direction)
    if not found:
        raise LookupError("No matching NISAR granules found for this window")
    pair = latest_acquisition_pair(found) if data.get("latest_pair", True) else found
    ordered = sort_products_by_proximity(pair, bbox)

    # Every granule is processed unless the client explicitly asks for fewer.
    max_granules = data.get("max_granules")
    if max_granules is None or str(max_granules).strip().lower() in ("", "all"):
        return ordered, len(ordered)
    return ordered[:max(1, int(max_granules))], len(ordered)


def _start_remote_job(data: dict):
    """Search ASF (if needed) and stream-process the granules in a background job."""
    try:
        try:
            products, available = _select_products(data)
        except LookupError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404

        token = _earthdata_token()
        if not token:
            return jsonify({"success": False, "error": "Earthdata token is required"}), 401

        maturity = data.get("maturity", "PROVISIONAL")
        workers = int(data.get("workers", 4))
        crop_bbox = _crop_bbox(data)
        job_id = _new_job()

        def _worker():
            try:
                ensure_output_maturity(DEFAULT_OUTPUT_DIR, maturity)
                _push_progress(job_id, "start", 1, f"Starting remote processing of {len(products)} granule(s)…")

                processed_count, failures = remote_process_products(
                    products,
                    token,
                    workers=workers,
                    progress_callback=lambda step, pct, detail: _push_progress(job_id, step, pct, detail),
                    crop_bbox=crop_bbox,
                )
                if failures and not processed_count:
                    raise RuntimeError("; ".join(failures))
                for failure in failures:
                    _push_progress(job_id, "warning", 90, f"Skipped failed granule: {failure}")

                _push_progress(job_id, "overlay", 93, "Rebuilding deformation overlay…")
                prep = ROOT_DIR / "trans" / "prepare_overlay.py"
                subprocess.run([sys.executable, str(prep), "--mosaic", "--full-swath"],
                               cwd=ROOT_DIR, check=False)

                fail_note = f" ({len(failures)} failed)" if failures else ""
                _finish_job(
                    job_id,
                    "done",
                    100,
                    f"✅ Processed {processed_count}/{len(products)} granule(s){fail_note}. Map overlay updated!",
                )
            except Exception as exc:
                _finish_job(job_id, "error", 0, f"❌ Error: {exc}", error=str(exc))

        threading.Thread(target=_worker, daemon=True).start()

        message = f"Started processing of {len(products)} granule(s)."
        if available > len(products):
            message += f" {available - len(products)} more matched but were left out by max_granules."
        return jsonify({
            "success": True,
            "job_id": job_id,
            "granules": len(products),
            "available_granules": available,
            "truncated": available > len(products),
            "message": message,
        }), 202

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/remote-process", methods=["POST"])
def remote_process_endpoint():
    """Search ASF and stream-process NISAR granule(s) without full download."""
    return _start_remote_job(request.get_json(force=True) or {})


@app.route("/")
def index():
    return send_from_directory(str(ROOT_DIR), "index.html")


@app.route("/map")
@app.route("/map/")
def map_redirect():
    return redirect("/map/map.html", code=302)


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(str(ROOT_DIR), path)


@app.route("/api/token-status", methods=["GET"])
def token_status():
    """Check if Earthdata token is configured in environment or file."""
    has_env = bool(os.environ.get("EARTHDATA_TOKEN"))
    has_file = TOKEN_FILE.is_file() and bool(TOKEN_FILE.read_text(encoding="utf-8").strip())
    masked = ""
    if has_env:
        tok = os.environ["EARTHDATA_TOKEN"].strip()
        masked = tok[:4] + "..." + tok[-4:] if len(tok) > 8 else "****"
    elif has_file:
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
        masked = tok[:4] + "..." + tok[-4:] if len(tok) > 8 else "****"

    return jsonify({
        "configured": has_env or has_file,
        "source": "Environment ($EARTHDATA_TOKEN)" if has_env else "Local File (earthdata_token.txt)" if has_file else "None",
        "masked_token": masked,
    })


@app.route("/api/save-token", methods=["POST"])
def save_token():
    """Save Earthdata bearer token locally."""
    data = request.get_json(force=True) or {}
    token = data.get("token", "").strip()
    if not token:
        return jsonify({"success": False, "error": "Token string is required"}), 400

    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token + "\n", encoding="utf-8")
    return jsonify({
        "success": True,
        "message": "Token saved to sat-data/earthdata_token.txt (gitignored)",
    })


@app.route("/api/trigger-search", methods=["POST"])
def trigger_search_endpoint():
    """Trigger earthquake scan and co-seismic InSAR granule matching."""
    data = request.get_json(force=True) or {}
    feed = data.get("feed", "4.5_month")
    min_mag = float(data.get("min_mag", 4.5))
    radius_deg = float(data.get("radius_deg", 1.0))
    window_days = int(data.get("window_days", 12))
    orbit_direction = data.get("orbit_direction") or None

    events = fetch_usgs_events(feed_name=feed, min_mag=min_mag)
    if not events:
        return jsonify({
            "success": True,
            "message": f"No earthquakes M >= {min_mag} found in feed '{feed}'",
            "events_count": 0,
            "target_event": None,
            "granules": [],
        })

    target_event = events[0]
    result = process_event_trigger(
        target_event,
        radius_deg=radius_deg,
        window_days=window_days,
        dry_run=True,
        orbit_direction=orbit_direction,
    )

    # Read the trigger manifest created
    trigger_manifest_path = MAP_DATA_DIR / "latest_trigger_event.json"
    manifest = {}
    if trigger_manifest_path.is_file():
        manifest = json.loads(trigger_manifest_path.read_text(encoding="utf-8"))

    return jsonify({
        "success": True,
        "message": f"Trigger executed for M{target_event['properties']['mag']} - {target_event['properties']['place']}",
        "events_count": len(events),
        "target_event": manifest,
    })


@app.route("/api/search-nisar", methods=["POST"])
def search_nisar_endpoint():
    """Search NASA ASF archive for NISAR granules given bounding box and dates."""
    data = request.get_json(force=True) or {}
    try:
        west = float(data["west"])
        south = float(data["south"])
        east = float(data["east"])
        north = float(data["north"])
        start_date = data.get("start_date", "2026-10-01")
        end_date = data.get("end_date", "2026-11-14")
        maturity = data.get("maturity", "PROVISIONAL")
        orbit_direction = data.get("orbit_direction") or None
        latest_pair = bool(data.get("latest_pair", False))

        bbox = (west, south, east, north)
        products = search_products(bbox, start_date, end_date, maturity)

        if orbit_direction:
            products = filter_orbit_direction(products, orbit_direction)
        if latest_pair and products:
            products = latest_acquisition_pair(products)

        # Optionally clip faults to this search window
        if data.get("clip_faults", True):
            try:
                clip_faults_to_bbox(bbox)
            except Exception as e:
                print(f"Notice: Fault clipping skipped ({e})")

        return jsonify({
            "success": True,
            "count": len(products),
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "products": [
                {
                    "fileName": p.get("fileName"),
                    "startTime": p.get("startTime"),
                    "stopTime": p.get("stopTime"),
                    "flightDirection": p.get("flightDirection"),
                    "sizeMB": round(float(p.get("bytes", 0)) / (1024 * 1024), 1) if p.get("bytes") else "Unknown",
                }
                for p in products
            ],
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/download-and-process", methods=["POST"])
def download_and_process_endpoint():
    """Kept for older clients; identical to /api/remote-process (nothing is fully downloaded)."""
    return _start_remote_job(request.get_json(force=True) or {})


@app.route("/api/clip-faults", methods=["POST"])
def clip_faults_endpoint():
    """Clip active faults dynamically to specified coordinates."""
    data = request.get_json(force=True) or {}
    try:
        west = float(data["west"])
        south = float(data["south"])
        east = float(data["east"])
        north = float(data["north"])
        bbox = (west, south, east, north)

        count = clip_faults_to_bbox(bbox)
        return jsonify({
            "success": True,
            "clipped_faults_count": count,
            "file": "map/faults_clip.geojson",
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser tab automatically")
    args = parser.parse_args(argv)

    url = f"http://127.0.0.1:{args.port}/"
    print(f"🚀 InSAR Hazard Sentinel Web Server running on {url}")
    if not args.no_open:
        webbrowser.open_new_tab(url)

    app.run(host="127.0.0.1", port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
