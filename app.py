#!/usr/bin/env python3
"""
app.py
Caption worker — Early ACKs, pulls media via signed URL, extracts audio (optional),
transcribes with Whisper, stores VTT in Supabase, and writes outputs/metrics.

Required env:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  OPENAI_API_KEY
  WORKER_TOKEN

Optional env:
  SUPABASE_OUTPUTS_BUCKET=outputs
  CONNECT_TIMEOUT_MS=30000
  TOTAL_TIMEOUT_MS=900000
  RETRY_COUNT=3
  RETRY_BASE_MS=500
  AUDIO_ONLY=true            # If "true", extract audio for Whisper (recommended)
  AUDIO_BITRATE_KBPS=64      # 64 → ~0.5 MB/min at 16 kHz mono (ignored for WAV)
  PORT=10000
  MAX_CONCURRENT_JOBS=5      # Maximum concurrent processing jobs (set to 1 initially)

Dependencies: pip install flask supabase requests gunicorn tenacity
Runtime: Requires ffmpeg (installed via Dockerfile)
"""

import os
import re
import sys
import time
import tempfile
import threading
import subprocess
import requests
import logging
import atexit
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from supabase import create_client, Client

# ---------- Env + Constants ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "").strip()

OUTPUTS_BUCKET = os.environ.get("SUPABASE_OUTPUTS_BUCKET", "outputs").strip()

CONNECT_TIMEOUT = int(os.environ.get("CONNECT_TIMEOUT_MS", "30000")) / 1000.0
TOTAL_TIMEOUT = int(os.environ.get("TOTAL_TIMEOUT_MS", "900000")) / 1000.0
RETRY_COUNT = int(os.environ.get("RETRY_COUNT", "3"))
RETRY_BASE_MS = int(os.environ.get("RETRY_BASE_MS", "500"))
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))  # Set to 1 for stability

AUDIO_ONLY = os.environ.get("AUDIO_ONLY", "true").lower() == "true"
AUDIO_BITRATE_KBPS = int(os.environ.get("AUDIO_BITRATE_KBPS", "64"))

# Hard fail if critical envs missing
missing = [k for k, v in [
    ("SUPABASE_URL", SUPABASE_URL),
    ("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY),
    ("OPENAI_API_KEY", OPENAI_API_KEY),
    ("WORKER_TOKEN", WORKER_TOKEN),
] if not v]
if missing:
    print(f"[FATAL] Missing required env vars: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)

# Configure logging for Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
supa: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Concurrency control with thread pool
_CONCURRENCY = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)
temp_files = []

# UUID validation regex
UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.I)

# One-time ffmpeg check
def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        logger.warning("ffmpeg not found; audio extraction disabled")
        return False
FFMPEG_OK = _ffmpeg_available()

# Cleanup temp files on shutdown
def cleanup_temp_files():
    for path in temp_files[:]:  # Copy to avoid modification during iteration
        try:
            if os.path.exists(path):
                os.remove(path)
                temp_files.remove(path)
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {path}: {e}")
atexit.register(cleanup_temp_files)

# ---------- Helpers ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ok(payload: dict, code: int = 200):
    return jsonify(payload), code

def err(msg: str, code: int = 400, detail: str | None = None):
    body = {"ok": False, "error": msg}
    if detail: body["detail"] = detail
    return jsonify(body), code

def is_uuid(s: str) -> bool:
    return bool(UUID_RE.match(str(s or "").strip()))

def backoff(attempt: int) -> float:
    return (RETRY_BASE_MS * (2 ** attempt)) / 1000.0

def supa_exec(res):
    return getattr(res, "data", []) or []

def validate_status(status: str) -> bool:
    return status in {"queued", "processing", "completed", "failed"}

def update_job(job_id: str, patch: dict):
    if not validate_status(patch.get("status", "")):
        raise ValueError(f"Invalid status: {patch.get('status')}")
    patch = dict(patch or {})
    patch["updated_at"] = now_iso()
    try:
        supa.table("transcription_jobs").update(patch).eq("id", job_id).execute()
        logger.info(f"Updated job {job_id} with {patch}")
    except Exception as e:
        logger.warning(f"update_job failed for {job_id}: {e}")
        raise

def claim_job(job_id: str) -> dict | None:
    now = now_iso()
    try:
        res = supa.table("transcription_jobs").update({
            "status": "processing",
            "claimed_by": "captioneerworker",
            "claimed_at": now,
            "updated_at": now
        }).eq("id", job_id).eq("status", "queued").execute()
        rows = supa_exec(res)
        if rows:
            return rows[0]
        res2 = supa.table("transcription_jobs").select(
            "id,lesson_id,input_video_path,status,claimed_by,claimed_at"
        ).eq("id", job_id).eq("status", "processing").execute()
        rows2 = supa_exec(res2)
        return rows2[0] if rows2 else None
    except Exception as e:
        logger.exception(f"claim_job failed for {job_id}")
        return None

def download_signed_url(signed_url: str, expected_bytes: int | None) -> tuple[str, int, int]:
    start = time.time()
    attempt = 0
    last_err = None
    fd, tmp_path = -1, None
    while attempt <= RETRY_COUNT:
        try:
            with requests.get(
                signed_url, stream=True, timeout=(CONNECT_TIMEOUT, TOTAL_TIMEOUT),
                verify=True, headers={"Range": "bytes=0-"}
            ) as r:
                r.raise_for_status()
                total = 0
                fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                temp_files.append(tmp_path)
                with os.fdopen(fd, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)
                if expected_bytes and abs(total - expected_bytes) / expected_bytes > 0.01:
                    raise RuntimeError(f"Size mismatch: got {total}, expected {expected_bytes}")
            ms = int((time.time() - start) * 1000)
            return tmp_path, total, ms
        except requests.RequestException as e:
            last_err = e
            logger.warning(f"Download attempt {attempt + 1}/{RETRY_COUNT} failed: {e}")
            time.sleep(backoff(attempt))
            attempt += 1
            if tmp_path and os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except Exception: logger.warning(f"Cleanup failed for {tmp_path}")
                temp_files.remove(tmp_path)
                tmp_path = None
    raise RuntimeError(f"Download failed after {RETRY_COUNT} retries: {last_err}")

def extract_audio(input_path: str, target_bitrate_kbps: int = AUDIO_BITRATE_KBPS) -> tuple[str, int, int]:
    if not FFMPEG_OK:
        raise RuntimeError("ffmpeg not available on worker; set AUDIO_ONLY=false")
    start = time.time()
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    temp_files.append(out_path)
    cmd = [
        "ffmpeg", "-y", "-i", input_path, "-vn",
        "-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000",
        out_path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    size = os.path.getsize(out_path)
    ms = int((time.time() - start) * 1000)
    if not os.path.exists(out_path) or size == 0:
        raise RuntimeError("Audio extraction failed or produced empty file")
    return out_path, size, ms

def whisper_to_vtt(file_path: str, mime: str, force_audio: bool) -> tuple[bytes, int]:
    start = time.time()
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    if force_audio:
        filename = "audio.wav"
        content_type = "audio/wav"
    else:
        content_type = mime or "video/mp4"
        ext = "mp4" if "mp4" in content_type else "wav" if "wav" in content_type else "mp3"
        filename = f"input.{ext}"
    with open(file_path, "rb") as f:
        files = {"file": (filename, f, content_type)}
        data = {"model": "whisper-1", "response_format": "vtt"}
        r = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers, files=files, data=data, timeout=TOTAL_TIMEOUT, verify=True
        )
    if r.status_code // 100 != 2:
        raise RuntimeError(f"Whisper failed: {r.status_code} {r.text[:500]}")
    ms = int((time.time() - start) * 1000)
    return r.text.encode("utf-8"), ms

def upload_vtt(job_id: str, vtt_bytes: bytes) -> str:
    key = f"{job_id}.vtt"
    up = supa.storage.from_(OUTPUTS_BUCKET).upload(key, vtt_bytes)
    if hasattr(up, "error") and up.error:
        raise RuntimeError(f"Upload failed: {up.error.message}")
    signed = supa.storage.from_(OUTPUTS_BUCKET).create_signed_url(key, 7 * 24 * 3600)
    url = signed.get("signedURL") or signed.get("signedUrl") or signed.get("signed_url") or signed.signed_url
    if not url:
        raise RuntimeError("Signed URL creation failed")
    return url

def process_job_async(job_id: str, signed_url: str, expected_bytes: int | None, content_type: str):
    with _CONCURRENCY:
        media_path, audio_path = None, None
        try:
            update_job(job_id, {"status": "processing"})
            logger.info(f"Processing job {job_id} started")
            # 1) Download media
            try:
                media_path, download_bytes, download_ms = download_signed_url(signed_url, expected_bytes)
            except Exception as e:
                update_job(job_id, {"status": "failed", "error": f"Download failed: {str(e)[:500]}"})
                return

            # 2) Extract audio if enabled
            whisper_input_path, whisper_force_audio, whisper_mime = media_path, False, content_type
            if AUDIO_ONLY:
                try:
                    audio_path, audio_bytes, audio_ms = extract_audio(media_path, AUDIO_BITRATE_KBPS)
                    whisper_input_path, whisper_force_audio, whisper_mime = audio_path, True, "audio/wav"
                except Exception as e:
                    update_job(job_id, {"status": "failed", "error": f"Audio extraction failed: {str(e)[:500]}"})
                    return

            # 3) Transcribe with Whisper
            try:
                vtt_bytes, transcribe_ms = whisper_to_vtt(whisper_input_path, whisper_mime, whisper_force_audio)
            except Exception as e:
                update_job(job_id, {"status": "failed", "error": f"Whisper failed: {str(e)[:500]}"})
                return

            # 4) Upload VTT
            try:
                vtt_url = upload_vtt(job_id, vtt_bytes)
            except Exception as e:
                update_job(job_id, {"status": "failed", "error": f"Upload failed: {str(e)[:500]}"})
                return

            # 5) Store metrics
            outputs = {
                "vtt": vtt_url,
                "metrics": {
                    "download_bytes": download_bytes,
                    "download_ms": download_ms,
                    **({"audio_bytes": audio_bytes, "audio_ms": audio_ms} if AUDIO_ONLY else {}),
                    "transcribe_ms": transcribe_ms,
                    "audio_only": AUDIO_ONLY
                }
            }
            update_job(job_id, {"status": "completed", "outputs": outputs})
            logger.info(f"Job {job_id} completed successfully")
        except Exception as e:
            update_job(job_id, {"status": "failed", "error": f"Unexpected error: {str(e)[:500]}"})
        finally:
            for path in [media_path, audio_path]:
                if path and os.path.exists(path):
                    try: os.remove(path)
                    except Exception: logger.warning(f"Cleanup failed for {path}")

# ---------- Routes ----------
@app.route("/health", methods=["GET"])
def health():
    return ok({
        "ok": True,
        "service": "captioneerworker",
        "env": {
            "SUPABASE_URL": bool(SUPABASE_URL),
            "SERVICE_ROLE": bool(SUPABASE_SERVICE_ROLE_KEY),
            "OPENAI_API_KEY": bool(OPENAI_API_KEY),
            "WORKER_TOKEN": bool(WORKER_TOKEN),
            "OUTPUTS_BUCKET": OUTPUTS_BUCKET,
            "AUDIO_ONLY": AUDIO_ONLY,
            "FFMPEG": FFMPEG_OK
        }
    })

@app.route("/enqueue", methods=["POST"])
def enqueue():
    auth = (request.headers.get("Authorization") or "").strip()
    if not auth.lower().startswith("bearer ") or auth.split(" ", 1)[1].strip() != WORKER_TOKEN:
        return err("unauthorized", 401)

    if not request.is_json:
        return err("content_type_not_json", 400)
    body = request.get_json(silent=True) or {}

    job_id = body.get("job_id")
    signed_url = body.get("signed_url")
    expected_bytes = body.get("bytes")
    content_type = (body.get("content_type") or "video/mp4").lower()

    if not job_id or not is_uuid(job_id):
        return err("bad_job_id", 400)
    if not signed_url:
        return err("missing_signed_url", 400)

    job = claim_job(job_id)
    if not job:
        return ok({"ok": True, "claimed": False, "job_id": job_id}, 202)

    t = threading.Thread(
        target=process_job_async,
        args=(job_id, signed_url, expected_bytes, content_type),
        daemon=True
    )
    t.start()
    return ok({"ok": True, "claimed": True, "job_id": job_id}, 202)

# ---------- Entry ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, workers=1)  # Single worker for stability
