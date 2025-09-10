# app.py
# Caption worker — accepts {job_id, signed_url, bytes, content_type, lesson_id},
# early-ACKs, pulls video via signed URL, transcribes with Whisper, stores VTT in Supabase.
# Env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY, WORKER_TOKEN
# Optional Env: SUPABASE_OUTPUTS_BUCKET=outputs, CONNECT_TIMEOUT_MS=30000, TOTAL_TIMEOUT_MS=900000, RETRY_COUNT=3, RETRY_BASE_MS=500

import os, re, time, tempfile, threading, uuid, requests
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from supabase import create_client, Client

# ---------- Env ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
OUTPUTS_BUCKET = os.environ.get("SUPABASE_OUTPUTS_BUCKET", "outputs")

CONNECT_TIMEOUT = int(os.environ.get("CONNECT_TIMEOUT_MS", "30000")) / 1000.0
TOTAL_TIMEOUT   = int(os.environ.get("TOTAL_TIMEOUT_MS",   "900000")) / 1000.0
RETRY_COUNT     = int(os.environ.get("RETRY_COUNT", "3"))
RETRY_BASE_MS   = int(os.environ.get("RETRY_BASE_MS", "500"))

app = Flask(__name__)
supa: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# single-job concurrency
_concurrency_lock = threading.Semaphore(1)

UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.I)

# ---------- Helpers ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ok(payload: dict, code: int = 200): return jsonify(payload), code
def err(msg: str, code: int = 400, detail: str | None = None):
    body = {"ok": False, "error": msg}
    if detail: body["detail"] = detail
    return jsonify(body), code

def is_uuid(x: str) -> bool:
    return bool(UUID_RE.match(str(x or "")))

def backoff(attempt: int) -> float:
    return (RETRY_BASE_MS * (2 ** attempt)) / 1000.0

def supa_exec(res):
    # supabase-py may return an object with .data or a plain list
    return getattr(res, "data", res)

def update_job(job_id: str, patch: dict):
    patch["updated_at"] = now_iso()
    try:
        supa.table("transcription_jobs").update(patch).eq("id", job_id).execute()
    except Exception as e:
        app.logger.warning(f"update_job failed: {e}")

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
        # already claimed elsewhere?
        res2 = supa.table("transcription_jobs").select(
            "id,lesson_id,input_video_path,status,claimed_by,claimed_at"
        ).eq("id", job_id).eq("status", "processing").execute()
        rows2 = supa_exec(res2) or []
        return rows2[0] if rows2 else None
    except Exception as e:
        app.logger.exception("claim_job failed")
        return None

def download_signed_url(url: str, expected_bytes: int | None):
    start = time.time()
    attempt = 0
    last_err = None
    fd, tmp_path = -1, None
    while attempt <= RETRY_COUNT:
        try:
            with requests.get(
                url, stream=True,
                timeout=(CONNECT_TIMEOUT, TOTAL_TIMEOUT),
                headers={"Range": "bytes=0-"}
            ) as r:
                r.raise_for_status()
                total = 0
                fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                with os.fdopen(fd, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)
            if expected_bytes:
                diff = abs(total - expected_bytes) / float(expected_bytes)
                if diff > 0.01:
                    raise RuntimeError(f"size mismatch: got {total}, expected {expected_bytes}")
            ms = int((time.time() - start) * 1000)
            return tmp_path, total, ms
        except Exception as e:
            last_err = e
            time.sleep(backoff(attempt))
            attempt += 1
            continue
    raise RuntimeError(f"signed_url_download_failed: {last_err}")

def whisper_to_vtt(file_path: str, content_type: str = "video/mp4"):
    start = time.time()
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    with open(file_path, "rb") as f:
        files = {"file": ("video", f, content_type)}
        data = {"model": "whisper-1", "response_format": "vtt"}
        r = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers, files=files, data=data, timeout=TOTAL_TIMEOUT
        )
    if r.status_code // 100 != 2:
        raise RuntimeError(f"whisper failed: {r.status_code} {r.text[:500]}")
    ms = int((time.time() - start) * 1000)
    return r.text.encode("utf-8"), ms

def upload_vtt(job_id: str, vtt_bytes: bytes) -> str:
    key = f"{job_id}.vtt"
    up = supa.storage.from_(OUTPUTS_BUCKET).upload(key, vtt_bytes)
    # Handle both shapes
    if hasattr(up, "error") and up.error:
        raise RuntimeError(f"upload failed: {up.error.message}")
    signed = supa.storage.from_(OUTPUTS_BUCKET).create_signed_url(key, 7 * 24 * 3600)
    url = None
    if isinstance(signed, dict):
        url = signed.get("signedURL") or signed.get("signedUrl") or signed.get("signed_url")
    else:
        url = getattr(signed, "signed_url", None)
    if not url:
        raise RuntimeError("signed url creation failed")
    return url

def process_job_async(job_id: str, signed_url: str, expected_bytes: int | None, content_type: str):
    with _concurrency_lock:
        tmp_path = None
        try:
            update_job(job_id, {"status": "processing"})
            try:
                tmp_path, downloaded_bytes, download_ms = download_signed_url(signed_url, expected_bytes)
            except Exception as e:
                update_job(job_id, {"status": "failed", "error": f"download failed: {str(e)[:500]}"})
                return

            try:
                vtt_bytes, transcribe_ms = whisper_to_vtt(tmp_path, content_type)
            except Exception as e:
                update_job(job_id, {"status": "failed", "error": f"whisper failed: {str(e)[:500]}"})
                return

            try:
                vtt_url = upload_vtt(job_id, vtt_bytes)
            except Exception as e:
                update_job(job_id, {"status": "failed", "error": f"upload failed: {str(e)[:500]}"})
                return

            outputs = {
                "vtt": vtt_url,
                "metrics": {
                    "download_bytes": downloaded_bytes,
                    "download_ms": download_ms,
                    "transcribe_ms": transcribe_ms
                }
            }
            update_job(job_id, {"status": "completed", "outputs": outputs})
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

# ---------- Routes ----------
@app.get("/health")
def health():
    return ok({
        "ok": True, "service": "captioneerworker",
        "env": {
            "SUPABASE_URL": bool(SUPABASE_URL),
            "SERVICE_ROLE": bool(SUPABASE_SERVICE_ROLE_KEY),
            "OPENAI_API_KEY": bool(OPENAI_API_KEY),
            "WORKER_TOKEN": bool(WORKER_TOKEN),
            "OUTPUTS_BUCKET": OUTPUTS_BUCKET
        }
    })

@app.post("/enqueue")
def enqueue():
    if not WORKER_TOKEN:
        return err("server_config_missing", 500)
    auth = (request.headers.get("Authorization") or "").strip()
    if not auth.lower().startswith("bearer ") or auth.split(" ", 1)[1].strip() != WORKER_TOKEN.strip():
        return err("unauthorized", 401)

    if not request.is_json:
        return err("content_type_not_json", 400)
    body = request.get_json(silent=True) or {}

    job_id = body.get("job_id")
    signed_url = body.get("signed_url")
    expected_bytes = body.get("bytes")
    content_type = body.get("content_type") or "video/mp4"

    if not job_id or not is_uuid(job_id):
        return err("bad_job_id", 400)
    if not signed_url:
        return err("missing_signed_url", 400)

    job = claim_job(job_id)
    if not job:
        return ok({"ok": True, "claimed": False, "job_id": job_id}, 202)

    # Early ACK; heavy work in background
    t = threading.Thread(target=process_job_async, args=(job_id, signed_url, expected_bytes, content_type), daemon=True)
    t.start()
    return ok({"ok": True, "claimed": True, "job_id": job_id}, 202)

# ---------- Entry ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
