# app.py
# Caption worker (Flask) — accepts {job_id, signed_url, bytes, content_type, lesson_id},
# downloads the video via signed URL, transcribes with Whisper, uploads VTT to Supabase.

import os, time, tempfile, requests
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from supabase import create_client, Client

# ---------- Env ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

OUTPUTS_BUCKET = os.environ.get("SUPABASE_OUTPUTS_BUCKET", "outputs")
VIDEOS_BUCKET = os.environ.get("SUPABASE_VIDEOS_BUCKET", "videos")  # not used here, kept for health

CONNECT_TIMEOUT = int(os.environ.get("CONNECT_TIMEOUT_MS", "30000")) / 1000.0
TOTAL_TIMEOUT   = int(os.environ.get("TOTAL_TIMEOUT_MS",   "900000")) / 1000.0
RETRY_COUNT     = int(os.environ.get("RETRY_COUNT", "3"))
RETRY_BASE_MS   = int(os.environ.get("RETRY_BASE_MS", "500"))

app = Flask(__name__)
supa: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ---------- Helpers ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ok(payload: dict, code: int = 200):
    return jsonify(payload), code

def err(msg: str, code: int = 400, detail: str | None = None):
    body = {"ok": False, "error": msg}
    if detail:
        body["detail"] = detail
    return jsonify(body), code

def auth_ok(req) -> bool:
    if not WORKER_TOKEN:
        return True
    token = (req.headers.get("Authorization") or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        token = (req.headers.get("X-Worker-Token") or "").strip()
    return token == WORKER_TOKEN.strip()

def backoff(attempt: int) -> float:
    return (RETRY_BASE_MS * (2 ** attempt)) / 1000.0

def safe_json(req) -> dict:
    try:
        data = req.get_json(silent=True)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def update_job(job_id: str, patch: dict):
    patch["updated_at"] = now_iso()
    try:
        supa.table("transcription_jobs").update(patch).eq("id", job_id).execute()
    except Exception as e:
        app.logger.warning(f"update_job failed: {e}")

def claim_job(job_id: str) -> dict | None:
    now = now_iso()
    try:
        upd = supa.table("transcription_jobs").update({
            "status": "processing",
            "claimed_by": "captioneerworker",
            "claimed_at": now,
            "updated_at": now
        }).eq("id", job_id).eq("status", "queued").execute()
        rows = getattr(upd, "data", None)
        if rows:
            return rows[0]
        # already claimed?
        chk = supa.table("transcription_jobs").select(
            "id,lesson_id,input_video_path,status,claimed_by,claimed_at"
        ).eq("id", job_id).eq("status", "processing").execute()
        rows = getattr(chk, "data", []) or []
        return rows[0] if rows else None
    except Exception as e:
        app.logger.exception("claim_job failed")
        return None

def download_signed_url(url: str, expected_bytes: int | None) -> tuple[str, int, int]:
    start = time.time()
    attempt = 0
    last_err = None
    while attempt <= RETRY_COUNT:
        try:
            with requests.get(
                url,
                stream=True,
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
            return tmp_path, total, int((time.time() - start) * 1000)
        except Exception as e:
            last_err = e
            time.sleep(backoff(attempt))
            attempt += 1
    raise RuntimeError(f"signed_url_download_failed: {last_err}")

def whisper_to_vtt(file_path: str, content_type: str = "video/mp4") -> tuple[bytes, int]:
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
    return r.text.encode("utf-8"), int((time.time() - start) * 1000)

def upload_vtt(job_id: str, vtt_bytes: bytes) -> str:
    key = f"{job_id}.vtt"
    up = supa.storage.from_(OUTPUTS_BUCKET).upload(key, vtt_bytes)
    # supabase-py returns dict-like; guard both shapes
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

# ---------- Routes ----------
@app.get("/health")
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
            "VIDEOS_BUCKET": VIDEOS_BUCKET
        }
    })

@app.post("/enqueue")
def enqueue():
    try:
        if not auth_ok(request):
            return err("unauthorized", 401)
        if not OPENAI_API_KEY:
            return err("missing OPENAI_API_KEY", 500)

        body = safe_json(request)
        job_id = body.get("job_id")
        signed_url = body.get("signed_url")
        expected_bytes = body.get("bytes")
        content_type = body.get("content_type") or "video/mp4"
        lesson_id = body.get("lesson_id")

        if not job_id or not signed_url:
            return err("job_id and signed_url required", 400)

        job = claim_job(job_id)
        if not job:
            # not ours to process
            return ok({"ok": True, "claimed": False, "job_id": job_id}, 202)

        update_job(job_id, {"status": "downloading"})
        tmp_path = None
        try:
            tmp_path, downloaded_bytes, download_ms = download_signed_url(signed_url, expected_bytes)
        except Exception as e:
            update_job(job_id, {"status": "failed", "error": f"download failed: {str(e)[:500]}"})
            return ok({"ok": True, "claimed": True, "job_id": job_id, "failed": True, "reason": "download failed"}, 202)

        try:
            update_job(job_id, {"status": "transcribing"})
            vtt_bytes, transcribe_ms = whisper_to_vtt(tmp_path, content_type)
        except Exception as e:
            update_job(job_id, {"status": "failed", "error": f"whisper failed: {str(e)[:500]}", "download_bytes": downloaded_bytes, "download_ms": download_ms})
            return ok({"ok": True, "claimed": True, "job_id": job_id, "failed": True, "reason": "whisper failed"}, 202)

        try:
            vtt_url = upload_vtt(job_id, vtt_bytes)
        except Exception as e:
            update_job(job_id, {"status": "failed", "error": f"upload failed: {str(e)[:500]}", "download_bytes": downloaded_bytes, "download_ms": download_ms})
            return err("upload failed", 500, str(e))

        update_job(job_id, {
            "status": "completed",
            "outputs": {"vtt": vtt_url, "download_bytes": downloaded_bytes, "download_ms": download_ms, "transcribe_ms": transcribe_ms}
        })

        return ok({"ok": True, "claimed": True, "job_id": job_id, "vtt": vtt_url, "download_bytes": downloaded_bytes, "download_ms": download_ms}, 202)

    finally:
        # cleanup temp file if created
        try:
            if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# ---------- Entry ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
