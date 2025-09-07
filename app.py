# app.py
# Caption worker (Flask) — /enqueue claims a job, transcribes with Whisper, uploads VTT.
# Env (required): SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY
# Env (optional): WORKER_TOKEN, SUPABASE_OUTPUTS_BUCKET (default "outputs"), SUPABASE_VIDEOS_BUCKET (default "videos")

import os
from datetime import datetime, timezone
from flask import Flask, request, jsonify
import requests
from supabase import create_client, Client  # pip install supabase

app = Flask(__name__)

# ---------- Environment ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

OUTPUTS_BUCKET = os.environ.get("SUPABASE_OUTPUTS_BUCKET", "outputs")
VIDEOS_BUCKET = os.environ.get("SUPABASE_VIDEOS_BUCKET", "videos")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    app.logger.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    app.logger.warning("OPENAI_API_KEY missing — Whisper will fail")

supa: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ---------- Helpers ----------
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def j_ok(body: dict, code: int = 200):
    return jsonify(body), code

def j_err(msg: str, code: int = 400, detail: str | None = None):
    body = {"ok": False, "error": msg}
    if detail:
        body["detail"] = detail
    return jsonify(body), code

def auth_ok(req) -> bool:
    """Accept Bearer <token>, raw Authorization, or X-Worker-Token."""
    if not WORKER_TOKEN:
        return True
    incoming = (req.headers.get("Authorization") or "").strip()
    if incoming.lower().startswith("bearer "):
        incoming = incoming[7:].strip()
    if not incoming:
        incoming = (req.headers.get("X-Worker-Token") or "").strip()
    return incoming == WORKER_TOKEN.strip()

def safe_json(req) -> dict:
    try:
        data = req.get_json(silent=True)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def fetch_video_bytes(lesson_id: str, input_video_path: str | None) -> bytes:
    """
    Get video bytes for the lesson.
    Priority:
      1) If input_video_path is an HTTP/HTTPS URL → GET it.
      2) If input_video_path is a storage key → make public URL from VIDEOS_BUCKET and GET it.
      3) Else read lessons.video_url and GET it.
    """
    # 1) direct URL
    if input_video_path and (input_video_path.startswith("http://") or input_video_path.startswith("https://")):
        r = requests.get(input_video_path, timeout=60)
        r.raise_for_status()
        return r.content

    # 2) storage key
    if input_video_path and not (input_video_path.startswith("http://") or input_video_path.startswith("https://")):
        try:
            pu = supa.storage.from_(VIDEOS_BUCKET).get_public_url(input_video_path)
            public_url = pu.get("publicUrl") if isinstance(pu, dict) else getattr(pu, "public_url", None)
            if public_url:
                r = requests.get(public_url, timeout=60)
                r.raise_for_status()
                return r.content
        except Exception as e:
            app.logger.warning(f"could not fetch by storage key: {e}")

    # 3) lessons.video_url
    try:
        q = supa.table("lessons").select("video_url").eq("id", lesson_id).execute()
        rows = getattr(q, "data", []) or []
        if rows and rows[0].get("video_url"):
            url = rows[0]["video_url"]
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.content
    except Exception as e:
        app.logger.exception("fetch lessons.video_url failed")

    raise RuntimeError("video bytes unavailable")

def whisper_vtt(video_bytes: bytes) -> bytes:
    """Call OpenAI Whisper (whisper-1) and return VTT bytes."""
    files = {
        "file": ("video.mp4", video_bytes, "application/octet-stream"),
    }
    data = {
        "model": "whisper-1",
        "response_format": "vtt"
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    r = requests.post("https://api.openai.com/v1/audio/transcriptions",
                      headers=headers, files=files, data=data, timeout=300)
    if r.status_code // 100 != 2:
        raise RuntimeError(f"whisper failed: {r.status_code} {r.text[:500]}")
    # r.text is the VTT content
    return r.text.encode("utf-8")

# ---------- Routes ----------
@app.route("/health", methods=["GET"])
def health():
    return j_ok({
        "ok": True,
        "service": "captioneerworker",
        "env": {
            "SUPABASE_URL": bool(SUPABASE_URL),
            "SERVICE_ROLE": bool(SUPABASE_SERVICE_ROLE_KEY),
            "OPENAI_API_KEY": bool(OPENAI_API_KEY),
            "WORKER_TOKEN": bool(WORKER_TOKEN),
            "OUTPUTS_BUCKET": OUTPUTS_BUCKET,
            "VIDEOS_BUCKET": VIDEOS_BUCKET,
        }
    })

@app.route("/ping", methods=["GET"])
def ping():
    try:
        return j_ok({"ok": True, "ts": utcnow_iso()})
    except Exception as e:
        app.logger.exception("ping failed")
        return j_err("ping failed", 500, str(e))

@app.route("/enqueue", methods=["POST"])
def enqueue():
    try:
        if not auth_ok(request):
            return j_err("unauthorized", 401)
        if not OPENAI_API_KEY:
            return j_err("missing OPENAI_API_KEY", 500)

        body = safe_json(request)
        job_id = body.get("job_id")
        if not job_id:
            return j_err("job_id required", 400)

        # 1) Atomically claim if queued
        now = utcnow_iso()
        upd = supa.table("transcription_jobs").update({
            "status": "processing",
            "claimed_by": "captioneerworker",
            "claimed_at": now,
            "updated_at": now
        }).eq("id", job_id).eq("status", "queued").execute()

        rows = getattr(upd, "data", None)
        if not rows:
            # double-check concurrent claim
            chk = supa.table("transcription_jobs") \
                .select("id,lesson_id,input_video_path,status,claimed_by,claimed_at") \
                .eq("id", job_id).eq("status", "processing").execute()
            rows = getattr(chk, "data", []) or []

        if not rows:
            return j_ok({"ok": True, "claimed": False, "job_id": job_id}, 202)

        job = rows[0]
        lesson_id = job.get("lesson_id")
        input_video_path = job.get("input_video_path")  # may be None

        # 2) Fetch video and transcribe with Whisper
        try:
            video_bytes = fetch_video_bytes(lesson_id, input_video_path)
        except Exception as e:
            # mark failed and return 202 (claimed true, failed)
            supa.table("transcription_jobs").update({
                "status": "failed",
                "error": f"video fetch failed: {str(e)[:500]}",
                "updated_at": now
            }).eq("id", job_id).execute()
            return j_ok({"ok": True, "claimed": True, "job_id": job_id, "failed": True, "reason": "video fetch failed"}, 202)

        try:
            vtt_bytes = whisper_vtt(video_bytes)
        except Exception as e:
            supa.table("transcription_jobs").update({
                "status": "failed",
                "error": f"whisper failed: {str(e)[:500]}",
                "updated_at": now
            }).eq("id", job_id).execute()
            return j_ok({"ok": True, "claimed": True, "job_id": job_id, "failed": True, "reason": "whisper failed"}, 202)

        # 3) Upload VTT to outputs bucket as <job_id>.vtt
        vtt_key = f"{job_id}.vtt"
        up = supa.storage.from_(OUTPUTS_BUCKET).upload(
            vtt_key,
            vtt_bytes
        )
        if getattr(up, "error", None):
            supa.table("transcription_jobs").update({
                "status": "failed",
                "error": f"upload failed: {up.error.message}",
                "updated_at": now
            }).eq("id", job_id).execute()
            return j_err("upload failed", 500, getattr(up.error, "message", "unknown"))

        # 4) Create a 7-day signed URL for the VTT and store in outputs
        signed = supa.storage.from_(OUTPUTS_BUCKET).create_signed_url(vtt_key, 7 * 24 * 3600)
        vtt_url = signed.get("signedURL") if isinstance(signed, dict) else getattr(signed, "signed_url", None)
        if not vtt_url:
            supa.table("transcription_jobs").update({
                "status": "failed",
                "error": "signed url creation failed",
                "updated_at": now
            }).eq("id", job_id).execute()
            return j_err("signed url failed", 500)

        # 5) Mark job completed with outputs
        supa.table("transcription_jobs").update({
            "status": "completed",
            "outputs": {"vtt": vtt_url},
            "updated_at": now
        }).eq("id", job_id).execute()

        return j_ok({"ok": True, "claimed": True, "job_id": job_id, "vtt": vtt_url}, 202)

    except Exception as e:
        app.logger.exception("enqueue failed")
        return j_err("enqueue failed", 500, str(e))

# ---------- Entry ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)