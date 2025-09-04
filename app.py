from flask import Flask, jsonify, request
import os, json
from datetime import datetime
from supabase import create_client

app = Flask(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OUTPUTS_BUCKET = os.environ.get("SUPABASE_OUTPUTS_BUCKET","outputs")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN","")

supa = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

@app.get("/health")
def health(): return jsonify(ok=True), 200

@app.get("/")
def index(): return "Captioneer worker is up", 200

@app.post("/enqueue")
def enqueue():
    if request.headers.get("Authorization","") != f"Bearer {WORKER_TOKEN}":
        print("AUTH FAIL")
        return jsonify(error="unauthorized"), 401

    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    if not job_id:
        print("NO JOB_ID")
        return jsonify(error="missing job_id"), 400

    print("JOB:", job_id)

    try:
        r1 = supa.table("transcription_jobs").update({
            "status":"processing",
            "claimed_by":"captioneerworker",
            "claimed_at": datetime.utcnow().isoformat()+"Z"
        }).eq("id", job_id).execute()
        print("UPDATE->processing:", r1)
    except Exception as e:
        print("ERR processing:", e)

    vtt = "WEBVTT\n\n00:00.000 --> 00:01.500\n[auto-caption stub]\n"
    key = f"{job_id}.vtt"
    vtt_url = None
    try:
        supa.storage.from_(OUTPUTS_BUCKET).upload(
            key, vtt.encode("utf-8"),
            {"content-type":"text/vtt","upsert":"true"}
        )
        s = supa.storage.from_(OUTPUTS_BUCKET).create_signed_url(key, 7*24*3600)
        vtt_url = s.get("signedURL") if isinstance(s, dict) else getattr(s, "signed_url", None)
        print("VTT URL:", vtt_url)
    except Exception as e:
        print("ERR VTT:", e)

    try:
        r2 = supa.table("transcription_jobs").update({
            "status":"completed",
            "outputs": json.dumps({"vtt": vtt_url}) if vtt_url else "{}"
        }).eq("id", job_id).execute()
        print("UPDATE->completed:", r2)
    except Exception as e:
        print("ERR completed:", e)

    return jsonify(accepted=True, id=job_id), 202

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","10000")))
