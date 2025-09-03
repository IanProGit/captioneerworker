from flask import Flask, jsonify, request
import os, json
from datetime import datetime
from supabase import create_client

app = Flask(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OUTPUTS_BUCKET = os.environ.get("SUPABASE_OUTPUTS_BUCKET","outputs")
supa = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

@app.get("/health")
def health(): return jsonify(ok=True), 200

@app.get("/")
def index(): return "Captioneer worker is up", 200

@app.post("/enqueue")
def enqueue():
    if request.headers.get("Authorization","") != f"Bearer {os.environ.get('WORKER_TOKEN')}":
        return jsonify(error="unauthorized"), 401
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    if not job_id: return jsonify(error="missing job_id"), 400

    supa.table("transcription_jobs").update({
        "status":"processing",
        "claimed_by":"captioneerworker",
        "claimed_at": datetime.utcnow().isoformat()+"Z"
    }).eq("id", job_id).execute()

    row = supa.table("transcription_jobs").select("user_id").eq("id", job_id).single().execute().data
    user_id = row["user_id"] if row else "unknown"

    vtt = "WEBVTT\n\n00:00.000 --> 00:01.500\n[auto-caption stub]\n"
    key = f"{user_id}/{job_id}.vtt"
    supa.storage.from_(OUTPUTS_BUCKET).upload(key, vtt.encode("utf-8"), {"content-type":"text/vtt","upsert":"true"})
    signed = supa.storage.from_(OUTPUTS_BUCKET).create_signed_url(key, 7*24*3600)
    vtt_url = signed.get("signedURL") if isinstance(signed, dict) else signed.signed_url

    supa.table("transcription_jobs").update({
        "status":"done",
        "outputs": json.dumps({"vtt": vtt_url})
    }).eq("id", job_id).execute()

    return jsonify(accepted=True, id=job_id), 202

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
