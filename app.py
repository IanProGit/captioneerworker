from flask import Flask, jsonify, request
import os, json, tempfile, subprocess, requests
from datetime import datetime
from supabase import create_client

app = Flask(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OUTPUTS_BUCKET = os.environ.get("SUPABASE_OUTPUTS_BUCKET","outputs")
VIDEOS_BUCKET  = os.environ.get("SUPABASE_VIDEOS_BUCKET","videos")
WORKER_TOKEN   = os.environ.get("WORKER_TOKEN","")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY","")

supa = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

@app.get("/health")
def health(): return jsonify(ok=True), 200

@app.get("/diag")
def diag():
    # prove env + ffmpeg presence without leaking secrets
    try:
        v = subprocess.check_output(["ffmpeg","-version"]).decode().splitlines()[0]
    except Exception as e:
        v = f"ffmpeg_error:{e}"
    return jsonify(
        ok=True,
        has_openai=bool(OPENAI_API_KEY),
        ffmpeg=v
    ), 200

@app.post("/enqueue")
def enqueue():
    if request.headers.get("Authorization","") != f"Bearer {WORKER_TOKEN}":
        return jsonify(error="unauthorized"), 401
    job_id = (request.get_json(silent=True) or {}).get("job_id")
    if not job_id: return jsonify(error="missing job_id"), 400

    supa.table("transcription_jobs").update({
        "status":"processing","claimed_by":"captioneerworker",
        "claimed_at": datetime.utcnow().isoformat()+"Z"
    }).eq("id", job_id).execute()

    row = supa.table("transcription_jobs").select("user_id,input_video_path").eq("id", job_id).single().execute().data
    if not row or not row.get("input_video_path"):
        supa.table("transcription_jobs").update({"status":"failed","error":"missing input_video_path"}).eq("id", job_id).execute()
        return jsonify(error="missing input_video_path"), 400

    user_id = row["user_id"]; key = row["input_video_path"]
    s = supa.storage.from_(VIDEOS_BUCKET).create_signed_url(key, 3600)
    url = s.get("signedURL") if isinstance(s, dict) else getattr(s, "signed_url", None)
    if not url:
        supa.table("transcription_jobs").update({"status":"failed","error":"signed url failed"}).eq("id", job_id).execute()
        return jsonify(error="signed url failed"), 500

    try:
        with tempfile.TemporaryDirectory() as td:
            mp4 = os.path.join(td,"in.mp4")
            wav = os.path.join(td,"in.wav")
            r = requests.get(url, timeout=120); r.raise_for_status()
            open(mp4,"wb").write(r.content)
            subprocess.check_call(["ffmpeg","-y","-i",mp4,"-vn","-ac","1","-ar","16000",wav],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY missing")

            files = {"file": open(wav,"rb")}
            data  = {"model":"whisper-1","response_format":"vtt"}
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            tr = requests.post("https://api.openai.com/v1/audio/transcriptions",
                               files=files, data=data, headers=headers, timeout=600)
            print("WHISPER_STATUS:", tr.status_code)
            print("WHISPER_SNIP:", tr.text[:220])
            tr.raise_for_status()
            vtt_bytes = tr.content

        out_key = f"{user_id}/{job_id}.vtt"
        supa.storage.from_(OUTPUTS_BUCKET).upload(out_key, vtt_bytes, {"content-type":"text/vtt","upsert":"true"})
        signed = supa.storage.from_(OUTPUTS_BUCKET).create_signed_url(out_key, 7*24*3600)
        vtt_url = signed.get("signedURL") if isinstance(signed, dict) else getattr(signed, "signed_url", None)

        supa.table("transcription_jobs").update({
            "status":"completed","outputs": json.dumps({"vtt": vtt_url})
        }).eq("id", job_id).execute()

        return jsonify(accepted=True, id=job_id), 202

    except Exception as e:
        supa.table("transcription_jobs").update({"status":"failed","error":str(e)}).eq("id", job_id).execute()
        print("WORKER_ERROR:", e)
        return jsonify(error=str(e), detail=repr(e)), 500
# redeploy test Thu  4 Sep 2025 11:44:28 CEST
