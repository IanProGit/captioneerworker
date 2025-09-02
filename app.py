# app.py — Captioner worker (Flask)
# - Auth via X-API-KEY
# - Download from Supabase Storage (public or signed URL)
# - Transcribe with OpenAI whisper-1 (segments) -> .txt / .vtt / .srt
# - Soft-sub MP4 (always), optional burned-in MP4
# - Upload all to outputs/ bucket, return public URLs
# - If job_id provided, updates public.jobs (status/progress/urls)

import os, io, time, tempfile, pathlib, subprocess, shutil, json
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from supabase import create_client, Client
import requests

# ---------- ENV ----------
SB_URL = os.environ["SUPABASE_URL"]
SB_KEY = os.environ["SUPABASE_SERVICE_KEY"]
BUCKET_V = os.environ.get("SUPABASE_VIDEOS_BUCKET", "videos")
BUCKET_O = os.environ.get("SUPABASE_OUTPUTS_BUCKET", "outputs")
API_KEY  = os.environ["WORKER_API_KEY"]
CAPTION_STYLE = os.environ.get(
    "CAPTION_STYLE",
    "Alignment=2,MarginV=10,FontName=Times New Roman,FontSize=16,Outline=2,Shadow=1,BorderStyle=1"
)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # add this in Render env

sb: Client = create_client(SB_URL, SB_KEY)
app = Flask(__name__)

# ---------- HELPERS ----------
def public_url(bucket: str, path: str) -> str:
    base = SB_URL.rstrip("/")
    return f"{base}/storage/v1/object/public/{bucket}/{path.lstrip('/')}"

def to_download_url(video_path: str) -> str:
    # Accept either "videos/xxx.mp4" or a full URL
    if video_path.startswith("http://") or video_path.startswith("https://"):
        return video_path
    # storage path; ensure bucket prefix removed before building public URL
    p = video_path
    if p.startswith(BUCKET_V + "/"):
        p = p[len(BUCKET_V) + 1 :]
    return public_url(BUCKET_V, p)

def dl(url: str, dst: pathlib.Path):
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 256):
                if chunk:
                    f.write(chunk)

def ts_to_vtt(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def ts_to_srt(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t % 1)*1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_text_assets(segments: List[Dict[str,Any]], txt_p: pathlib.Path, vtt_p: pathlib.Path, srt_p: pathlib.Path):
    # TXT
    with open(txt_p, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg["text"].strip() + "\n")
    # VTT
    with open(vtt_p, "w", encoding="utf-8") as v:
        v.write("WEBVTT\n\n")
        for seg in segments:
            v.write(f"{ts_to_vtt(seg['start'])} --> {ts_to_vtt(seg['end'])}\n{seg['text'].strip()}\n\n")
    # SRT
    with open(srt_p, "w", encoding="utf-8") as s:
        for i, seg in enumerate(segments, start=1):
            s.write(f"{i}\n{ts_to_srt(seg['start'])} --> {ts_to_srt(seg['end'])}\n{seg['text'].strip()}\n\n")

def upload(p: pathlib.Path, dest: str, content_type: str = "application/octet-stream") -> str:
    with open(p, "rb") as f:
        sb.storage.from_(BUCKET_O).upload(dest, f, {"upsert": True, "contentType": content_type})
    return sb.storage.from_(BUCKET_O).get_public_url(dest)

def burn_with_ffmpeg(src_mp4: pathlib.Path, srt_p: pathlib.Path, out_mp4: pathlib.Path):
    srt_esc = srt_p.as_posix().replace("'", r"'\''")
    vf = f"subtitles=filename='{srt_esc}':charenc=UTF-8:force_style='{CAPTION_STYLE}'"
    cmd = ["ffmpeg", "-y", "-i", src_mp4.as_posix(), "-vf", vf, "-c:a", "copy", out_mp4.as_posix()]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def mux_softsub(src_mp4: pathlib.Path, srt_p: pathlib.Path, out_mp4: pathlib.Path):
    cmd = ["ffmpeg", "-y", "-i", src_mp4.as_posix(), "-i", srt_p.as_posix(), "-c", "copy", "-c:s", "mov_text", out_mp4.as_posix()]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def transcribe_whisper1(mp4_path: pathlib.Path) -> List[Dict[str,Any]]:
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {
        "file": (mp4_path.name, open(mp4_path, "rb"), "audio/mpeg"),
        "model": (None, "whisper-1"),
        "response_format": (None, "verbose_json")
    }
    r = requests.post(url, headers=headers, files=files, timeout=1800)
    r.raise_for_status()
    data = r.json()
    segs = data.get("segments") or []
    if not segs and data.get("text"):
        segs = [{"start": 0.0, "end": 0.0, "text": data["text"]}]
    # Ensure floats/strings shape
    out = []
    for seg in segs:
        out.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": str(seg.get("text", "")).strip()
        })
    return out

# ---------- FLASK ----------
@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/process")
def process():
    # Auth
    if request.headers.get("X-API-KEY") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401

    t0 = time.time()
    data = request.get_json(force=True)
    job_id     = data.get("job_id")
    video_path = data["video_path"]                # storage path or full URL
    safe_name  = data.get("safe_name", "lesson")
    burn       = bool(data.get("burn", False))

    # Update job (optional)
    try:
        if job_id:
            sb.table("jobs").update({"status": "processing", "progress": 5}).eq("id", job_id).execute()
    except Exception:
        pass

    work = pathlib.Path(tempfile.mkdtemp(prefix="captioner_"))
    in_mp4 = work / "in.mp4"
    txt_p  = work / f"{safe_name}.txt"
    vtt_p  = work / f"{safe_name}.vtt"
    srt_p  = work / f"{safe_name}.srt"
    soft_p = work / f"{safe_name}_softsub.mp4"
    burn_p = work / f"{safe_name}_burned_bw.mp4"

    urls = {}
    try:
        # 1) download
        url = to_download_url(video_path)
        dl(url, in_mp4)

        # 2) transcribe -> segments
        segs = transcribe_whisper1(in_mp4)
        if not segs:
            raise RuntimeError("whisper-1 returned no segments")
        build_text_assets(segs, txt_p, vtt_p, srt_p)

        # mid-progress
        try:
            if job_id:
                sb.table("jobs").update({"progress": 60}).eq("id", job_id).execute()
        except Exception:
            pass

        # 3) soft-sub always
        mux_softsub(in_mp4, srt_p, soft_p)
        urls["softsub_url"] = upload(soft_p, f"{safe_name}_softsub.mp4", "video/mp4")

        # 4) optional burned-in
        if burn:
            burn_with_ffmpeg(in_mp4, srt_p, burn_p)
            urls["burned_url"] = upload(burn_p, f"{safe_name}_burned_bw.mp4", "video/mp4")

        # 5) upload text assets
        urls["srt_url"] = upload(srt_p, f"{safe_name}.srt", "text/plain")
        urls["vtt_url"] = upload(vtt_p, f"{safe_name}.vtt", "text/vtt")
        urls["txt_url"] = upload(txt_p, f"{safe_name}.txt", "text/plain")

        # job done
        try:
            if job_id:
                sb.table("jobs").update({"status":"done", "progress":100, **urls}).eq("id", job_id).execute()
        except Exception:
            pass

        elapsed = int(time.time() - t0)
        return jsonify({"ok": True, "elapsed_sec": elapsed, "safe_name": safe_name, **urls})

    except requests.HTTPError as e:
        msg = f"http {e.response.status_code}: {e.response.text[:300]}"
        if job_id:
            sb.table("jobs").update({"status":"error","error":msg}).eq("id", job_id).execute()
        return jsonify({"error": msg}), 502
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if job_id:
            sb.table("jobs").update({"status":"error","error":msg}).eq("id", job_id).execute()
        return jsonify({"error": msg}), 500
    finally:
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)