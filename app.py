# app.py — Captioneer Worker (Whisper API version)
# Reliable: 3-min videos, soft-subs by default, optional burn later.

import os, pathlib, tempfile, time, shutil, subprocess, requests
from flask import Flask, request, jsonify
from supabase import create_client, Client

# ---------- ENV ----------
SB_URL  = os.environ["SUPABASE_URL"]
SB_KEY  = os.environ["SUPABASE_SERVICE_KEY"]
BUCKET_V = os.environ.get("SUPABASE_VIDEOS_BUCKET", "videos")
BUCKET_O = os.environ.get("SUPABASE_OUTPUTS_BUCKET", "outputs")
API_KEY  = os.environ["WORKER_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CAPTION_STYLE = os.environ.get(
    "CAPTION_STYLE",
    "FontName=Times New Roman,FontSize=16,Outline=2,Shadow=1,BorderStyle=1,Alignment=2,MarginV=10"
)

sb: Client = create_client(SB_URL, SB_KEY)
app = Flask(__name__)

# ---------- HELPERS ----------
def public_url(bucket: str, path: str) -> str:
    return f"{SB_URL}/storage/v1/object/public/{bucket}/{path}"

def to_url(video_path: str) -> str:
    if video_path.startswith("http"): return video_path
    return public_url(BUCKET_V, video_path.replace(f"{BUCKET_V}/",""))

def download(url: str, dst: pathlib.Path):
    r = requests.get(url, stream=True, timeout=600)
    r.raise_for_status()
    with open(dst,"wb") as f:
        for chunk in r.iter_content(1024*256):
            if chunk: f.write(chunk)

def ts_vtt(t: float) -> str:
    h,m,s = int(t//3600), int((t%3600)//60), t%60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def ts_srt(t: float) -> str:
    h,m,s,ms = int(t//3600), int((t%3600)//60), int(t%60), int((t%1)*1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_assets(segments, txt,vtt,srt):
    with open(txt,"w",encoding="utf-8") as f:
        for seg in segments: f.write(seg["text"].strip()+"\n")
    with open(vtt,"w",encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{ts_vtt(seg['start'])} --> {ts_vtt(seg['end'])}\n{seg['text'].strip()}\n\n")
    with open(srt,"w",encoding="utf-8") as f:
        for i,seg in enumerate(segments,1):
            f.write(f"{i}\n{ts_srt(seg['start'])} --> {ts_srt(seg['end'])}\n{seg['text'].strip()}\n\n")

def upload(p: pathlib.Path, dest: str, ctype="text/plain") -> str:
    with open(p,"rb") as f:
        sb.storage.from_(BUCKET_O).upload(dest,f,{"upsert":True,"contentType":ctype})
    return sb.storage.from_(BUCKET_O).get_public_url(dest)

def transcribe(mp4: pathlib.Path):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {
        "file": (mp4.name, open(mp4,"rb"), "audio/mpeg"),
        "model": (None, "whisper-1"),
        "response_format": (None, "verbose_json")
    }
    r = requests.post(url, headers=headers, files=files, timeout=1800)
    r.raise_for_status()
    data = r.json()
    segs = data.get("segments") or []
    if not segs and data.get("text"):
        segs=[{"start":0.0,"end":0.0,"text":data["text"]}]
    return [{"start":float(s["start"]), "end":float(s["end"]), "text":s["text"]} for s in segs]

# ---------- ROUTES ----------
@app.get("/health")
def health(): return jsonify({"ok":True})

@app.post("/process")
def process():
    if request.headers.get("X-API-KEY")!=API_KEY: return jsonify({"error":"unauthorized"}),401
    d=request.get_json(force=True); safe=d.get("safe_name","lesson"); burn=bool(d.get("burn",False))
    vpath=d["video_path"]; t0=time.time()
    work=pathlib.Path(tempfile.mkdtemp()); inmp4=work/"in.mp4"
    txt,vtt,srt=[work/f"{safe}{ext}" for ext in (".txt",".vtt",".srt")]
    soft, burned=[work/f"{safe}_softsub.mp4", work/f"{safe}_burned.mp4"]

    try:
        download(to_url(vpath), inmp4)
        segs=transcribe(inmp4); build_assets(segs,txt,vtt,srt)
        urls={"txt_url":upload(txt,f"{safe}.txt"),
              "vtt_url":upload(vtt,f"{safe}.vtt","text/vtt"),
              "srt_url":upload(srt,f"{safe}.srt")}
        # Soft-sub always
        subprocess.run(["ffmpeg","-y","-i",inmp4.as_posix(),"-i",srt.as_posix(),
                        "-c","copy","-c:s","mov_text",soft.as_posix()],
                        check=True)
        urls["softsub_url"]=upload(soft,f"{safe}_softsub.mp4","video/mp4")
        # Burn optional
        if burn:
            vf=f"subtitles={srt.as_posix()}:charenc=UTF-8:force_style='{CAPTION_STYLE}'"
            subprocess.run(["ffmpeg","-y","-i",inmp4.as_posix(),"-vf",vf,"-c:a","copy",burned.as_posix()],
                           check=True)
            urls["burned_url"]=upload(burned,f"{safe}_burned.mp4","video/mp4")
        return jsonify({"ok":True,"safe_name":safe,"elapsed_sec":int(time.time()-t0),**urls})
    except Exception as e:
        return jsonify({"error":str(e)}),500
    finally:
        shutil.rmtree(work,ignore_errors=True)

if __name__=="__main__":
    port=int(os.environ.get("PORT",8080)); app.run(host="0.0.0.0",port=port)