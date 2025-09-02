from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

# Process endpoint (stub for now)
@app.route("/process", methods=["POST"])
def process():
    api_key = request.headers.get("X-API-KEY")
    if api_key != os.environ.get("WORKER_API_KEY"):
        return jsonify({"error": "Unauthorized"}), 403

    data = request.json
    job_id = data.get("job_id")
    video_path = data.get("video_path")
    safe_name = data.get("safe_name")

    # Placeholder: In future we’ll add ffmpeg + transcription logic here
    return jsonify({
        "status": "processing started",
        "job_id": job_id,
        "video_path": video_path,
        "safe_name": safe_name
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)