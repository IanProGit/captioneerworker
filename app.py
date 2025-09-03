from flask import Flask, jsonify, request
import os

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(ok=True), 200

@app.get("/")
def index():
    return "Captioneer worker is up", 200

@app.post("/enqueue")
def enqueue():
    # Validate Authorization header against WORKER_TOKEN env var
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {os.environ.get('WORKER_TOKEN')}":
        return jsonify(error="unauthorized"), 401

    # Expect JSON with { "job_id": "..." }
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    if not job_id:
        return jsonify(error="missing job_id"), 400

    # For now just acknowledge
    return jsonify(accepted=True, id=job_id), 202

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)