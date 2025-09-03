from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(ok=True), 200

@app.get("/")
def index():
    return "Captioneer worker is up", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
