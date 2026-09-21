import os
import threading

from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image, UnidentifiedImageError
from werkzeug.middleware.proxy_fix import ProxyFix

from recognizer import ImageRecognizer

MAX_SIDE = 1600  # downscale big images; the model resizes anyway

app = Flask(__name__, template_folder="web")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB uploads
# Behind a hosting proxy the real client IP is in X-Forwarded-For (needed for rate limiting).
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

recognizer = ImageRecognizer()
# One shared model; serialize calls so concurrent requests don't pile onto the CPU.
model_lock = threading.Lock()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
@limiter.limit("10 per minute; 100 per day")
def analyze():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return jsonify(error="Choose or paste an image first."), 400

    try:
        image = Image.open(upload.stream).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        return jsonify(error="That file isn't a readable image."), 400

    image.thumbnail((MAX_SIDE, MAX_SIDE))

    with model_lock:
        return jsonify(caption=recognizer.describe(image))


@app.errorhandler(429)
def too_many(_):
    return jsonify(error="Too many requests. Please wait a minute and try again."), 429


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="That file is too large (16MB max)."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
