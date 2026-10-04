import json
import os
import random
import threading
from contextlib import contextmanager

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from itsdangerous import BadSignature, URLSafeTimedSerializer
from PIL import Image, UnidentifiedImageError
from werkzeug.middleware.proxy_fix import ProxyFix

from emotions import EmotionRecognizer
from llm import Qwen, Turns
from narrate import how_i_feel, story_intention, what_i_saw
from recognize_feelings import feelings_events
from recognizer import MOODS, ImageRecognizer
from writer import GENRES, MAX_IDEAS, StoryWriter

MAX_SIDE = 1600  # downscale big images; the model resizes anyway
# Bound on the description the page sends back, to keep prompts small.
MAX_DESCRIPTION = 2000  # detailed descriptions run to about 700 characters
WRITER_WAIT = 30  # seconds a story request waits for the writer before "busy"
# Seconds a sentence of /analyze waits for Qwen (busy writing a story) before
# it falls back to a plain sentence; describing a picture never fails on it.
ANALYZE_WAIT = 5
IDEA_SECONDS = 60 * 60  # how long a planned idea can still be written
GENRE_CHOICES = 3  # genres the page offers for the story

app = Flask(__name__, template_folder="web")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB uploads
app.json.sort_keys = False  # keep story ideas in aspect order (genre, then characters)
# Behind a hosting proxy the real client IP is in X-Forwarded-For (needed for rate limiting).
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

recognizer = ImageRecognizer()
emotions = EmotionRecognizer()
qwen = Qwen()  # shared by the story writer and recognize_feelings
writer = StoryWriter(qwen)
# Qwen (llama.cpp) has a single context, so requests that use it take turns.
# Florence-2 and CLIP are safe to run from several threads at once.
writer_lock = threading.Lock()
# Planned ideas are signed, so /story/write only writes what /story/plan
# planned, not any text a client sends. With more than one server process,
# set SECRET_KEY so an idea planned by one can be written by another.
signer = URLSafeTimedSerializer(os.environ.get("SECRET_KEY") or os.urandom(32))


class WriterBusy(Exception):
    pass


@contextmanager
def writer_turn():
    """Hold the story writer, or raise WriterBusy after WRITER_WAIT seconds."""
    if not writer_lock.acquire(timeout=WRITER_WAIT):
        raise WriterBusy
    try:
        yield
    finally:
        writer_lock.release()


@app.get("/")
def index():
    response = app.make_response(render_template("index.html"))
    # Always fetch the current page: a cached copy talks to the server in an
    # outdated way after an update.
    response.headers["Cache-Control"] = "no-store"
    return response


def _read_image():
    """(PIL image, None) from the uploaded "file", or (None, error response)."""
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return None, (jsonify(error="Choose or paste an image first."), 400)
    try:
        image = Image.open(upload.stream).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        return None, (jsonify(error="That file isn't a readable image."), 400)
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image, None


def _read_mood(mood):
    if not isinstance(mood, str) or mood not in MOODS:
        return jsonify(error="Pick one of the listed moods."), 400
    return None


def _read_genre(genre):
    if genre is not None and genre not in GENRES:
        return jsonify(error="Pick one of the listed genres."), 400
    return None


def _read_description(description):
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > MAX_DESCRIPTION
    ):
        return jsonify(error="Describe an image first."), 400
    return None


@app.post("/analyze")
@limiter.limit("10 per minute; 100 per day")
def analyze():
    """
    What the app makes of a picture, streamed as JSON lines so the page can
    show each part as soon as it's ready: the caption and description, then
    what the app says, section by section, each as Qwen writes it
    ({"saw": piece}, {"feelings": piece}, ...). Before a section, a status
    says what the app is doing ({"status": "Letting it sink in…", "for":
    "i_feel"}); the moods and genres to pick from follow their sentences.
    The last line is {"done": true}, or {"error": ...} if something failed.
    """
    image, error = _read_image()
    if error:
        return error

    def events():
        caption, description = recognizer.describe(image)
        yield {"caption": caption, "description": description}
        moods = recognizer.moods(image)
        turns = Turns(qwen, writer_lock, ANALYZE_WAIT)
        for piece in what_i_saw(turns, description, caption):
            yield {"saw": piece}
        # How the people in the picture feel; nothing when there's no person.
        yield from feelings_events(turns, recognizer, emotions, image, description)
        yield {"status": "Letting it sink in…", "for": "i_feel"}
        for piece in how_i_feel(turns, description, moods[0][0]):
            yield {"i_feel": piece}
        yield {"moods": [{"mood": m, "score": p} for m, p in moods]}
        yield {"status": "Thinking of a story…", "for": "story_intention"}
        for piece in story_intention(turns, caption):
            yield {"story_intention": piece}
        yield {"genres": random.sample(list(GENRES), GENRE_CHOICES)}
        yield {"done": True}

    def lines():
        # Once streaming has started, errors can't change the status code.
        try:
            for event in events():
                yield json.dumps(event) + "\n"
        except Exception:
            app.logger.exception("analyze failed")
            yield json.dumps({"error": "Something went wrong on the server. Please try again."}) + "\n"

    return Response(stream_with_context(lines()), mimetype="application/x-ndjson")


@app.post("/story/plan")
@limiter.limit("10 per minute; 100 per day")
def story_plan():
    """
    Brainstorm and pick up to MAX_IDEAS story ideas, plus as many spares; the
    page then writes each. Takes the image and its description from
    /analyze: ideas start from the description and are checked against the
    image with CLIP. Each idea comes with a token to write it with.
    """
    image, error = _read_image()
    mood, description = request.form.get("mood"), request.form.get("description")
    genre = request.form.get("genre")
    error = error or _read_mood(mood) or _read_genre(genre) or _read_description(description)
    if error:
        return error
    count = request.form.get("count", str(MAX_IDEAS))
    if count not in {str(n) for n in range(1, MAX_IDEAS + 1)}:
        return jsonify(error=f"Ask for 1 to {MAX_IDEAS} ideas."), 400

    description = description.strip()
    image_vec = recognizer.image_vector(image)  # once, for every grounding check
    with writer_turn():
        plan = writer.plan(
            description,
            mood,
            int(count),
            match=lambda texts: recognizer.match(image_vec, texts),
            genre=genre,
        )
    if plan is None:
        return jsonify(error="Couldn't come up with ideas that fit this picture. Try again."), 503

    def signed(idea):
        token = signer.dumps({"description": description, "mood": mood, "idea": idea})
        return {"idea": idea, "token": token}

    return jsonify(
        options=plan["options"],
        rejected=plan["rejected"],
        ideas=[signed(idea) for idea in plan["ideas"]],
        spares=[signed(idea) for idea in plan["spares"]],
    )


@app.post("/story/write")
@limiter.limit("30 per minute; 300 per day")
def story_write():
    """
    Write the story seed for one idea from /story/plan, given its token.
    422 means this idea didn't work out, and the page tries a spare instead.
    """
    token = (request.get_json(silent=True) or {}).get("token")
    try:
        if not isinstance(token, str):
            raise BadSignature("no token")
        planned = signer.loads(token, max_age=IDEA_SECONDS)
    except BadSignature:  # includes expired tokens
        return jsonify(error="That story idea isn't valid or has expired. Brainstorm again."), 400

    with writer_turn():
        seed = writer.write(planned["description"], planned["mood"], planned["idea"])
    if seed is None:
        return jsonify(error="Couldn't write this story."), 422
    return jsonify(story=seed)


@app.errorhandler(429)
def too_many(_):
    return jsonify(error="Too many requests. Please wait a minute and try again."), 429


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="That file is too large (16MB max)."), 413


@app.errorhandler(WriterBusy)
def writer_busy(_):
    return jsonify(error="The story writer is busy with another request. Try again in a minute."), 503


@app.errorhandler(500)
def server_error(_):
    return jsonify(error="Something went wrong on the server. Please try again."), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
