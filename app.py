import json
import os

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from itsdangerous import BadSignature, URLSafeTimedSerializer
from werkzeug.middleware.proxy_fix import ProxyFix

from brain.brain import GENRES, MAX_IDEAS, MOODS, Brain, Busy
from effectors import mouth
from senses import eye

# Bound on the description the page sends back, to keep prompts small.
MAX_DESCRIPTION = 2000  # detailed descriptions run to about 700 characters
IDEA_SECONDS = 60 * 60  # how long a planned idea can still be written
MAX_SPOKEN = 1000  # characters said out loud at a time; everything said about a picture runs to about 600

app = Flask(__name__, template_folder="web")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB uploads
app.json.sort_keys = False  # keep story ideas in aspect order (genre, then characters)
# Behind a hosting proxy the real client IP is in X-Forwarded-For (needed for rate limiting).
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

# Speaking out loud comes out of this machine's speaker, whoever asks. Run as
# `python app.py` only this machine can ask, so it's on if there's a speaker;
# served to others (gunicorn), it's off unless SPEAK_ALOUD=1 (a robot, say).
# VOICE picks the voice it speaks in, from brain/networks/voices.py.
SPEAKS = __name__ == "__main__" or os.environ.get("SPEAK_ALOUD") == "1"

# Everything the app makes of a picture; this file is only its web server.
brain = Brain(mouth=mouth.find() if SPEAKS else None, voice=os.environ.get("VOICE"))
# Planned ideas are signed, so /story/write only writes what /story/plan
# planned, not any text a client sends. With more than one server process,
# set SECRET_KEY so an idea planned by one can be written by another.
signer = URLSafeTimedSerializer(os.environ.get("SECRET_KEY") or os.urandom(32))


@app.get("/")
def index():
    # The page only offers to speak out loud when there's a mouth to do it with.
    response = app.make_response(render_template("index.html", speaks=brain.mouth is not None))
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
        return eye.see(upload.stream), None
    except eye.Unreadable:
        return None, (jsonify(error="That file isn't a readable image."), 400)


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
    what the app says, section by section, each once Qwen has written it
    ({"saw": "I saw ..."}, {"feelings": ...}, ...). Before a section, a status
    says what the app is doing ({"status": "Letting it sink in…", "for":
    "i_feel"}); the moods and genres to pick from follow their sentences.
    The last line is {"done": true}, or {"error": ...} if something failed.
    """
    image, error = _read_image()
    if error:
        return error

    def lines():
        # Once streaming has started, errors can't change the status code.
        try:
            # Coming from the page, it's a picture; the brain can't tell that by looking.
            for event in brain.perceive(image=image, looking_at="a picture the user showed you"):
                yield json.dumps(event) + "\n"
            yield json.dumps({"done": True}) + "\n"
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
    plan = brain.plan_stories(image, description, mood, int(count), genre)
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

    seed = brain.write_story(planned["description"], planned["mood"], planned["idea"])
    if seed is None:
        return jsonify(error="Couldn't write this story."), 422
    return jsonify(story=seed)


@app.post("/speak")
@limiter.limit("40 per minute; 400 per day")  # up to four things are said about each picture
def speak():
    """
    The user would like to hear this out loud: a hint, which the brain takes
    if it has a mouth. The sound comes out of the speaker of the machine the
    app runs on, not the user's device, after anything still being said.
    Answers at once, {"aloud": false} if there's no mouth.
    """
    text = (request.get_json(silent=True) or {}).get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_SPOKEN:
        return jsonify(error="There's nothing to say out loud."), 400
    return jsonify(aloud=brain.say_aloud(text.strip()))


@app.post("/hush")
@limiter.limit("40 per minute; 400 per day")
def hush():
    """Stop speaking out loud, mid-word, and drop what was still to be said."""
    brain.hush()
    return "", 204


@app.errorhandler(429)
def too_many(_):
    return jsonify(error="Too many requests. Please wait a minute and try again."), 429


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="That file is too large (16MB max)."), 413


@app.errorhandler(Busy)
def writer_busy(_):
    return jsonify(error="The story writer is busy with another request. Try again in a minute."), 503


@app.errorhandler(500)
def server_error(_):
    return jsonify(error="Something went wrong on the server. Please try again."), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
