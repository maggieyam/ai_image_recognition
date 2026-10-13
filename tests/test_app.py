import io
import json
import sys

import pytest
from PIL import Image

import brain.brain
from brain.brain import Busy


class FakeBrain:
    """Stands in for the brain, so the web server is tested without any model."""

    def __init__(self):
        self.perceived, self.fails, self.busy = [], False, False

    def perceive(self, **stimulus):
        self.perceived.append(stimulus)
        yield {"caption": "A grey cat.", "description": "The image shows a grey cat."}
        if self.fails:
            raise RuntimeError("a model crashed")
        yield {"saw": "I saw a grey cat."}

    def write_story(self, description, mood, idea):
        if self.busy:
            raise Busy
        return {"title": f"{mood} {idea['genre']}"}


@pytest.fixture
def web(monkeypatch):
    """app.py, started with a FakeBrain in place of the real one."""
    monkeypatch.setattr(brain.brain, "Brain", FakeBrain)
    sys.modules.pop("app", None)
    import app

    yield app
    sys.modules.pop("app", None)


def picture():
    file = io.BytesIO()
    Image.new("RGB", (40, 30), "grey").save(file, "PNG")
    file.seek(0)
    return file


def analyze(web, file, name="cat.png"):
    response = web.app.test_client().post("/analyze", data={"file": (file, name)})
    lines = [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line]
    return response, lines


def test_the_brain_gets_the_picture_and_the_page_gets_what_it_makes_of_it_then_done(web):
    response, lines = analyze(web, picture())
    assert response.status_code == 200 and response.mimetype == "application/x-ndjson"
    assert lines == [
        {"caption": "A grey cat.", "description": "The image shows a grey cat."},
        {"saw": "I saw a grey cat."},
        {"done": True},
    ]
    (stimulus,) = web.brain.perceived
    assert stimulus["image"].size == (40, 30) and stimulus["image"].mode == "RGB"
    assert stimulus["looking_at"]  # the page says what it is; the brain can't tell by looking


def test_a_file_that_isnt_a_picture_never_reaches_the_brain(web):
    response, lines = analyze(web, io.BytesIO(b"not a picture"), name="notes.txt")
    assert response.status_code == 400 and lines == [{"error": "That file isn't a readable image."}]
    assert web.brain.perceived == []


def test_a_failure_partway_ends_the_stream_with_an_error_not_done(web):
    web.brain.fails = True
    response, lines = analyze(web, picture())
    assert response.status_code == 200  # streaming had started
    assert "caption" in lines[0] and "error" in lines[-1] and {"done": True} not in lines


def test_the_page_is_told_when_the_brain_is_busy_with_another_story(web):
    web.brain.busy = True
    token = web.signer.dumps({"description": "A grey cat.", "mood": "cozy", "idea": {"genre": "comedy"}})
    busy = web.app.test_client().post("/story/write", json={"token": token})
    assert busy.status_code == 503 and "busy" in busy.get_json()["error"]
