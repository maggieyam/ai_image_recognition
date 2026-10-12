import re

import pytest
import torch
from PIL import Image

from brain.affect.mood import MOODS
from brain.brain import Brain, build_faculties
from brain.networks.qwen import Busy, SharedQwen


class FakeFlorence:
    def describe(self, image):
        return "The image shows a man with a beard. He is smiling."

    def caption(self, image):
        return "A man with a beard."

    def find_faces(self, image):
        return []


class FakeYuNet:
    def __init__(self, boxes):
        self.boxes = boxes

    def find_faces(self, image):
        return self.boxes


class FakeEmotions:
    def recognize(self, face):
        return {"emotion": "Happiness", "valence": 0.6, "arousal": 0.3}


class FakeClip:
    """Every picture feels mostly playful; texts match a picture by their length."""

    def text_vectors(self, texts):
        return torch.tensor([[float(any(s in t for s in synonyms)) for synonyms in MOODS.values()] for t in texts])

    def image_vector(self, image):
        return torch.tensor([float(mood == "playful") for mood in MOODS])

    def probabilities(self, image_vec, label_vecs):
        return (10 * label_vecs @ image_vec).softmax(dim=0)

    def match(self, image_vec, texts):
        return [len(t) / 100 for t in texts]


class FakeQwen:
    """Says the same thing every time, in pieces."""

    def stream(self, messages, temperature, max_tokens):
        yield from re.findall(r"\S+\s*", "Hello there.")


def brain_with(faces):
    """A Brain with fake networks; `faces` are the face boxes YuNet finds."""
    brain = Brain.__new__(Brain)
    brain.clip, brain.qwen = FakeClip(), SharedQwen(FakeQwen())
    brain.faculties = build_faculties(FakeFlorence(), brain.clip, FakeYuNet(faces), FakeEmotions())
    return brain


def kinds(events):
    """What each event is, in order, with the pieces of one sentence counted once."""
    names = [f"status for {e['for']}" if "status" in e else next(iter(e)) for e in events]
    return [n for i, n in enumerate(names) if i == 0 or n != names[i - 1]]


def test_what_the_brain_makes_of_a_picture_in_the_order_the_page_shows_it():
    events = list(brain_with(faces=[[100, 100, 300, 400]]).perceive(image=Image.new("RGB", (500, 500))))
    assert kinds(events) == [
        "caption",
        "saw",
        "status for feelings", "characters", "feelings",
        "status for i_feel", "i_feel", "moods",
        "status for story_intention", "story_intention", "genres",
    ]
    assert events[0] == {
        "caption": "A man with a beard.",
        "description": "The image shows a man with a beard. He is smiling.",
    }
    characters = next(e["characters"] for e in events if "characters" in e)
    assert characters == [{
        "box": [100, 100, 300, 400], "character": "a man with a beard",
        "emotion": "Happiness", "valence": 0.6, "arousal": 0.3,
    }]
    assert next(e["moods"] for e in events if "moods" in e)[0]["mood"] == "playful"
    assert "".join(e.get("saw", "") for e in events) == "Hello there."


def test_nothing_is_said_about_feelings_when_no_face_is_found():
    events = list(brain_with(faces=[]).perceive(image=Image.new("RGB", (500, 500))))
    assert {"characters": []} in events and not any("feelings" in e for e in events)
    assert kinds(events)[-1] == "genres"  # the rest still follows


def test_what_the_brain_says_is_plain_while_a_story_holds_qwen(monkeypatch):
    monkeypatch.setattr("brain.brain.SPEECH_WAIT", 0.1)
    brain = brain_with(faces=[])
    with brain.qwen.hold(wait=1):
        events = list(brain.perceive(image=Image.new("RGB", (500, 500))))
    assert "".join(e.get("saw", "") for e in events) == "I saw a man with a beard."
    assert "".join(e.get("i_feel", "") for e in events) == "It makes me feel playful."


class FakeWriter:
    def plan(self, description, mood, count, match, genre):
        return {"scores": match(["a nurse", "a retired police officer"]), "count": count, "genre": genre}

    def write(self, description, mood, idea):
        return {"title": f"{mood} {idea['genre']}"}


def test_stories_are_checked_against_the_picture_and_written_while_holding_qwen():
    brain = brain_with(faces=[])
    brain.writer = FakeWriter()
    plan = brain.plan_stories("image", "A man.", "happy", 2, "comedy")
    assert plan == {"scores": [0.07, 0.24], "count": 2, "genre": "comedy"}
    assert brain.write_story("A man.", "happy", {"genre": "comedy"}) == {"title": "happy comedy"}
    assert not brain.qwen.lock.locked()


def test_a_story_is_told_the_brain_is_busy_when_qwen_stays_taken(monkeypatch):
    monkeypatch.setattr("brain.brain.STORY_WAIT", 0.1)
    brain = brain_with(faces=[])
    brain.writer = FakeWriter()
    with brain.qwen.hold(wait=1):
        with pytest.raises(Busy):
            brain.write_story("A man.", "happy", {"genre": "comedy"})
