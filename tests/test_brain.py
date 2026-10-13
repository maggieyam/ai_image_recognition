import json

import torch
from PIL import Image

from brain.affect.mood import MOODS
from brain.brain import Brain, build_faculties
from brain.networks.qwen import SharedQwen


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
    """Every picture feels mostly playful."""

    def text_vectors(self, texts):
        return torch.tensor([[float(any(s in t for s in synonyms)) for synonyms in MOODS.values()] for t in texts])

    def image_vector(self, image):
        return torch.tensor([float(mood == "playful") for mood in MOODS])

    def probabilities(self, image_vec, label_vecs):
        return (10 * label_vecs @ image_vec).softmax(dim=0)


class FakeQwen:
    """Says the same things about every picture, a few characters at a time."""

    def stream(self, messages, temperature, max_tokens, schema=None):
        reply = json.dumps({
            "saw": "I saw a man.", "they_look": "He looks happy.", "i_feel": "I feel playful.",
            "story_offer": "Shall I write a story?",
        })
        yield from (reply[i:i + 9] for i in range(0, len(reply), 9))


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


def test_the_brain_perceives_and_feels_then_says_it_all():
    events = list(brain_with(faces=[[100, 100, 300, 400]]).perceive(image=Image.new("RGB", (500, 500)), looking_at="a picture"))
    assert kinds(events) == [
        "caption",
        "status for feelings", "characters",
        "saw", "feelings",
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
    assert {"saw": "I saw a man."} in events and {"feelings": "He looks happy."} in events
