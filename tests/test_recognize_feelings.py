import json
import re

import pytest

from recognize_feelings import express, how_people_feel, main_characters, mentions_a_person


class FakeQwen:
    """Streams the given replies in turn, and records each call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def stream(self, messages, temperature, max_tokens):
        self.calls.append({"prompt": messages[-1]["content"], "temperature": temperature})
        yield from re.findall(r"\S+\s*", self.replies.pop(0))


class FakeRecognizer:
    def __init__(self, people):
        self.people, self.calls = people, 0

    def find_people(self, image):
        self.calls += 1
        return [dict(p) for p in self.people]


class FakeEmotions:
    """Reads the emotion written on each fake face, one face per call."""

    def __init__(self):
        self.faces = []

    def recognize(self, face):
        self.faces.append(face)
        return {"emotion": face, "valence": 0.0, "arousal": 0.0}


MAN = {"human": "a man with a beard", "position": "left", "face": "Anger"}
GIRL = {"human": "a girl crying", "position": "right", "face": "Sadness"}


@pytest.mark.parametrize(
    "description, person",
    [
        ("A woman kneels next to a dog lying on the road.", True),
        ("Two men and a little girl sit on a bench.", True),
        ("A mother bear and her two cubs on a rocky hillside.", False),
        ("A close-up of a pile of fresh strawberries.", False),
        ("Three glasses of whiskey on a black background.", False),
    ],
)
def test_mentions_a_person(description, person):
    assert mentions_a_person(description) is person


def test_each_persons_face_is_read_then_qwen_says_it_once():
    qwen, emotions = FakeQwen("The man is angry, and the girl is sad."), FakeEmotions()
    result = how_people_feel(qwen, FakeRecognizer([MAN, GIRL]), emotions, "image", "A man and a girl.")
    assert emotions.faces == ["Anger", "Sadness"]  # one face at a time
    assert len(qwen.calls) == 1 and qwen.calls[0]["temperature"] == 0
    assert json.loads(qwen.calls[0]["prompt"]) == [
        {"human": "a man with a beard", "feeling": "angry"},
        {"human": "a girl crying", "feeling": "sad"},
    ]
    assert result["answer"] == "The man is angry, and the girl is sad."
    assert [p["emotion"] for p in result["people"]] == ["Anger", "Sadness"]
    assert all("face" not in p for p in result["people"])


def test_no_person_in_the_description_means_nothing_runs():
    qwen, recognizer, emotions = FakeQwen(), FakeRecognizer([MAN]), FakeEmotions()
    result = how_people_feel(qwen, recognizer, emotions, "image", "A brown dog sleeps on a couch.")
    assert result == {"people": [], "answer": None}
    assert recognizer.calls == 0 and emotions.faces == [] and qwen.calls == []


def test_nothing_is_said_without_a_main_characters_face():
    qwen, emotions = FakeQwen(), FakeEmotions()
    result = how_people_feel(qwen, FakeRecognizer([]), emotions, "image", "A man seen from behind.")
    assert result == {"people": [], "answer": None}
    assert emotions.faces == [] and qwen.calls == []


def test_express_sends_who_and_feeling_as_json_and_streams_the_answer():
    qwen = FakeQwen("The woman looks sad.")
    pieces = list(express(qwen, [{"human": "a woman in a red dress", "emotion": "Sadness"}]))
    assert pieces == ["The ", "woman ", "looks ", "sad."]
    assert json.loads(qwen.calls[0]["prompt"]) == [{"human": "a woman in a red dress", "feeling": "sad"}]


def test_express_falls_back_to_plain_sentences_when_qwen_says_nothing():
    pieces = express(FakeQwen(""), [{"human": "a man with a beard", "emotion": "Anger"}])
    assert "".join(pieces) == "A man with a beard looks angry."


def test_main_characters_drop_background_faces():
    main = {"position": "center", "box": [100, 100, 300, 350]}  # 250 px tall
    crowd = {"position": "left", "box": [10, 500, 30, 525]}  # 25 px tall
    assert main_characters([main, crowd], image_height=1000) == [main]
