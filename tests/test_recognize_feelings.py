import json
import re

from recognize_feelings import express, feelings_events


class FakeQwen:
    """Streams the given replies in turn, and records each call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def stream(self, messages, temperature, max_tokens):
        self.calls.append({"prompt": messages[-1]["content"], "temperature": temperature})
        yield from re.findall(r"\S+\s*", self.replies.pop(0))


class FakeRecognizer:
    def __init__(self, characters):
        self.characters, self.calls = characters, 0

    def find_characters(self, image):
        self.calls += 1
        return [dict(c) for c in self.characters]


class FakeEmotions:
    """Reads the emotion written on each fake face, one face per call."""

    def __init__(self):
        self.faces = []

    def recognize(self, face):
        self.faces.append(face)
        return {"emotion": face, "valence": 0.0, "arousal": 0.0}


MAN = {"character": "a man with a beard", "face": "Anger"}
GIRL = {"character": "a girl crying", "face": "Sadness"}


def run(qwen, characters, emotions=None):
    """The events /analyze would send for these characters, and the fake models."""
    recognizer, emotions = FakeRecognizer(characters), emotions or FakeEmotions()
    events = list(feelings_events(qwen, recognizer, emotions, "image"))
    return events, recognizer, emotions


def test_each_characters_face_is_read_then_qwen_says_it_once():
    qwen = FakeQwen("The man is angry, and the girl is sad.")
    events, _, emotions = run(qwen, [MAN, GIRL])
    assert emotions.faces == ["Anger", "Sadness"]  # one face at a time
    assert len(qwen.calls) == 1 and qwen.calls[0]["temperature"] == 0
    assert json.loads(qwen.calls[0]["prompt"]) == [
        {"character": "a man with a beard", "feeling": "angry"},
        {"character": "a girl crying", "feeling": "sad"},
    ]
    assert events[0] == {"status": "Looking at their faces…", "for": "feelings"}
    characters = events[1]["characters"]
    assert [c["emotion"] for c in characters] == ["Anger", "Sadness"]
    assert all("face" not in c for c in characters)
    assert "".join(e["feelings"] for e in events[2:]) == "The man is angry, and the girl is sad."


def test_an_animals_feeling_is_recognized():
    dog = {"character": "a brown dog on a red couch", "face": "Fear"}
    events, _, _ = run(FakeQwen("The dog is afraid."), [dog])
    assert [c["emotion"] for c in events[1]["characters"]] == ["Fear"]


def test_faces_are_always_looked_for_and_nothing_is_said_without_one():
    qwen = FakeQwen()
    events, recognizer, emotions = run(qwen, [])
    assert recognizer.calls == 1
    assert events == [{"status": "Looking at their faces…", "for": "feelings"}, {"characters": []}]
    assert emotions.faces == [] and qwen.calls == []


def test_a_face_that_cant_be_read_leaves_out_only_that_character():
    class OneBadFace(FakeEmotions):
        def recognize(self, face):
            if face == "Anger":
                raise ValueError("empty crop")
            return super().recognize(face)

    events, _, _ = run(FakeQwen("The girl is sad."), [MAN, GIRL], emotions=OneBadFace())
    assert [c["character"] for c in events[1]["characters"]] == ["a girl crying"]


def test_express_sends_who_and_feeling_as_json_and_streams_the_answer():
    qwen = FakeQwen("The woman looks sad.")
    pieces = list(express(qwen, [{"character": "a woman in a red dress", "emotion": "Sadness"}]))
    assert pieces == ["The ", "woman ", "looks ", "sad."]
    assert json.loads(qwen.calls[0]["prompt"]) == [{"character": "a woman in a red dress", "feeling": "sad"}]


def test_express_falls_back_to_plain_sentences_when_qwen_says_nothing():
    pieces = express(FakeQwen(""), [{"character": "a man with a beard", "emotion": "Anger"}])
    assert "".join(pieces) == "A man with a beard looks angry."
