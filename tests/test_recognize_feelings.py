import json
import re

import pytest

from recognize_feelings import express, feelings_events, mentions_a_person


class FakeQwen:
    """
    Streams the given replies in turn, and records each call. Asked whether a
    description mentions a person, answers `person` (None: no answer).
    """

    def __init__(self, *replies, person=True):
        self.replies, self.person = list(replies), person
        self.calls, self.asked = [], []

    def ask(self, messages, schema, temperature, max_tokens):
        self.asked.append(messages[-1]["content"])
        return None if self.person is None else {"mentioned": self.person}

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


MAN = {"human": "a man with a beard", "face": "Anger"}
GIRL = {"human": "a girl crying", "face": "Sadness"}


@pytest.mark.parametrize("answer, expected", [(True, True), (False, False), (None, True)])
def test_mentions_a_person_asks_qwen_and_looks_anyway_without_an_answer(answer, expected):
    qwen = FakeQwen(person=answer)
    assert mentions_a_person(qwen, "The image shows a mother and her son on a bed.") is expected
    assert qwen.asked == ["You saw a mother and her son on a bed. Did you see any person?"]


def run(qwen, people, description, emotions=None):
    """The events /analyze would send for these people, and the fake models."""
    recognizer, emotions = FakeRecognizer(people), emotions or FakeEmotions()
    events = list(feelings_events(qwen, recognizer, emotions, "image", description))
    return events, recognizer, emotions


def test_each_persons_face_is_read_then_qwen_says_it_once():
    qwen = FakeQwen("The man is angry, and the girl is sad.")
    events, _, emotions = run(qwen, [MAN, GIRL], "A man and a girl.")
    assert emotions.faces == ["Anger", "Sadness"]  # one face at a time
    assert len(qwen.calls) == 1 and qwen.calls[0]["temperature"] == 0
    assert json.loads(qwen.calls[0]["prompt"]) == [
        {"human": "a man with a beard", "feeling": "angry"},
        {"human": "a girl crying", "feeling": "sad"},
    ]
    assert events[0] == {"status": "Looking at their faces…", "for": "feelings"}
    people = events[1]["people"]
    assert [p["emotion"] for p in people] == ["Anger", "Sadness"]
    assert all("face" not in p for p in people)
    assert "".join(e["feelings"] for e in events[2:]) == "The man is angry, and the girl is sad."


def test_no_person_in_the_description_means_nothing_runs():
    qwen = FakeQwen(person=False)
    events, recognizer, emotions = run(qwen, [MAN], "A brown dog sleeps on a couch.")
    assert events == [{"people": []}]
    assert recognizer.calls == 0 and emotions.faces == [] and qwen.calls == []


def test_nothing_is_said_without_a_main_characters_face():
    qwen = FakeQwen()
    events, _, emotions = run(qwen, [], "A man seen from behind.")
    assert events == [{"status": "Looking at their faces…", "for": "feelings"}, {"people": []}]
    assert emotions.faces == [] and qwen.calls == []


def test_a_face_that_cant_be_read_leaves_out_only_that_person():
    class OneBadFace(FakeEmotions):
        def recognize(self, face):
            if face == "Anger":
                raise ValueError("empty crop")
            return super().recognize(face)

    events, _, _ = run(FakeQwen("The girl is sad."), [MAN, GIRL], "A man and a girl.", emotions=OneBadFace())
    assert [p["human"] for p in events[1]["people"]] == ["a girl crying"]


def test_express_sends_who_and_feeling_as_json_and_streams_the_answer():
    qwen = FakeQwen("The woman looks sad.")
    pieces = list(express(qwen, [{"human": "a woman in a red dress", "emotion": "Sadness"}]))
    assert pieces == ["The ", "woman ", "looks ", "sad."]
    assert json.loads(qwen.calls[0]["prompt"]) == [{"human": "a woman in a red dress", "feeling": "sad"}]


def test_express_falls_back_to_plain_sentences_when_qwen_says_nothing():
    pieces = express(FakeQwen(""), [{"human": "a man with a beard", "emotion": "Anger"}])
    assert "".join(pieces) == "A man with a beard looks angry."
