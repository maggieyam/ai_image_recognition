import json
import re

import pytest

from brain.imagination.story import GENRES
from brain.language.speech import Say, how_i_feel, how_they_feel, offer_story, say, what_i_saw
from brain.workspace import Workspace


class FakeQwen:
    """Streams the given reply in pieces, and records each call."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def stream(self, messages, temperature, max_tokens):
        self.calls.append({"facts": messages[1]["content"], "temperature": temperature})
        yield from re.findall(r"\S+\s*", self.reply)


DESCRIPTION = "A man with a beard takes a close-up selfie, smiling."
CAPTION = "A close-up selfie of a man's face."
MOODS = [("playful", 0.5), ("happy", 0.12)]
MAN = {"character": "a man with a beard", "emotion": "Anger"}
WOMAN = {"character": "a woman in a red dress", "emotion": "Sadness"}


def words(events, section):
    """What was said in a section, from its events."""
    return "".join(e.get(section, "") for e in events)


def test_each_section_is_one_qwen_call_with_only_its_facts():
    qwen = FakeQwen("Hello there")
    for section, said in (
        ("saw", what_i_saw(qwen, DESCRIPTION, CAPTION)),
        ("feelings", how_they_feel(qwen, [MAN, WOMAN])),
        ("i_feel", how_i_feel(qwen, DESCRIPTION, MOODS)),
        ("story_intention", offer_story(qwen, CAPTION)),
    ):
        pieces = [e[section] for e in said if section in e]
        assert pieces == ["Hello ", "there"]  # streamed piece by piece
    assert [c["temperature"] for c in qwen.calls] == [0, 0, 0, 0]
    assert qwen.calls[0]["facts"] == f"Description: {DESCRIPTION}"
    assert json.loads(qwen.calls[1]["facts"]) == [
        {"character": "a man with a beard", "feeling": "angry"},
        {"character": "a woman in a red dress", "feeling": "sad"},
    ]
    assert qwen.calls[2]["facts"] == f"What you saw: {DESCRIPTION}\nHow it made you feel: playful"
    assert qwen.calls[3]["facts"] == f"The picture: {CAPTION}"  # only the caption; no genres


@pytest.mark.parametrize(
    "section, said, expected",
    [
        ("saw", lambda q: what_i_saw(q, DESCRIPTION, CAPTION), "I saw a close-up selfie of a man's face."),
        ("feelings", lambda q: how_they_feel(q, [MAN]), "A man with a beard looks angry."),
        ("i_feel", lambda q: how_i_feel(q, DESCRIPTION, MOODS), "It makes me feel playful."),
        ("story_intention", lambda q: offer_story(q, CAPTION),
         "I'd like to write you a story about it. Which genre would you like?"),
    ],
)
def test_plain_sentence_when_qwen_says_nothing(section, said, expected):
    assert words(said(FakeQwen("")), section) == expected


def test_say_falls_back_when_qwen_says_nothing():
    assert "".join(say(FakeQwen(""), [None, {"content": ""}], 10, "It makes me feel calm.")) == "It makes me feel calm."
    assert "".join(say(FakeQwen("I feel calm."), [None, {"content": ""}], 10, "fallback")) == "I feel calm."


def test_nothing_is_said_about_feelings_without_a_character():
    qwen = FakeQwen("Nobody is here.")
    assert list(how_they_feel(qwen, [])) == [] and qwen.calls == []


def test_how_i_feel_is_about_the_top_mood_and_the_moods_follow_the_sentence():
    events = list(how_i_feel(FakeQwen("I feel playful."), DESCRIPTION, MOODS))
    assert events[0] == {"status": "Letting it sink in…", "for": "i_feel"}
    assert words(events[1:-1], "i_feel") == "I feel playful."
    assert events[-1] == {"moods": [{"mood": "playful", "score": 0.5}, {"mood": "happy", "score": 0.12}]}


def test_the_story_offer_is_followed_by_genres_to_pick_from():
    events = list(offer_story(FakeQwen("Shall I write one?"), CAPTION))
    assert events[0] == {"status": "Thinking of a story…", "for": "story_intention"}
    assert words(events[1:-1], "story_intention") == "Shall I write one?"
    offered = events[-1]["genres"]
    assert len(set(offered)) == 3 and set(offered) <= set(GENRES)


def test_say_gives_a_section_its_facts_from_the_workspace_and_the_voice():
    saying = Say(what_i_saw, needs=("description", "caption"))
    assert saying.needs == ("voice", "description", "caption") and str(saying) == "Say what_i_saw"
    qwen = FakeQwen("I saw a man.")
    events = list(saying.run(Workspace(voice=qwen, description=DESCRIPTION, caption=CAPTION)))
    assert events == [{"saw": "I "}, {"saw": "saw "}, {"saw": "a "}, {"saw": "man."}]
    assert qwen.calls[0]["facts"] == f"Description: {DESCRIPTION}"
