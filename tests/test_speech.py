import json

from brain.imagination.story import GENRES
from brain.language.speech import SCHEMA, Speak, _parts
from brain.workspace import Workspace

REPLY = {
    "saw": "I saw a man taking a selfie.",
    "they_look": "The man looks angry.",
    "i_feel": "It makes me feel playful, he's pulling a face.",
    "story_offer": "I'd love to write a story. Which genre?",
}


def in_pieces(text, size=7):
    return [text[i:i + size] for i in range(0, len(text), size)]


class FakeQwen:
    """Streams the given reply as JSON text, a few characters at a time; records each call."""

    def __init__(self, reply, cut=None):
        self.text, self.calls = json.dumps(reply)[:cut], []

    def stream(self, messages, temperature, max_tokens, schema=None):
        self.calls.append({"messages": messages, "temperature": temperature, "schema": schema})
        yield from in_pieces(self.text)


LOOKING_AT = "a picture the user showed you"
DESCRIPTION = "A man with a beard takes a close-up selfie, smiling."
CAPTION = "A close-up selfie of a man's face."
MOODS = [("playful", 0.5), ("happy", 0.12)]
MAN = {"character": "a man with a beard", "emotion": "Anger", "box": [1, 2, 3, 4]}


def speak(voice, characters=(MAN,)):
    """The events of everything said about a picture with these characters."""
    workspace = Workspace(
        voice=voice, looking_at=LOOKING_AT, description=DESCRIPTION, caption=CAPTION,
        characters=list(characters), moods=MOODS,
    )
    return list(Speak().run(workspace))


def test_everything_is_said_in_one_qwen_call_given_the_facts_on_the_workspace():
    qwen = FakeQwen(REPLY)
    speak(qwen)
    (call,) = qwen.calls
    assert call["temperature"] == 0 and call["schema"] == SCHEMA
    facts = call["messages"][-1]["content"]
    assert all(fact in facts for fact in (LOOKING_AT, DESCRIPTION, "a man with a beard", "angry", "playful"))


def test_each_part_reaches_the_page_in_its_section_with_its_choices_after_it():
    events = speak(FakeQwen(REPLY))
    assert events[:7] == [
        {"saw": REPLY["saw"]},
        {"feelings": REPLY["they_look"]},
        {"status": "Letting it sink in…", "for": "i_feel"},
        {"i_feel": REPLY["i_feel"]},
        {"moods": [{"mood": "playful", "score": 0.5}, {"mood": "happy", "score": 0.12}]},
        {"status": "Thinking of a story…", "for": "story_intention"},
        {"story_intention": REPLY["story_offer"]},
    ]
    offered = events[7]["genres"]
    assert len(events) == 8 and len(set(offered)) == 3 and set(offered) <= set(GENRES)


def test_a_part_is_passed_on_as_soon_as_its_complete_not_when_the_reply_is():
    text = json.dumps({**REPLY, "saw": 'I saw a "selfie".'})
    pieces = iter(in_pieces(text, size=5))
    parts = _parts(pieces)
    assert next(parts) == ("saw", 'I saw a "selfie".')  # quotes inside a part don't end it
    assert "story_offer" in "".join(pieces)  # most of the reply hadn't been written yet


def test_plain_sentences_when_qwen_says_nothing():
    events = speak(FakeQwen(REPLY, cut=0))  # Qwen was busy
    said = {k: v for e in events for k, v in e.items() if k in ("saw", "feelings", "i_feel", "story_intention")}
    assert said == {
        "saw": "I saw a close-up selfie of a man's face.",
        "feelings": "A man with a beard looks angry.",
        "i_feel": "It makes me feel playful.",
        "story_intention": "I'd like to write you a story about it. Which genre would you like?",
    }
    assert any("moods" in e for e in events) and "genres" in events[-1]  # the choices still follow


def test_a_reply_thats_cut_off_is_finished_in_plain_sentences():
    cut = json.dumps(REPLY).index("he's pulling")  # Qwen ran out of time partway through how it feels
    events = speak(FakeQwen(REPLY, cut=cut))
    assert {"saw": REPLY["saw"]} in events and {"feelings": REPLY["they_look"]} in events
    assert {"i_feel": "It makes me feel playful."} in events
    assert {"story_intention": "I'd like to write you a story about it. Which genre would you like?"} in events


def test_how_the_characters_look_is_said_even_if_qwen_leaves_it_out():
    events = speak(FakeQwen({**REPLY, "they_look": ""}))
    assert {"feelings": "A man with a beard looks angry."} in events


def test_nothing_is_said_about_characters_when_there_are_none():
    events = speak(FakeQwen(REPLY), characters=())  # Qwen says how "the man" looks anyway
    assert not any("feelings" in e for e in events)
    assert [next(iter(e)) for e in events[:2]] == ["saw", "status"]  # the rest still follows
