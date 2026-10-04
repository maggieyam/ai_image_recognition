import re

import pytest

from narrate import how_i_feel, story_intention, what_i_saw


class FakeQwen:
    """Streams the given reply in pieces, and records each call."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def stream(self, messages, temperature, max_tokens):
        self.calls.append({"facts": messages[1]["content"], "temperature": temperature})
        yield from re.findall(r"\S+\s*", self.reply)


DESCRIPTION = "A man with a beard takes a close-up selfie, smiling."
CAPTION = "A close-up selfie of a man's face."


def test_each_section_is_one_qwen_call_with_only_its_facts():
    qwen = FakeQwen("Hello there")
    for section in (
        what_i_saw(qwen, DESCRIPTION, "A close-up selfie of a man's face."),
        how_i_feel(qwen, DESCRIPTION, "playful"),
        story_intention(qwen, CAPTION),
    ):
        assert list(section) == ["Hello ", "there"]  # streamed piece by piece
    assert [c["temperature"] for c in qwen.calls] == [0, 0, 0]
    assert qwen.calls[0]["facts"] == f"Description: {DESCRIPTION}"
    assert qwen.calls[1]["facts"] == f"What you saw: {DESCRIPTION}\nHow it made you feel: playful"
    assert qwen.calls[2]["facts"] == f"The picture: {CAPTION}"  # only the caption; no genres


@pytest.mark.parametrize(
    "section, expected",
    [
        (lambda q: what_i_saw(q, DESCRIPTION, "A close-up selfie of a man's face."),
         "I saw a close-up selfie of a man's face."),
        (lambda q: how_i_feel(q, DESCRIPTION, "playful"), "It makes me feel playful."),
        (lambda q: story_intention(q, CAPTION),
         "I'd like to write you a story about it. Which genre would you like?"),
    ],
)
def test_plain_sentence_when_qwen_says_nothing(section, expected):
    assert "".join(section(FakeQwen(""))) == expected
