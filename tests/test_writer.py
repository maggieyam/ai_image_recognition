import pytest

import writer
from sentences import split_sentences
from writer import GROUNDING_MIN, MAX_TRIES, StoryWriter

DESCRIPTION = "A man in a white robe walks alone across sand dunes under a clear blue sky."
SEED = {
    "title": "Salt Road",
    "characters": [{"name": "Idris", "about": "a salt trader"}],
    "premise": "A salt trader must cross the desert before the caravan leaves without him.",
    "want": "Idris wants to reach the oasis market.",
    "obstacle": "A sandstorm has buried the old road.",
    "storyline": "Idris sets out at dawn. The storm catches him at noon. "
    "He follows the stars to the market. He sells the salt.",
    "first_line": "The wind was already rising when Idris loaded the camel. It was early.",
}


@pytest.fixture
def story_writer():
    """A StoryWriter without its model; tests stub the methods that call it."""
    return StoryWriter.__new__(StoryWriter)


def test_split_sentences_keeps_titles():
    assert split_sentences("He met Mrs. Brown. She smiled!") == ["He met Mrs. Brown.", "She smiled!"]


def test_pick_avoids_repeats_and_returns_spares(story_writer, monkeypatch):
    monkeypatch.setattr(writer.random, "sample", lambda seq, k: list(seq)[:k])  # no shuffle
    monkeypatch.setattr(story_writer, "_ask", lambda *a, **k: {"picks": [1, 2]})
    combos = [
        {"genre": g, "characters": c} for g in ("comedy", "mystery") for c in ("twins", "rivals")
    ]
    picks, spares = story_writer._pick(DESCRIPTION, "happy", combos, 2)
    # The model's second pick reuses "comedy", so the next fresh combination is taken.
    assert picks == [
        {"genre": "comedy", "characters": "twins"},
        {"genre": "mystery", "characters": "rivals"},
    ]
    assert spares == [
        {"genre": "comedy", "characters": "rivals"},
        {"genre": "mystery", "characters": "twins"},
    ]


def test_pick_fills_count_when_the_model_fails(story_writer, monkeypatch):
    monkeypatch.setattr(story_writer, "_ask", lambda *a, **k: None)
    combos = [{"genre": g, "characters": "twins"} for g in ("comedy", "mystery", "heist")]
    picks, _ = story_writer._pick(DESCRIPTION, "happy", combos, 3)
    assert len(picks) == 3


def test_grounded_ideas_rejects_only_ideas_that_fail_the_check(story_writer, monkeypatch):
    scores = {"a": 0.30, "b": 0.29, "c": 0.28, "d": 0.27, "e": 0.26, "f": 0.10}
    monkeypatch.setattr(story_writer, "_brainstorm", lambda *a: list(scores))
    kept, rejected = story_writer._grounded_ideas(
        DESCRIPTION, "happy", "ideas", lambda texts: [scores[t] for t in texts]
    )
    assert kept == ["a", "b", "c", "d"]
    assert rejected == ["f"]  # "e" fits the picture but is over the cap: not "rejected"


def test_grounded_ideas_retries_once_then_gives_up(story_writer, monkeypatch):
    calls = []
    monkeypatch.setattr(story_writer, "_brainstorm", lambda *a: calls.append(1) or ["a", "b"])
    result = story_writer._grounded_ideas(
        DESCRIPTION, "happy", "ideas", lambda texts: [GROUNDING_MIN - 0.05] * len(texts)
    )
    assert result is None
    assert len(calls) == MAX_TRIES == 2


def test_write_retries_once_then_gives_up(story_writer, monkeypatch):
    calls = []
    monkeypatch.setattr(story_writer, "_ask", lambda *a, **k: calls.append(1))
    assert story_writer.write(DESCRIPTION, "happy", {"genre": "comedy", "characters": "x"}) is None
    assert len(calls) == MAX_TRIES


def test_write_trims_to_whole_sentences(story_writer, monkeypatch):
    monkeypatch.setattr(story_writer, "_ask", lambda *a, **k: dict(SEED))
    seed = story_writer.write(DESCRIPTION, "happy", {"genre": "adventure", "characters": "x"})
    assert seed["storyline"] == (
        "Idris sets out at dawn. The storm catches him at noon. He follows the stars to the market."
    )
    assert seed["first_line"] == "The wind was already rising when Idris loaded the camel."


@pytest.mark.parametrize(
    "change",
    [
        {"want": "Unknown."},
        {"characters": [{"name": "None", "about": ""}]},
        {"storyline": "A man in a white robe walks alone across the sand dunes under a blue sky."},
        {"first_line": "The rain had not stopped for three days, and neither had Moss's watch."},
    ],
    ids=["non-answer", "no name", "restates the photo", "copies the example"],
)
def test_sensible_rejects_non_stories(change):
    assert writer._sensible(SEED, DESCRIPTION)
    assert not writer._sensible({**SEED, **change}, DESCRIPTION)
