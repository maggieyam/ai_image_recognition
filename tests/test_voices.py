import importlib

import pytest

from brain.networks import voices


class Echo:
    """A voice that loads no model."""

    def sounds(self, text):
        yield f"the sound of {text!r}", 22050


def test_a_voice_is_picked_by_name_and_loaded(monkeypatch):
    monkeypatch.setitem(voices.VOICES, "echo", (__name__, "Echo"))
    assert list(voices.load("echo").sounds("hello")) == [("the sound of 'hello'", 22050)]


def test_with_no_name_its_the_usual_voice(monkeypatch):
    monkeypatch.setitem(voices.VOICES, "echo", (__name__, "Echo"))
    monkeypatch.setattr(voices, "DEFAULT", "echo")
    assert type(voices.load()).__name__ == "Echo"


def test_a_voice_that_isnt_listed_is_refused_by_name_with_the_ones_that_are():
    with pytest.raises(ValueError, match="no voice called 'pipr'.*piper"):
        voices.load("pipr")


def test_every_listed_voice_is_a_class_that_makes_sounds():
    assert voices.DEFAULT in voices.VOICES
    for module, cls in voices.VOICES.values():
        assert callable(getattr(importlib.import_module(module), cls).sounds)
