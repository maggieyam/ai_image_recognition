import sys
import threading
import types

import pytest

from effectors import mouth
from effectors.mouth import Mouth

WAIT = 5  # seconds a test waits for the mouth's thread before giving up


class FakeSpeaker:
    """Stands in for sounddevice: notes what it's asked to play, and each sound lasts until let go."""

    def __init__(self):
        self.played, self.stopped = [], 0
        self.started, self.finish = threading.Semaphore(0), threading.Event()

    def play(self, samples, rate):
        self.played.append((samples, rate))
        self.finish.clear()
        self.started.release()

    def wait(self):
        self.finish.wait(WAIT)

    def stop(self):
        self.stopped += 1
        self.finish.set()

    def heard(self, sounds):
        """Wait until `sounds` more sounds have started playing."""
        for _ in range(sounds):
            assert self.started.acquire(timeout=WAIT), "the mouth didn't say it"


def test_it_says_things_in_order_one_after_another():
    speaker = FakeSpeaker()
    lips = Mouth(speaker)
    lips.say([("I saw", 22050), ("a cat.", 22050)])
    lips.say([("It looks happy.", 22050)])
    speaker.heard(1)
    assert speaker.played == [("I saw", 22050)]  # the next waits for this one to end
    for _ in range(2):
        speaker.finish.set()
        speaker.heard(1)
    assert speaker.played == [("I saw", 22050), ("a cat.", 22050), ("It looks happy.", 22050)]


def test_nobody_waits_for_the_sound_to_be_made():
    speaker, made = FakeSpeaker(), []

    def sounds():
        made.append(threading.current_thread())
        yield "I saw a cat.", 22050

    Mouth(speaker).say(sounds())  # returns before the sound exists
    speaker.heard(1)
    assert made and made[0] is not threading.current_thread()


def test_a_hush_stops_it_mid_word_and_drops_what_was_waiting():
    speaker = FakeSpeaker()
    lips = Mouth(speaker)
    lips.say([("I saw", 22050), ("a cat.", 22050)])
    lips.say([("It looks happy.", 22050)])
    speaker.heard(1)
    lips.hush()
    assert speaker.stopped == 1
    lips.say([("A new picture.", 22050)])
    speaker.heard(1)
    assert speaker.played == [("I saw", 22050), ("A new picture.", 22050)]


def test_something_that_cant_be_said_doesnt_silence_what_follows():
    def broken():
        raise RuntimeError("the voice failed")
        yield

    speaker = FakeSpeaker()
    lips = Mouth(speaker)
    lips.say(broken())
    lips.say([("It looks happy.", 22050)])
    speaker.heard(1)
    assert speaker.played == [("It looks happy.", 22050)]


def test_a_machine_that_cant_play_sound_has_no_mouth(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # not installed, or no PortAudio
    assert mouth.find() is None


def test_a_machine_with_no_speaker_has_no_mouth(monkeypatch):
    class PortAudioError(Exception):
        pass

    def query_devices(kind):
        raise PortAudioError("Error querying device -1")

    fake = types.SimpleNamespace(PortAudioError=PortAudioError, query_devices=query_devices)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    assert mouth.find() is None


def test_a_machine_with_a_speaker_has_a_mouth_that_plays_through_it(monkeypatch):
    fake = types.SimpleNamespace(PortAudioError=Exception, query_devices=lambda kind: {"name": "Speakers"})
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    assert mouth.find().speaker is fake
