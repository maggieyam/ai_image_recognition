import threading

from llm import Turns, say


class FakeQwen:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def stream(self, messages, temperature, max_tokens):
        self.calls += 1
        yield from self.reply


def test_turns_streams_and_frees_the_lock_once_qwen_is_done():
    lock = threading.Lock()
    assert list(Turns(FakeQwen(["Hi ", "there"]), lock, wait=1).stream([], 0, 10)) == ["Hi ", "there"]
    assert lock.acquire(blocking=False)


def test_turns_frees_the_lock_even_if_the_caller_stops_reading():
    lock = threading.Lock()
    pieces = Turns(FakeQwen(["a", "b", "c"]), lock, wait=1).stream([], 0, 10)
    next(pieces)
    pieces.close()  # e.g. the page went away
    assert lock.acquire(timeout=1)


def test_turns_says_nothing_when_qwen_stays_busy():
    lock, qwen = threading.Lock(), FakeQwen(["Hi"])
    lock.acquire()  # the story writer has Qwen
    assert list(Turns(qwen, lock, wait=0.1).stream([], 0, 10)) == []
    assert qwen.calls == 0


def test_say_falls_back_when_qwen_says_nothing():
    assert "".join(say(FakeQwen([]), [], 10, "It makes me feel calm.")) == "It makes me feel calm."
    assert "".join(say(FakeQwen(["I feel ", "calm."]), [], 10, "fallback")) == "I feel calm."
