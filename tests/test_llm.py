import threading
import time

from llm import Qwen, Turns, say


class FakeQwen:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def stream(self, messages, temperature, max_tokens):
        self.calls += 1
        yield from self.reply

    def ask(self, messages, schema, temperature, max_tokens):
        self.calls += 1
        return {"answer": "".join(self.reply)}


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


def test_turns_asks_in_turn_and_frees_the_lock():
    lock = threading.Lock()
    assert Turns(FakeQwen(["yes"]), lock, wait=1).ask([], {}, 0, 10) == {"answer": "yes"}
    assert lock.acquire(blocking=False)


def test_turns_has_no_answer_when_qwen_stays_busy():
    lock, qwen = threading.Lock(), FakeQwen(["yes"])
    lock.acquire()  # the story writer has Qwen
    assert Turns(qwen, lock, wait=0.1).ask([], {}, 0, 10) is None
    assert qwen.calls == 0


def test_turns_says_nothing_when_qwen_stays_busy():
    lock, qwen = threading.Lock(), FakeQwen(["Hi"])
    lock.acquire()  # the story writer has Qwen
    assert list(Turns(qwen, lock, wait=0.1).stream([], 0, 10)) == []
    assert qwen.calls == 0


def test_say_falls_back_when_qwen_says_nothing():
    assert "".join(say(FakeQwen([]), [], 10, "It makes me feel calm.")) == "It makes me feel calm."
    assert "".join(say(FakeQwen(["I feel ", "calm."]), [], 10, "fallback")) == "I feel calm."


def test_turns_waits_once_per_caller_then_stops_waiting():
    lock = threading.Lock()
    lock.acquire()  # the story writer has Qwen
    turns = Turns(FakeQwen(["Hi"]), lock, wait=0.2)
    start = time.monotonic()
    for _ in range(3):
        assert list(turns.stream([], 0, 10)) == []
    assert time.monotonic() - start < 0.4  # one 0.2 s wait, not three


def test_turns_lets_waiting_story_requests_go_first():
    lock, qwen = threading.Lock(), FakeQwen(["Hi"])
    assert list(Turns(qwen, lock, wait=1, defer=lambda: True).stream([], 0, 10)) == []
    assert qwen.calls == 0 and lock.acquire(blocking=False)  # the lock was never taken


class FakeLlama:
    def __init__(self, pieces):
        self.pieces, self.kwargs = pieces, None

    def create_chat_completion(self, messages, **kwargs):
        self.kwargs = kwargs
        return iter([{"choices": [{"delta": {"content": p}}]} for p in self.pieces])


def test_ask_and_stream_share_one_generation_path():
    qwen = Qwen.__new__(Qwen)
    qwen.llm = FakeLlama(['{"answer": ', '"yes"}'])
    assert qwen.ask([], {"type": "object"}, 0, 10) == {"answer": "yes"}
    assert qwen.llm.kwargs["response_format"] == {"type": "json_object", "schema": {"type": "object"}}
    qwen.llm = FakeLlama(["I ", "saw"])
    assert list(qwen.stream([], 0, 10)) == ["I ", "saw"]
    assert "response_format" not in qwen.llm.kwargs


def test_a_story_request_that_starts_waiting_meanwhile_still_goes_first():
    lock, qwen = threading.Lock(), FakeQwen(["Hi"])
    lock.acquire()  # the story writer has Qwen
    checks = iter([False, False, True])  # a story request starts waiting during the wait
    turns = Turns(qwen, lock, wait=5, defer=lambda: next(checks, True))
    start = time.monotonic()
    assert list(turns.stream([], 0, 10)) == []
    assert time.monotonic() - start < 1 and not turns.gave_up and qwen.calls == 0
