import threading
import time

import pytest

from brain.networks.qwen import Busy, Qwen, SharedQwen, Turns


class FakeQwen:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def stream(self, messages, temperature, max_tokens, schema=None):
        self.calls += 1
        self.schema = schema
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
    assert time.monotonic() - start < 1 and qwen.calls == 0  # it made way at once, not after 5 s


def test_a_story_holds_qwen_and_frees_it_afterwards():
    shared = SharedQwen(FakeQwen(["Hi"]))
    with shared.hold(wait=1):
        assert shared.lock.locked() and shared.waiting == 0  # holding isn't waiting
        assert list(shared.in_passing(wait=0.1).stream([], 0, 10)) == []  # a sentence can't cut in
    assert list(shared.in_passing(wait=1).stream([], 0, 10)) == ["Hi"]


def test_a_story_is_told_qwen_is_busy_once_it_has_waited():
    shared = SharedQwen(FakeQwen(["Hi"]))
    with shared.hold(wait=1):
        with pytest.raises(Busy):
            with shared.hold(wait=0.1):
                pass
        assert shared.waiting == 0
    assert shared.lock.acquire(blocking=False)  # the story that gave up didn't take or free it


def test_sentences_make_way_for_a_story_thats_waiting():
    shared = SharedQwen(FakeQwen(["Hi"]))
    sentences = shared.in_passing(wait=5)
    shared.waiting = 1  # a story is waiting in hold()
    start = time.monotonic()
    assert list(sentences.stream([], 0, 10)) == []
    assert time.monotonic() - start < 1 and shared.qwen.calls == 0


def test_turns_passes_a_json_schema_on_to_qwen():
    qwen = FakeQwen(['{"saw": ', '"a cat"}'])
    turns = Turns(qwen, threading.Lock(), wait=1)
    assert "".join(turns.stream([], 0, 10, schema={"type": "object"})) == '{"saw": "a cat"}'
    assert qwen.schema == {"type": "object"}
