"""
Qwen, the small language model the brain shares: the story writer and
everything the brain says ask it questions, so it's loaded once. llama.cpp
has a single context, so callers take turns (SharedQwen).
"""
import json
import platform
import queue
import threading
import time
from contextlib import contextmanager

from llama_cpp import Llama

# 4-bit GGUF, the format planned for the Jetson. 0.5B was tried first and
# could not follow the story seed format (too many characters, no conflict).
MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
# Wall-clock limit per call. Calls normally take 3-30s; the limit keeps one
# runaway generation from holding the model (and every other request).
CALL_SECONDS = 60
# How often a Turns waiting for Qwen checks whether to let others go first.
WAIT_STEP = 0.1


class Qwen:
    def __init__(self):
        # Offload every layer when llama.cpp has a GPU backend (e.g. CUDA on the
        # Jetson). Not on Intel Macs: Metal on their AMD GPUs returns garbage.
        intel_mac = platform.system() == "Darwin" and platform.machine() == "x86_64"
        self.llm = Llama.from_pretrained(
            MODEL_REPO,
            MODEL_FILE,
            n_ctx=2048,
            n_gpu_layers=0 if intel_mac else -1,
            verbose=False,
        )

    def ask(self, messages, schema, temperature, max_tokens):
        """Qwen's reply as a dict matching the JSON schema, or None if cut off."""
        text = "".join(self.stream(messages, temperature, max_tokens, schema))
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def stream(self, messages, temperature, max_tokens, schema=None):
        """
        Qwen's reply, yielded piece by piece as it's generated: plain text,
        or with a JSON schema, the text of JSON matching it.
        """
        # The schema becomes a grammar, so the reply is always well-formed
        # JSON unless max_tokens or the time limit cuts it off.
        response_format = {"type": "json_object", "schema": schema} if schema else None
        yield from self._generate(messages, temperature, max_tokens, response_format)

    def _generate(self, messages, temperature, max_tokens, response_format=None):
        """The reply's text, piece by piece; streamed so the time limit can stop it."""
        deadline = time.monotonic() + CALL_SECONDS
        extra = {"response_format": response_format} if response_format else {}
        chunks = self.llm.create_chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=True,
            **extra,
        )
        for chunk in chunks:
            piece = chunk["choices"][0]["delta"].get("content")
            if piece:
                yield piece
            if time.monotonic() > deadline:
                chunks.close()
                return


class Turns:
    """
    Qwen for a caller that shares it with others (llama.cpp has one context):
    each reply is generated in a thread that holds `lock` only while Qwen
    writes, not while the caller sends the pieces on (to a slow client, say).
    The reply is empty, so the caller can fall back to plain sentences, when:
    - `defer()` is true (others go first, e.g. a story waiting in SharedQwen.hold);
    - the lock isn't free within `wait` seconds.
    """

    def __init__(self, qwen, lock, wait, defer=lambda: False):
        self.qwen, self.lock, self.wait, self.defer = qwen, lock, wait, defer

    def stream(self, messages, temperature, max_tokens, schema=None):
        if not self._take_turn():
            return
        pieces = queue.Queue()

        def generate():
            try:
                for piece in self.qwen.stream(messages, temperature, max_tokens, schema):
                    pieces.put(piece)
            finally:
                self.lock.release()
                pieces.put(None)

        threading.Thread(target=generate, daemon=True).start()
        while (piece := pieces.get()) is not None:
            yield piece

    def _take_turn(self):
        """
        Take the lock, waiting in short steps so a story request that starts
        waiting meanwhile still goes first (the lock itself isn't fair).
        """
        deadline = time.monotonic() + self.wait
        while not self.defer():
            if self.lock.acquire(timeout=WAIT_STEP):
                return True
            if time.monotonic() >= deadline:
                return False
        return False


class Busy(Exception):
    """Qwen stayed taken for longer than the caller could wait."""


class SharedQwen:
    """
    Who gets Qwen next. Stories hold it for a whole plan or story, and go
    first; what the brain says about a picture takes it for one reply, and
    makes way while a story is waiting (see Turns).
    """

    def __init__(self, qwen):
        self.qwen = qwen
        # Qwen (llama.cpp) has a single context, so callers that use it take turns.
        self.lock = threading.Lock()
        self.waiting = 0  # callers waiting in hold()
        self.waiting_lock = threading.Lock()

    @contextmanager
    def hold(self, wait):
        """Hold Qwen, or raise Busy after `wait` seconds."""
        with self.waiting_lock:
            self.waiting += 1
        try:
            got_it = self.lock.acquire(timeout=wait)
        finally:
            with self.waiting_lock:
                self.waiting -= 1
        if not got_it:
            raise Busy
        try:
            yield
        finally:
            self.lock.release()

    def in_passing(self, wait):
        """
        Qwen for a caller that can do without it: a Turns that waits up to
        `wait` seconds for it, and lets whoever is waiting in hold() go first.
        """
        return Turns(self.qwen, self.lock, wait, defer=lambda: self.waiting > 0)
