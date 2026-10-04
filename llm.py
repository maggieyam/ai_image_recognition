"""
Qwen, the small language model the app shares: the story writer and recognize_feelings
both ask it questions, so it's loaded once. llama.cpp has a single context, so
callers take turns (app.py's lock).
"""
import json
import platform
import queue
import threading
import time

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
        # The schema becomes a grammar, so the reply is always well-formed
        # JSON unless max_tokens or the time limit cuts it off.
        response_format = {"type": "json_object", "schema": schema}
        text = "".join(self._generate(messages, temperature, max_tokens, response_format))
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def stream(self, messages, temperature, max_tokens):
        """Qwen's reply as plain text, yielded piece by piece as it's generated."""
        yield from self._generate(messages, temperature, max_tokens)

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
    The reply is empty, so the caller can fall back to a plain sentence, when:
    - `defer()` is true (others go first, e.g. a story request waiting);
    - the lock isn't free within `wait` seconds. After that, this Turns stops
      waiting: one wait per caller, not one per reply.
    """

    def __init__(self, qwen, lock, wait, defer=lambda: False):
        self.qwen, self.lock, self.wait, self.defer = qwen, lock, wait, defer
        self.gave_up = False

    def stream(self, messages, temperature, max_tokens):
        if not self._take_turn():
            return
        pieces = queue.Queue()

        def generate():
            try:
                for piece in self.qwen.stream(messages, temperature, max_tokens):
                    pieces.put(piece)
            finally:
                self.lock.release()
                pieces.put(None)

        threading.Thread(target=generate, daemon=True).start()
        while (piece := pieces.get()) is not None:
            yield piece

    def ask(self, messages, schema, temperature, max_tokens):
        """Qwen.ask in turn; None if cut off, or if Qwen wasn't free (see above)."""
        if not self._take_turn():
            return None
        try:
            return self.qwen.ask(messages, schema, temperature, max_tokens)
        finally:
            self.lock.release()

    def _take_turn(self):
        """
        Take the lock, waiting in short steps so a story request that starts
        waiting meanwhile still goes first (the lock itself isn't fair).
        """
        if self.gave_up:  # already waited once: take Qwen only if it's free now
            return not self.defer() and self.lock.acquire(blocking=False)
        deadline = time.monotonic() + self.wait
        while not self.defer():
            if self.lock.acquire(timeout=WAIT_STEP):
                return True
            if time.monotonic() >= deadline:
                self.gave_up = True
                return False
        return False


def say(qwen, messages, max_tokens, fallback):
    """
    Qwen's reply as plain text, piece by piece as it's written, at temperature
    0; `fallback` if it says nothing (it was cut off, or Qwen was busy).
    """
    said = False
    for piece in qwen.stream(messages, temperature=0, max_tokens=max_tokens):
        said = said or bool(piece.strip())
        yield piece
    if not said:
        yield fallback
