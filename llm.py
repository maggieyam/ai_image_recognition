"""
Qwen, the small language model the app shares: the story writer and recognize_feelings
both ask it questions, so it's loaded once. llama.cpp has a single context, so
callers take turns (app.py's lock).
"""
import json
import platform
import time

from llama_cpp import Llama

# 4-bit GGUF, the format planned for the Jetson. 0.5B was tried first and
# could not follow the story seed format (too many characters, no conflict).
MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
# Wall-clock limit per call. Calls normally take 3-30s; the limit keeps one
# runaway generation from holding the model (and every other request).
CALL_SECONDS = 60


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
        # JSON unless max_tokens or the time limit cuts it off. Streamed so the
        # time limit can stop it.
        deadline = time.monotonic() + CALL_SECONDS
        chunks = self.llm.create_chat_completion(
            messages,
            response_format={"type": "json_object", "schema": schema},
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=True,
        )
        text = ""
        for chunk in chunks:
            text += chunk["choices"][0]["delta"].get("content") or ""
            if time.monotonic() > deadline:
                chunks.close()
                return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def stream(self, messages, temperature, max_tokens):
        """Qwen's reply as plain text, yielded piece by piece as it's generated."""
        deadline = time.monotonic() + CALL_SECONDS
        chunks = self.llm.create_chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=True,
        )
        for chunk in chunks:
            piece = chunk["choices"][0]["delta"].get("content")
            if piece:
                yield piece
            if time.monotonic() > deadline:
                chunks.close()
                return

