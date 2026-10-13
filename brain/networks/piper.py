"""
Piper, one of the voices (voices.py): words as sound, for the mouth to say
(effectors/mouth.py). A small speech model run with ONNX, built for boards like the Raspberry Pi:
on a laptop CPU it makes 9 seconds of speech in under a second, so the sound
stays well ahead of the speaking.

The voice was trained from scratch on LJ Speech, which is public domain. Its
weights (63 MB) download to the Hugging Face cache, outside the repo.
"""
from huggingface_hub import hf_hub_download
from piper import PiperVoice

VOICE_REPO = "rhasspy/piper-voices"
VOICE_REVISION = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"  # pinned, so the voice doesn't change under us
VOICE_FILE = "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx"


class Piper:
    def __init__(self):
        model = hf_hub_download(VOICE_REPO, VOICE_FILE, revision=VOICE_REVISION)
        config = hf_hub_download(VOICE_REPO, VOICE_FILE + ".json", revision=VOICE_REVISION)
        self.voice = PiperVoice.load(model, config)

    def sounds(self, text):
        """
        `text` as sound, a sentence at a time, each made when it's asked for:
        (samples, rate), the samples 16-bit mono.
        """
        for chunk in self.voice.synthesize(text):
            yield chunk.audio_int16_array, chunk.sample_rate
