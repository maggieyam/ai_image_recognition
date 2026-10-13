"""
The mouth: sound out of this machine's speaker. It's handed sound to make
(the brain makes it from words: brain/networks/piper.py) and knows nothing of
language.

Saying something takes as long as it sounds, so nobody waits for the mouth:
what's to be said queues up and is said in order, on a thread of its own.
"""
import logging
import queue
import threading

log = logging.getLogger(__name__)


def find():
    """This machine's mouth, or None if it has no speaker to say anything through."""
    try:
        # Needs the PortAudio library, which a server often doesn't have.
        import sounddevice
    except (ImportError, OSError):
        return None
    try:
        sounddevice.query_devices(kind="output")
    except (sounddevice.PortAudioError, ValueError):  # no output device
        return None
    return Mouth(sounddevice)


class Mouth:
    def __init__(self, speaker):
        """speaker: what plays the sound, with sounddevice's play(), wait() and stop()."""
        self.speaker = speaker
        self.waiting = queue.Queue()
        # Goes up at each hush: anything queued before it is never said.
        self.hushes = 0
        self.lock = threading.Lock()
        threading.Thread(target=self._speak, daemon=True).start()

    def say(self, sounds):
        """
        Say `sounds` after what's already waiting, and return at once.
        sounds: (samples, rate) one after another; a generator is only run
        when its turn comes, so the sound can be made while the last is said.
        """
        self.waiting.put((self.hushes, sounds))

    def hush(self):
        """Stop mid-word, and drop everything still waiting to be said."""
        with self.lock:
            self.hushes += 1
            self.speaker.stop()

    def _speak(self):
        while True:
            hushes, sounds = self.waiting.get()
            try:
                for samples, rate in sounds:
                    # Under the lock, so a hush can't land between the check and the sound.
                    with self.lock:
                        if hushes != self.hushes:
                            break
                        self.speaker.play(samples, rate)
                    self.speaker.wait()
            except Exception:  # one thing that can't be said doesn't silence the rest
                log.exception("couldn't say something out loud")
