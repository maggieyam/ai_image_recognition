"""
The voices the brain can speak in: the networks that turn words into sound
for the mouth (effectors/mouth.py). Which one is picked by name when the
brain starts, so switching is a setting (VOICE=piper), not a code change.

A voice is a class that loads its model when made and has one method:

    def sounds(self, text):
        # `text` as sound, a sentence at a time, each made when it's asked
        # for, so speaking starts before the last sentence exists.
        yield samples, rate  # a numpy array (mono, int16 or float32) and its sample rate

To add one, write its class in a file of its own here and list it in VOICES.
"""
import importlib

# name: (module, class). Imported only when picked, so a voice whose package
# isn't installed doesn't get in the way of the others.
VOICES = {
    "piper": ("brain.networks.piper", "Piper"),
}
DEFAULT = "piper"


def load(name=None):
    """The voice called `name` (DEFAULT if None), loaded. Raises ValueError if there's no such voice."""
    name = name or DEFAULT
    if name not in VOICES:
        raise ValueError(f"There's no voice called {name!r}. The voices are: {', '.join(sorted(VOICES))}")
    module, cls = VOICES[name]
    return getattr(importlib.import_module(module), cls)()
