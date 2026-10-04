"""
Recognizing how the people in a photo feel. This module does the
orchestration; Qwen (llm.py) only puts the face model's results into words.

1. YuNet and Florence-2 (recognizer.py): find the main characters by their
   faces, and say who each one is.
2. Code: for each person, the emotion model (emotions.py) reads their face.
3. Qwen: put the people and their feelings, as JSON, into words, streamed as
   it writes. Qwen sees no photo, only those facts, so it can relate them
   ("the man is angry, and the girl is crying, so she must be sad").

Why a face model reads the emotion: guessing feelings from Florence-2's
description alone, Qwen 1.5B got 7 of 40 OASIS face photos right; worded from
the face model's results, 23 of 40 (as many as the face model). Qwen runs at
temperature 0, so the same input gets the same answer.

Every photo is searched for faces; nothing first checks whether there's a
person (Qwen 1.5B couldn't reliably tell from the description; experiments on
issue #5). A face found on an animal or a statue is read like a person's: the
emotion is a guess at how a face looks, which fits them too. Objects rarely
get one: 1 of 62 OASIS object photos (a concrete wall).
"""
import json
import logging

from llm import say
from sentences import upper_first

EXPRESS_SYSTEM = (
    "You get a JSON list of the main characters in a photo, with who each one is and how they feel. "
    "Say how they feel in one or two short, natural sentences. Use only the facts given. "
    "Write in English. Reply with only what you'd say."
)
# The face model's emotion labels as feelings.
FEELING_WORDS = {
    "Anger": "angry", "Contempt": "contemptuous", "Disgust": "disgusted", "Fear": "afraid",
    "Happiness": "happy", "Neutral": "neutral", "Sadness": "sad", "Surprise": "surprised",
}


def feelings_events(qwen, recognizer, emotions, image):
    """
    How the people in a photo feel, as /analyze events: a status while their
    faces are read, the people found ([] if none), then Qwen's words piece by
    piece. `qwen` is an llm.Turns, `recognizer` finds the people (YuNet and
    Florence-2), `emotions` reads each face, one at a time.
    """
    yield {"status": "Looking at their faces…", "for": "feelings"}
    # Feelings are optional: if a face can't be read, that person is left out,
    # and the rest of what the app says still follows.
    try:
        found = recognizer.find_people(image)
    except Exception:
        logging.exception("finding the people failed")
        found = []
    people = []
    for person in found:
        try:
            person.update(emotions.recognize(person.pop("face")))
        except Exception:
            logging.exception("reading a face failed")
            continue
        people.append(person)
    yield {"people": people}
    if people:
        for piece in express(qwen, people):
            yield {"feelings": piece}


def express(qwen, people):
    """
    How the people feel, in words yielded as Qwen writes them, from who each
    one is and the emotion read from their face:
    [{"character": "a woman in a red dress", "emotion": "Sadness"}, ...].
    """
    facts = [{"character": p["character"], "feeling": _feeling(p)} for p in people]
    messages = [
        {"role": "system", "content": EXPRESS_SYSTEM},
        {"role": "user", "content": json.dumps(facts)},
    ]
    fallback = " ".join(f"{upper_first(f['character'])} looks {f['feeling']}." for f in facts)
    yield from say(qwen, messages, 20 + 30 * len(people), fallback)


def _feeling(person):
    return FEELING_WORDS.get(person["emotion"], person["emotion"].lower())
