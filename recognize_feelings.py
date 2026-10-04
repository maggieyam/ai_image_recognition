"""
Recognizing how the people in a photo feel. This module does the
orchestration; Qwen (llm.py) only puts the face model's results into words.

1. Code: does Florence-2's description mention a person? If not (animals,
   objects, places), nothing is said about feelings.
2. Florence-2 (recognizer.py): find the main characters by their faces, and
   say who each one is.
3. Code: for each person, the emotion model (emotions.py) reads their face.
4. Qwen: put the people and their feelings, as JSON, into words, streamed as
   it writes. Qwen sees no photo, only those facts, so it can relate them
   ("the man is angry, and the girl is crying, so she must be sad").

Why code orchestrates: told to call the emotion tool only for living humans,
Qwen 1.5B called it for every photo; and left to call it itself, it never
did. Answering from the description alone, it got 7 of 40 OASIS face photos
right; from the face model's results, 23 of 40 (as many as the face model).
Qwen runs at temperature 0, so the same input gets the same answer.
"""
import json
import re

from llm import say
from sentences import upper_first

# Words for people in Florence-2's descriptions; a plural ("soldiers") counts
# as its singular. If none appears, there's no human in the picture. Family
# words ("mother", "couple") and "crowd" are left out: descriptions use them
# for animals too ("a mother bear and her cubs").
HUMAN_WORDS = {
    "person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children",
    "kid", "baby", "babies", "toddler", "teenager", "adult", "lady", "ladies", "gentleman",
    "gentlemen", "guy", "bride", "groom", "student", "soldier", "player", "athlete",
    "worker", "businessman", "businessmen", "businesswoman", "businesswomen", "doctor",
    "nurse", "officer", "policeman", "policemen", "firefighter", "chef", "dancer", "singer",
    "musician", "tourist", "farmer", "teacher", "jockey", "surfer", "skier", "climber",
}
EXPRESS_SYSTEM = (
    "You get a JSON list of the people in a photo, with who each one is and how they feel. "
    "Say how they feel in one or two short, natural sentences. Use only the facts given. "
    "Write in English. Reply with only what you'd say."
)
# The face model's emotion labels as feelings.
FEELING_WORDS = {
    "Anger": "angry", "Contempt": "contemptuous", "Disgust": "disgusted", "Fear": "afraid",
    "Happiness": "happy", "Neutral": "neutral", "Sadness": "sad", "Surprise": "surprised",
}


def feelings_events(qwen, recognizer, emotions, image, description):
    """
    How the people in a photo feel, as /analyze events: a status while their
    faces are read, the people found, then Qwen's words piece by piece. Just
    {"people": []} when the description mentions no person. `recognizer`
    finds the people (Florence-2), `emotions` reads each face, one at a time.
    """
    if not mentions_a_person(description):
        yield {"people": []}
        return
    yield {"status": "Looking at their faces…", "for": "feelings"}
    people = recognizer.find_people(image)
    for person in people:
        person.update(emotions.recognize(person.pop("face")))
    yield {"people": people}
    if people:
        for piece in express(qwen, people):
            yield {"feelings": piece}


def mentions_a_person(description):
    """Whether Florence-2's description of a photo mentions a person."""
    words = re.findall(r"[a-z]+", description.lower())
    return any(w in HUMAN_WORDS or (w.endswith("s") and w[:-1] in HUMAN_WORDS) for w in words)


def express(qwen, people):
    """
    How the people feel, in words yielded as Qwen writes them, from who each
    one is and the emotion read from their face:
    [{"human": "a woman in a red dress", "emotion": "Sadness"}, ...].
    """
    facts = [{"human": p["human"], "feeling": _feeling(p)} for p in people]
    messages = [
        {"role": "system", "content": EXPRESS_SYSTEM},
        {"role": "user", "content": json.dumps(facts)},
    ]
    fallback = " ".join(f"{upper_first(f['human'])} looks {f['feeling']}." for f in facts)
    yield from say(qwen, messages, 20 + 30 * len(people), fallback)


def _feeling(person):
    return FEELING_WORDS.get(person["emotion"], person["emotion"].lower())
