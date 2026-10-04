"""
Recognizing how the people in a photo feel. This module does the
orchestration; Qwen (llm.py) only puts the face model's results into words.

1. Qwen: does Florence-2's description mention a living person? If not
   (animals, objects, places), nothing is said about feelings.
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
import logging

from llm import say
from sentences import LEAD_IN, lower_first, upper_first

# Asked of Qwen with the description, answered yes or no: "You saw a close-up
# of a baby's face. ..." The description's "The image is" is left out, and it
# ends with a full stop, so none is added. An earlier
# wording ('In this description "...", did it mention a living person?') did as
# well as a list of person words on 169 OASIS photos, without a list to keep
# up ("son", "surgeons"); asked about a "photo", or with a free-text answer,
# Qwen did much worse.
PERSON_QUESTION = "You saw {} Did you see any person?"
PERSON_SCHEMA = {
    "type": "object",
    "properties": {"mentioned": {"type": "boolean"}},
    "required": ["mentioned"],
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
    {"people": []} when the description mentions no person. `qwen` is an
    llm.Turns, `recognizer` finds the people (Florence-2), `emotions` reads
    each face, one at a time.
    """
    if not mentions_a_person(qwen, description):
        yield {"people": []}
        return
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


def mentions_a_person(qwen, description):
    """
    Whether Florence-2's description of a photo mentions a living person, as
    Qwen reads it. If Qwen can't answer (busy, or cut off), the faces are
    looked for anyway: finding none just means nothing is said.
    """
    seen = lower_first(LEAD_IN.sub("", description))
    messages = [{"role": "user", "content": PERSON_QUESTION.format(seen)}]
    reply = qwen.ask(messages, PERSON_SCHEMA, temperature=0, max_tokens=10)
    return reply is None or reply["mentioned"]


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
