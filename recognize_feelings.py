"""
Recognizing how the people in a photo feel. This module does the
orchestration; Qwen (llm.py) only puts the face model's results into words.

1. Code: does Florence-2's description mention a person? If not (animals,
   objects, places), nothing is said about feelings.
2. Florence-2 (recognizer.py): find the main characters by their faces, and
   say who each one is.
3. Code: for each person, the emotion model (emotions.py) reads their face.
4. Qwen: put the people and their feelings, as JSON, into words, streamed as
   it writes. Qwen sees
   no photo, only those facts, so it can relate them ("the man is angry, and
   the girl is crying, so she must be sad").

Why code orchestrates: told to call the emotion tool only for living humans,
Qwen 1.5B called it for every photo; and left to call it itself, it never
did. Answering from the description alone, it got 7 of 40 OASIS face photos
right; from the face model's results, 23 of 40 (as many as the face model).
Qwen runs at temperature 0, so the same input gets the same answer.
"""
import json
import re

# Words for people in Florence-2's descriptions. If none appears, there's no
# human in the picture. Family words ("mother", "couple") and "crowd" are left
# out: descriptions use them for animals too ("a mother bear and her cubs").
HUMAN_WORDS = {
    "person", "persons", "people", "man", "men", "woman", "women", "boy", "boys", "girl",
    "girls", "child", "children", "kid", "kids", "baby", "babies", "toddler", "teenager",
    "lady", "ladies", "gentleman", "guy",
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
# Faces shorter than this share of the photo's height are background people
# (a crowd behind the main character). Provisional: to be set from real photos.
MIN_FACE_HEIGHT = 0.1


def how_people_feel(qwen, recognizer, emotions, image, description):
    """
    How the people in a photo feel: {"people": [{"human", "position", "emotion",
    "valence", "arousal"}, ...], "answer": words or None}.
    """
    people = find_feelings(recognizer, emotions, image, description)
    return {"people": people, "answer": "".join(express(qwen, people)) if people else None}


def find_feelings(recognizer, emotions, image, description):
    """
    The main characters and the emotion on each one's face, or [] when the
    description mentions no person. `recognizer` finds the people (Florence-2),
    `emotions` reads each face, one at a time.
    """
    if not mentions_a_person(description):
        return []
    people = recognizer.find_people(image)
    for person in people:
        person.update(emotions.recognize(person.pop("face")))
    return people


def mentions_a_person(description):
    """Whether Florence-2's description of a photo mentions a person."""
    return any(word in HUMAN_WORDS for word in re.findall(r"[a-z]+", description.lower()))


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
    said = False
    for piece in qwen.stream(messages, temperature=0, max_tokens=20 + 30 * len(people)):
        said = said or bool(piece.strip())
        yield piece
    if not said:
        yield " ".join(f"{f['human'][:1].upper()}{f['human'][1:]} looks {f['feeling']}." for f in facts)


def _feeling(person):
    return FEELING_WORDS.get(person["emotion"], person["emotion"].lower())


def main_characters(faces, image_height):
    """The faces big enough to be main characters; each face has a "box" [x1, y1, x2, y2]."""
    return [f for f in faces if f["box"][3] - f["box"][1] >= MIN_FACE_HEIGHT * image_height]
