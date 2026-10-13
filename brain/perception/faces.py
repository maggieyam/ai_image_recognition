"""
Reading faces: who the main characters in a photo are and how each one looks
like they feel. Putting that into words is language/speech.py.

1. YuNet and Florence-2: find the main characters by their faces, and say who
   each one is.
2. The face-emotion model: read each character's face.

Why a face model reads the emotion: guessing feelings from Florence-2's
description alone, Qwen 1.5B got 7 of 40 OASIS face photos right; worded from
the face model's results, 23 of 40 (as many as the face model).

Every photo is searched for faces; nothing first checks whether there's a
person (Qwen 1.5B couldn't reliably tell from the description; experiments on
issue #5). A face found on an animal or a statue is read like a person's: the
emotion is a guess at how a face looks, which fits them too. Objects rarely
get one: 1 of 62 OASIS object photos (a concrete wall).
"""
import logging

from brain.faculty import Faculty
from brain.language.sentences import lower_first

# Faces shorter than this share of the photo's height are background people
# (a crowd behind the main character). Provisional: to be set from real photos.
MIN_FACE_HEIGHT = 0.1
# At most this many main characters (the biggest faces): each one costs a
# Florence-2 caption and a face-model pass before any feelings are said.
MAX_PEOPLE = 4
# The face model's emotion labels as feelings.
FEELING_WORDS = {
    "Anger": "angry", "Contempt": "contemptuous", "Disgust": "disgusted", "Fear": "afraid",
    "Happiness": "happy", "Neutral": "neutral", "Sadness": "sad", "Surprise": "surprised",
}


class ReadFaces(Faculty):
    """
    The main characters in the photo, left to right, [] if no face is found:
    [{"character": "a woman in a white dress", "box": [x1, y1, x2, y2],
      "emotion": "Sadness", "valence": -0.7, "arousal": 0.4}, ...].
    """

    needs = ("image",)
    gives = ("characters",)

    def __init__(self, yunet, florence, face_emotion):
        """
        yunet finds the faces, florence finds the ones it misses and says who
        each belongs to, and face_emotion reads each face, one at a time.
        """
        self.yunet, self.florence, self.face_emotion = yunet, florence, face_emotion

    def run(self, workspace):
        yield {"status": "Looking at their faces…", "for": "feelings"}
        # Feelings are optional: if a face can't be read, that character is left out,
        # and the rest of what the brain says still follows.
        try:
            found = self.find_characters(workspace["image"])
        except Exception:
            logging.exception("finding the characters failed")
            found = []
        characters = []
        for c in found:
            try:
                c.update(self.face_emotion.recognize(c.pop("face")))
            except Exception:
                logging.exception("reading a face failed")
                continue
            characters.append(c)
        workspace.post(characters=characters)
        yield {"characters": characters}

    def find_characters(self, image):
        """
        The main characters in a PIL image, found by their faces, as
        [{"character": "a woman in a white dress", "box": [x1, y1, x2, y2],
          "face": <the face, cropped>}, ...], left to right. YuNet finds the
        faces; when it misses a close-up filling the frame, Florence-2's
        object detection looks for them instead.
        """
        faces = _clean_boxes(self.yunet.find_faces(image), image.size)
        characters = main_characters([{"box": b} for b in faces], image.height)
        # YuNet misses close-ups filling the frame, sometimes finding just a
        # small face in the background. Several small faces are a crowd, though,
        # so Florence-2 (slower) only looks when YuNet found at most one.
        if not characters and len(faces) <= 1:
            faces = _clean_boxes(self.florence.find_faces(image), image.size)
            characters = main_characters([{"box": b} for b in faces], image.height)
        for c in characters:
            c.update(character=self._who(image, c["box"]), face=image.crop(c["box"]))
        return characters

    def _who(self, image, box):
        """
        Who a face belongs to ("a woman in a black jacket"): Florence-2's caption
        of the face and what's just below it (hair, shoulders, clothes). The
        crop is kept narrow: a wider one took in the person next to them, and
        both got "a man and a woman standing next to each other".
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        around = image.crop(
            (max(0, x1 - w // 4), max(0, y1 - h // 2), min(image.width, x2 + w // 4), min(image.height, y2 + 2 * h))
        )
        return lower_first(self.florence.caption(around).rstrip("."))


def feeling(emotion):
    """The face model's emotion label as a feeling: "Sadness" -> "sad"."""
    return FEELING_WORDS.get(emotion, emotion.lower())


def main_characters(faces, image_height):
    """
    The faces big enough to be main characters, at most MAX_PEOPLE (the
    biggest), left to right. Each face has a "box" [x1, y1, x2, y2].
    """
    big = [f for f in faces if _height(f) >= MIN_FACE_HEIGHT * image_height]
    biggest = sorted(big, key=_height, reverse=True)[:MAX_PEOPLE]
    return sorted(biggest, key=lambda f: f["box"][0])


def _height(face):
    return face["box"][3] - face["box"][1]


def _clean_boxes(boxes, size):
    """
    Face boxes [x1, y1, x2, y2] in whole pixels, kept inside the picture;
    empty ones are dropped, as an empty crop breaks the face model.
    """
    width, height = size
    clamped = [
        [max(0, round(x1)), max(0, round(y1)), min(width, round(x2)), min(height, round(y2))]
        for x1, y1, x2, y2 in boxes
    ]
    return [b for b in clamped if b[2] > b[0] and b[3] > b[1]]
