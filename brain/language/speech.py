"""
What the brain says about a picture, section by section, in the first person.
Code decides what there is to say; Qwen (networks/qwen.py) only puts it into
words. Each section is its own narrow Qwen call at temperature 0, given only
the facts it needs, so the same picture gets the same words; Qwen never sees
the picture. Each section is yielded piece by piece as Qwen writes it, so the
page can show it while it's being written.

1. what_i_saw: Florence-2's description -> "I saw ..."
2. how_they_feel: the characters and the emotion read on each face
   (perception/faces.py), as JSON -> how they feel, if there are any. Qwen
   sees only those facts, so it can relate them ("the man is angry, and the
   girl is crying, so she must be sad").
3. how_i_feel: the brain's mood (CLIP) and why -> "I feel ..., because ..."
4. offer_story: the brain offers to write a story and asks which genre the
   user would like. Qwen only gets the short caption: given the whole
   description, it described the picture again instead. Nor does it get the
   genres (the page shows them to pick from): given them, it chose one
   itself and then asked.

If Qwen gives no answer, each section falls back to a plain sentence.

Each is a plain function of the facts it needs, yielding events for the page
(faculty.py); Say puts one into the brain's cycle at the point where it's said.
"""
import json
import random

from brain.faculty import Faculty
from brain.imagination.story import GENRES
from brain.language.sentences import lower_first, upper_first
from brain.perception.faces import feeling

SAW_SYSTEM = (
    "Say what you saw in one or two short sentences, in the first person, starting "
    'with "I saw". Use only the description given. Write in English. Reply with only '
    "what you'd say."
)
EXPRESS_SYSTEM = (
    "You get a JSON list of the main characters in a photo, with who each one is and how they feel. "
    "Say how they feel in one or two short, natural sentences. Use only the facts given. "
    "Write in English. Reply with only what you'd say."
)
FEEL_SYSTEM = (
    "In one short sentence, in the first person, say how the picture made you feel "
    "and why, using the feeling given and what the picture shows. Write in English. Reply "
    "with only what you'd say."
)
STORY_SYSTEM = (
    "Express to the user that you're inspired to write a story, and politely ask them "
    "to pick a genre. Use one or two short sentences, in the first person. Don't describe "
    "the picture, and don't name any genre. Write in English. Reply with only what you'd say."
)
GENRE_CHOICES = 3  # genres the page offers for the story


class Say(Faculty):
    """
    One of the sections below as a faculty: `section(voice, *needs)`, each
    read from the workspace. "voice" is this stimulus's turn at Qwen: anything
    with Qwen's stream() (a qwen.Turns, when Qwen is shared).
    """

    def __init__(self, section, needs):
        self.section, self.needs = section, ("voice", *needs)

    def run(self, workspace):
        return self.section(*(workspace[name] for name in self.needs))

    def __str__(self):
        return f"Say {self.section.__name__}"


def what_i_saw(voice, description, caption):
    fallback = f"I saw {lower_first(caption)}"
    yield from _say("saw", voice, SAW_SYSTEM, f"Description: {description}", 80, fallback)


def how_they_feel(voice, characters):
    """
    How the characters feel, from who each one is and the emotion read from
    their face: [{"character": "a woman in a red dress", "emotion": "Sadness"}, ...].
    Nothing is said when no face was found.
    """
    if not characters:
        return
    facts = [{"character": c["character"], "feeling": feeling(c["emotion"])} for c in characters]
    fallback = " ".join(f"{upper_first(f['character'])} looks {f['feeling']}." for f in facts)
    yield from _say("feelings", voice, EXPRESS_SYSTEM, json.dumps(facts), 20 + 30 * len(characters), fallback)


def how_i_feel(voice, description, moods):
    """moods: the brain's moods, strongest first, as [(mood, probability), ...]."""
    mood = moods[0][0]
    yield {"status": "Letting it sink in…", "for": "i_feel"}
    facts = f"What you saw: {description}\nHow it made you feel: {mood}"
    yield from _say("i_feel", voice, FEEL_SYSTEM, facts, 60, f"It makes me feel {mood}.")
    # The moods to pick from follow their sentence.
    yield {"moods": [{"mood": m, "score": p} for m, p in moods]}


def offer_story(voice, caption):
    yield {"status": "Thinking of a story…", "for": "story_intention"}
    fallback = "I'd like to write you a story about it. Which genre would you like?"
    yield from _say("story_intention", voice, STORY_SYSTEM, f"The picture: {caption}", 70, fallback)
    # The genres to pick from follow their sentence, too.
    yield {"genres": random.sample(list(GENRES), GENRE_CHOICES)}


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


def _say(section, voice, system, facts, max_tokens, fallback):
    """A section worded by Qwen from only its facts, each piece as an event for the page."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": facts}]
    for piece in say(voice, messages, max_tokens, fallback):
        yield {section: piece}
