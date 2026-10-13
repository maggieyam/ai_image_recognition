"""
What the brain says about what it saw, in the first person: what it saw, how
the characters in it look (if any were found: their feelings are read from
their faces, so it's how they look, not how they feel), how it makes the
brain feel, and an offer to write a story. Code decides what there is to
say; Qwen (networks/qwen.py) only puts it into words, from the facts alone.
Qwen never gets a picture: only what the eyes saw, as the faculties before it
left it on the workspace. What the eyes were looking at comes with the stimulus
("looking_at"): the web page says it's a picture the user showed; an eye on
the world would say something else, and the prompt doesn't decide it.

One prompt covers all four parts, and the prompt keeps them apart. What made
that work with Qwen 1.5B: two worked examples, the characters given as a
plain line, and a firm rule for their feelings. On 16 OASIS photos it then
said how the characters looked every time (12 of 12, always as "looks", never
"feels"), never mixed their feelings with the brain's own (8 of 8 where they
differ), and asked for a genre more often than four separate prompts did (14
of 16, against 5). With the characters as a JSON list and a looser rule, it
left their feelings out in 5 of 12.
Temperature 0, so the same picture gets the same words.

Qwen isn't given the genres (the page shows them to pick from): given them,
it chose one itself and then asked.

The reply is JSON, one part after another, and each part goes to the page as
soon as it's complete, while Qwen writes the next. A part Qwen leaves out
(it was busy, or cut off) is said as a plain sentence instead.
"""
import itertools
import json
import random
import re

from brain.faculty import Faculty
from brain.imagination.story import GENRES
from brain.language.sentences import lower_first, upper_first
from brain.perception.faces import feeling

SYSTEM = (
    "You tell the user, in the first person, what you just saw and how you feel about it. You get what you "
    "were looking at, what your eyes saw, the main characters you saw with how each one looks, and how it made "
    "you feel. "
    "Reply in JSON with four parts, each one or two short sentences, in English:\n"
    '"saw": what you saw, starting with "I saw". Only what your eyes saw.\n'
    '"they_look": if characters are listed, say how each one looks, using exactly the feeling given for them. '
    'You only see their faces, so say "looks happy", never "feels happy" or "is happy". '
    'Only if it says "none found", leave it "".\n'
    '"i_feel": how what you saw made you feel and why: exactly the feeling given for you. It is your feeling, '
    "not the characters'.\n"
    '"story_offer": tell the user you\'re inspired to write a story and ask them to pick a genre. '
    "Don't describe what you saw or the story, and don't name any genre."
)
# The parts of the reply, in the order Qwen writes them, and the page section each is shown in.
SECTIONS = {"saw": "saw", "they_look": "feelings", "i_feel": "i_feel", "story_offer": "story_intention"}
SCHEMA = {
    "type": "object",
    "properties": {part: {"type": "string"} for part in SECTIONS},
    "required": list(SECTIONS),
}
# One part of the reply, once its closing quote has arrived.
PART = re.compile(r'"(%s)"\s*:\s*"((?:[^"\\]|\\.)*)"' % "|".join(SECTIONS))
# What the page shows where a part will appear, while Qwen writes it.
STATUS = {"i_feel": "Letting it sink in…", "story_offer": "Thinking of a story…"}
MAX_TOKENS = 300  # replies run to about 150 tokens
GENRE_CHOICES = 3  # genres the page offers for the story


def _facts(looking_at, description, characters, mood):
    """The facts Qwen puts into words. characters: [{"character": ..., "feeling": ...}, ...]."""
    who = "; ".join(f"{c['character']} ({c['feeling']})" for c in characters) or "none found"
    return (
        f"You were looking at: {looking_at}\nWhat your eyes saw: {description}\n"
        f"Characters and how they look: {who}\nHow it made you feel: {mood}"
    )


# Two worked examples, as (what the eyes saw, the characters, the brain's mood,
# the reply); small models follow a demonstration far better than instructions
# alone. In one a character feels differently from the brain; in the other
# there are no characters.
EXAMPLES = [
    (
        "A grey cat sits on a windowsill, looking out at the rain. A girl in a yellow coat stands beside it, "
        "holding an umbrella.",
        [{"character": "a close up of a girl in a yellow coat", "feeling": "happy"}],
        "lonely",
        {
            "saw": "I saw a grey cat on a windowsill looking out at the rain, and a girl in a yellow coat "
            "beside it with an umbrella.",
            "they_look": "The girl in the yellow coat looks happy.",
            "i_feel": "It makes me feel lonely, with the rain outside and the cat just watching it.",
            "story_offer": "I'd love to write you a story about this. Which genre would you like?",
        },
    ),
    (
        "A red tractor stands in a wheat field at sunset. The sky is orange and there are hills in the distance.",
        [],
        "peaceful",
        {
            "saw": "I saw a red tractor standing in a wheat field at sunset, with hills in the distance.",
            "they_look": "",
            "i_feel": "It makes me feel peaceful, with the warm light and nothing moving.",
            "story_offer": "Seeing this makes me want to write a story. What genre should it be?",
        },
    ),
]


def _examples(looking_at):
    """The worked examples as messages, looking at the same kind of thing as now."""
    messages = []
    for description, characters, mood, reply in EXAMPLES:
        messages += [
            {"role": "user", "content": _facts(looking_at, description, characters, mood)},
            {"role": "assistant", "content": json.dumps(reply)},
        ]
    return messages


class Speak(Faculty):
    """
    Everything the brain says about what it saw, in one Qwen call. "voice" is
    this stimulus's turn at Qwen: anything with Qwen's stream() (a qwen.Turns,
    when Qwen is shared).
    """

    needs = ("voice", "looking_at", "description", "caption", "characters", "moods")

    def run(self, workspace):
        moods = workspace["moods"]
        mood = moods[0][0]
        characters = [
            {"character": c["character"], "feeling": feeling(c["emotion"])} for c in workspace["characters"]
        ]
        # What's said in place of a part Qwen leaves out.
        plain = {
            "saw": f"I saw {lower_first(workspace['caption'])}",
            "they_look": " ".join(f"{upper_first(c['character'])} looks {c['feeling']}." for c in characters),
            "i_feel": f"It makes me feel {mood}.",
            "story_offer": "I'd like to write you a story about it. Which genre would you like?",
        }
        # The moods and genres to pick from follow their sentences.
        choices = {
            "i_feel": {"moods": [{"mood": m, "score": p} for m, p in moods]},
            "story_offer": {"genres": random.sample(list(GENRES), GENRE_CHOICES)},
        }
        messages = [
            {"role": "system", "content": SYSTEM},
            *_examples(workspace["looking_at"]),
            {"role": "user", "content": _facts(workspace["looking_at"], workspace["description"], characters, mood)},
        ]
        reply = workspace["voice"].stream(messages, temperature=0, max_tokens=MAX_TOKENS, schema=SCHEMA)
        order, said = list(SECTIONS), set()
        # Qwen's parts, each as soon as it's complete, then plain ones for any it left out.
        for part, text in itertools.chain(_parts(reply), plain.items()):
            if part in said:
                continue
            said.add(part)
            text = text.strip() or plain[part]
            if part == "they_look" and not characters:
                text = ""  # nothing is said when no face was found
            if text:
                yield {SECTIONS[part]: text}
            if part in choices:
                yield choices[part]
            following = order[order.index(part) + 1 :][:1]
            if following and following[0] in STATUS and following[0] not in said:
                yield {"status": STATUS[following[0]], "for": SECTIONS[following[0]]}


def _parts(pieces):
    """
    (part, text) for each part of Qwen's JSON reply, as soon as that part is
    complete: `pieces` is the reply's text as it's generated.
    """
    reply, found = "", 0
    for piece in pieces:
        reply += piece
        for match in list(PART.finditer(reply))[found:]:
            found += 1
            try:
                yield match[1], json.loads(f'"{match[2]}"')
            except json.JSONDecodeError:  # not a string after all: left to the plain sentence
                pass
