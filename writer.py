import difflib
import itertools
import json
import random
import re

from sentences import split_sentences

# Each step tries twice. A prompt that fails twice tends to keep failing the
# same way, so after that the step gives up and the page moves on to a
# different idea rather than asking again.
MAX_TRIES = 2
MAX_IDEAS = 3
OPTIONS_PER_ASPECT = 4
BRAINSTORM_IDEAS = 6  # brainstormed per aspect; the best grounded ones are kept
# CLIP grounding check on brainstormed ideas. CLIP spots things that aren't in
# the photo (a bird in a desert photo, fish for a man in a robe) but can't
# confirm roles ("a journalist"), so good ideas also score low in absolute
# terms. Ideas are kept if they score at least GROUNDING_MIN and are within
# GROUNDING_MARGIN of the best idea for the same photo. Calibrated on 6 photos:
# best ideas 0.25-0.34, rejected ones 0.15-0.22.
GROUNDING_MIN = 0.18
GROUNDING_MARGIN = 0.05
GROUNDING_KEEP = 2  # but always keep this many (if above GROUNDING_MIN), for variety
MAX_STORY_SENTENCES = 3

# Each genre with what a story needs to count as one. A 1.5B model given only
# the label ignores it ("time travel" stories had no time travel), so the
# writer gets this line too.
GENRES = {
    "love story": "two characters fall for each other, or a love is put at risk",
    "comedy": "a funny misunderstanding or absurd situation drives the story",
    "thriller": "someone is in danger and time is running out",
    "mystery": "something puzzling has happened and someone must find out why",
    "adventure": "a journey or quest into the unknown",
    "fantasy": "magic or magical creatures are real",
    "fairy tale": "a wish, a curse, an enchantment or a talking animal",
    "science fiction": "a future technology or discovery changes everything",
    "family drama": "a conflict between members of a family",
    "coming-of-age story": "a young person finds out who they are",
    "heist": "someone plans to steal something valuable",
    "ghost story": "a ghost haunts a place or a person",
    "slice of life": "a small, everyday problem in ordinary life",
    # Hugely popular (穿越 / 穿书 in Chinese web fiction, isekai in Japanese).
    "time travel": "someone from today wakes up in this scene's time, or the person "
    "in the photo is thrown into another era",
    # Saying "invented" here made the model write "an invented book"; the
    # ORIGINAL rule already keeps the story world made up.
    "transported into a story": "someone from our world wakes up inside the world of "
    "a novel, TV drama or video game, as the person in the photo",
}
# What to vary before writing. A list is sampled at random in code; a string
# is what the model is asked to brainstorm from the photo. Genres are sampled
# because the model can't be surprising: asked to invent genres it echoed the
# photo ("man walking alone in the desert"), and asked to choose from this
# list it always took slice of life, love story or coming-of-age.
# Each brainstormed aspect is its own short call, so adding one (e.g.
# "setting") costs one more call, not a longer prompt.
ASPECTS = {
    "genre": list(GENRES),
    "characters": "ideas for who the people, animals or things in the photo are: their "
    "jobs, roles or relationship to each other. Only who is in the photo, no one else. "
    "A few words each, no names",
}

# Every system prompt also asks for English: Qwen sometimes answers in Chinese.
ORIGINAL = (
    "Everything must be original: never use characters, names, titles or plots from "
    "existing books, films, TV shows, games or legends, and never real people. "
)
# Florence-2 calls film and TV stills "a still from a drama", which made
# every idea a story about actors.
REAL_SCENE = (
    "If the photo looks like a film or TV still, treat it as a real moment: "
    "people are who they appear to be, not actors, and there is no camera. "
)
BRAINSTORM_SYSTEM = (
    "You brainstorm ideas for short stories inspired by a photo and a mood. "
    "Base every idea on what the photo description shows. "
    + REAL_SCENE
    + ORIGINAL
    + "Make every idea clearly different from the others. Write in English and reply in JSON."
)
PICK_SYSTEM = (
    "You are a story editor choosing which story ideas to develop. "
    "Pick the ones that fit the photo and mood and would make the most interesting "
    "stories, and prefer picks with different genres. Write in English and reply in JSON."
)
WRITE_SYSTEM = (
    "You turn a story idea into a short story pitch. "
    "The thing or person in the photo must be a main character or the main setting. "
    + REAL_SCENE
    + "Use the given characters. The genre comes with what it needs; start by writing "
    "the premise: one sentence on how this story delivers that. "
    "First decide what the main character wants and what stands in their way; "
    "the mood decides how big that problem is (a cozy or happy story can have a gentle one). "
    "Give the characters fresh, unusual names that suit the setting. "
    + ORIGINAL
    + "The storyline is 2 or 3 sentences built on that want and obstacle. The first "
    "line is the story's first sentence, in the given mood. Write in English and reply in JSON."
)
# One worked example; small models follow a demonstration far better than
# instructions alone.
WRITE_EXAMPLE = [
    {
        "role": "user",
        "content": "Photo description: A grey cat sits on a windowsill, looking out at "
        "the rain.\nMood: lonely\n"
        "Genre: family drama (a conflict between members of a family)\n"
        "Characters: an old cat and his owner in hospital",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "title": "The Last Window on Alder Street",
                "characters": [
                    {
                        "name": "Moss",
                        "about": "an old grey cat who has lived in the same flat for fourteen years",
                    },
                    {
                        "name": "Mrs. Petrova",
                        "about": "his owner, who went into hospital a week ago and hasn't come back",
                    },
                ],
                "premise": "Mrs. Petrova's son wants to sell the flat while she is in "
                "hospital, and only her cat stands for her side.",
                "want": "Moss wants Mrs. Petrova to come home.",
                "obstacle": "The flat is being sold, and the movers are coming.",
                "storyline": "Every day Moss waits at the window for Mrs. Petrova, while the "
                "neighbour who feeds him talks on the phone about selling the flat. When the "
                "movers arrive, Moss must choose between the only home he knows and slipping "
                "out into the rain to find her.",
                "first_line": "The rain had not stopped for three days, and neither had "
                "Moss's watch at the window.",
            }
        ),
    },
]
SEED_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "about": {"type": "string"}},
                "required": ["name", "about"],
            },
            "minItems": 1,
            "maxItems": 3,
        },
        # Generated in this order, so the model commits to how the story fits
        # its genre and to a conflict before writing the storyline; without
        # these, many storylines ignored the genre or had no conflict.
        "premise": {"type": "string"},
        "want": {"type": "string"},
        "obstacle": {"type": "string"},
        "storyline": {"type": "string"},
        "first_line": {"type": "string"},
    },
    "required": ["title", "characters", "premise", "want", "obstacle", "storyline", "first_line"],
}


class StoryWriter:
    """
    Story ideas in small steps, each a fresh short prompt:
    brainstorm each aspect -> pick the best combinations -> write each pick.
    Only the results (plain dicts) pass between steps.
    """

    def __init__(self, qwen):
        """qwen: the shared llm.Qwen."""
        self.qwen = qwen

    def plan(self, description, mood, count=MAX_IDEAS, match=None, genre=None):
        """
        Brainstorm options for each aspect and pick the `count` best
        combinations. `match(texts)` scores texts against the photo (CLIP);
        brainstormed ideas scoring below GROUNDING_MIN are rejected. Returns
        {"options": {aspect: [...]}, "rejected": {aspect: [...]},
         "ideas": [{"genre": ..., "characters": ...}, ...], "spares": [...]}
        or None if no grounded ideas came up. "spares" are up to `count` more
        combinations, to write in place of an idea that can't be written.
        With `genre`, every idea is in that genre instead of a random few.
        """
        options, rejected = {}, {}
        for aspect, source in ASPECTS.items():
            if aspect == "genre" and genre:
                options[aspect] = [genre]
                continue
            if isinstance(source, list):
                options[aspect] = random.sample(source, OPTIONS_PER_ASPECT)
                continue
            # With one chosen genre, each idea makes only one combination, so
            # keep more of them: up to enough for the ideas and their spares,
            # but insist only on enough for the ideas, so spares still have to
            # be within GROUNDING_MARGIN of the best.
            keep = (BRAINSTORM_IDEAS, count) if genre else (OPTIONS_PER_ASPECT, GROUNDING_KEEP)
            grounded = self._grounded_ideas(description, mood, source, match, *keep)
            if grounded is None:
                return None
            options[aspect], rejected[aspect] = grounded
        combos = [dict(zip(options, c)) for c in itertools.product(*options.values())]
        ideas, spares = self._pick(description, mood, combos, count)
        return {"options": options, "rejected": rejected, "ideas": ideas, "spares": spares}

    def _grounded_ideas(
        self, description, mood, ask, match, most=OPTIONS_PER_ASPECT, least=GROUNDING_KEEP
    ):
        """
        (kept, rejected) brainstormed ideas, best grounded first, or None.
        Up to `most` are kept, and at least `least` if they pass GROUNDING_MIN.
        Rejected ideas are the ones that failed the CLIP check; grounded ideas
        beyond `most` are left out of both.
        """
        for _ in range(MAX_TRIES):
            ideas = self._brainstorm(description, mood, ask)
            if not ideas:
                continue
            if match is None:
                return ideas[:most], []
            scores = dict(zip(ideas, match(ideas)))
            ranked = sorted(ideas, key=scores.get, reverse=True)
            cutoff = max(GROUNDING_MIN, scores[ranked[0]] - GROUNDING_MARGIN)
            kept = [i for i in ranked if scores[i] >= cutoff][:most]
            if len(kept) < least:
                kept = [i for i in ranked if scores[i] >= GROUNDING_MIN][:least]
            if kept:
                return kept, [i for i in ranked if i not in kept and scores[i] < cutoff]
        return None

    def write(self, description, mood, idea):
        """
        Return a story seed for one picked idea, or None:
        {"title", "characters": [{"name", "about"}], "storyline", "first_line"}
        """
        messages = [
            {"role": "system", "content": WRITE_SYSTEM},
            *WRITE_EXAMPLE,
            {"role": "user", "content": _scene(description, mood) + "\n" + _idea_text(idea, "\n", hint=True)},
        ]
        for _ in range(MAX_TRIES):
            seed = self._ask(messages, SEED_SCHEMA, temperature=0.8, max_tokens=300)
            if seed and _complete(seed) and _sensible(seed, description):
                seed["storyline"] = _first_sentences(seed["storyline"], MAX_STORY_SENTENCES)
                seed["first_line"] = _first_sentences(seed["first_line"], 1)
                return seed
        return None

    def _brainstorm(self, description, mood, ask):
        n = BRAINSTORM_IDEAS
        messages = [
            {"role": "system", "content": BRAINSTORM_SYSTEM},
            {"role": "user", "content": f"{_scene(description, mood)}\nList {n} {ask}."},
        ]
        schema = _list_schema("ideas", {"type": "string"}, n)
        # One call; _grounded_ideas retries. A few repeats are fine.
        reply = self._ask(messages, schema, temperature=0.9, max_tokens=250)
        # Strip the "1. " numbering the model sometimes adds.
        ideas = [re.sub(r"^\d+[.)]\s*", "", i).strip() for i in (reply or {}).get("ideas", [])]
        return list(dict.fromkeys(i for i in ideas if i))

    def _pick(self, description, mood, combos, count):
        # Shuffled: the model favours whatever is listed first.
        combos = random.sample(combos, len(combos))
        listing = "\n".join(f"{i}. {_idea_text(c, '; ')}" for i, c in enumerate(combos, 1))
        messages = [
            {"role": "system", "content": PICK_SYSTEM},
            {
                "role": "user",
                "content": f"{_scene(description, mood)}\nStory ideas:\n{listing}\n"
                f"Pick the {count} best ideas by number.",
            },
        ]
        schema = _list_schema("picks", {"type": "integer"}, count)
        reply = self._ask(messages, schema, temperature=0.2, max_tokens=40) or {}
        # Take the model's picks in its order, then the rest, skipping any that
        # reuse a genre or characters already picked (the model often repeats
        # one genre); only if that can't fill `count`, allow repeats.
        chosen = [i - 1 for i in reply.get("picks", []) if 1 <= i <= len(combos)]
        order = list(dict.fromkeys(chosen + list(range(len(combos)))))
        picks = []
        for strict in (True, False):
            for combo in (combos[i] for i in order):
                reuses = any(combo[a] == p[a] for p in picks for a in combo)
                if len(picks) < count and combo not in picks and not (strict and reuses):
                    picks.append(combo)
        spares = [combos[i] for i in order if combos[i] not in picks][:count]
        return picks, spares

    def _ask(self, messages, schema, temperature, max_tokens):
        return self.qwen.ask(messages, schema, temperature, max_tokens)


def _scene(description, mood):
    return f"Photo description: {description}\nMood: {mood}"


def _idea_text(idea, sep, hint=False):
    """The idea as prompt text; with `hint`, each genre says what it needs."""
    parts = []
    for aspect, value in idea.items():
        if hint and aspect == "genre" and value in GENRES:
            value = f"{value} ({GENRES[value]})"
        parts.append(f"{aspect.capitalize()}: {value}")
    return sep.join(parts)


def _list_schema(key, items, n):
    return {
        "type": "object",
        "properties": {key: {"type": "array", "items": items, "minItems": n, "maxItems": n}},
        "required": [key],
    }


# Phrases the model writes when it gives up on a part instead of answering.
NON_ANSWERS = re.compile(
    r"\b(?:unknown|not applicable|n/a|not (?:explicitly )?(?:stated|given|clear)"
    r"|no (?:want|obstacle|goal|conflict))\b"
    r"|^\W*(?:none|nothing)\b(?!\s+more)",  # "None stated yet", but not "nothing more than"
    re.I,
)
EXAMPLE_TEXTS = [
    v for v in json.loads(WRITE_EXAMPLE[1]["content"]).values() if isinstance(v, str)
]
# A part drawing more than this share of its words from the photo description
# is a restatement, not a story. Restated storylines measured 0.62-0.74, real
# ones 0-0.29.
MAX_DESCRIPTION_WORDS = 0.5
STOPWORDS = set(
    "a an the and or but of in on at to with his her their its is are was were be as "
    "he she they it this that for from by into who what has have had".split()
)
TEXT_PARTS = ("title", "premise", "want", "obstacle", "storyline", "first_line")


def _sensible(seed, description):
    """
    Reject seeds where the model gave up: a non-answer ("unknown"), a part
    copied from the worked example, or the photo description repeated in
    place of a story.
    """
    if any(NON_ANSWERS.search(c["name"]) for c in seed["characters"]):
        return False
    for key in ("premise", "want", "obstacle"):
        if NON_ANSWERS.search(seed[key]):
            return False
    for key in ("premise", "want", "obstacle", "storyline"):
        if _word_overlap(seed[key], description) > MAX_DESCRIPTION_WORDS:
            return False
    return not any(_copied(seed[key], EXAMPLE_TEXTS) for key in TEXT_PARTS)


def _word_overlap(text, source):
    """Share of the content words in `text` that also appear in `source`."""
    words = _content_words(text)
    known = set(_content_words(source))
    return sum(w in known for w in words) / len(words) if words else 0


def _content_words(text):
    return [
        w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 2 and w not in STOPWORDS
    ]


def _copied(text, sources):
    """Share of the sentences in `text` that closely match a sentence in `sources`."""
    sentences = split_sentences(text)
    known = [s.lower() for source in sources for s in split_sentences(source)]
    copies = sum(
        any(difflib.SequenceMatcher(None, s.lower(), k).ratio() > 0.75 for k in known)
        for s in sentences
    )
    return copies / len(sentences) if sentences else 0


def _complete(seed):
    return all(seed[k].strip() for k in TEXT_PARTS) and all(
        c["name"].strip() for c in seed["characters"]
    )


def _first_sentences(text, n):
    # The model often runs past the requested length; keep whole sentences only.
    return " ".join(split_sentences(text)[:n])
