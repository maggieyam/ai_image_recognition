"""
What the app says about a picture, section by section, in the first person.
Each section is its own narrow Qwen call (llm.py) at temperature 0, given
only the facts it needs; Qwen never sees the picture. How the people in the
picture feel is recognize_feelings.py. Each section is yielded piece by piece
as Qwen writes it, so the page can show it while it's being written.

1. what_i_saw: Florence-2's description -> "I saw ..."
2. (recognize_feelings: how the people feel, if there are any)
3. how_i_feel: the app's mood (CLIP) and why -> "I feel ..., because ..."
4. story_intention: the app offers to write a story and asks which genre the
   user would like. Qwen only gets the short caption: given the whole
   description, it described the picture again instead. Nor does it get the
   genres (the page shows them to pick from): given them, it chose one
   itself and then asked.

If Qwen gives no answer, each section falls back to a plain sentence.
"""
from llm import say
from sentences import lower_first

SAW_SYSTEM = (
    "Say what you saw in one or two short sentences, in the first person, starting "
    'with "I saw". Use only the description given. Write in English. Reply with only '
    "what you'd say."
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


def what_i_saw(qwen, description, caption):
    yield from _say(qwen, SAW_SYSTEM, f"Description: {description}", 80, f"I saw {lower_first(caption)}")


def how_i_feel(qwen, description, mood):
    facts = f"What you saw: {description}\nHow it made you feel: {mood}"
    yield from _say(qwen, FEEL_SYSTEM, facts, 60, f"It makes me feel {mood}.")


def story_intention(qwen, caption):
    facts = f"The picture: {caption}"
    fallback = "I'd like to write you a story about it. Which genre would you like?"
    yield from _say(qwen, STORY_SYSTEM, facts, 70, fallback)


def _say(qwen, system, facts, max_tokens, fallback):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": facts}]
    yield from say(qwen, messages, max_tokens, fallback)
