"""
Sight: what's in the picture, in words. Florence-2 describes it; the caption
is the first sentence of its description.
"""
import re

from brain.faculty import Faculty
from brain.language.sentences import split_sentences, upper_first

QUOTED = re.compile(r'\s*"[^"]*"')
STILL_FROM = re.compile(r"\bstill from\b", re.I)
AT_CAMERA = re.compile(r"\b(?:directly )?(?:at|into|toward|towards) the camera\b")
# "The image shows ...", "The image is ...", "It shows ..."
LEAD_IN = re.compile(
    r"^(?:(?:The|This) (?:image|photo|picture) (?:shows|depicts|is(?: of)?)|It shows)\s+"
)


class Sight(Faculty):
    """
    The picture as a caption, one sentence for the page, and a description, a
    paragraph on who and what is in it, what they wear and the setting, which
    story ideas start from.
    """

    needs = ("image",)
    gives = ("caption", "description")

    def __init__(self, florence):
        self.florence = florence

    def run(self, workspace):
        caption, description = _caption_and_description(self.florence.describe(workspace["image"]))
        workspace.post(caption=caption, description=description)
        yield {"caption": caption, "description": description}


def _caption_and_description(text):
    """
    (caption, description) from Florence-2's detailed description.

    Florence-2 describes film and TV stills as such ('a still from the drama
    "The King's Woman"', 'looking at the camera'), and the story writer then
    borrows the quoted title's characters and writes about actors and cameras.
    So those sentences are dropped and camera glances become "straight ahead".
    The caption is the first sentence left.
    """
    sentences = split_sentences(text)
    kept = [s for s in sentences if '"' not in s and not STILL_FROM.search(s)]
    if not kept:  # every sentence named a title: keep them, minus the titles
        kept = [QUOTED.sub("", s) for s in sentences]
    description = AT_CAMERA.sub("straight ahead", " ".join(kept))
    caption = LEAD_IN.sub("", split_sentences(description)[0])
    return upper_first(caption), description
