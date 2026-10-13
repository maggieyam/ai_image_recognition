"""
The brain itself: it loads the networks, lists the faculties, and is all the
web server talks to. What comes in through the senses (senses/) goes on a
workspace, and each faculty in turn adds what it makes of it.

To give the brain a new ability, write a Faculty (faculty.py) and list it in
build_faculties; the other faculties stay as they are. One that needs a new
sense is skipped whenever that sense took in nothing.
"""
from brain.affect.mood import MOODS, Mood
from brain.faculty import check_wiring, run
from brain.imagination.story import GENRES, MAX_IDEAS, StoryWriter
from brain.language.speech import Speak
from brain.networks.clip import Clip
from brain.networks.face_emotion import FaceEmotion
from brain.networks.florence import Florence
from brain.networks.qwen import Busy, Qwen, SharedQwen
from brain.networks.yunet import YuNet
from brain.perception.faces import ReadFaces
from brain.perception.sight import Sight
from brain.workspace import Workspace

# What the web server needs besides the Brain: Busy, and what it checks requests against.
__all__ = ["Brain", "Busy", "GENRES", "MAX_IDEAS", "MOODS"]

# What a stimulus is made of: what the eyes took in, and what they were
# looking at ("a picture the user showed you"), which only whoever passes it
# in can know.
STIMULUS = ("image", "looking_at")
STORY_WAIT = 30  # seconds a story waits for Qwen before Busy
# Seconds what the brain says about a stimulus waits for Qwen (busy writing a
# story) before it's said in plain sentences instead; perceiving never fails
# on it.
SPEECH_WAIT = 5


def build_faculties(florence, clip, yunet, face_emotion):
    """What the brain makes of a stimulus, in order: it perceives and feels, then speaks."""
    faculties = [
        Sight(florence),
        Mood(clip),
        ReadFaces(yunet, florence, face_emotion),
        Speak(),
    ]
    # "voice" is this stimulus's turn at Qwen, which the brain adds itself.
    check_wiring(faculties, STIMULUS + ("voice",))
    return faculties


class Brain:
    def __init__(self):
        # Florence-2 and CLIP are safe to run from several threads at once;
        # Qwen isn't, so everything that uses it takes turns (SharedQwen).
        florence, clip, qwen = Florence(), Clip(), Qwen()
        self.clip = clip
        self.qwen = SharedQwen(qwen)
        self.writer = StoryWriter(qwen)
        self.faculties = build_faculties(florence, clip, YuNet(), FaceEmotion())
        print(f"Brain ready (device: {florence.device})")

    def perceive(self, **stimulus):
        """
        What the brain makes of what the senses took in (image=<PIL image>,
        looking_at="a picture the user showed you"), as the events its
        faculties yield (faculty.py), each as soon as it's ready.
        """
        workspace = Workspace(voice=self.qwen.in_passing(SPEECH_WAIT), **stimulus)
        yield from run(self.faculties, workspace)

    def plan_stories(self, image, description, mood, count, genre=None):
        """
        Story ideas for a picture, checked against it with CLIP: StoryWriter.plan's
        result, or None. Raises Busy if Qwen stays taken.
        """
        image_vec = self.clip.image_vector(image)  # once, for every grounding check
        with self.qwen.hold(STORY_WAIT):
            return self.writer.plan(
                description,
                mood,
                count,
                match=lambda texts: self.clip.match(image_vec, texts),
                genre=genre,
            )

    def write_story(self, description, mood, idea):
        """The story seed for one planned idea, or None. Raises Busy if Qwen stays taken."""
        with self.qwen.hold(STORY_WAIT):
            return self.writer.write(description, mood, idea)
