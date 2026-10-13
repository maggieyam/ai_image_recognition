"""Quick look at sight and mood, without the rest of the brain: python -m brain images/*.jpg"""
import sys

from brain.affect.mood import Mood
from brain.faculty import run
from brain.networks.clip import Clip
from brain.networks.florence import Florence
from brain.perception.sight import Sight
from brain.workspace import Workspace
from senses import eye

faculties = [Sight(Florence()), Mood(Clip())]
for path in sys.argv[1:]:
    workspace = Workspace(image=eye.see(path))
    for _ in run(faculties, workspace):
        pass
    moods = ", ".join(f"{m} {p:.0%}" for m, p in workspace["moods"])
    print(f"{path}\n  caption: {workspace['caption']}\n  mood:    {moods}\n  detail:  {workspace['description']}")
