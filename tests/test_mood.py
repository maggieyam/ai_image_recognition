import torch

from brain.affect.mood import MOODS, Mood
from brain.workspace import Workspace


class FakeClip:
    """Each text's vector says only which mood it's about; a picture is given as its vector."""

    def text_vectors(self, texts):
        return torch.tensor(
            [[float(any(s in text for s in synonyms)) for synonyms in MOODS.values()] for text in texts]
        )

    def image_vector(self, image):
        return image

    def probabilities(self, image_vec, label_vecs):
        return (10 * label_vecs @ image_vec).softmax(dim=0)


def picture(**strengths):
    return torch.tensor([strengths.get(mood, 0.0) for mood in MOODS])


def test_the_moods_that_come_through_most_strongest_first():
    workspace = Workspace(image=picture(lonely=0.9, gloomy=0.5, peaceful=0.3))
    assert Mood(FakeClip()).run(workspace) is None  # nothing for the page yet
    moods = workspace["moods"]
    assert [m for m, _ in moods] == ["lonely", "gloomy", "peaceful"]
    assert all(p == round(p, 3) for _, p in moods) and moods[0][1] > moods[1][1] > moods[2][1]
