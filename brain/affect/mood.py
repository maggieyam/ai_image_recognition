"""
Mood: how the picture makes the brain feel. CLIP scores the picture itself
against each mood, not a description of it.
"""
import torch

from brain.faculty import Faculty

# Zero-shot mood labels. Each mood is scored as the average of CLIP text
# embeddings over (synonym x template), which is steadier than one prompt.
MOODS = {
    "happy": ["happy", "joyful", "cheerful"],
    "playful": ["playful", "fun", "lighthearted"],
    "exciting": ["exciting", "energetic", "thrilling"],
    "peaceful": ["peaceful", "calm", "serene"],
    "cozy": ["cozy", "warm", "comfortable"],
    "romantic": ["romantic", "tender", "loving"],
    "busy": ["busy", "hectic", "crowded"],
    "lonely": ["lonely", "isolated", "solitary"],
    "sad": ["sad", "sorrowful", "melancholy"],
    "gloomy": ["gloomy", "bleak", "dreary"],
    "tense": ["tense", "scary", "threatening"],
    "mysterious": ["mysterious", "eerie", "enigmatic"],
}
MOOD_TEMPLATES = [
    "a {} scene.",
    "a photo of a {} scene.",
    "a photo that feels {}.",
    "a {} moment.",
]


class Mood(Faculty):
    """The moods that come through most, strongest first: [(mood, probability), ...]."""

    needs = ("image",)
    gives = ("moods",)

    def __init__(self, clip):
        self.clip = clip
        # One unit vector per mood, computed once. A linear probe trained on
        # CLIP image features would replace this matrix (and CLIP's own scale).
        self.mood_names = list(MOODS)
        self.mood_vectors = self._embed_moods()

    def run(self, workspace):
        workspace.post(moods=self.moods(workspace["image"]))

    def moods(self, image, top_k=3):
        """Return the top_k moods for a PIL image as [(mood, probability), ...]."""
        probs = self.clip.probabilities(self.clip.image_vector(image), self.mood_vectors)
        top = probs.topk(top_k)
        return [
            (self.mood_names[i], round(p, 3))
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]

    def _embed_moods(self):
        vectors = []
        for synonyms in MOODS.values():
            prompts = [t.format(s) for s in synonyms for t in MOOD_TEMPLATES]
            mean = self.clip.text_vectors(prompts).mean(dim=0)
            vectors.append(mean / mean.norm())
        return torch.stack(vectors)
