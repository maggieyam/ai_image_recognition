"""
Recognizing the emotion on a face, one face at a time: HSEmotion's
EmotiEffLib, a pretrained facial-emotion model trained on AffectNet. It gives
one of 8 emotions plus valence and arousal. ONNX, so loading it runs no
pickled code; the weights download to ~/.emotiefflib, outside the repo.
"""
import numpy as np
from emotiefflib.facial_analysis import EmotiEffLibRecognizer

EMOTION_MODEL = "enet_b0_8_va_mtl"


class EmotionRecognizer:
    def __init__(self):
        self.model = EmotiEffLibRecognizer(engine="onnx", model_name=EMOTION_MODEL)

    def recognize(self, face):
        """The emotion on a face (PIL image): {"emotion": "Sadness", "valence": -0.7, "arousal": 0.4}."""
        emotions, scores = self.model.predict_emotions([np.asarray(face)], logits=False)
        return {
            "emotion": emotions[0],
            "valence": round(float(scores[0][-2]), 3),
            "arousal": round(float(scores[0][-1]), 3),
        }
