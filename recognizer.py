import os

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

BLIP_ID = "Salesforce/blip-image-captioning-base"
# Optionally load the model weights from a local folder instead of the Hugging Face Hub.
BLIP_WEIGHTS = os.environ.get("BLIP_WEIGHTS", BLIP_ID)


class ImageRecognizer:
    def __init__(self, device=None):
        """device: "cuda", "cpu", or None (auto-detect)."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.blip_processor = BlipProcessor.from_pretrained(BLIP_ID)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_WEIGHTS
        ).to(self.device)
        self.blip_model.eval()

        print(f"ImageRecognizer ready (device: {self.device})")

    def describe(self, image):
        """Return a short caption for a PIL image."""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # repetition_penalty stops BLIP looping on scenes of one repeated
            # object ("strawberries, strawberries, ..."). Don't combine it with
            # num_beams: beam search defeats no_repeat_ngram_size here.
            out = self.blip_model.generate(
                **inputs, max_new_tokens=50, repetition_penalty=1.5
            )
        return self.blip_processor.decode(out[0], skip_special_tokens=True)
