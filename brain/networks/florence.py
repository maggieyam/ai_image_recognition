"""Florence-2, the vision-language model that puts pictures into words and finds objects in them."""
import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration

# Replaced BLIP-base: about the same size, but its detailed descriptions say
# what's actually in the picture, where BLIP often guessed a film title.
FLORENCE_ID = "florence-community/Florence-2-base"
# Object detection lists every object, about 7 tokens each, so 512 tokens
# cover about 70 objects; "human face" entries past the limit would be lost.
DETECT_TOKENS = 512


class Florence:
    def __init__(self, device=None):
        """device: "cuda", "cpu", or None (auto-detect)."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(FLORENCE_ID)
        self.model = Florence2ForConditionalGeneration.from_pretrained(FLORENCE_ID).to(self.device)
        self.model.eval()

    def describe(self, image):
        """
        A paragraph on who and what is in a PIL image, what they wear and the
        setting: Florence-2's most detailed mode. Its short-caption mode
        guesses titles ("the king's woman china web drama") and invents
        settings (a beach behind a couple in a city), much like BLIP did.
        """
        return self._generate(image, "<MORE_DETAILED_CAPTION>", max_new_tokens=160)

    def caption(self, image):
        """A few words on what a PIL image shows; good enough for a crop of one person."""
        return self._generate(image, "<CAPTION>", max_new_tokens=20)

    def find_faces(self, image):
        """The boxes [x1, y1, x2, y2] of the faces its object detection finds in a PIL image."""
        found = self._generate(image, "<OD>", DETECT_TOKENS, parse=True)
        return [b for label, b in zip(found["labels"], found["bboxes"]) if label.lower() == "human face"]

    def _generate(self, image, task, max_new_tokens, parse=False):
        """Florence-2's output for a task: text, or with `parse`, its parsed result (boxes)."""
        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        if not parse:
            return self.processor.decode(out[0], skip_special_tokens=True).strip()
        raw = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(raw, task=task, image_size=image.size)[task]
