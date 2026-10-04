import hashlib
import os
import re
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    Florence2ForConditionalGeneration,
)

from sentences import LEAD_IN, lower_first, split_sentences, upper_first

MODELS_DIR = Path(__file__).parent / "models"
# Replaced BLIP-base: about the same size, but its detailed descriptions say
# what's actually in the picture, where BLIP often guessed a film title.
FLORENCE_ID = "florence-community/Florence-2-base"
CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_LOCAL_DIR = MODELS_DIR / "clip-vit-base-patch32"
# OpenCV's YuNet face detector (MIT licence, 230 KB), downloaded to models/ on
# first use. It finds the faces in everyday photos (people from the waist up)
# that Florence-2 only labels "person"; Florence-2 still finds the faces in
# close-ups, which YuNet misses.
YUNET_FILE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
# Pinned to a commit and checked, so a moved or half-downloaded file isn't used.
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DOWNLOAD_SECONDS = 60  # a stalled download fails instead of hanging startup
FACE_SCORE = 0.8  # YuNet's confidence needed to count as a face

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

# Faces shorter than this share of the photo's height are background people
# (a crowd behind the main character). Provisional: to be set from real photos.
MIN_FACE_HEIGHT = 0.1
# At most this many main characters (the biggest faces): each one costs a
# Florence-2 caption and a face-model pass before any feelings are said.
MAX_PEOPLE = 4
# Object detection lists every object, about 7 tokens each, so 512 tokens
# cover about 70 objects; "human face" entries past the limit would be lost.
DETECT_TOKENS = 512

QUOTED = re.compile(r'\s*"[^"]*"')
STILL_FROM = re.compile(r"\bstill from\b", re.I)
AT_CAMERA = re.compile(r"\b(?:directly )?(?:at|into|toward|towards) the camera\b")


def _torch_can_load_bin():
    major, minor = torch.__version__.split(".")[:2]
    return (int(major), int(minor)) >= (2, 6)


def _weights_source(hub_id, model_cls, local_dir):
    """
    The Hub id for a model's weights, or a local safetensors copy on older torch.

    transformers refuses to torch.load the Hub's pytorch_model.bin below
    torch 2.6 (CVE-2025-32434), so older installs convert it once locally.
    """
    if _torch_can_load_bin():
        return hub_id
    if not (local_dir / "model.safetensors").exists():
        bin_path = hf_hub_download(hub_id, "pytorch_model.bin")
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        model = model_cls(model_cls.config_class.from_pretrained(hub_id))
        model.load_state_dict(state, strict=False)
        model.save_pretrained(local_dir)  # writes model.safetensors
    return local_dir


def _embedding(features):
    # transformers 5 wraps the projected embedding in an output object.
    return features if isinstance(features, torch.Tensor) else features.pooler_output


class ImageRecognizer:
    def __init__(self, device=None):
        """device: "cuda", "cpu", or None (auto-detect)."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.florence_processor = AutoProcessor.from_pretrained(FLORENCE_ID)
        self.florence_model = Florence2ForConditionalGeneration.from_pretrained(
            FLORENCE_ID
        ).to(self.device)
        self.florence_model.eval()

        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
        self.clip_model = CLIPModel.from_pretrained(
            _weights_source(CLIP_ID, CLIPModel, CLIP_LOCAL_DIR)
        ).to(self.device)
        self.clip_model.eval()
        self.face_detector = cv2.FaceDetectorYN.create(
            str(_yunet_file()), "", (320, 320), score_threshold=FACE_SCORE
        )
        # One detector for every request: its input size is set per picture.
        self.face_lock = threading.Lock()
        # One unit vector per mood, computed once. A linear probe trained on
        # CLIP image features would replace this matrix (and logit_scale).
        self.mood_names = list(MOODS)
        self.mood_vectors = self._embed_moods()

        print(f"ImageRecognizer ready (device: {self.device})")

    def describe(self, image):
        """
        Return (caption, description) for a PIL image: one sentence for the
        page, and a paragraph on who and what is in it, what they wear and
        the setting, which story ideas start from.

        Both come from Florence-2's most detailed mode. Its short-caption mode
        guesses titles ("the king's woman china web drama") and invents
        settings (a beach behind a couple in a city), much like BLIP did.
        """
        return _caption_and_description(
            self._florence(image, "<MORE_DETAILED_CAPTION>", max_new_tokens=160)
        )

    def match(self, image_vec, texts):
        """
        CLIP similarity of each text to an image, given its image_vector()
        (about 0.15 unrelated, 0.3 a good match).
        """
        inputs = self.clip_processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            text_vecs = _embedding(self.clip_model.get_text_features(**inputs))
        text_vecs = text_vecs / text_vecs.norm(dim=-1, keepdim=True)
        return (text_vecs @ image_vec).tolist()

    def moods(self, image, top_k=3):
        """Return the top_k moods for a PIL image as [(mood, probability), ...]."""
        image_vec = self.image_vector(image)
        with torch.no_grad():
            logits = self.clip_model.logit_scale.exp() * self.mood_vectors @ image_vec
            probs = logits.softmax(dim=0)
        top = probs.topk(top_k)
        return [
            (self.mood_names[i], round(p, 3))
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]

    def find_people(self, image):
        """
        The main characters in a PIL image, found by their faces, as
        [{"human": "a woman in a white dress", "box": [x1, y1, x2, y2],
          "face": <the face, cropped>}, ...], left to right. YuNet finds the
        faces; when it misses a close-up filling the frame, Florence-2's
        object detection looks for them instead.
        """
        faces = self._yunet_faces(image)
        people = main_characters([{"box": b} for b in faces], image.height)
        # YuNet misses close-ups filling the frame, sometimes finding just a
        # small face in the background. Several small faces are a crowd, though,
        # so Florence-2 (slower) only looks when YuNet found at most one.
        if not people and len(faces) <= 1:
            people = main_characters([{"box": b} for b in self._florence_faces(image)], image.height)
        for person in people:
            person.update(human=self._who(image, person["box"]), face=image.crop(person["box"]))
        return people

    def _yunet_faces(self, image):
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        with self.face_lock:
            self.face_detector.setInputSize(image.size)
            _, found = self.face_detector.detect(bgr)
        if found is None:
            return []
        return _clean_boxes([[x, y, x + w, y + h] for x, y, w, h in found[:, :4]], image.size)

    def _florence_faces(self, image):
        found = self._florence(image, "<OD>", DETECT_TOKENS, parse=True)
        boxes = [b for label, b in zip(found["labels"], found["bboxes"]) if label.lower() == "human face"]
        return _clean_boxes(boxes, image.size)

    def _who(self, image, box):
        """
        Who a face belongs to ("a woman in a black jacket"): Florence-2's caption
        of the face and what's just below it (hair, shoulders, clothes). The
        crop is kept narrow: a wider one took in the person next to them, and
        both got "a man and a woman standing next to each other".
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        around = image.crop(
            (max(0, x1 - w // 4), max(0, y1 - h // 2), min(image.width, x2 + w // 4), min(image.height, y2 + 2 * h))
        )
        return lower_first(self._florence(around, "<CAPTION>", max_new_tokens=20).rstrip("."))

    def _florence(self, image, task, max_new_tokens, parse=False):
        """Florence-2's output for a task: text, or with `parse`, its parsed result (boxes)."""
        inputs = self.florence_processor(text=task, images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            out = self.florence_model.generate(**inputs, max_new_tokens=max_new_tokens)
        if not parse:
            return self.florence_processor.decode(out[0], skip_special_tokens=True).strip()
        raw = self.florence_processor.batch_decode(out, skip_special_tokens=False)[0]
        return self.florence_processor.post_process_generation(
            raw, task=task, image_size=image.size
        )[task]

    def image_vector(self, image):
        """The unit CLIP embedding of a PIL image."""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_vec = _embedding(self.clip_model.get_image_features(**inputs))[0]
        return image_vec / image_vec.norm()

    def _embed_moods(self):
        vectors = []
        for synonyms in MOODS.values():
            prompts = [t.format(s) for s in synonyms for t in MOOD_TEMPLATES]
            inputs = self.clip_processor(
                text=prompts, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                text = _embedding(self.clip_model.get_text_features(**inputs))
            text = text / text.norm(dim=-1, keepdim=True)
            mean = text.mean(dim=0)
            vectors.append(mean / mean.norm())
        return torch.stack(vectors)


def _yunet_file():
    """
    YuNet's model file. Downloaded on first use to a temp file of this
    process's own, checked, then moved in place: an interrupted or bad
    download is never kept, and processes starting together don't clash.
    """
    if YUNET_FILE.exists():
        return YUNET_FILE
    YUNET_FILE.parent.mkdir(exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=YUNET_FILE.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(YUNET_URL, timeout=DOWNLOAD_SECONDS) as r:
            data = r.read()
            out.write(data)
        if hashlib.sha256(data).hexdigest() != YUNET_SHA256:
            raise RuntimeError(f"{YUNET_URL} didn't download correctly; start again to retry")
        os.replace(temp, YUNET_FILE)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return YUNET_FILE


def _clean_boxes(boxes, size):
    """
    Face boxes [x1, y1, x2, y2] in whole pixels, kept inside the picture;
    empty ones are dropped, as an empty crop breaks the face model.
    """
    width, height = size
    clamped = [
        [max(0, round(x1)), max(0, round(y1)), min(width, round(x2)), min(height, round(y2))]
        for x1, y1, x2, y2 in boxes
    ]
    return [b for b in clamped if b[2] > b[0] and b[3] > b[1]]


def main_characters(faces, image_height):
    """
    The faces big enough to be main characters, at most MAX_PEOPLE (the
    biggest), left to right. Each face has a "box" [x1, y1, x2, y2].
    """
    big = [f for f in faces if _height(f) >= MIN_FACE_HEIGHT * image_height]
    biggest = sorted(big, key=_height, reverse=True)[:MAX_PEOPLE]
    return sorted(biggest, key=lambda f: f["box"][0])


def _height(face):
    return face["box"][3] - face["box"][1]


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


if __name__ == "__main__":
    # Quick look at the pipeline: python recognizer.py images/*.jpg
    recognizer = ImageRecognizer()
    for path in sys.argv[1:]:
        image = Image.open(path).convert("RGB")
        moods = ", ".join(f"{m} {p:.0%}" for m, p in recognizer.moods(image))
        caption, description = recognizer.describe(image)
        print(f"{path}\n  caption: {caption}\n  mood:    {moods}\n  detail:  {description}")
