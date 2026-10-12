"""CLIP, which puts pictures and texts in one space, so how well a text fits a picture can be scored."""
import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPModel, CLIPProcessor

from brain.networks import MODELS_DIR

CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_LOCAL_DIR = MODELS_DIR / "clip-vit-base-patch32"


class Clip:
    def __init__(self, device=None):
        """device: "cuda", "cpu", or None (auto-detect)."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(CLIP_ID)
        self.model = CLIPModel.from_pretrained(
            _weights_source(CLIP_ID, CLIPModel, CLIP_LOCAL_DIR)
        ).to(self.device)
        self.model.eval()

    def image_vector(self, image):
        """The unit CLIP embedding of a PIL image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_vec = _embedding(self.model.get_image_features(**inputs))[0]
        return image_vec / image_vec.norm()

    def text_vectors(self, texts):
        """The unit CLIP embedding of each text, one row per text."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            text_vecs = _embedding(self.model.get_text_features(**inputs))
        return text_vecs / text_vecs.norm(dim=-1, keepdim=True)

    def match(self, image_vec, texts):
        """
        CLIP similarity of each text to an image, given its image_vector()
        (about 0.15 unrelated, 0.3 a good match).
        """
        return (self.text_vectors(texts) @ image_vec).tolist()

    def probabilities(self, image_vec, label_vecs):
        """
        How likely each label is for an image, given its image_vector() and
        one unit vector per label: the labels compared with each other, as
        CLIP was trained to compare them, so they add up to 1.
        """
        with torch.no_grad():
            logits = self.model.logit_scale.exp() * label_vecs @ image_vec
            return logits.softmax(dim=0)


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
