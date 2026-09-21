from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import BlipConfig, BlipForConditionalGeneration, BlipProcessor

BLIP_ID = "Salesforce/blip-image-captioning-base"
BLIP_LOCAL_DIR = Path(__file__).parent / "models" / "blip-base"


def _torch_can_load_bin():
    major, minor = torch.__version__.split(".")[:2]
    return (int(major), int(minor)) >= (2, 6)


def _blip_weights_source():
    """
    The Hub id for the BLIP weights, or a local safetensors copy on older torch.

    transformers refuses to torch.load the Hub's pytorch_model.bin below
    torch 2.6 (CVE-2025-32434), so older installs convert it once locally.
    """
    if _torch_can_load_bin():
        return BLIP_ID
    if not (BLIP_LOCAL_DIR / "model.safetensors").exists():
        bin_path = hf_hub_download(BLIP_ID, "pytorch_model.bin")
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        model = BlipForConditionalGeneration(BlipConfig.from_pretrained(BLIP_ID))
        model.load_state_dict(state, strict=False)
        model.save_pretrained(BLIP_LOCAL_DIR)  # writes model.safetensors
    return BLIP_LOCAL_DIR


class ImageRecognizer:
    def __init__(self, device=None):
        """device: "cuda", "cpu", or None (auto-detect)."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.blip_processor = BlipProcessor.from_pretrained(BLIP_ID)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            _blip_weights_source()
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
