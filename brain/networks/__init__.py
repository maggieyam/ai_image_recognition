"""
The pretrained networks the faculties work with, one class per model. Each is
loaded once and shared: Florence-2 both describes the picture and says who a
face belongs to, CLIP both scores moods and checks story ideas against the
picture. Methods are named for the job (describe, find_faces), so a model can
be swapped for another class with the same methods.
"""
from pathlib import Path

# Weights downloaded or converted on first use: next to the code, git-ignored.
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
