import os
from typing import Final

IMG_SIZE: Final[int] = 512

DATA_PATH: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
WEIGHTS_DIR = os.path.join(DATA_PATH, "stable_diffusion_weights")
IMAGES_DIR = os.path.join(DATA_PATH, "stable_diffusion_inferences")

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
