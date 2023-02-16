import hashlib
import os
from typing import Optional

from dream_booth_inference import WEIGHTS_DIR


def get_weights(person_name: str, num_steps: Optional[int] = None) -> str:
    """aaaa"""
    person_dir = os.path.join(WEIGHTS_DIR, person_name)
    if num_steps is None:
        num_steps = max([int(f.name) for f in os.scandir(person_dir) if f.is_dir() and f.name.isnumeric()])

    person_weights = os.path.join(person_dir, str(num_steps))
    assert os.path.isdir(person_weights)
    return person_weights


def hash_prompt(positive_prompt: str, negative_prompt: str) -> str:
    """aaaa"""
    return hashlib.sha256((positive_prompt+"//"+negative_prompt).encode("utf-8")).hexdigest()
