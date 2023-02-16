from PIL import Image
import os
from random import randrange
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline
from torch import autocast

from dream_booth_inference import IMAGES_DIR
from dream_booth_inference import IMG_SIZE
from dream_booth_inference.utils import get_weights
from dream_booth_inference.utils import hash_prompt

import numpy as np


class StableDiffusionGenerator:

    def __init__(self, person_name: str, negative_prompt: str) -> None:
        self.g_cuda = torch.Generator(device='cuda')
        self.negative_prompt = negative_prompt
        self.person_name = person_name
        model_path = get_weights(person_name)
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                                            safety_checker=None,
                                                            torch_dtype=torch.float16).to("cuda")
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

    def render_single(self,
                      positive_prompt: str,
                      seed: int,
                      num_inference_steps: int = 50,
                      guidance_scale: int = 10):
        self.g_cuda.manual_seed(seed)  # Set seed
        with autocast("cuda"), torch.inference_mode():
            images = self.pipe(
                positive_prompt,
                height=IMG_SIZE,
                width=IMG_SIZE,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=1,  # Can only handle a single image on my GPU
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=self.g_cuda
            ).images
        return images[0]

    def render_batch(self, list_prompts: List[str], seeds: Optional[List[int]] = None, n_iter: int = 10):
        person_folder = os.path.join(IMAGES_DIR, self.person_name)
        os.makedirs(person_folder, exist_ok=True)

        if seeds is None:  # Use n_iter if seeds is not defined
            seeds = []
            rng = np.random.default_rng()
            seeds = rng.choice(100000, size=n_iter, replace=False).tolist()

        for prompt in list_prompts:
            key = hash_prompt(prompt, self.negative_prompt)
            print(key)
            prompt_folder = os.path.join(person_folder, key)
            os.makedirs(prompt_folder, exist_ok=True)

            self._save_prompt(prompt_folder, 'prompt', prompt)

            for seed in seeds:
                img = self.render_single(prompt, seed)
                img.save(os.path.join(prompt_folder, f"{seed}.png"))

    def compare_prompts_on_same_seed(self, list_prompts: List[str], seed: int):
        seed_folder = os.path.join(IMAGES_DIR, self.person_name, "seeds", str(seed))
        os.makedirs(seed_folder, exist_ok=True)

        for prompt in list_prompts:
            key = hash_prompt(prompt, self.negative_prompt)

            self._save_prompt(seed_folder, key, prompt)

            img = self.render_single(prompt, seed)
            img.save(os.path.join(seed_folder, f"{key}.png"))

    def fine_tune_best_seeds(self, prompt: str,
                             best_seeds: List[int],
                             list_guidance_scale: List[int],
                             list_num_inference_steps: List[int]):
        person_folder = os.path.join(IMAGES_DIR, self.person_name)
        key = hash_prompt(prompt, self.negative_prompt)
        print(key)
        prompt_tuning_folder = os.path.join(person_folder, key, "tuning")
        os.makedirs(prompt_tuning_folder, exist_ok=True)

        config = "gscale_" + "_".join(map(str, list_guidance_scale))
        config += "__nsteps_" + "_".join(map(str, list_num_inference_steps))

        for seed in best_seeds:
            dst = Image.new('RGB', (IMG_SIZE * len(list_num_inference_steps),
                                    IMG_SIZE * len(list_guidance_scale)))
            for x, num_inf_steps in enumerate(list_num_inference_steps):
                for y, guidance_scale in enumerate(list_guidance_scale):
                    img = self.render_single(prompt, seed, num_inf_steps, guidance_scale)
                    dst.paste(img, (x*IMG_SIZE, y*IMG_SIZE))
            dst.save(os.path.join(prompt_tuning_folder, f"{seed}__{config}.png"))

    def _save_prompt(self, output_folder: str, basename: str, positive_prompt: str):
        with open(os.path.join(output_folder, basename+'.txt'), 'w') as f:
            f.write(f"Positive prompt:\n{positive_prompt}\n\nNegative prompt:\n{self.negative_prompt})")
