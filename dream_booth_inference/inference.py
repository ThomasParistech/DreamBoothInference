import glob
from typing import Union
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

DEFAULT_NUM_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 10


class StableDiffusionGenerator:

    def __init__(self, person_name: str, negative_prompt: str, weights_steps: Optional[int] = None) -> None:
        self.g_cuda = torch.Generator(device='cuda')
        self.negative_prompt = negative_prompt
        self.person_name = person_name
        model_path, self.num_steps = get_weights(person_name, num_steps=weights_steps)
        print(f"Load weights {self.num_steps} of {person_name}")
        self.person_weights_images_folder = os.path.join(IMAGES_DIR, self.person_name, str(self.num_steps))
        os.makedirs(self.person_weights_images_folder, exist_ok=True)

        self.pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                                            safety_checker=None,
                                                            torch_dtype=torch.float16).to("cuda")
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

    def render_single(self,
                      positive_prompt: str,
                      seed: int,
                      num_inference_steps: Optional[int] = None,
                      guidance_scale: Optional[int] = None):
        if num_inference_steps is None:
            num_inference_steps = DEFAULT_NUM_STEPS
        if guidance_scale is None:
            guidance_scale = DEFAULT_GUIDANCE_SCALE

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

    def render_batch(self, prompt: str, seeds: Optional[List[int]] = None, n_iter: int = 10):
        key = hash_prompt(prompt, self.negative_prompt)
        print(key)
        prompt_folder = os.path.join(self.person_weights_images_folder, key)
        os.makedirs(prompt_folder, exist_ok=True)

        if seeds is None:  # Use n_iter if seeds is not defined
            seeds = []
            rng = np.random.default_rng()
            # Don't generate seeds that have already been tested
            known_seeds = [int(name.split(".")[0]) for name in os.listdir(prompt_folder) if name.endswith(".png")]
            all_seeds = list(set(range(100000)) - set(known_seeds))
            seeds = rng.choice(all_seeds, size=n_iter, replace=False).tolist()

        self._save_prompt(prompt_folder, 'prompt', prompt)

        n_images = len(glob.glob(os.path.join(prompt_folder, "*.png")))
        n_seeds = len(seeds)
        n_digits = len(str(n_seeds))
        for k, seed in enumerate(seeds):
            print(f"----- {k+1:0>{n_digits}}/{n_seeds} - Seed {seed} - {n_images+k+1} images for prompt {key} -----")
            img = self.render_single(prompt, seed)
            img.save(os.path.join(prompt_folder, f"{seed}.png"))

    def compare_prompts_on_same_seed(self, list_prompts: List[str], seed: int):
        seed_folder = os.path.join(self.person_weights_images_folder, "seeds", str(seed))
        os.makedirs(seed_folder, exist_ok=True)

        for prompt in list_prompts:
            key = hash_prompt(prompt, self.negative_prompt)
            print(key)
            self._save_prompt(seed_folder, key, prompt)

            img = self.render_single(prompt, seed)
            img.save(os.path.join(seed_folder, f"{key}.png"))

    def fine_tune_best_seeds(self, prompt: str,
                             best_seeds: List[int],
                             list_list_guidance_scale: List[List[int]],
                             list_list_num_inference_steps: List[List[int]]):
        key = hash_prompt(prompt, self.negative_prompt)
        print(key)
        prompt_tuning_folder = os.path.join(self.person_weights_images_folder, key, "tuning")
        os.makedirs(prompt_tuning_folder, exist_ok=True)

        for list_guidance_scale in list_list_guidance_scale:
            for list_num_inference_steps in list_list_num_inference_steps:
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
