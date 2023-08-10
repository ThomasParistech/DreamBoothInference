# from typing import List
# import requests
# from PIL import Image
# from io import BytesIO
# from diffusers import StableDiffusionUpscalePipeline
# import torch

# # load model and scheduler
# model_id = "stabilityai/stable-diffusion-x4-upscaler"
# pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipeline = pipeline.to("cuda")
# pipeline.enable_attention_slicing()
# pipeline.set_use_memory_efficient_attention_xformers(True)

# # let's download an  image
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
# low_res_img = low_res_img.resize((128, 128))

# prompt = "a white cat"

# upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
# upscaled_image.save("upsampled_cat.png")


# def upscale(images: List[Image]) -> List[Image]:
#     """Upscale an image"""

import cv2
import os
old_folder = "data/best_selection"
new_folder = "data/best_selection_upscaled"

os.makedirs(new_folder, exist_ok=True)
for filename in os.listdir(old_folder):
    img = cv2.imread(os.path.join(old_folder, filename))
    img = cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(new_folder, filename), img)
