
from dream_booth_inference.inference import StableDiffusionGenerator

person_name = "onex"

list_prompts = [
    "analog style, (portrait of onex man), combover haircut, stubble beard, looking at camera, street background, in the twenties, in the 1920s, old photograph, photorealistic, centered, highly detailed, award winning photo, cinematic lighting, no color, black and white",
]


negative_prompt = "piercing,tatoo,glasses,3d, game, out of frame, lowres, text, error, cropped, worst quality, " \
    "low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, " \
    "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, " \
    "extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, " \
    "extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"


gen = StableDiffusionGenerator(person_name, negative_prompt)

# gen.compare_prompts_on_same_seed(list_prompts, 36330)  # 45856

gen.render_batch(list_prompts, n_iter=10)

# gen.render_batch(list_prompts, seeds=list(range(1000, 1400)))

# gen.render_batch(list_prompts, seeds=[5037, 3171, 28684, 10316, 14444,
#                  26622, 29409, 32949, 51685, 53518, 80457, 84514, 88213])

# gen.fine_tune_best_seeds(list_prompts[0],
#                          best_seeds=[476],
#                          list_guidance_scale=[10],
#                          list_num_inference_steps=[50, 60, 70, 80, 90, 100, 110])
