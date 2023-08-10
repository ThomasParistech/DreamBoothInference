
from dream_booth_inference.inference import StableDiffusionGenerator


person_name = "zwx"

# hand drawn sketch
# ink color, watercolors

list_prompts = [
    # " artistic modern portrait of zwx man, symmetrical capricorn horns, zodiac, goat"
    # " Medium portrait of zwx man, with a goat head with horns, front view, galaxy background, watercolor, art by boris vallejo and greg rutkowski",
    # "Medium portrait of zwx man, (with a goat head with horns:3), zodiac capricorn, galaxy background, watercolor, trending on artstation, art by boris vallejo and greg rutkowski",
    # "Medium portrait of zwx man, (with a goat head with horns:5), galaxy background, detailed, art by Android Jones",
    # "portrait of zwx man, (with large curly horns:3), zodiac signs, goat, galaxy background, art by Peter Mohrbacher"
    # "portrait of zwx man, (with large curly capricorn horns:2), zodiac signs, goat, galaxy background, art by Peter Mohrbacher"
    # "portrait of zwx man, (with large curly capricorn horns:3), ibex, galaxy background, art by Peter Mohrbacher",
    # "portrait of zwx man, (with large curly capricorn horns pointing down:3), ibex, galaxy background, art by Peter Mohrbacher",
    # "portrait of zwx man, (with large curly capricorn horns pointing down:3), capricorn, galaxy background, art by Peter Mohrbacher",
    # "portrait of zwx man, (with large curly ram wild sheep horns:3), zodiac signs, goat, galaxy background, art by Peter Mohrbacher",

    # "portrait of zwx man, (with large curly horns:3), goat, galaxy background, art by Peter Mohrbacher",
    "Medium shot monochrome, (portrait photo of zwx man:3) in 1920 at a Great Gatsby party",


    # "portrait of zwx man, (with large curly wild sheep horns:4), zodiac signs, goat, galaxy background, art by Peter Mohrbacher",
    # "portrait of zwx man, (with large curly wild sheep horns:5), zodiac signs, goat, galaxy background, art by Peter Mohrbacher",



    # "portrait of zwx man, (with large curly capricorn horns:3), zodiac signs, goat, galaxy background, art by Peter Mohrbacher"


    # "(portrait of zwx man:2), (with large curly horns pointing downwards:3), goat, galaxy background, art by Peter Mohrbacher",

  
    # '(portrait of ukj woman:10), art by Alphonse Mucha',
    # "analog style, (face portrait of a ukj woman), lady, beautiful curly hairs, sherlock holmes style, in 1920s, (street background), old photograph, photorealistic, centered, highly detailed, award winning photo, smooth, cinematic lighting, masterpiece, no color, 80mm lense, sharp focus, looking into camera, black and white",
    # "Medium shot,(portrait of zwx man:10)) pretty, highly detailed, artstation, concept art, sharp focus, art by tomasz alen kopera and justin gerard",
    # "(portrait of cmdr man:10) wearing medevial clothes, vintage character art",
    # "Medium shot, (portrait of zwx man:10), pretty face, hyper detailed,  front view, vivid colors, optimistic, fun, vintage character art, by artgerm",
    # "(portrait of zwx man:10), knight, with a sword, strong, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski and alphonse mucha",

    # "(portrait of cmdr man:10), pretty face, perfect skin, hyper detailed, front view, art by Tomer Hanuka",
    # "Medium shot, (portrait of ukj woman:10), pearlescent skin, fun, smiling, pretty face, hyper detailed, front view, art by justin gerard",
    # "Medium shot, (portrait of onex man:10), pretty face, (in an anime), volleyball court in the background,  hyper detailed",
    # # "detailed coloured pencil drawing, (portrait of wwx man), as the Little Prince, pretty face, artstation, concept art, sharp focus",
    # "(portrait of ukj woman:10), queen, beautiful long curly hairs, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), elf, beautiful long curly hairs, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), magician, beautiful long curly hairs, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), troll, beautiful long curly hairs, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), princess, beautiful long curly hairs, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), sexy zombie, beautiful long curly hairs, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), closed-mouth smile, beautiful long curly hairs, armor, sexy, pearlescent skin, gold, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), beautiful long curly hairs, closed-mouth smile, armor, sexy, pearlescent skin, gold, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), beautiful long curly hairs, armor, closed-mouth smile, sexy, pearlescent skin, gold, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), beautiful long curly hairs, armor, sexy, closed-mouth smile, pearlescent skin, gold, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), beautiful long curly hairs, armor, sexy, pearlescent skin, closed-mouth smile, gold, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), beautiful long curly hairs, armor, sexy, pearlescent skin, gold, closed-mouth smile, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), beautiful long curly hairs, armor, sexy, pearlescent skin, gold, pretty face, closed-mouth smile, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), beautiful long curly hairs, armor, sexy, pearlescent skin, gold, pretty face, elegant, closed-mouth smile, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), beautiful long curly hairs, armor, sexy, pearlescent skin, gold, pretty face, elegant, artstation, closed-mouth smile, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski",
    # "(portrait of ukj woman:10), beautiful long curly hairs, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by greg rutkowski and justin gerard",
    # "(portrait of ukj woman:10), beautiful long curly hairs, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by greg rutkowski",
    # "(portrait of ukj woman:10), beautiful long curly hairs, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by justin gerard",
    # "(portrait of ukj woman:10), beautiful long curly hairs, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley",

    # "Medium shot, (portrait of szn woman:10), with transparent glasses, smiling, fun, pretty face, hyper detailed, front view, art by Tomer Hanuka",

    # "Medium shot, front view, beautiful (portrait of zwx man:10) with red hair, by magali villeneuve and greg rutkowski and artgerm and alphonse mucha, intricate, elegant, highly detailed, photorealistic, trending on artstation, trending on cgsociety, 8 k, sharp focus",
    # "(portrait of zwx man:10) made of wood, leather clothes, shiny gold, headshot, insanely nice professional hair style, dramatic hair color, digital painting, of a old 17th century, old cyborg merchant, amber jewels, Chinese Three Kingdoms, baroque, ornate clothing, scifi, realistic, hyper detail, chiaroscuro, concept art, art by Franz Hals and Jon Foster and Ayami Kojima and Amano and Karol Bak",
    # "(portrait of cmdr man:10) made of wood, leather clothes, shiny gold, headshot, insanely nice professional hair style, dramatic hair color, digital painting, of a old 17th century, old cyborg merchant, amber jewels, Chinese Three Kingdoms, baroque, ornate clothing, scifi, realistic, hyper detail, chiaroscuro, concept art, art by Franz Hals and Jon Foster and Ayami Kojima and Amano and Karol Bak",
    # "(portrait of cmdr man:10), pretty face, clear eyes, cute, fine details, angry, complex, details, by mandy jurgens, artgerm, william mortensen, wayne barlowe, trending on artstation and greg rutkowski and zdislav beksinski",

    # "(portrait of ukj woman:10), queen, beautiful long curly hairs, sexy, pearlescent skin, pretty face, elegant, artstation, heroic fantasy, concept art, sharp focus, art by simon bisley and greg rutkowski and alphonse mucha",

]

# Tomer Hanuka
# Cyril Rolando
# greg rutkowski
# simon bisley
# william mortensen


negative_prompt = "tatoo, 3d, game, out of frame, lowres, text, error, cropped, worst quality, " \
    "low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, " \
    "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, " \
    "extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, " \
    "extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"


gen = StableDiffusionGenerator(person_name, negative_prompt,
                               #    weights_steps=1650
                               )

# gen.compare_prompts_on_same_seed(list_prompts, 7557)

gen.render_batch(list_prompts[0], n_iter=100)

# gen.render_batch(list_prompts[0], seeds=[13933, 49333, 57547])

# gen.render_batch(list_prompts[0], seeds=[2617, 19041,  84819, 92246, 56374, 79317, 50095, 76312, 27031, ])

# gen.fine_tune_best_seeds(list_prompts[0],
#                          best_seeds=[7557],
#                          list_list_guidance_scale=[[10, 12, 14]],
#                          list_list_num_inference_steps=[[29,30,31]]
#                          )


# gen.fine_tune_best_seeds(list_prompts[0],
#                          best_seeds=[44844],
#                          list_list_guidance_scale=[[10], [11]],
#                          list_list_num_inference_steps=[[30], [35], [40], [45], [50], [55],
#                                                         [60], [65], [70], [75], [80]]
#                          #  list_list_num_inference_steps=[[27], [28], [29], [30], [31], [32], [33], [34], [35]]
#                          )
