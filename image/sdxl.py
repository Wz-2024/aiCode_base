from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("/data_disk/dyy/models/sdxl", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda:1")

prompt='A surreal dreamscape where the sky is split into day and night.' \
'On the left side, a bright sun shines over golden fields with people flying kites, ' \
'while on the right side, ' \
'a deep blue night sky is filled with stars and glowing jellyfish floating in the air. ' \
'In the center, a giant clock tower stands, ' \
'with its hands pointing to different times for each side. ' \
'A person wearing a half-day, ' \
'half-night cloak is walking down the path that separates the two worlds.'




image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
image.save('./sdxl-long_text.png')