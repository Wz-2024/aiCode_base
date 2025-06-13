from diffusers import FluxPipeline
import torch
prompt=' A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.'
prompt='Crocodile bomber bomber bomber bomber'
pipe = FluxPipeline.from_pretrained("/data_disk/dyy/models/flux.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #

image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"{prompt}.png")