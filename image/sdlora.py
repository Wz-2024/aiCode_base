from diffusers import StableDiffusionXLPipeline
from diffusers import EulerAncestralDiscreteScheduler
import torch
from diffusers.utils import load_image

pipe=StableDiffusionXLPipeline.from_pretrained(
    "/data_disk/dyy/models/sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda:1")
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Load the LoRA
pipe.load_lora_weights('/data_disk/dyy/models/sdlora', 
                       weight_name='Studio Ghibli style.safetensors', 
                       adapter_name="Studio Ghibli style")

# Activate the LoRA
pipe.set_adapters(["Studio Ghibli style"], adapter_weights=[2.0])
init_img=load_image("/data_disk/dyy/python_projects/bili_dif/z_output_img/lxy.png")
prompt = "medieval rich kingpin sitting in a tavern, Studio Ghibli style"
prompt=r"Convert this image into a Studio Ghibli-style illustration. A young person in a black puffy jacket and orange sweater feeds seagulls by a waterfront promenade. The seagulls, with white-gray feathers and red beaks, fly dynamically. The background has a calm lake, floating platforms with yellow stars, distant mountains, and a blue sky with soft clouds. The promenade features white railings and a red-blue path. Use vibrant colors, delicate lines, and a whimsical, nostalgic vibe."
# propmt=r"A young person in a black puffy jacket and orange sweater feeds seagulls by a waterfront promenade. The seagulls, with white-gray feathers and red beaks, fly dynamically. The background has a calm lake, floating platforms with yellow stars, distant mountains, and a blue sky with soft clouds."
# prompt = "medieval rich kingpin sitting in a tavern, Studio Ghibli style"

negative_prompt = "nsfw"
width = 512
height = 512
num_inference_steps = 1
guidance_scale = 0
images = pipe(prompt, 
              negative_prompt=negative_prompt, 
              width=width, 
              image=init_img,
              height=height, 
              guidance_scale=guidance_scale, 
              num_inference_steps=num_inference_steps).images[0]
images.save('result.png')
