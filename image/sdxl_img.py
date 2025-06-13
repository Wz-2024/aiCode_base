from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForImage2Image.from_pretrained("/data_disk/dyy/models/sdxl", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda:1")

init_image = load_image("/data_disk/dyy/python_projects/bili_dif/g2.jpg").resize((512, 512))

prompt = "A beautiful young girl with glasses, delicate and expressive features, vibrant cartoon style, soft colors, Studio Ghibli-inspired, 8k"
#strength 0.3表示添加30%噪声,再修改图像
image = pipe(prompt, image=init_image, num_inference_steps=10, strength=0.6, guidance_scale=3.0).images[0]
image.save('./img2img2.png')