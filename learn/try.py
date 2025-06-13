import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

basemodel='/data_disk/dyy/models/FLUX.1-dev'
pipe=DiffusionPipeline.from_pretrained(basemodel,torch_dtype=torch.float16)

lora_repo='/data_disk/dyy/models/Flux-Ghibli'
trigger_word='Ghibli'
pipe.load_lora_weights(lora_repo)

device=torch.device('cuda')
pipe.to(device)



init_image = load_image("/data_disk/dyy/python_projects/bili_dif/g2.jpg").resize((1024, 1024))  # 调整为模型支持的分辨率

# 5. 设置提示词并生成图像
prompt = "A beautiful forest in Ghibli style"  # 结合你的触发词 'Ghibli'
image = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,  # 控制图像修改的强度（0.0-1.0，越大变化越多）
    num_inference_steps=30,  # 推理步数
    guidance_scale=7.5  # 提示词引导强度
).images[0]

# 6. 保存生成结果
image.save("output_image.png")