import torch
from diffusers import DiffusionPipeline
from diffusers import AutoPipelineForImage2Image
from PIL import Image

# 加载基础模型
base_model = "/data_disk/dyy/models/flux.1-schnell"
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

# 加载 LoRA 权重
lora_repo = "/data_disk/dyy/models/Flux-Ghibli"
trigger_word = "Ghibli Art"
pipe.load_lora_weights(lora_repo)

# 设置设备
device = torch.device("cuda:1")
pipe.to(device)

# 加载初始图像 (假设你有一个图像文件路径)
init_image_path = "./test.png"  # 替换为你的初始图像路径
init_image = Image.open(init_image_path).convert("RGB")

# 设置 img2img 参数并生成图像
prompt = f"{trigger_word}, a beautiful girl with glasses "  # 你想要生成的提示词
strength = 0.75  # strength 参数控制初始图像保留程度 (0.0-1.0)，值越小越接近初始图像
num_inference_steps = 50  # 推理步数

# 调用 img2img 管道
output_image = pipe(
    prompt=prompt,
    # init_image=init_image,
    # strength=strength,
    num_inference_steps=num_inference_steps,
    guidance_scale=7.5  # 控制提示词的引导强度
).images[0]

# 保存生成的图像
output_image.save("girl.png")