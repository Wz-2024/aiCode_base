from diffusers import AutoPipelineForImage2Image
import torch
from PIL import Image

# device='cuda:1' if torch.cuda.is_available() else 'cpu'
# print(device)
#加载基础模型
base_model='/data_disk/dyy/models/FLUX.1-dev'
pipe=AutoPipelineForImage2Image.from_pretrained(base_model,torch_dtype=torch.float16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power


#加载lora权重
lora_repo='/data_disk/dyy/models/Flux_ghibli_new'
trigger_word="Ghibli Art"
pipe.load_lora_weights(lora_repo)   

#设置设备
# pipe.to(device)

print('Pipeline loaded!!!')
#设置管道的参数
init_image_path='/data_disk/dyy/python_projects/bili_dif/z_output_img/lxy.png'
init_image=Image.open(init_image_path).convert('RGB')
width, height = init_image.size
#调用管道
prompt=f"{trigger_word},a young boy in a black puffer jacket, orange sweater and sunglasses feeds seagulls by a waterfront. Several birds hover around, eagerly reaching for food. The scene features a calm sea, distant mountains, and a clear blue sky, with a paved walkway and modern streetlights in the background"
# prompt='Ghibli Art, a young man in a dark suit with a white shirt and a red-blue striped tie, standing in a cozy room with a wooden staircase and a wall full of framed photos, soft sunlight streaming in, warm and gentle colors, whimsical atmosphere, detailed hand-drawn style, serene and nostalgic mood'
# prompt='Ghibli style, a handsome yong boy'
# prompt='Ghibli style, A pretty girl is taking a selfie in a makeup mirror'
# prompt=r"Transform this image into a Studio Ghibli-style illustration with a whimsical, hand-painted aesthetic. The scene depicts a young person standing on a scenic waterfront promenade by a serene lake or sea, surrounded by a flock of seagulls in mid-flight. The person is wearing a black puffy jacket over an orange sweater, extending their hand to feed the seagulls, with a joyful expression. The seagulls have white and gray feathers with black-tipped wings and red beaks, captured in dynamic poses as they soar and flutter around. The background features a calm body of water with gentle ripples, small floating platforms with yellow star-shaped decorations, and distant mountains under a bright blue sky with soft, fluffy clouds. The promenade has a modern design with white railings, a red and blue pathway, and decorative street lamps with a subtle Asian influence. Add Ghibli-style details like vibrant yet soft colors, delicate linework, and a magical, nostalgic atmosphere, with a touch of wind-swept motion in the seagulls' feathers and the person's clothing. Emphasize the harmony between the person, nature, and the serene landscape, evoking a sense of wonder and connection."
# prompt=r'Turn this into a Studio Ghibli-style illustration: a young person in a black puffy jacket and orange sweater feeds seagulls on a waterfront promenade by a serene lake. The seagulls, with white-gray feathers and red beaks, fly dynamically. The background has calm water, floating platforms with yellow stars, distant mountains, and a blue sky with clouds. The promenade features white railings, a red-blue pathway, and Asian-style street lamps, with a whimsical, magical atmosphere.'
pipe=pipe.to("cuda:1")
sgrength=0.70
time_inference=30
guidance_scale=8.0
#展示图像
img=pipe(prompt,image=init_image,
        strength=sgrength,
        num_inference_steps=time_inference,
        guidance_scale=8.0,
        width=width,
        height=height,  
        generator=torch.Generator("cuda:1").manual_seed(318)

        ).images[0]

#保存图像
img.save(f'{trigger_word}lxy3.png')