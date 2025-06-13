import torch
from diffusers import FluxPipeline

model_id = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`

pipe = FluxPipeline.from_pretrained("/data_disk/dyy/models/FLUX.1-dev", 
                                    torch_dtype=torch.bfloat16).to("cuda:1")
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt =\
"In a winter forest blanketed with thick snow, " \
#寒风,雪花❌
" cold wind howls, and fine snowflakes drift through the sky. " \
#高松树✅
"The forest is filled with tall, long-needled pine trees, " \
#霜,小灌木✅
"their branches adorned with frost, while low shrubs are buried under snow, " \
#✅
"and scattered fallen logs lie covered in white. A clear stream runs through the forest, " \
#✅                               ❌
"its surface edged with thin ice, yet fish still leap from the water from time to time. " \
#✅                                                                    草根❌
"Various animals are foraging: a herd of deer digs through the snow for frozen grassroots, " \
#✅
"a bear rummages under fallen logs for food, and a pack of wolves prowls in the distance. " \
#❌
"A hunting dog stands by the stream, having just finished eating a fish, " \
#嘴角有血,,,,呼出雾气   ❌
"its lips stained with fresh red blood, its breath forming clouds of mist in the frigid air. " \
#✅
"In the distance, squirrels leap between branches, and crows circle overhead, " \
#                                                                                   浆果❌        蕨类❌
"their low caws echoing through the trees. The forest is dotted with frozen berry bushes and withered ferns, " \
#
"adding a touch of desolate yet vibrant life to the scene."
prompt='A surreal dreamscape where the sky is split into day and night.' \
'On the left side, a bright sun shines over golden fields with people flying kites, ' \
'while on the right side, ' \
'a deep blue night sky is filled with stars and glowing jellyfish floating in the air. ' \
'In the center, a giant clock tower stands, ' \
'with its hands pointing to different times for each side. ' \
'A person wearing a half-day, ' \
'half-night cloak is walking down the path that separates the two worlds.'

seed = 42
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=50, #use a larger number if you are using [dev]
    generator=torch.Generator("cuda:1").manual_seed(seed)
).images[0]
image.save("flux-dev.png")