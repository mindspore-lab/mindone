import config
import os
import cv2
import einops
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import random
import datetime

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_model
from cldm.ddim_hacked import DDIMSampler

print(datetime.datetime.now())
# set enviroment variable
os.environ["SD_VERSION"] = "1.5"
# ms.set_context(mode=ms.context.PYNATIVE_MODE, pynative_synchronize=True, device_target='Ascend', device_id=6)
ms.set_context(mode=ms.context.GRAPH_MODE, device_target='Ascend', device_id=6)

apply_canny = CannyDetector()

# create model
model = create_model('/home/mindspore/congw/project/mindone/examples/stable_diffusion_v2/configs/cldm_v15.yaml')
# print(model)
model.set_train(False)
# load_model(model, './models/control_sd15_canny.pth')

sampler = DDIMSampler(model)

image_resolution = 256 # 256~768
input_image = np.random.randint(0, 255, (image_resolution, image_resolution, 3), dtype=np.uint8)

num_samples = 1 # 1~12
strength = 1 # 1~2
guess_mode = False
low_threshold =  20 # 1~255
high_threshold = 100 # 1~255
ddim_steps = 20 # 1~100
scale = 9.0 # 0.1~30
eta = 0.0
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
prompt = ''

img = resize_image(HWC3(input_image), image_resolution)
H, W, C = img.shape


detected_map = apply_canny(img, low_threshold, high_threshold)
detected_map = HWC3(detected_map)

control = ms.Tensor(detected_map.copy()).float() / 255.0
control = control.reshape((C,H,W))
control = ops.stack([control for _ in range(num_samples)], axis=0)

cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
print(f"cong's test message: cond['c_crossattn'][0].shape: {cond['c_crossattn'][0].shape}")
un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
shape = (4, H // 8, W // 8)

model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
samples, intermediates = sampler.sample(ddim_steps, num_samples,
                            shape, cond, verbose=False, eta=eta,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=un_cond)

x_samples = model.decode_first_stage(samples)
x_samples = (ops.transpose(x_samples, (0, 2, 3, 1)) * 127.5 + 127.5).asnumpy().clip(0, 255).astype(np.uint8)

results = [x_samples[i] for i in range(num_samples)]
retults = [255 - detected_map] + results

print(results)

print(type(results), len(results)) # list, 1
from PIL import Image 
img = Image.fromarray(retults[0])
img.save('/home/mindspore/congw/data/test_canny.jpg')

print(datetime.datetime.now())
