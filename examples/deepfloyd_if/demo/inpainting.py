from deepfloyd_if.modules import IFStageI, IFStageII  # , StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import inpainting
from PIL import Image
import os
import numpy as np

RESOURCES_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'deepfloyd_if', 'resources'))


t5 = T5Embedder()
if_I = IFStageI('IF-I-M-v1.0')
if_II = IFStageII('IF-II-M-v1.0')
# if_III = StableStageIII('stable-diffusion-x4-upscaler')

raw_pil_image = Image.open(os.path.join(RESOURCES_ROOT, "seriouscatcover.jpeg")).convert('RGB').resize((1024, 1024))
pil_image = raw_pil_image.resize(
    (64, 64), resample=getattr(Image, 'Resampling', Image).BICUBIC, reducing_gap=None
)
pil_image = np.array(pil_image)
pil_image = pil_image.astype(np.float32) / 127.5 - 1
pil_image = np.transpose(pil_image, [2, 0, 1])
inpainting_mask = np.zeros_like(pil_image)
inpainting_mask[:, 10:20, 10:20] = 1
inpainting_mask[:, 16:18, 20:26] = 1
inpainting_mask[:, 10:20, 26:38] = 1
masked_image = (1 - inpainting_mask) * pil_image
masked_image = ((np.transpose(masked_image, [1, 2, 0]) + 1) * 127.5).astype(np.uint8)
masked_image = Image.fromarray(masked_image)
masked_image.save("inpainting-masked-image.jpg")
inpainting_mask = np.expand_dims(inpainting_mask, 0)


result = inpainting(
    t5=t5, if_I=if_I,
    if_II=if_II,
    if_III=None,
    support_pil_img=raw_pil_image,
    inpainting_mask=inpainting_mask,
    prompt=[
        'blue sunglasses',
        'yellow sunglasses',
        'red sunglasses',
        'green sunglasses',
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "10,10,10,10,10,0,0,0,0,0",
        'support_noise_less_qsample_steps': 0,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        'aug_level': 0.0,
        "sample_timestep_respacing": '100',
    },
)
if_I.show(result['I'], 2, 3, filename="inpainting-a-i.jpg")
if_II.show(result['II'], 2, 6, filename="inpainting-a-ii.jpg")
# if_III.show(result['III'], 2, 14, filename="inpainting-a-iii.jpg")
