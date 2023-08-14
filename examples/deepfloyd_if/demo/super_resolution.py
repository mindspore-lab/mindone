from deepfloyd_if.modules import IFStageI, IFStageII  # , StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import super_resolution
from PIL import Image
import os

RESOURCES_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'deepfloyd_if', 'resources'))


def show_superres(low, high, filename):
    w, h = high.size
    low = low.resize((w, h))
    result = Image.new(high.mode, (w, h * 2))
    result.paste(low, box=(0, 0))
    result.paste(high, box=(0, h))
    result.save(filename)


t5 = T5Embedder()
if_I = IFStageI('IF-I-M-v1.0')
if_II = IFStageII('IF-II-M-v1.0')
# if_III = StableStageIII('stable-diffusion-x4-upscaler')

raw_pil_image = Image.open(os.path.join(RESOURCES_ROOT, "seriouscatcover.jpeg")).convert('RGB')


middle_res = super_resolution(
    t5,
    if_III=if_II,
    prompt=['white cat, detailed picture, 4k dslr, best quality'],
    support_pil_img=raw_pil_image,
    img_scale=4.,
    img_size=64,
    if_III_kwargs={
        'sample_timestep_respacing': 'smart100',
        'aug_level': 0.5,
        'guidance_scale': 6.0,
    },
)
show_superres(raw_pil_image, middle_res['III'][0], "super_resolution-a-ii.jpg")
# high_res = super_resolution(
#     t5,
#     if_III=if_III,
#     prompt=[''],
#     support_pil_img=middle_res['III'][0],
#     img_scale=4.,
#     img_size=256,
#     if_III_kwargs={
#         "guidance_scale": 9.0,
#         "noise_level": 20,
#         "sample_timestep_respacing": "75",
#     },
# )
# show_superres(raw_pil_image, high_res['III'][0], "super_resolution-a-iii.jpg")
