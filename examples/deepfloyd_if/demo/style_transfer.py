from deepfloyd_if.modules import IFStageI, IFStageII  # , StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import style_transfer
from PIL import Image
import os

RESOURCES_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'deepfloyd_if', 'resources'))


t5 = T5Embedder()
if_I = IFStageI('IF-I-M-v1.0')
if_II = IFStageII('IF-II-M-v1.0')
# if_III = StableStageIII('stable-diffusion-x4-upscaler')

raw_pil_image = Image.open(os.path.join(RESOURCES_ROOT, "sketch-mountains-input.jpeg")).convert('RGB')


####################################################################################################
# II-a. Style Transfer using stage-I cascade with not fully noising
####################################################################################################

count = 4

result = style_transfer(
    t5=t5, if_I=if_I, if_II=if_II, if_III=None,
    support_pil_img=raw_pil_image,
    style_prompt=[
        'A fantasy landscape in style lego',
        'A fantasy landscape in style zombie',
        'A fantasy landscape in style origami',
        'A fantasy landscape in style anime',
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 10.0,
        "sample_timestep_respacing": "10,10,10,10,10,10,10,10,0,0",
        'support_noise_less_qsample_steps': 5,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": 'smart50',
        "support_noise_less_qsample_steps": 5,
    },
)
if_II.show(result['II'], count, 20, filename="style_transfer-a-ii.jpg")


####################################################################################################
# II-b. Style Transfer using stage-II cascade (skip stage-I) with high guidance-scale and aug-level
####################################################################################################

count = 4

result = style_transfer(
    t5=t5, if_I=None, if_II=if_II, if_III=None,
    support_pil_img=raw_pil_image,
    style_prompt=[
        'A fantasy landscape in style lego',
        'A fantasy landscape in style zombie',
        'A fantasy landscape in style origami',
        'A fantasy landscape in style anime',
    ],
    seed=42,
    if_II_kwargs={
        "guidance_scale": 10.0,
        "sample_timestep_respacing": '100',
        "aug_level": 0.85,

    },
)
if_II.show(result['II'], count, 20, filename="style_transfer-b-ii.jpg")


####################################################################################################
# II-c. Style Transfer using double prompting for more control style
####################################################################################################

raw_pil_image = Image.open(os.path.join(RESOURCES_ROOT, "seriouscatcover.jpeg")).convert('RGB')

count = 4
prompt = 'white cat'

result = style_transfer(
    t5=t5, if_I=if_I, if_II=if_II, if_III=None,
    support_pil_img=raw_pil_image,
    prompt=[prompt]*count,
    style_prompt=[
        f'in style lego',
        f'in style zombie',
        f'in style origami',
        f'in style anime',
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 10.0,
        "sample_timestep_respacing": "10,10,10,10,10,0,0,0,0,0",
        'support_noise_less_qsample_steps': 5,
        'positive_mixer': 0.8,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": 'smart50',
        "support_noise_less_qsample_steps": 5,
        'positive_mixer': 1.0,
    },
)
if_II.show(result['II'], 2, 14, filename="style_transfer-c-ii.jpg")
