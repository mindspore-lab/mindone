from deepfloyd_if.modules import IFStageI, IFStageII  # , StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream


t5 = T5Embedder()
if_I = IFStageI('IF-I-M-v1.0')
if_II = IFStageII('IF-II-M-v1.0')
# if_III = StableStageIII('stable-diffusion-x4-upscaler')


####################################################################################################
# I-a. Core method. Dream with prompt using all three cascades: stage-I, stage-II, stage-III:
# 64-->256-->1024px
####################################################################################################

prompt = 'ultra close-up color photo portrait of rainbow owl with deer horns in the woods'
count = 4

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=None,
    prompt=[prompt]*count,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
    if_III_kwargs={
        "guidance_scale": 9.0,
        "noise_level": 20,
        "sample_timestep_respacing": "75",
    },
)

if_I.show(result['I'], size=3, filename="dream-a0-i.jpg")
if_II.show(result['II'], size=6, filename="dream-a0-ii.jpg")
# if_III.show(result['III'], size=14, filename="dream-a0-iii.jpg")

prompt = "a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says 'very deep learning'"
count = 4

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=None,
    prompt=[prompt]*count,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
)
if_I.show(result['I'], size=3, filename="dream-a1-i.jpg")
if_II.show(result['II'], size=6, filename="dream-a1-ii.jpg")
# if_III.show(result['III'], size=14, filename="dream-a1-iii.jpg")


####################################################################################################
# I-b. Dream with control of style:
# "style_prompt" - text
# "positive_mixer" - [0, 1.0], recommended 0.4 or 0.25..0.5
####################################################################################################

prompt = 'ultra close-up color photo portrait of rainbow owl with deer horns in the woods'
style_prompt = 'in style lego'
# style_prompt = 'in style cubism'
count = 4

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=None,
    prompt=[prompt]*count,
    style_prompt=[style_prompt]*count,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
        "positive_mixer": 0.4,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
        "positive_mixer": 1.0,
    },
)

if_II.show(result['II'], size=6, filename="dream-b-ii.jpg")
# if_III.show(result['III'], size=14, filename="dream-b-iii.jpg")


####################################################################################################
# I-c. Dream with control style and adding style in main prompt
# "style_prompt" - text
# "positive_mixer" - [0, 1.0], recommended 0.25 or 0.25..0.5
####################################################################################################

prompt = 'ultra close-up color photo portrait of rainbow owl with deer horns in the woods'
style_prompt = 'in style lego'
# style_prompt = 'in style cubism'
count = 4

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=None,
    prompt=[f'{style_prompt}, {prompt}']*count,
    style_prompt=[style_prompt]*count,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
        "positive_mixer": 0.4,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
        "positive_mixer": 1.0,
    },
)
if_II.show(result['II'], size=6, filename="dream-c-ii.jpg")
# if_III.show(result['III'], size=14, filename="dream-c-iii.jpg")


####################################################################################################
# I-d. Dream with different prompts in one batch
####################################################################################################

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=None,
    prompt=[
        "A vibrant professional close-up photo portrait of a brutal dog in a helmet driving the harley davidson motorcycle in the street of the new york city, bokeh, 35mm, golden hour, in style of William Eggleston, fujifilm",
        'colour fashion photo of a Teddy Bear with t-shirt with text "Deep Floyd"',
        "a little green budgie parrot driving small red toy car in new york street, photo",
        "A black background photo of rainbow owl with deer horns, full hd, 4k, detailed picture",
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
)
if_I.show(result['I'], size=3, filename="dream-d0-i.jpg")
if_II.show(result['II'], size=6, filename="dream-d0-ii.jpg")
# if_III.show(result['III'], size=14, filename="dream-d0-iii.jpg")

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=None,
    prompt=[
        "A vibrant professional close-up photo portrait of a brutal dog in a helmet driving the harley davidson motorcycle in the street of the new york city, bokeh, 35mm, golden hour, in style of William Eggleston, fujifilm",
        'colour fashion photo of a Teddy Bear with t-shirt with text "Deep Floyd"',
        "a little green budgie parrot driving small red toy car in new york street, photo",
        "A black background photo of rainbow owl with deer horns, full hd, 4k, detailed picture",
    ],
    style_prompt=[
        "in style cubism",
        "in style lego",
        "in style minecraft",
        "in style origami",
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
        "positive_mixer": 0.4,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
        "positive_mixer": 1.0,
    },
)
if_I.show(result['I'], size=3, filename="dream-d1-i.jpg")
if_II.show(result['II'], size=6, filename="dream-d1-ii.jpg")
# if_III.show(result['III'], size=14, filename="dream-d1-iii.jpg")


####################################################################################################
# I-e. Dream with negative prompts - help model, exclude unnecessary concept
####################################################################################################

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=None,
    prompt=[
        "A vibrant professional close-up photo portrait of a brutal dog in a helmet driving the harley davidson motorcycle in the street of the new york city, bokeh, 35mm, golden hour, in style of William Eggleston, fujifilm",
        'colour fashion photo of a Teddy Bear with t-shirt with text "Deep Floyd"',
        "a little green budgie parrot driving small red toy car in new york street, photo",
        "A black background photo of rainbow owl with deer horns, full hd, 4k, detailed picture",
    ],
    style_prompt=[
        "in style cubism",
        "in style lego",
        "in style minecraft",
        "in style origami",
    ],
    negative_prompt=[
        "yellow",
        "yellow",
        "yellow",
        "yellow",
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
        "positive_mixer": 0.4,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
        "positive_mixer": 1.0,
    },
)
if_II.show(result['II'], size=6, filename="dream-e-ii.jpg")
# if_III.show(result['III'], size=14, filename="dream-e-iii.jpg")


####################################################################################################
# I-f. Dream with aspect ratio
####################################################################################################

prompt = 'ultra close-up color photo portrait of rainbow owl with deer horns in the woods'
count = 2

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=None,
    prompt=[prompt]*count,
    negative_prompt=['green']*count,
    seed=42,
    aspect_ratio='16:9',
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
)
if_II.show(result['II'], size=6, filename="dream-f-ii.jpg")
# if_III.show(result['III'], size=14, filename="dream-f-iii.jpg")
