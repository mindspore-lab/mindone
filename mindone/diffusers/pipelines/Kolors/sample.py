import os

import mindspore as ms

from mindone.diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionXLKolorsPipeline,
    UNet2DConditionModel,
)
from mindone.diffusers.pipelines.Kolors.modeling_chatglm import ChatGLMModel
from mindone.diffusers.pipelines.Kolors.tokenization_chatglm import ChatGLMTokenizer

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt_dir = f"{root_dir}/weights/Kolors"

text_encoder = ChatGLMModel.from_pretrained(f"{ckpt_dir}/text_encoder", mindspore_dtype=ms.float16).half()
tokenizer = ChatGLMTokenizer.from_pretrained(f"{ckpt_dir}/text_encoder")
vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()

pipe = StableDiffusionXLKolorsPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    force_zeros_for_empty_prompt=False,
)

prompt = "一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着“mindone”"
image = pipe(
    prompt=prompt, height=1024, width=1024, num_inference_steps=50, guidance_scale=5.0, num_images_per_prompt=1
)[0][0]

image.save(f"{root_dir}/outputs/sample_test.jpg")
