# coding=utf-8
# Copyright 2025 MMaDA Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
from models import MAGVITv2, MMadaModelLM, get_mask_schedule
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from training.prompting_utils import UniversalPrompting
from training.utils import get_config
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import mint


def resize_vocab(model, config):
    print(f"Resizing token embeddings to {config.new_vocab_size}")
    model.resize_token_embeddings(config.new_vocab_size)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def draw_caption_on_image(
    images, captions, output_dir=".", font_size=20, text_color=(255, 255, 255), bg_color=(0, 0, 0, 128)
):
    try:
        font = ImageFont.load_default(font_size)
    except Exception:
        font = ImageFont.load_default()
    font_size = 16
    output_paths = []
    for i, (image, caption) in enumerate(zip(images, captions)):
        img = image.copy()
        draw = ImageDraw.Draw(img, "RGBA")

        text_height = font_size
        # text_width = draw.textlength(caption, font=font)
        margin = 10
        img_width, img_height = img.size
        bg_position = (0, img_height - text_height - 2 * margin, img_width, img_height)
        text_position = (margin, img_height - text_height - margin)

        draw.rectangle(bg_position, fill=bg_color)

        draw.text(text_position, caption, fill=text_color, font=font)

        output_path = f"{output_dir}/image_with_caption_{i}.png"
        img.save(output_path)
        output_paths.append(output_path)

    return output_paths


if __name__ == "__main__":
    config = get_config()

    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.set_train(False)
    for param in vq_model.get_parameters():
        param.requires_grad = False
    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, mindspore_dtype=ms.bfloat16)

    mask_token_id = model.config.mask_token_id
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = config.batch_size

    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps

    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()
    output_images, output_responses = [], []
    for step in tqdm(range(0, len(validation_prompts), config.training.batch_size)):
        prompts = validation_prompts[step : step + config.training.batch_size]

        image_tokens = mint.ones((len(prompts), config.model.mmada.num_vq_tokens), dtype=ms.int64) * mask_token_id
        input_ids, attention_mask = uni_prompting((prompts, image_tokens), "t2i_gen")
        if config.training.guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = uni_prompting(([""] * len(prompts), image_tokens), "t2i_gen")
        else:
            uncond_input_ids = None
            uncond_attention_mask = None

        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_schedule(schedule, **args)
        else:
            mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

        gen_token_ids = model.t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            uncond_attention_mask=uncond_attention_mask,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            seq_len=config.model.mmada.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )

        gen_token_ids = mint.clamp(gen_token_ids, max=config.model.mmada.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)
        output_images.append(images)
        output_responses.extend(prompts)

    images = mint.cat(output_images, dim=0)
    images = mint.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).asnumpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]
    output_dir = "./inference_t2i_outputs/"
    os.makedirs(output_dir, exist_ok=True)
    draw_caption_on_image(pil_images, output_responses, output_dir)

    print("Generated images are saved in ", output_dir)
