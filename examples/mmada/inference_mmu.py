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
os.environ["SAFETENSORS_WEIGHTS_NAME"] = "pytorch_model.safetensors"  # vq_model
import numpy as np
from models import MAGVITv2, MMadaModelLM
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from training.prompting_utils import UniversalPrompting
from training.utils import get_config, image_transform
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
    images,
    captions,
    output_dir=".",
    font_size=20,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0, 128),
    file_list=None,
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
        if file_list is None:
            output_path = f"{output_dir}/image_with_caption_{i}.png"
        else:
            output_path = f"{output_dir}/{file_list[i]}"
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

    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability
    file_list = os.listdir(config.mmu_image_root)
    file_list = [file_name for file_name in file_list if file_name.endswith("png") or file_name.endswith("jpg")]
    responses = ["" for i in range(len(file_list))]
    images = []
    config.question = config.question.split(" *** ")
    for i, file_name in enumerate(tqdm(file_list)):
        image_path = os.path.join(config.mmu_image_root, file_name)
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution)
        image = ms.tensor(image).unsqueeze(0)
        images.append(image)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        batch_size = 1

        for question in config.question:
            input_ids = uni_prompting.text_tokenizer(
                [
                    "<|start_header_id|>user<|end_header_id|>\n"
                    + "Please describe this image in detail."
                    + "<eot_id><|start_header_id|>assistant<|end_header_id|>\n"
                ]
            )["input_ids"]
            input_ids = ms.tensor(input_ids)

            input_ids = mint.cat(
                [
                    (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|mmu|>"]).to(ms.int64),
                    (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|soi|>"]).to(ms.int64),
                    image_tokens.to(ms.int64),
                    (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|eoi|>"]).to(ms.int64),
                    (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|sot|>"]).to(ms.int64),
                    input_ids.to(ms.int64),
                ],
                dim=1,
            )
            output_ids = model.mmu_generate(input_ids, max_new_tokens=1024, steps=512, block_length=1024)
            text = uni_prompting.text_tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
            )
            print(text[0])
            responses[i] += text[0]

    images = mint.cat(images, dim=0)
    images = mint.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).asnumpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]
    output_dir = "./inference_mmu_outputs/"
    os.makedirs(output_dir, exist_ok=True)
    draw_caption_on_image(pil_images, responses, output_dir, file_list=file_list)

    print("Generated captions are saved in", output_dir)
