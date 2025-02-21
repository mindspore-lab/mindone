# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import base64
import io
import json
from typing import Dict, List

import PIL.Image

import mindspore as ms

"""
from janus.models import MultiModalityCausalLM, VLChatProcessor


def load_pretrained_model(model_path: str):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    return tokenizer, vl_chat_processor, vl_gpt
"""


def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """

    Support file path or base64 images.

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_data in message["images"]:
            if image_data.startswith("data:image"):
                # Image data is in base64 format
                _, image_data = image_data.split(",", 1)
                image_bytes = base64.b64decode(image_data)
                pil_img = PIL.Image.open(io.BytesIO(image_bytes))
            else:
                # Image data is a file path
                pil_img = PIL.Image.open(image_data)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images


def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
        return data


def set_model_param_dtype(model, dtype=ms.bfloat16, keep_norm_fp32=False):
    if model is not None:
        assert isinstance(model, ms.nn.Cell)

        k_num, c_num = 0, 0
        for _, p in model.parameters_and_names():
            # filter norm/embedding position_ids param
            if keep_norm_fp32 and ("norm" in p.name):
                # print(f"param {p.name} keep {p.dtype}") # disable print
                k_num += 1
            elif "position_ids" in p.name:
                k_num += 1
            else:
                c_num += 1
                p.set_dtype(dtype)

        print(f"Convert '{type(model).__name__}' param to {dtype}, keep/modify num {k_num}/{c_num}.")

    return model
