import os
import ssl
import urllib.request
from typing import Optional

from PIL import Image
from transformers import AutoProcessor

import mindspore as ms
import mindspore.nn as nn

from mindone.transformers import LlavaOnevisionForConditionalGeneration

MODEL_NAME = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


def get_image(url: str, fname: Optional[str] = None) -> Image.Image:
    if fname is None:
        fname = os.path.basename(url)

    if not os.path.isfile(fname):
        ssl._create_default_https_context = ssl._create_unverified_context  # disable ssl verify
        urllib.request.urlretrieve(url, fname)
    image = Image.open(fname)
    return image


def main():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    with nn.no_init_parameters():
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            mindspore_dtype=ms.float16,
            attn_implementation="eager",  # TODO: does not support flash attention yet.
        )

    # prepare image and text prompt, using the appropriate prompt template
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = get_image(url)

    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are these?"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="np")
    for k, v in inputs.items():
        inputs[k] = ms.Tensor(v)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    print(processor.decode(output[0][2:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
