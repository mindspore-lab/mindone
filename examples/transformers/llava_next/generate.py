import os
import ssl
import urllib.request
from typing import Optional

from PIL import Image
from transformers import LlavaNextProcessor

import mindspore as ms
import mindspore.nn as nn

from mindone.transformers import LlavaNextForConditionalGeneration

MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"


def get_image(url: str, fname: Optional[str] = None) -> Image.Image:
    if fname is None:
        fname = os.path.basename(url)

    if not os.path.isfile(fname):
        ssl._create_default_https_context = ssl._create_unverified_context  # disable ssl verify
        urllib.request.urlretrieve(url, fname)
    image = Image.open(fname)
    return image


def main():
    processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)

    with nn.no_init_parameters():
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_NAME, mindspore_dtype=ms.float16, attn_implementation="flash_attention_2"
        )

    # prepare image and text prompt, using the appropriate prompt template
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = get_image(url, fname="llava_v1_5_radar.jpg")

    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown in this image?"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="np")
    for k, v in inputs.items():
        inputs[k] = ms.Tensor(v)
    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    print(processor.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
