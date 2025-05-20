import os
import ssl
import urllib.request

from PIL import Image
from transformers import AutoProcessor

import mindspore as ms
import mindspore.nn as nn

from mindone.transformers import LlavaOnevisionForConditionalGeneration

MODEL_NAME = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


def main():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    with nn.no_init_parameters():
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            mindspore_dtype=ms.float16,
            attn_implementation="eager",  # TODO: does not support flash attention yet.
        )

    # prepare image and text prompt, using the appropriate prompt template
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image_path = "demo.jpg"
    if not os.path.isfile(image_path):
        ssl._create_default_https_context = ssl._create_unverified_context  # disable ssl verify
        urllib.request.urlretrieve(url, image_path)
    image = Image.open(image_path)

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
