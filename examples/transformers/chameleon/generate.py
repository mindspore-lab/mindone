import os
import ssl
import urllib.request

from PIL import Image
from transformers import ChameleonProcessor

import mindspore as ms
import mindspore.nn as nn

from mindone.transformers import ChameleonForConditionalGeneration

MODEL_NAME = "facebook/chameleon-7b"


def main():
    processor = ChameleonProcessor.from_pretrained(MODEL_NAME)

    with nn.no_init_parameters():
        model = ChameleonForConditionalGeneration.from_pretrained(
            MODEL_NAME, mindspore_dtype=ms.bfloat16, attn_implementation="flash_attention_2"
        )

    # prepare image and text prompt
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_path = "demo.jpg"
    if not os.path.isfile(image_path):
        ssl._create_default_https_context = ssl._create_unverified_context  # disable ssl verify
        urllib.request.urlretrieve(url, image_path)
    image = Image.open(image_path)
    prompt = "What do you see in this image?<image>"

    inputs = processor(images=image, text=prompt, return_tensors="np")
    for k, v in inputs.items():
        inputs[k] = ms.Tensor(v)

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=50)
    print(processor.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
