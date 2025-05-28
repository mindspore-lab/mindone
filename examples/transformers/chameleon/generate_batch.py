import os
import ssl
import urllib.request
from typing import Optional

from PIL import Image
from transformers import ChameleonProcessor

import mindspore as ms
import mindspore.nn as nn

from mindone.transformers import ChameleonForConditionalGeneration

MODEL_NAME = "facebook/chameleon-7b"


def get_image(url: str, fname: Optional[str] = None) -> Image.Image:
    if fname is None:
        fname = os.path.basename(url)

    if not os.path.isfile(fname):
        ssl._create_default_https_context = ssl._create_unverified_context  # disable ssl verify
        urllib.request.urlretrieve(url, fname)
    image = Image.open(fname)
    return image


def main():
    processor = ChameleonProcessor.from_pretrained(MODEL_NAME, padding_side="left")

    with nn.no_init_parameters():
        model = ChameleonForConditionalGeneration.from_pretrained(
            MODEL_NAME, mindspore_dtype=ms.bfloat16, attn_implementation="flash_attention_2"
        )

    # Get three different images
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image_stop = get_image(url)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_cats = get_image(url)

    url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
    image_snowman = get_image(url)

    # Prepare a batched prompt, where the first one is a multi-image prompt and the second is not
    prompts = ["What do these images have in common?<image><image>", "<image>What is shown in this image?"]

    # We can simply feed images in the order they have to be used in the text prompt
    # Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
    inputs = processor(images=[image_stop, image_cats, image_snowman], text=prompts, padding=True, return_tensors="np")
    for k, v in inputs.items():
        inputs[k] = ms.Tensor(v)

    # autoregressively complete prompt
    generate_ids = model.generate(**inputs, max_new_tokens=50)
    print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))


if __name__ == "__main__":
    main()
