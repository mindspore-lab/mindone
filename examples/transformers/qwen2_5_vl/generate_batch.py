import os
import ssl
import urllib.request
from typing import Optional

from PIL import Image
from transformers import AutoProcessor

import mindspore as ms
import mindspore.nn as nn

from mindone.transformers import Qwen2_5_VLForConditionalGeneration
from mindone.transformers.models.qwen2_vl.qwen_vl_utils import process_vision_info

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def get_image(url: str, fname: Optional[str] = None) -> Image.Image:
    if fname is None:
        fname = os.path.basename(url)

    if not os.path.isfile(fname):
        ssl._create_default_https_context = ssl._create_unverified_context  # disable ssl verify
        urllib.request.urlretrieve(url, fname)
    image = Image.open(fname)
    return image


def main():
    with nn.no_init_parameters():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, mindspore_dtype=ms.bfloat16, attn_implementation="flash_attention_2"
        )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, padding_side="left")

    get_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "demo.jpeg",
    )
    messages1 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    messages2 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "demo.jpeg",
                },
                {"type": "text", "text": "Is this a AI generated image?"},
            ],
        }
    ]
    # Combine messages for batch processing
    messages = [messages1, messages2]

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="np",
    )
    for k, v in inputs.items():
        inputs[k] = ms.Tensor(v)

    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_texts)


if __name__ == "__main__":
    main()
