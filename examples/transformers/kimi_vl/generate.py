import os
import ssl
import urllib.request

from kimi_vl import KimiVLConfig, KimiVLForConditionalGeneration
from PIL import Image
from transformers import AutoProcessor

import mindspore as ms
import mindspore.nn as nn

DEBUG = False  # for debugging only
MODEL_PATH = "moonshotai/Kimi-VL-A3B-Instruct"
# MODEL_PATH = "moonshotai/Kimi-VL-A3B-Thinking"


def main():
    if DEBUG:
        ms.runtime.launch_blocking()
        config = KimiVLConfig.from_pretrained(MODEL_PATH, attn_implementation="flash_attention_2")
        config.text_config.num_hidden_layers = 2  # one for FFN, one for MOE
        config.vision_config.num_hidden_layers = 1
        model = KimiVLForConditionalGeneration._from_config(config, torch_dtype=ms.bfloat16)
    else:
        with nn.no_init_parameters():
            model = KimiVLForConditionalGeneration.from_pretrained(
                MODEL_PATH, mindspore_dtype=ms.bfloat16, attn_implementation="flash_attention_2"
            )  # "eager" / "flash_attention_2"

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    image_path = "demo.png"
    if not os.path.isfile(image_path):
        ssl._create_default_https_context = ssl._create_unverified_context  # disable ssl verify
        urllib.request.urlretrieve(
            "https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/resolve/main/figures/demo.png", image_path
        )

    image = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "What is the dome building in the picture? Think step by step."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="np")
    inputs = processor(images=image, text=text, return_tensors="np", padding=True, truncation=True)
    for k, v in inputs.items():
        inputs[k] = ms.Tensor(v)
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=1.0)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(response[0])


if __name__ == "__main__":
    main()
