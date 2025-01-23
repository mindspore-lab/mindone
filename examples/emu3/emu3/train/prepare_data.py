# -*- coding: utf-8 -*-

import argparse
import json
import os

from PIL import Image
import mindspore as ms
from mindspore import _no_grad, jit_class

from emu3.tokenizer import Emu3VisionVQModel, Emu3VisionVQImageProcessor

@jit_class
class no_grad(_no_grad):
    """
    A context manager that suppresses gradient memory allocation in PyNative mode.
    """

    def __init__(self):
        super().__init__()
        self._pynative = ms.get_context("mode") == ms.PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='vision tokenizer path')
    parser.add_argument('--data-path', type=str, help='data path')
    parser.add_argument('--output-path', type=str, help='tokenized data save path')
    parser.add_argument('--image-area', type=int, default=720 * 720)

    args = parser.parse_args()
    return args


def smart_resize(image, image_area: int = 720 * 720):
    w, h = image.size
    current_area = h * w
    target_ratio = (image_area / current_area) ** 0.5

    th = int(round(h * target_ratio))
    tw = int(round(w * target_ratio))

    image = image.resize((tw, th))
    return image


def main():
    args = prepare_args()

    image_processor = Emu3VisionVQImageProcessor.from_pretrained(args.model_path)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(args.model_path)
    image_tokenizer.set_train(False)

    os.makedirs(f"{args.output_path}/feature", exist_ok=True)
    os.makedirs(f"{args.output_path}/list", exist_ok=True)

    datalist = {
        "prefix": f"{args.output_path}/feature",
        "path_list": []
    }

    with open(args.data_path) as f:
        input_data = json.load(f)

    for inp in input_data:
        name = inp["name"]
        prompt = inp["text"]

        image = Image.open(inp["image"]).convert("RGB")
        image = smart_resize(image, args.image_area)

        image = image_processor(image, return_tensors="np")["pixel_values"]
        with no_grad():
            image = ms.Tensor(image)
            token_ids = image_tokenizer.encode(image)

        token_ids = token_ids.squeeze(0).asnumpy()
        data = {
            "name": name,
            "images": token_ids,
            "texts": prompt
        }

        ms.save_checkpoint(data, f"{args.output_path}/feature/{name}.ckpt")
        datalist["path_list"].append(f"{name}.ckpt")

    with open(f"{args.output_path}/list/train.json", 'w') as f:
        json.dump(datalist, f)


if __name__ == "__main__":
    main()
