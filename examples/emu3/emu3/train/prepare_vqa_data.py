"""
This script is to prepare image generation dataset.

Input example data json format: e.g. llava-OneVision-Data llava_onevision_mathv360k_train.json
[
    {
        "id": "data1"，
        "image": "image_1.jpg"
        "conversations": [
            {
                "from": "human",
                "value": "<image>\ntext prompt and instruction #1."
            },
            {
                "from": "gpt",
                "value": "response #1."
            }
        ]
    },
    {
        "id": "data2"，
        "image": "image_2.jpg"
        "conversations": [
            {
                "from": "human",
                "value": "<image>\ntext prompt and instruction #2."
            },
            {
                "from": "gpt",
                "value": "response #2."
            }
        ]
    },
...
]

Output dataset
- output_path/list/train.json:
    {
        "prefix": output_path/feature,
        "path_list": ["data1.ckpt", "data2.ckpt", ...]
    }
- /output_path/feature/data1.ckpt:
    {"name": name, "images": token_ids, "texts": prompt, "response": answer prompt}
- /output_path/feature/data2.ckpt:
    {"name": name, "images": token_ids, "texts": prompt, "response": answer prompt}
...

Usage:
cd examples/emu3
python emu3/train/prepare_data.py --model-path DIR-TO-Emu3-VisionTokenizer --data-path DIR-TO-DATA.json --output-path DATA-DIR
"""

import argparse
import json
import os

from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import _no_grad, jit_class


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
    parser.add_argument("--model-path", type=str, help="vision tokenizer path")
    parser.add_argument("--data-path", type=str, help="data path")
    parser.add_argument("--output-path", type=str, help="tokenized data save path")
    parser.add_argument("--image-area", type=int, default=480 * 320)
    parser.add_argument("--split", type=str, default="train", help="split to train to test set")

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

    datalist = {"prefix": os.path.join(args.output_path, "feature"), "path_list": []}

    with open(args.data_path) as f:
        input_data = json.load(f)
    base_dir = os.path.dirname(args.data_path)
    cnt = 0
    # input_data = input_data[0]
    for inp in tqdm(input_data):
        name = inp["id"]
        prompts = inp["conversations"]
        img_dir = os.path.join(base_dir, "images", inp["image"])
        image = Image.open(img_dir).convert("RGB")
        image = smart_resize(image, args.image_area)

        image = image_processor(image, return_tensors="np")["pixel_values"]
        with no_grad():
            image = ms.Tensor(image, dtype=image_tokenizer.dtype)
            token_ids = image_tokenizer.encode(image)

        token_ids = token_ids.squeeze(0).asnumpy()
        input_prompt = ""
        out_prompt = ""
        for prompt in prompts:
            if prompt["from"] == "human":
                input_prompt = prompt["value"].replace("<image>", "")
            elif prompt["from"] == "gpt":
                out_prompt = prompt["value"]
            else:
                raise ValueError(f"Unrecognize prompt from {prompt['from']}")
        data = {"name": name, "images": token_ids, "texts": input_prompt, "response": out_prompt}

        ms.save_checkpoint([], f"{args.output_path}/feature/{name}.ckpt", append_dict=data)
        datalist["path_list"].append(f"{name}.ckpt")
        cnt += 1

    json_file = f"{args.output_path}/list/{args.split}.json"
    with open(json_file, "w") as f:
        json.dump(datalist, f)
    print(f"Generated {json_file} with {cnt} data items.")


if __name__ == "__main__":
    main()
