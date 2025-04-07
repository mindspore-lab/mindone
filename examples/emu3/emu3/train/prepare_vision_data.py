"""
This script is to prepare image generation dataset.

Input example data json format:
{
[
    "name": "data1"，
    "text": "text prompt #1",
    "image": "image_1.jpg"
],
[
    "name": "data2"，
    "text": "text prompt #2",
    "image": "image_2.jpg"
],
...
}

Output dataset
- output_path/list/train.json:
    {
        "prefix": output_path/feature,
        "path_list": ["data1.ckpt", "data2.ckpt", ...]
    }
- /output_path/feature/data1.ckpt:
    {"name": name, "images": token_ids, "texts": prompt}
- /output_path/feature/data2.ckpt:
    {"name": name, "images": token_ids, "texts": prompt}
...

Usage:
cd examples/emu3
python emu3/train/prepare_vision_data.py --model-path DIR-TO-Emu3-VisionTokenizer --data-path DIR-TO-DATA.json --output-path DATA-DIR
"""

import argparse
import json
import os

from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image
from tqdm import tqdm

import mindspore as ms

from mindone.diffusers.training_utils import pynative_no_grad as no_grad


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="vision tokenizer path")
    parser.add_argument("--data-path", type=str, help="data path")
    parser.add_argument("--output-path", type=str, help="tokenized data save path")
    parser.add_argument("--image-area", type=int, default=720 * 720)
    parser.add_argument("--split", type=str, default="train", help="split to train to test set")

    args = parser.parse_args()
    return args


def smart_resize(image, image_area: int = 720 * 720, factor: int = 8):
    w, h = image.size
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    current_area = h_bar * w_bar
    target_ratio = (image_area / current_area) ** 0.5

    th = int(round(h * target_ratio / factor) * factor)
    tw = int(round(w * target_ratio / factor) * factor)

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
    for inp in tqdm(input_data):
        name = inp["name"]
        prompt = inp["text"]
        img_dir = os.path.join(base_dir, inp["image"])
        image = Image.open(img_dir).convert("RGB")
        image = smart_resize(image, args.image_area)

        image = image_processor(image, do_resize=False, return_tensors="np")["pixel_values"]
        with no_grad():
            image = ms.Tensor(image, dtype=image_tokenizer.dtype)
            token_ids = image_tokenizer.encode(image)

        token_ids = token_ids.squeeze(0)
        data = {"name": name, "images": token_ids, "texts": prompt}

        ms.save_checkpoint([], f"{args.output_path}/feature/{name}.ckpt", append_dict=data)
        datalist["path_list"].append(f"{name}.ckpt")
        cnt += 1

    json_file = f"{args.output_path}/list/{args.split}.json"
    with open(json_file, "w") as f:
        json.dump(datalist, f)
    print(f"Generated {json_file} with {cnt} data items.")


if __name__ == "__main__":
    main()
