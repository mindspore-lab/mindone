# Adapted from https://github.com/baaivision/Emu3 to work with MindSpore.
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
python emu3/train/prepare_text_data.py --data-path DIR-TO-DATA.json --output-path DATA-DIR
"""

import argparse
import json
import os

from tqdm import tqdm

import mindspore as ms


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="data path")
    parser.add_argument("--output-path", type=str, help="tokenized data save path")
    parser.add_argument("--split", type=str, default="train", help="split to train to test set")

    args = parser.parse_args()
    return args


def main():
    args = prepare_args()

    os.makedirs(f"{args.output_path}/feature", exist_ok=True)
    os.makedirs(f"{args.output_path}/list", exist_ok=True)

    datalist = {"prefix": os.path.join(args.output_path, "feature"), "path_list": []}

    with open(args.data_path) as f:
        input_data = json.load(f)
    cnt = 0
    for inp in tqdm(input_data):
        name = inp["id"]
        input_prompt = inp["texts"]
        out_prompt = inp["response"]
        token_ids = False

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
