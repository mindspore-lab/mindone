"""
Convert hugging face parquet format dataset to raw format that is trainable for Kohya finetune method (with a metadata file)

Usage: python convert_hf_datset.py

The conversion is run serially and is slow now.
"""
import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
    "YaYaB/onepiece-blip-captions": ("image", "text"),
}


def convert_hf_dataset_to_raw(
    dn,
    save_folder,
    cache_dir,
):
    metadata_fn = "metadata.json"
    dataset = load_dataset(dn, cache_dir=cache_dir)

    column_names = dataset["train"].column_names
    print(column_names)

    dataset_columns = DATASET_NAME_MAPPING.get(dn, None)
    if dataset_columns is None:
        raise NotImplementedError(
            "please check your dataset column names of img and text and add to DATASET_NAME_MAPPING"
        )

    image_column = dataset_columns[0]
    caption_column = dataset_columns[1]

    num_samples = len(dataset["train"])  # only train split in pokemon dataset
    print("Num samples: ", num_samples)

    metadata = {}
    for i in tqdm(range(num_samples)):
        pil_obj = dataset["train"][image_column][i]
        text = dataset["train"][caption_column][i]

        img_fn = f"img{i}.png"
        pil_obj.save(f"{save_folder}/{img_fn}")

        metadata[img_fn] = {"caption": text}

    with open(os.path.join(save_folder, metadata_fn), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("Completed! Output saved in ", save_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, default="YaYaB/onepiece-blip-captions")
    parser.add_argument("--cache_dir", type=str, default="./download")
    parser.add_argument("--save_folder", type=str, default="./onepiece_raw_data")
    args = parser.parse_args()

    convert_hf_dataset_to_raw(dn=args.dataset_name, cache_dir=args.cache_dir, save_folder=args.save_folder)
