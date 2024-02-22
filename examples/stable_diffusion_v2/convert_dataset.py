"""
Convert hugging face parquet format dataset to raw format that is trainable with stable diffusion in mindone

Usage: python python convert_hf_datset.py

The conversion is run serially and is slow now.
"""
import os

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def convert_hf_dataset_to_raw(
    dn="lambdalabs/pokemon-blip-captions",
    save_folder="pokemon_raw",
):
    csv_fn = "img_txt.csv"

    dataset = load_dataset(dn, cache_dir="./downloads")

    column_names = dataset["train"].column_names
    print(column_names)

    dataset_columns = DATASET_NAME_MAPPING.get(dn, None)
    image_column = dataset_columns[0]
    caption_column = dataset_columns[0]

    num_samples = len(dataset["train"])  # only train split in pokemon dataset
    print("Num samples: ", num_samples)

    img_fns = []
    texts = []
    for i in tqdm(range(num_samples)):
        pil_obj = dataset["train"][image_column][i]
        text = dataset["train"][caption_column][i]

        img_fn = f"img{i}.png"
        pil_obj.save(f"{save_folder}/{img_fn}")

        img_fns.append(img_fn)
        texts.append(text)

    frame = pd.DataFrame({"dir": img_fns, "text": texts})
    frame.to_csv(os.path.join(save_folder, csv_fn), index=False, sep=",")

    print("Completed! Output saved in ", save_folder)


if __name__ == "__main__":
    convert_hf_dataset_to_raw()
