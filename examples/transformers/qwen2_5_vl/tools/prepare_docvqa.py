import argparse
import json
import os

import tqdm
from datasets import load_dataset
from PIL import Image


def main():
    parser = argparse.ArgumentParser("Convert DocVQA to the training format.")
    parser.add_argument("-o", "--outdir", default="./doc_vqa", help="path of the output directory.")
    parser.add_argument("-d", "--dataset_name", default="lmms-lab/DocVQA", help="name or path of the dataset")
    args = parser.parse_args()

    image_folder = os.path.join(args.outdir, "images")
    os.makedirs(image_folder, exist_ok=True)
    dataset = load_dataset(args.dataset_name, "DocVQA", split="validation")
    contents = []
    for row in tqdm.tqdm(dataset):
        question_id = row["questionId"]
        image: Image.Image = row["image"]
        question = "<image>\n" + row["question"]
        answers = row["answers"]
        image_path = os.path.join(image_folder, f"{question_id}.jpg")
        for answer in answers:
            conversation_pair = {
                "image": image_path,
                "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer}],
            }
            contents.append(conversation_pair)
        image.save(image_path)
    with open(os.path.join(args.outdir, "doc_vqa.json"), "w", encoding="utf8") as f:
        json.dump(contents, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
