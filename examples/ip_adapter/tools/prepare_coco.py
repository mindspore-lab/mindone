#!/usr/bin/env python
import argparse
import json
import os
from typing import Iterator, Tuple


def read_coco_annotations(path: str, image_root: str) -> Iterator[Tuple[str, str]]:
    with open(path, "r") as f:
        content = json.load(f)["annotations"]
        for record in content:
            image_id = record["image_id"]
            caption = record["caption"]
            image_name = f"{image_id:012d}.jpg"
            image_path = os.path.join(image_root, image_name)
            if not os.path.isfile(image_path):
                print(f"Cannot find `{image_path}`, skip.")
                continue
            caption = caption.strip().replace(",", "").replace("\n", "")
            yield image_name, caption


def main():
    parser = argparse.ArgumentParser(description="Converting COCO json annatotion into plain txt")
    parser.add_argument("--label", required=True, help="Path of the label file")
    parser.add_argument("--image", required=True, help="Path of the image root")
    parser.add_argument("--out", default="img_txt.csv", help="Output path of the txt file")
    args = parser.parse_args()

    with open(args.out, "w") as f:
        f.write("dir,text\n")
        for image_name, caption in read_coco_annotations(args.label, args.image):
            f.write(f"{image_name},{caption}\n")


if __name__ == "__main__":
    main()
