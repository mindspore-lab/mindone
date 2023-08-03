import csv
import glob
import os
import sys

sys.path.append(".")

from eval.fid.utils import Download


def gen_dummpy_data():
    # prepare dummy images
    data_dir = "data/Canidae"
    dataset_url = (
        "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    )
    if not os.path.exists(data_dir):
        Download().download_and_extract_archive(dataset_url, "./")

    # prepare dummy labels
    label_path = "tests/st/dummy_labels/img_txt.csv"
    image_dir = f"{data_dir}/val/dogs"
    new_label_path = f"{data_dir}/val/dogs/img_txt.csv"
    img_paths = glob.glob(os.path.join(image_dir, "*.JPEG"))
    with open(new_label_path, "w") as f_w:
        with open(label_path, "r") as f_r:
            spamread = csv.reader(f_r)
            spamwriter = csv.writer(f_w)
            for i, row in enumerate(spamread):
                if i > 0:
                    row[0] = os.path.basename(img_paths[i - 1])
                spamwriter.writerow(row)
    print(f"Dummpy annotation file is generated in {new_label_path}")

    return data_dir


if __name__ == "__main__":
    gen_dummpy_data()
