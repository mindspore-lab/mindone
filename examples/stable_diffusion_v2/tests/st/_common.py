import csv
import glob
import os
import sys

sys.path.insert(0, ".")

from tools.eval.fid.utils import Download


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


def down_checkpoint(version="1.5"):
    root = os.path.dirname(os.path.abspath(__file__))
    url = {
        "1.5": "https://download-mindspore.osinfra.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt",
        "2.0": "https://download-mindspore.osinfra.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt",
    }
    pretrained_model_dir = os.path.join(root, "../../models/")
    if version == "1.5":
        pretrained_model_path = os.path.join(root, "../../models/sd_v1.5-d0ab7146.ckpt")
    elif version == "2.0":
        pretrained_model_path = os.path.join(root, "../../models/sd_v2_base-57526ee4.ckpt")
    else:
        raise ValueError(f"SD {version} not included in test")

    Download().download_and_extract_archive(url[version], pretrained_model_dir)

    return pretrained_model_path


if __name__ == "__main__":
    gen_dummpy_data()
