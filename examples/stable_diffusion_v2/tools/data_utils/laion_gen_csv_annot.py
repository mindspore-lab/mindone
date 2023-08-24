import argparse
import glob
import json
import json as js
import os

import pandas as pd
from tqdm import tqdm

# from pyspark.sql import SparkSession

# check completeness of download images
filter_width = 512


def check_download_result(data_dir="/data3/datasets/laion_art", img_fmt="jpg", download_fmt="files"):
    assert os.path.exists(data_dir), f"{data_dir} not exists"
    img_paths = sorted(glob.glob(os.path.join(data_dir, f"*/*.{img_fmt}")))
    num_imgs = len(img_paths)
    print("Get image num: ", num_imgs)

    # check total fails

    # check parquets in download image folder
    # spark = SparkSession.builder.config("spark.driver.memory", "2G") .master("local[4]").appName('spark-stats').getOrCreate()
    # df = spark.read.parquet(data_dir)
    fp = data_dir + "/00000.parquet"
    print(fp)
    df = pd.read_parquet(fp)
    print(df.count())
    print(df.show())


def gen_csv(
    data_dir,
    img_fmt="jpg",
    one_csv_per_part=True,
    folder_prefix="",
    merge_all=True,
    merge_fn="merged_imgp_text.csv",
    start_sample_idx=0,
    end_sample_idx=-1,
    del_part_csvs=True,
    json_data_path=None,
):
    """
    Args:
        start_sample_idx: index of the first sample to include for training in the first tar of the first part
        end_sample_idx: index of the last sample to include for training in the last tar of the last part

    Input data structure:
    ```text
    data_dir
    ├── part_1/ # part folder
    │   ├── 00000 # sub folder extracted for tar file
    │   │   ├── 000000000.jpg
    │   │   ├── 000000001.jpg
    │   │   ├── 000000002.jpg
    │   │   └── ...
    │   ├── ...
    │
    ├── part_2/
    ...
    ```
    """
    assert os.path.exists(data_dir), f"{data_dir} not exists"
    num_imgs = 0
    len_postfix = len(img_fmt) + 1

    folders = [fp for fp in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, fp))]
    if folder_prefix != "":
        folders = [f for f in folders if f.startswith(folder_prefix)]

    # sort folders by part_idx
    folders = sorted(folders, key=lambda folder_name: int(folder_name.split("_")[1]))
    part_ids = [int(folder_name.split("_")[1]) for folder_name in folders]
    print("part ids: ", part_ids)

    # for part_id in range(1, num_parts+1):
    def _gather_img_text_in_folder(root_dir, folder, trim_start=0, trim_end=-1):
        # import pdb
        # pdb.set_trace()
        if trim_end == -1:
            img_paths = sorted(glob.glob(os.path.join(root_dir, folder, f"*.{img_fmt}")))[trim_start:]
        else:
            img_paths = sorted(glob.glob(os.path.join(root_dir, folder, f"*.{img_fmt}")))[trim_start : trim_end + 1]
        print("Image folder: ", folder, ", num imgs: ", len(img_paths))
        rel_img_paths = []
        texts = []
        if len(img_paths) > 0:
            for img_fp in tqdm(img_paths):
                # text_fp = img_fp[:-len_postfix] + ".txt"
                json_fp = img_fp[:-len_postfix] + ".json"

                # import pdb
                # pdb.set_trace()
                rel_img_paths.append(os.path.join(folder, os.path.basename(img_fp)))

                # with open(text_fp, 'r') as f:
                #    text = f.read()
                try:
                    with open(json_fp, "r") as f:
                        meta = json.load(f)
                        text = meta["caption"]
                except Exception:
                    print("json file open failed or not exist, path: ", json_fp, flush=True)
                    text = "Fake text"

                texts.append(text)

        return rel_img_paths, texts

    def _save_to_csv(img_paths, texts, save_fp):
        assert len(img_paths) == len(texts), f"{len(img_paths)} != {len(texts)}"
        frame = pd.DataFrame({"dir": img_paths, "text": texts})
        frame.to_csv(save_fp, index=False, sep=",")
        print("csv saved in ", save_fp)

    all_csv_paths = []
    # pdb.set_trace()

    for i, folder in enumerate(folders):
        # try to get image-text from level one folder
        # rel_img_paths, texts = _gather_img_text_in_folder(data_dir, folder)
        # if len(texts) > 0:
        #    save_fp = os.path.join(data_dir, folder + ".csv")
        #    _save_to_csv(rel_img_paths, texts, save_fp)
        #    pdb.set_trace()
        #    all_csv_paths.append(save_fp)
        #    num_imgs += len(rel_img_paths)

        # second level
        subfolders = [
            dn
            for dn in sorted(os.listdir(os.path.join(data_dir, folder)))
            if os.path.isdir(os.path.join(data_dir, folder, dn))
        ]
        # img_folders = [os.path.join(folder, dn)  for dn in subfolders]
        print("Folder: ", folder, ", num sub folders: ", len(subfolders))

        rel_img_paths_all = []
        texts_all = []
        for j, subfolder in enumerate(subfolders):  # tar extracted
            trim_start = 0
            trim_end = -1
            if (i == 0) and (j == 0):  # is_first_subset:
                trim_start = start_sample_idx
            if (i == len(folders) - 1) and (j == len(subfolders) - 1):  # is_last_subset:
                trim_end = end_sample_idx

            # import pdb
            # pdb.set_trace()
            untrimed_img_samples = len(
                sorted(glob.glob(os.path.join(os.path.join(data_dir, folder), subfolder, f"*.{img_fmt}")))
            )
            part_tar_samples = int(js.load(open(json_data_path))[folder.split("_")[1]][str(int(subfolder))])
            try:
                print(
                    "untrimed_img_samples", untrimed_img_samples, "json part_tar_samples", part_tar_samples, flush=True
                )
                assert untrimed_img_samples == part_tar_samples
            except Exception:
                print(
                    f"number of imges in {os.path.join(data_dir, folder, subfolder)} is {untrimed_img_samples}, "
                    f"which is not eq to json value {part_tar_samples}",
                    flush=True,
                )

            rel_img_paths, texts = _gather_img_text_in_folder(
                os.path.join(data_dir, folder), subfolder, trim_start=trim_start, trim_end=trim_end
            )

            if len(rel_img_paths) > 0:
                if one_csv_per_part:
                    # csv saved along with folder
                    rel_img_paths = [folder + "/" + p for p in rel_img_paths]
                    rel_img_paths_all.extend(rel_img_paths)
                    texts_all.extend(texts)
                else:
                    # csv saved under folder
                    # rel_img_paths= [os.path.join(p.split("/")[1:]) for p in rel_img_paths]
                    save_fp = os.path.join(data_dir, folder, subfolder + ".csv")
                    _save_to_csv(rel_img_paths, texts, save_fp)
                    all_csv_paths.append(save_fp)

                num_imgs += len(rel_img_paths)
        if len(rel_img_paths_all) > 0:
            print("Saving csv...")
            save_fp = os.path.join(data_dir, folder + ".csv")
            print(len(rel_img_paths_all), len(texts_all))
            _save_to_csv(rel_img_paths_all, texts_all, save_fp)
            all_csv_paths.append(save_fp)

    print("Num text-image pairs with trim: ", num_imgs)
    print("All csv files are saved in ", data_dir)

    if merge_all:
        print("csv files to merged: ", all_csv_paths)
        df = pd.concat(map(pd.read_csv, all_csv_paths), ignore_index=True)
        save_fp = os.path.join(data_dir, merge_fn)
        df.to_csv(save_fp, index=False, sep=",")
        print("Finished. Removing cached csv files...")
        if del_part_csvs:
            for cache_csv in all_csv_paths:
                if os.path.exists(cache_csv):
                    os.remove(cache_csv)
        print("Merged CSV file saved in: ", save_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save csv")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Volumes/Extreme_SSD/LAION/sd2.1_base_train",
        help="dir containing the downloaded images",
    )
    parser.add_argument(
        "--folder_prefix", type=str, default="part", help="folder prefix to filter unwanted folders. e.g. part"
    )
    parser.add_argument(
        "--save_csv_per_img_folder",
        type=bool,
        default=False,
        help="If False, save a csv file for each part, which will result in a large csv file (~400MB). \
            If True, save a csv file for each image folder, which will result in hundreads of csv files for one part of dataset.",
    )
    parser.add_argument(
        "--start_sample_idx",
        type=int,
        default=0,
        help="index of the first sample to include for training in the first tar of the first part",
    )
    parser.add_argument(
        "--end_sample_idx",
        type=int,
        default=-1,
        help="index of the last sample to include for training in the last tar of the last part",
    )
    args = parser.parse_args()

    # data_dir = '/data3/datasets/laion_art_filtered'
    # data_dir = args.data_dir
    # check_download_result(data_dir)
    gen_csv(
        "/cache",
        one_csv_per_part=not args.save_csv_per_img_folder,
        folder_prefix=args.folder_prefix,
        start_sample_idx=args.start_sample_idx,
        end_sample_idx=args.end_sample_idx,
    )
