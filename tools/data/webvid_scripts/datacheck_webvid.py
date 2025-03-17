import argparse
import json
import os

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="WebVid Data Check")
    parser.add_argument(
        "--set",
        type=str,
        required=True,
        choices=["10M_train", "10M_val", "2M_train", "2M_val"],
        help="Dataset set name",
    )
    parser.add_argument("--part", type=str, help="Part ID of the dataset to check, required for train sets")
    parser.add_argument("--root", type=str, default="./webvid-10m", help="Root directory of the dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    meta_root_dir = os.path.join(args.root, "metadata")
    dataset_root_dir = os.path.join(args.root, "dataset")

    # Handle validation and training data paths
    if "val" in args.set:
        metadata_path = os.path.join(meta_root_dir, f"results_{args.set}.csv")
        data_dir = os.path.join(dataset_root_dir, args.set)
    else:
        assert args.part, "Please provide a part ID via --part for training datasets."
        metadata_path = os.path.join(meta_root_dir, f"results_{args.set}/part{args.part}.csv")
        data_dir = os.path.join(dataset_root_dir, f"{args.set}/part{args.part}")

    print(f"Checking the data completeness of {data_dir}. It takes some time...")

    # get metadata line num, i.e., video num
    meta_video_num = int(os.popen(f"wc -l {metadata_path}").read().split()[0]) - 1

    data_filenames = os.listdir(data_dir)
    if "_tmp" in data_filenames:
        raise Exception(f"The data in {data_dir} has not been completely downloaded yet.")

    download_video_num = 0
    parquet_num, json_num, tar_num = 0, 0, 0
    for name in tqdm(data_filenames):
        filepath = os.path.join(data_dir, name)
        if name.endswith(".parquet"):
            parquet_num += 1
        elif name.endswith(".json"):
            with open(filepath, "r") as f:
                try:
                    _ = json.load(f)
                except Exception:
                    raise Exception(
                        f"Invalid json file: {filepath}. Please download the whole {args.set}/part{args.part} again."
                    )
            json_num += 1
        elif name.endswith(".tar"):
            r = os.popen(
                f"python -m tarfile -l {filepath} | wc -l"
            ).read()  # If error message appears, the tar file is invalid. Please download again.
            if int(r) % 3 != 0:
                raise Exception(
                    f"Broken tar file: {filepath}. Please download the whole {args.set}/part{args.part} again."
                )

            download_video_num += int(r) // 3  # one video data consists of three parts: json, txt, mp4.
            tar_num += 1
        else:
            raise Exception(
                f"Unknown file: {filepath}. The valid file formats are parquet, json and tar. "
                f"Please delete the file or download the whole {args.set}/part{args.part} again."
            )

    assert parquet_num == json_num == tar_num, (
        f"The data in {data_dir} is incomplete, because the numbers of parquet, json and tar are not identical. "
        f"Please download the whole {args.set}/part{args.part} again."
    )

    success_rate = download_video_num / meta_video_num if meta_video_num else 0
    print("INFO: If no error message appears, the downloaded data is complete and valid. Otherwise, please redownload.")
    print(f"*** {args.set} {'part ' + args.part if args.part else ''} ***")
    print(f"Number of videos in metadata {metadata_path}: {meta_video_num}")
    print(f"Number of downloaded files: {download_video_num}")
    print(f"Number of tar files: {tar_num}")
    print(f"Downloading success rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()
