import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../../")))
from mindone.data.video_reader import VideoReader


def convert(csv_path: str, dataset_path: str, out_path: str, num_workers: int = 10):
    def read_data(info: dict):
        video_path, caption = info["video"], info["caption"]
        with VideoReader(os.path.join(str(dataset_path), video_path)) as reader:
            return {
                "video": video_path,
                "length": len(reader),
                "width": reader.shape[0],
                "height": reader.shape[1],
                "caption": caption,
            }

    with open(csv_path, "r") as csv_file:
        video_list = list(csv.DictReader(csv_file))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        data = list(tqdm(executor.map(read_data, video_list), total=len(video_list)))

    fieldnames = ["video", "length", "width", "height", "caption"]
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for item in data:
            writer.writerow(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adds video statistics to the CSV file.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the original CSV file with videos.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the directory containing the videos.")
    parser.add_argument("--out_path", type=str, default="./videos_new.csv", help="Path to the output CSV file.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers for multithreading.")
    cfg = parser.parse_args()
    convert(**vars(cfg))
