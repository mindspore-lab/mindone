from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data.video_reader import VideoReader
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fc, path_type
from tqdm import tqdm

Path_dr = path_type("dr")  # path to a directory that exists and is readable


def convert(dataset_path: Path_dr, out_path: Path_fc = "./videos.csv", num_workers: int = 10):
    """
    Generates a CSV file containing information about videos in a dataset.

    Parameters:
        dataset_path: The path to the dataset root directory.
        out_path: The path to the output CSV file. Default: "./videos.csv".
        num_workers: The number of worker threads to use for parallel data reading. Default: 10.
    """

    def read_data(path):
        caption = open(path.parent / (path.stem + ".txt")).read()[2:-3]  # remove (' and ',)
        caption = caption.replace("\\n", "").replace("\\r", "").replace("\\t", "")
        with VideoReader(str(path)) as reader:
            return f"{path.relative_to(dataset_path)},{len(reader)},{caption}\n"

    video_list = list(Path(dataset_path).rglob("*.mp4"))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        data = list(tqdm(executor.map(read_data, video_list), total=len(video_list)))

    with open(out_path, "w") as f:
        f.write("video,length,caption\n")
        f.writelines(data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_function_arguments(convert)
    cfg = parser.parse_args()
    convert(**cfg.as_dict())
