"""
Script: data_check_part.py
Description:
This script checks the contents of each part of the data.
It performs the following checks for each part:
1. Number of tar files in the part
2. Extraction of each tar file
3. Number of jpg files in each tar file
4. Existence of corresponding json or txt files for each jpg file in each tar file

Usage: python data_check_part.py --output_path output_dir --part part_id
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import tarfile

import moxing as mox


class RootFilter(logging.Filter):
    def filter(self, record):
        # Filter out the WARNING and ERROR information
        # of the root logger to avoid repeated printing by mox.file.copy_parallel
        if record.levelno in (logging.WARNING, logging.ERROR) and record.name == "root":
            return False
        return True


root_filter = RootFilter()

root_logger = logging.getLogger()
root_logger.addFilter(root_filter)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

_logger = logging.getLogger("mindone")
_logger.propagate = False
_logger.addHandler(handler)


def check_json(part_id, data):
    json_txt_missing = 0
    for value in data.values():  # check json or txt missing
        if str(value).startswith("missing"):
            json_txt_missing += 1
            _logger.warning(value)

    corrupt = 0
    for key, value in data.items():  # check corrupted tar files
        if value == -1:
            corrupt += 1
            _logger.warning(f"{key} is corrupted.")

    keys_list = list(data.keys())
    keys_list = [item.split("/")[-1] for item in keys_list]

    missing = 0
    tar_name_list = []
    for i in range(data["cnt"]):  # check missing tar
        tar_name = str(i).zfill(5) + ".tar"
        if tar_name not in keys_list:
            missing += 1
            _logger.warning(f"{tar_name} is missing")
            tar_name_list.append(tar_name)

    _logger.info(f"=========Part {part_id} check results=========")
    _logger.info("Total tar = %d", data["cnt"])
    _logger.info("Total corrupted cnt: %d", corrupt)
    _logger.info("Total missing tar cnt: %d", missing)
    _logger.info("Total missing json/txt cnt: %d", json_txt_missing)


def check_tar(file_path):
    def _is_image_file(fn):
        return fn.endswith(".jpg")

    def _is_text_file(fn):
        return fn.endswith(".txt")

    def _is_json_file(fn):
        return fn.endswith(".json")

    tar_number = {}
    try:
        tar = tarfile.open(file_path, "r")
        all_names = [fn for fn in tar.getnames()]
        json_map = {}
        txt_map = {}
        image_files = []
        unknown = []
        for name in all_names:
            if _is_image_file(name):
                image_files.append(name)
            elif _is_json_file(name):
                json_map[os.path.splitext(name)[0] + ".jpg"] = name
            elif _is_text_file(name):
                txt_map[os.path.splitext(name)[0] + ".jpg"] = name
            else:
                unknown.append(name)
        missing_file = False
        for image in image_files:
            if image not in json_map.keys():
                tar_number[file_path] = f"missing {os.path.splitext(image)[0]}.json"
                missing_file = True
                break
            if image not in txt_map.keys():
                tar_number[file_path] = f"missing {os.path.splitext(image)[0]}.txt"
                missing_file = True
                break
        if missing_file:
            return tar_number
        img_cnt = len(image_files)
        _logger.info(f"{file_path} jpg_number: {img_cnt}")
        tar_number[file_path] = img_cnt
    except Exception:
        tar_number[file_path] = -1  # tar package corrupted flag
        _logger.error("Error in extracting", file_path)
    return tar_number


def sync_data(from_path, to_path, output_path, part):
    """
    Copy data from `from_path` to `to_path`.
    1) if `from_path` is remote url and `to_path` is local path, download data from remote obs to local directory
    2) if `from_path` is local path and `to_path` is remote url, upload data from local directory to remote obs .
    """
    # Each server contains 8 devices as most.
    _logger.info("===data synchronization begin===")
    num_samples_json = {}

    valid_to_path = []
    for f, t in zip(from_path, to_path):
        try:
            mox.file.copy_parallel(f, t)
            _logger.info("finish copy" + t)
            valid_to_path.append(t)
        except Exception:
            _logger.warning(f"path {f} not exist!")
    _logger.info("===finish data synchronization===")

    num_samples_json["cnt"] = len(valid_to_path)

    pool = mp.Pool(min(mp.cpu_count(), len(to_path)))
    results = pool.map(check_tar, valid_to_path, chunksize=1)
    pool.close()
    pool.join()

    for each_dict in results:
        if isinstance(each_dict, dict):
            num_samples_json.update(each_dict)

    json_str = json.dumps(num_samples_json)
    with open(os.path.join(output_path, f"check_num_samples_part{part}.json"), "w") as json_file:
        json_file.write(json_str)
    _logger.info(f"===finish writing the following check_num_samples_part{part}.json===")
    _logger.info(num_samples_json)
    check_json(part, num_samples_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--part", type=int, help="the part ID of the dataset to be inspected")
    args = parser.parse_args()

    LOCAL_RANK = int(os.getenv("RANK_ID", 0))

    if LOCAL_RANK == 0:  # only use card 0 to check
        src_list = []
        dst_list = []
        part_list = [args.part]

        for part in part_list:
            _logger.info(f"checking part {part}......")
            for t in range(533):
                src = f"obs://laion-2b/sd2.1_base_train/part_{part}/{t:05d}.tar"
                dst = f"/cache/part_{part}/{t:05d}.tar"
                src_list.append(src)
                dst_list.append(dst)
            os.makedirs(f"/cache/part_{part}", exist_ok=True)
            sync_data(src_list, dst_list, args.output_path, part_list[0])
