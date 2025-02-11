import glob
import json
import os
import sys

import webdataset as wds
from tqdm import tqdm


def get_tar_file_list(data_dir):
    # get tar file recursively
    tar_files = []
    tar_files.extend(glob.glob(os.path.join(data_dir, "*.tar")))

    folders = [fp for fp in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, fp))]
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        tar_files.extend(get_tar_file_list(folder_path))

    return tar_files


def get_tar_nsample(tar_file):
    # TODO: improve efficiency.
    wds_iterator = wds.WebDataset(tar_file)
    n = 0
    for cur in wds_iterator:
        n += 1
    return n


def generate_sharlist(data_dir):
    tar_files = get_tar_file_list(data_dir)
    out = {
        "__kind__": "wids-shard-index-v1",
        "wids_version": 1,
        "shardlist": [],
    }
    print("INFO: Start to scan tar files...", flush=True)
    # TODO: 1) use multi-process. 2) consider multiple machine access.
    for tf in tqdm(tar_files):
        tar_info_fp = tf.replace(".tar", ".txt")
        if not os.path.exists(tar_info_fp):
            # scan
            nsamples = get_tar_nsample(tf)

            with open(tar_info_fp, "w") as fp:
                fp.write(str(nsamples))
        else:
            with open(tar_info_fp, "r") as fp:
                nsamples = int(fp.read())

        out["shardlist"].append({"url": tf, "nsamples": nsamples})
    save_fp = os.path.join(data_dir, "data_info.json")
    with open(save_fp, "w") as fp:
        json.dump(out, fp)

    return save_fp


if __name__ == "__main__":
    data_path = sys.argv[1]
    generate_sharlist(data_path)
