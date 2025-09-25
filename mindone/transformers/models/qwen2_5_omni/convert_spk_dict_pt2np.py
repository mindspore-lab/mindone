"""Adapted from https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_omni/convert_spk_dict_pt2np.py."""

import argparse
import json
import zipfile

import numpy as np
import torch


def load_n_save_speakers(path, zip_spk_path):
    speaker_map = torch.load(path)
    print("Speaker torch {} loaded".format(list(speaker_map.keys())))

    save_dict_to_zip(zip_spk_path, speaker_map)

    # validate
    data_dict = load_dict_from_zip(zip_spk_path)
    for key, value in data_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    np.allclose(speaker_map[key][k].cpu().numpy(), v)
                else:
                    assert speaker_map[key][k] == v
        else:
            if isinstance(value, np.ndarray):
                np.allclose(speaker_map[key].cpu().numpy(), value)
            else:
                assert speaker_map[key] == value
    print("Speaker numpy {} loaded".format(list(speaker_map.keys())))
    print("All data is same as origin")


def save_dict_to_zip(filepath: str, pt_data: dict):
    r"""
    Using `torch.load` to load a pt file will get a nested dictionary containing the following data types
    (Tensor, int, float, string). Considering the security of data storage and loading, avoid using pickle,
    convert Tensor to numpy.ndarray for storage, use JSON to process non-array data, and package them into
    a single zip file.
    """
    npz_data = {}
    json_data = {}

    def _flatten(d, prefix=""):
        for key, value in d.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                _flatten(value, full_key)
            elif torch.is_tensor(value):
                npz_data[full_key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                npz_data[full_key] = value
            elif isinstance(value, (int, float, str)):
                json_data[full_key] = value
            else:
                raise TypeError(f"Unsupported type {type(value)}")

    _flatten(pt_data)

    # save json and npz files as a zip file
    with zipfile.ZipFile(filepath, "w") as zf:
        if npz_data:
            with zf.open("arrays.npz", "w") as f:
                np.savez(f, **npz_data)
        if json_data:
            with zf.open("meta.json", "w") as f:
                f.write(json.dumps(json_data).encode("utf-8"))


def load_dict_from_zip(filepath: str) -> dict:
    result = {}
    with zipfile.ZipFile(filepath, "r") as zf:
        # load numpy data
        if "arrays.npz" in zf.namelist():
            with zf.open("arrays.npz") as f:
                npz_data = np.load(f)
                for key in npz_data.files:
                    *path, final_key = key.split("/")
                    current = result
                    for p in path:
                        current = current.setdefault(p, {})
                    current[final_key] = npz_data[key]

        # load json data
        if "meta.json" in zf.namelist():
            with zf.open("meta.json") as f:
                json_data = json.load(f)
                for key, value in json_data.items():
                    *path, final_key = key.split("/")
                    current = result
                    for p in path:
                        current = current.setdefault(p, {})
                    current[final_key] = value
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spk_path", type=str, default="Qwen/Qwen2.5-Omni-7B/spk_dict.pt", help="path to torch speaker checkpoint"
    )
    parser.add_argument(
        "--zip_spk_path", type=str, default="Qwen/Qwen2.5-Omni-7B/spk_dict.zip", help="path to zip speaker checkpoint"
    )
    args = parser.parse_args()
    load_n_save_speakers(args.spk_path, args.zip_spk_path)
