import argparse

import numpy as np
import torch


def load_n_save_speakers(path, np_path):
    speaker_map = {}
    for key, value in torch.load(path).items():
        if isinstance(value, dict):
            value_ms_dict = {}
            for k, v in value.items():
                if torch.is_tensor(v):
                    value_ms_dict[k] = v.cpu().numpy()
                else:
                    value_ms_dict[k] = v
            speaker_map[key] = value_ms_dict
        else:
            if torch.is_tensor(value):
                speaker_map[key] = value.cpu().numpy()
            else:
                speaker_map[key] = value
    # print("Speaker torch2ms loaded:", speaker_map)
    print("Speaker torch {} loaded".format(list(speaker_map.keys())))

    np.save(np_path, speaker_map)

    # validate
    np_dict = np.load(np_path, allow_pickle=True).item()
    speaker_map = {}
    for key, value in np_dict.items():
        speaker_map[key] = value
    # print("Speaker np loaded:", speaker_map)
    print("Speaker numpy {} loaded".format(list(speaker_map.keys())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spk_path", type=str, default="Qwen/Qwen2.5-Omni-7B/spk_dict.pt", help="path to torch speaker checkpoint"
    )
    parser.add_argument(
        "--np_spk_path", type=str, default="Qwen/Qwen2.5-Omni-7B/spk_dict.npy", help="path to numpy speaker checkpoint"
    )
    args = parser.parse_args()
    load_n_save_speakers(args.spk_path, args.np_spk_path)
