import argparse
import os

import numpy as np
import pandas as pd
from pipeline.datasets.imagededup.methods import AHash, DHash, PHash, WHash
from pipeline.datasets.utils import extract_frames, is_video, pil_loader
from tqdm import tqdm


class VideoTextDataset:
    def __init__(self, meta_path):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample["path"]

        # extract the middle frame
        if not is_video(path):
            image = [pil_loader(path)]
        else:
            num_frames = sample["num_frames"] if "num_frames" in sample else None
            # extract the first frame of every video
            image = extract_frames(path, frame_inds=[0], backend="opencv", num_frames=num_frames)

        return path, np.array(image)[0]  # Return the first (and the only) image

    def __len__(self):
        return len(self.meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--skip_if_existing", action="store_true")
    parser.add_argument(
        "--hash",
        type=str,
        default="phash",
        help="Hash algorithm to use, choose from 'phash', 'ahash', 'dhash', or 'whash'",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=15,
        help="Max distance threshold for detecting duplication after encoding and hashing, between 1 to 64",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_dedup{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    if args.threshold < 1 or args.threshold > 64:
        print(f"Invalid threshold value '{args.threshold}'. Must be between 1 to 64. Exit.")
        exit()

    dataset = VideoTextDataset(meta_path)
    if args.hash == "phash":
        hasher = PHash()
    elif args.hash == "ahash":
        hasher = AHash()
    elif args.hash == "dhash":
        hasher = DHash()
    elif args.hash == "whash":
        hasher = WHash()
    else:
        print(f"Invalid hash {args.hash}. Must be one of 'phash', 'ahash', 'dhash', 'whash'. Exit.")
        exit()

    encodings = {}
    for index in tqdm(range(len(dataset))):
        path, image = dataset[index]
        encoding = hasher.encode_image(image_array=image)
        encodings[path] = encoding

    # find duplicates & filter
    duplicates = hasher.find_duplicates_to_remove(encoding_map=encodings, max_distance_threshold=args.threshold)
    duplicate_paths = set(duplicates)

    deduplicated_meta = dataset.meta[~dataset.meta["path"].isin(duplicate_paths)]
    deduplicated_meta.to_csv(out_path, index=False)
    print(f"Deduplicated videos saved to '{out_path}'.")


if __name__ == "__main__":
    main()
