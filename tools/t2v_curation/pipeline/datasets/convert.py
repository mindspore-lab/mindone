import argparse
import os
import time

import pandas as pd

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
VID_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", "m2ts")


def scan_recursively(root):
    num = 0
    for entry in os.scandir(root):
        if entry.is_file():
            yield entry
        elif entry.is_dir():
            num += 1
            if num % 100 == 0:
                print(f"Scanned {num} directories.")
            yield from scan_recursively(entry.path)


def get_filelist(file_path, exts=None):
    filelist = []
    time_start = time.time()

    obj = scan_recursively(file_path)
    for entry in obj:
        ext = os.path.splitext(entry.name)[-1].lower()
        if exts is None or ext in exts:
            filelist.append(entry.path)

    time_end = time.time()
    print(f"Scanned {len(filelist)} files in {time_end - time_start:.2f} seconds.")
    return filelist


def process_general_images(root, output):
    root = os.path.expanduser(root)
    if not os.path.exists(root):
        return
    path_list = get_filelist(root, IMG_EXTENSIONS)
    fname_list = [os.path.splitext(os.path.basename(x))[0] for x in path_list]
    df = pd.DataFrame(dict(id=fname_list, path=path_list))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} samples to {output}")


def process_general_videos(root, output):
    root = os.path.expanduser(root)
    if not os.path.exists(root):
        return
    path_list = get_filelist(root, VID_EXTENSIONS)
    path_list = list(set(path_list))  # remove duplicates
    fname_list = [os.path.splitext(os.path.basename(x))[0] for x in path_list]
    relpath_list = [os.path.relpath(x, root) for x in path_list]
    df = pd.DataFrame(dict(id=fname_list, path=path_list, relpath=relpath_list))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} samples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # mammalnet not yet implemented
    parser.add_argument("dataset", type=str, choices=["image", "video"])
    parser.add_argument("root", type=str)
    parser.add_argument("--output", type=str, default=None, required=True, help="Output path")
    args = parser.parse_args()

    if args.dataset == "image":
        process_general_images(args.root, args.output)
    elif args.dataset == "video":
        process_general_videos(args.root, args.output)
    else:
        raise ValueError("Invalid dataset")
