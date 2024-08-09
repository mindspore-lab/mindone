import os
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Split one large CSV to smaller ones")
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--stride", type=int, default=500000, help="Max entries per CSV")
    parser.add_argument("--root_dir", type=str, default="./", help="Root directory for reading the metadata file")
    parser.add_argument("--output_dir", type=str, default="./", help="Output directory for split CSV files")
    return parser.parse_args()

def main():
    args = parse_args()
    meta_filename = os.path.basename(args.meta_path)  # Extracts filename from the given path
    meta_filepath = os.path.join(args.root_dir, meta_filename)
    output_dir = os.path.join(args.output_dir, os.path.splitext(meta_filename)[0])
    os.makedirs(output_dir, exist_ok=True)

    linenum = int(os.popen(f"wc -l {meta_filepath}").read().split()[0])
    stride = 500000
    i = 2
    partid = 0

    print(f"Spliting the metadata file: {meta_filepath}. Total number of lines: {linenum}")

    while i <= linenum:
        output_filename = os.path.join(output_dir, f"part{partid}.csv")
        os.system(f"head -n 1 {meta_filepath} > {output_filename}")
        os.system(f"sed -n '{i}, {i + stride}p' {meta_filepath} >> {output_filename}")
        i += stride + 1
        partid += 1

    assert (
            partid == linenum // stride + 1
    ), f"The metadata file shoule be splited into {linenum // stride + 1} parts, but got {partid} parts."
    print(f"Finished! The metadata file is splited into {partid} parts, saved in '{output_dir}'.")

if __name__ == "__main__":
    main()
