import argparse
import os


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
    stride = args.stride
    i = 2  # start from the second line, assuming the first line is a header
    partid = 0

    print(f"Splitting the metadata file: {meta_filepath}. Total number of lines: {linenum}")

    while i <= linenum:
        output_filename = os.path.join(output_dir, f"part{partid}.csv")
        os.system(f"head -n 1 {meta_filepath} > {output_filename}")  # Write the header to each part
        end_line = min(i + stride - 1, linenum)  # Calculate the correct end line, ensuring not to exceed total lines
        os.system(f"sed -n '{i},{end_line}p' {meta_filepath} >> {output_filename}")
        i += stride
        partid += 1

    expected_parts = (linenum - 1) // stride + 1  # Calculate expected number of parts
    assert (
        partid == expected_parts
    ), f"The metadata file should be split into {expected_parts} parts, but got {partid} parts."
    print(f"Finished! The metadata file is split into {partid} parts, saved in '{output_dir}'.")


if __name__ == "__main__":
    main()
