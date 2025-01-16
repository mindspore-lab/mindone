import argparse
import io
import os

import pandas as pd


def filter_valid_lines(input_file, output_file, expected_fields=6):
    """
    This method tries to concat multiple lines into a single line until gathering the expected fields.
    This corresponds to the issue identified in this dataset,
    so it is recommended to use this function to fully utilize the dataset.
    :param input_file: input file directory & name
    :param output_file:  output file directory & name
    :return: total lines in the original csv and total lines in the output csv
    """
    total_lines = 0
    valid_lines = 0
    valid_rows = []
    buffer = ""

    with open(input_file, "r", encoding="utf-8") as infile:
        header = infile.readline().strip()
        valid_rows.append(header)
        total_lines += 1

        for line in infile:
            total_lines += 1
            buffer += line.strip()

            if buffer.count(",") >= expected_fields - 1:
                try:
                    pd.read_csv(io.StringIO(buffer), header=None)
                    valid_rows.append(buffer)
                    valid_lines += 1
                except pd.errors.ParserError:
                    continue
                buffer = ""
            else:
                buffer += " "

    with open(output_file, "w", encoding="utf-8") as outfile:
        for row in valid_rows:
            outfile.write(row + "\n")

    return total_lines, valid_lines


def filter_valid_lines_old(input_file, output_file):
    """
    This method discards lines that contain errors.
    Using this method will result in a csv that can work, but it does not utilize all available data.
    :param input_file: input file directory & name
    :param output_file:  output file directory & name
    :return: total lines in the original csv and total lines in the output csv
    """
    total_lines = 0
    valid_lines = 0
    valid_rows = []

    with open(input_file, "r", encoding="utf-8") as infile:
        header = infile.readline().strip()
        valid_rows.append(header)
        total_lines += 1
        for line in infile:
            total_lines += 1
            try:
                pd.read_csv(io.StringIO(line), header=None)
                valid_rows.append(line.strip())
                valid_lines += 1
            except pd.errors.ParserError:
                print(line)
                continue

    with open(output_file, "w", encoding="utf-8") as outfile:
        for row in valid_rows:
            outfile.write(row + "\n")

    return total_lines, valid_lines


def main(root, set_type, part):
    input_file = os.path.join(root, f"metadata/results_{set_type}/part{part}.csv")
    output_file = os.path.join(root, f"metadata/results_{set_type}/part{part}_fixed.csv")

    if not os.path.exists(input_file):
        print(f"Error: The file {input_file} does not exist.")
        return

    total_lines, valid_lines = filter_valid_lines(input_file, output_file)
    percentage_kept = (valid_lines / total_lines) * 100 if total_lines > 0 else 0

    print(f"Original CSV entries: {total_lines}")
    print(f"Filtered CSV entries: {valid_lines}")
    print(f"Percentage of entries kept: {percentage_kept:.2f}%")
    print(f"Filtered CSV saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a CSV file to ensure proper formatting.")
    parser.add_argument("--root", type=str, default="./webvid-10m", help="Root directory path")
    parser.add_argument(
        "--set", type=str, required=True, choices=["2M_train", "2M_val", "10M_train", "10M_val"], help="Dataset type"
    )
    parser.add_argument("--part", type=int, required=True, help="Part number of the CSV file")

    args = parser.parse_args()
    main(args.root, args.set, args.part)
