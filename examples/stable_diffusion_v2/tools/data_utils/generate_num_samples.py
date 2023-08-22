import argparse
import ast
import json
import os

"""
Usage:
Assuming that all check_num_samples_part{part}.json are in the path ./json_list
1. Generate num_samples_part.json for part1, part2, part3:
python generate_num_samples.py --part_list "[1, 2, 3]" --json_path ./json_list
2. Generate num_samples_part.json for all 64 parts:
python generate_num_samples.py --all_part True --json_path ./json_list
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part_list",
        required=False,
        type=str,
        help="A list of part IDs indicating complete data. " 'For example, input "[1,2,3]" for parts 1, 2, and 3.',
    )
    parser.add_argument(
        "--all_part", default="False", type=str, help="Generate num_samples.json for all 64 parts (True/False)"
    )
    parser.add_argument("--json_path", required=True, type=str, help="Path to the folder containing JSON files")
    args = parser.parse_args()

    complete_part = None
    if args.part_list is not None:
        complete_part = ast.literal_eval(args.part_list)
    if args.all_part == "True":
        complete_part = [i for i in range(1, 65)]
    if complete_part is None:
        raise ValueError("--part_list is not set and --all_part is False ")

    num_samples_json = {}
    for part in complete_part:
        file_name = f"check_num_samples_part{part}.json"
        curr_json = json.load(open(os.path.join(args.json_path, file_name)))
        num_samples_json[part] = {}
        for key, value in curr_json.items():
            if key != "cnt":
                tar_number = int(key.split("/")[-1].split(".")[0])
                num_samples_json[part][tar_number] = value
    outputFile = open("num_samples_part.json", "w")
    json.dump(num_samples_json, outputFile, indent=2)

    print(f"Finished writing num_samples_part.json in path {os.getcwd()}")
