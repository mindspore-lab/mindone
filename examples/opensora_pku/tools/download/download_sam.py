import argparse
import os
import subprocess

import pandas as pd
import tqdm

# download text file from https://ai.meta.com/datasets/segment-anything-downloads/
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_text", type=str, required=True, help="The input text file. ")
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    required=True,
    help="The local directory where the downloaded datasets will be saved.",
)
args = parser.parse_args()

text_file = args.input_text
local_folder = args.output_dir

if not os.path.exists(local_folder):
    os.makedirs(local_folder)
df = pd.read_csv(text_file, sep="\t")

non_success = 0
for index, item in tqdm.tqdm(df.iterrows(), total=len(df)):
    file_name = item["file_name"]
    cdn_link = item["cdn_link"]
    target_path = os.path.join(local_folder, file_name)
    # cmd = f'curl --continue-at - -L -o {target_path} "{cdn_link}"'
    # os.system(cmd)

    try:
        subprocess.run(["curl", "--continue-at", "-", "-L", "-o", f"{target_path}", f"{cdn_link}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        non_success += 1
    print(f"Try to download {index+1} files, {non_success} files were failed.")
