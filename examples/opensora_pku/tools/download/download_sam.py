import os
import subprocess

import pandas as pd
import tqdm

# download text file from https://ai.meta.com/datasets/segment-anything-downloads/
text_file = "D://PKUopensora//Open-Sora-Plan-v1.1.0//SA-1B//sa-1b-links.txt"
local_folder = "D://PKUopensora//Open-Sora-Plan-v1.1.0//SA-1B"
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
