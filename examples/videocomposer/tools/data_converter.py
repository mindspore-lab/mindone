import glob
import os
import sys

from tqdm import tqdm

data_dir = sys.argv[1]
output_data_path = os.path.join(data_dir, "video_caption.csv")

with open(output_data_path, "w") as f:
    f.write("video,caption\n")  # column names
    count = 0
    for video_path in tqdm(glob.glob(os.path.join(data_dir, "*.mp4"))):
        caption_path = os.path.splitext(video_path)[0] + ".txt"
        if not os.path.isfile(caption_path):
            print(f"WARNING: Skip video '{video_path}' because its text label file doesn't exist in '{data_dir}'.")
            continue
        read_caption_cmd = f"cat {caption_path}"
        caption_text = os.popen(read_caption_cmd).read()
        caption_text = (
            '"' + caption_text.replace('"', "'") + '"'
        )  # There is comma(,) in the caption text, which cause bug when loading csv file. Replace " by ', and then add "" to avoid ambiguity.

        video_name = os.path.basename(video_path)
        f.write(f"{video_name},{caption_text}\n")
        count += 1

print(f"INFO: The annotation file for **{count}** videos is generated sucessfully as '{output_data_path}'.")
