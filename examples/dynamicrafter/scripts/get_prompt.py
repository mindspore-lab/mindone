import csv
from pathlib import Path

csv_file = Path("/root/lhy/data/mixkit-100videos/video_caption_train.csv")
output_file = Path("/root/lhy/data/mixkit-100videos_dc_infer/text_prompts_train.txt")

with open(csv_file, "r") as f:
    data = list(csv.DictReader(f))

with open(output_file, "w") as f:
    for d in data:
        f.write(d["caption"] + "\n")   
