import csv
import os
from pathlib import Path


video_dir = Path("/root/lhy/data/mixkit-100videos/mixkit")
csv_file = Path("/root/lhy/data/mixkit-100videos/video_caption_train.csv")
output_dir = Path("/root/lhy/data/mixkit-100videos_dc_infer")

with open(csv_file, "r") as f:
    data = list(csv.DictReader(f))

for d in data:
    video_path = video_dir / d['video']
    output_file = (output_dir / d['video']).with_suffix('.png')
    ffmpeg_cmd = f"ffmpeg -i {video_path} -ss 1 -f image2 {output_file}"
    # ffmpeg_cmd = f"ffmpeg -threads 8 -i {video_path} {frames_dir}/%d.jpg > /dev/null 2>&1"
    os.popen(ffmpeg_cmd)