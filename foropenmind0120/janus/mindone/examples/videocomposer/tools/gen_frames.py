import glob
import os
import time

from tqdm import tqdm

root_dir = "/data1/webvid-10m/dataset"
set_name = "2M_val"
input_dir = os.path.join(root_dir, set_name)

start = time.time()

for part_dir in glob.glob(os.path.join(input_dir, "part*")):
    tars = glob.glob(os.path.join(part_dir, "*.tar"))
    for tar in tars:
        print("*" * 40)
        print(f"Extracting tar file: '{tar}'")
        print("Size:{:.2f} GB".format(os.path.getsize(tar) / (1024**3)))
        video_dir = os.path.splitext(tar)[0]
        os.makedirs(video_dir, exist_ok=True)
        tar_cmd = f"tar -xf {tar} -C {video_dir}"
        os.popen(tar_cmd)
        # subprocess.run(tar_cmd, shell=True)

        print(f"\nExtracting frames of videos in {video_dir}")
        for video in tqdm(glob.glob(os.path.join(video_dir, "*.mp4"))):
            frames_dir = os.path.splitext(video)[0]
            os.makedirs(frames_dir, exist_ok=True)
            ffmpeg_cmd = f"ffmpeg -threads 8 -i {video} {frames_dir}/%d.jpg > /dev/null 2>&1"
            os.popen(ffmpeg_cmd)

end = time.time()
print(f"time: {end-start}s")
