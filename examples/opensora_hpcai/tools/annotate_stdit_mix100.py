import glob
import json
import math
import os
import random

root_dir = "/home_host/ddd/workspace/datasets/mixkit-100videos/mixkit/"
annot_fp = "/home_host/ddd/workspace/datasets/mixkit-100videos/anno_jsons/video_mixkit_65f_54735.json"
output_csv = "video_caption.csv"

# read paths
video_fps = sorted(glob.glob(os.path.join(root_dir, "*/*.mp4")))
# remove header
video_fps = [fp.replace(root_dir, "") for fp in video_fps]
print(video_fps)

fp_out = open(output_csv, "w")
fp_out.write("video,caption\n")

matched_videos = []
matched_captions = []
with open(annot_fp, "r") as fp:
    annot_list = json.load(fp)
    for i, annot in enumerate(annot_list):
        video_path = annot["path"]
        if video_path in video_fps and video_path not in matched_videos:
            caption = annot["cap"]
            fp_out.write('{},"{}"\n'.format(video_path, caption))
            matched_videos.append(video_path)
            matched_captions.append(caption)

fp_out.close()
print("Num samples", len(matched_videos))
print("csv saved in ", output_csv)

# split into train and test
train_ratio = 0.8
num_samples = len(matched_videos)
num_train = math.ceil(num_samples * train_ratio)
num_test = num_samples - num_train
vc_list = [(matched_videos[i], matched_captions[i]) for i in range(num_samples)]
random.shuffle(vc_list)

train_set = sorted(vc_list[:num_train])
test_set = sorted(vc_list[num_train:])


def write_csv(vcl, save_path):
    with open(save_path, "w") as fp:
        fp.write("video,caption\n")
        for vc in vcl:
            fp.write('{},"{}"\n'.format(vc[0], vc[1]))


write_csv(train_set, output_csv.replace(".csv", "_train.csv"))
write_csv(test_set, output_csv.replace(".csv", "_test.csv"))
