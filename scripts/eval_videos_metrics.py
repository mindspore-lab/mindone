import argparse
import os

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor

from mindone.metrics import ClipScoreFrame, ClipScoreText
from mindone.transformers import CLIPModel

VIDEO_EXTENSIONS = {".mp4"}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="openai/clip-vit-base-patch32/",
    help="the name of a (Open/)CLIP model as shown in HuggingFace." "Default: openai/clip-vit-base-patch32/",
)
parser.add_argument("--video_data_dir", type=str, default=None, help="path to data folder." "Default: None")
parser.add_argument("--video_caption_path", type=str, default=None, help="path to video caption path." "Default: None")
parser.add_argument("--metric", type=str, default="clip_score_text", choices=["clip_score_text", "clip_score_frame"])
args = parser.parse_args()

assert args.video_data_dir is not None

model = CLIPModel.from_pretrained(args.model_name)
processor = CLIPProcessor.from_pretrained(args.model_name)
clip_score_text = ClipScoreText(model, processor)
clip_score_frame = ClipScoreFrame(model, processor)

scores = []
df = pd.read_csv(args.video_caption_path)
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    video_name = row["video"]
    edited_prompt = row["caption"]
    if os.path.splitext(video_name)[1] in VIDEO_EXTENSIONS:
        video_path = f"{args.video_data_dir}/{video_name}"
    else:
        print(f"Not support format: {video_name}. ")
        continue
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(video_path)
    frames = []
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
            index += 1
        else:
            break
    cap.release()
    frames = [i.resize((224, 224)) for i in frames]
    if args.metric == "clip_score_text":
        score = clip_score_text.score(frames, edited_prompt)
        scores.append(score)
    elif args.metric == "clip_score_frame":
        score = clip_score_frame.score(frames)
        scores.append(score)
    else:
        raise NotImplementedError(args.metric)

print("{}: {}".format(args.metric, sum(scores) / len(scores)))
