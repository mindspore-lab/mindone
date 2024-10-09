import os

import cv2
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import CLIPProcessor

from mindspore import Tensor, ops

from mindone.transformers import CLIPModel

VIDEO_EXTENSIONS = {".mp4"}


class ClipScore:
    def __init__(self, model_name="openai/clip-vit-base-patch32/"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.fill_diagonal = ops.FillDiagonal(0.0)

    def frame_consistency(self, frames):
        inputs = self.processor(images=frames)
        inputs = {k: Tensor(v) for k, v in inputs.items()}
        image_features = self.model.get_image_features(**inputs).asnumpy()
        cosine_sim_matrix = cosine_similarity(image_features)
        cosine_sim_matrix = self.fill_diagonal(Tensor(cosine_sim_matrix))  # set diagonal elements to 0
        score = cosine_sim_matrix.sum() / (len(frames) * (len(frames) - 1))
        return score

    def textual_alignment(self, frames, prompt):
        inputs = self.processor(text=[prompt], images=frames)
        inputs = {k: Tensor(v) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        score = outputs[0].asnumpy().mean()

        return score

    def calucation_score(self, video_folder, csv_path, metric="clip_score_text"):
        scores = []
        df = pd.read_csv(csv_path)
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            video_name = row["video"]
            edited_prompt = row["caption"]
            if os.path.splitext(video_name)[1] in VIDEO_EXTENSIONS:
                video_path = f"{video_folder}/{video_name}"
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
            if metric == "clip_score_text":
                score = self.textual_alignment(frames, edited_prompt)
                scores.append(score)
            elif metric == "clip_score_frame":
                score = self.frame_consistency(frames)
                scores.append(score)
            else:
                raise NotImplementedError(metric)

        return sum(scores) / len(scores)
