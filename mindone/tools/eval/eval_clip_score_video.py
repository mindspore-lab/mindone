# eval clip score for video
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from mindspore import Tensor

from examples.stable_diffusion_v2.tools._common.clip import CLIPImageProcessor, CLIPModel, CLIPTokenizer, parse

try:
    import torch

    is_torch_available = True
except ImportError:
    is_torch_available = False

try:
    from transformers import CLIPModel as CLIPModelPT
    from transformers import CLIPProcessor as CLIPProcessorPT

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


VIDEO_EXTENSIONS = {".mp4", ".gif"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="examples/stable_diffusion_v2/tools/_common/clip/configs/clip_vit_l_14.yaml",
        help="YAML config files for ms backend",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-large-patch14/",
        help="the name of a (Open/)CLIP model as shown in HuggingFace for pt backend."
        "Default: openai/clip-vit-large-patch14/",
    )
    parser.add_argument("--video_data_dir", type=str, default=None, help="path to data folder." "Default: None")
    parser.add_argument(
        "--video_caption_path", type=str, default=None, help="path to video caption path." "Default: None"
    )
    parser.add_argument(
        "--metric", type=str, default="clip_score_text", choices=["clip_score_text", "clip_score_frame"]
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ms",
        choices=["pt", "ms"],
        help="backend to do CLIP model inference for CLIP score compute." "Default: ms",
    )
    parser.add_argument("--ckpt_path", type=str, default=None, help="Load model checkpoint." "Default: None")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="examples/stable_diffusion_v2/ldm/models/clip/bpe_simple_vocab_16e6.txt.gz",
        help="Load tokenizer.",
    )
    args = parser.parse_args()

    assert args.video_data_dir is not None
    print(f"Backend: {args.backend}")
    if args.backend == "pt":
        assert is_torch_available is True, "torch is not installed, please install torch"
        assert is_transformers_available is True, "transformers is not installed, please install transformers"
        model = CLIPModelPT.from_pretrained(args.model_name)
        processor = CLIPProcessorPT.from_pretrained(args.model_name)
    elif args.backend == "ms":
        config = parse(args.config, args.ckpt_path)
        model = CLIPModel(config)
        processor = CLIPImageProcessor()
        text_processor = CLIPTokenizer(args.tokenizer_path, pad_token="!")

        def process_text(text):
            return Tensor(text_processor(text, padding="max_length", max_length=77)["input_ids"]).reshape(1, -1)

    else:
        raise NotImplementedError(args.backend)

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

        if args.backend == "pt":
            if args.metric == "clip_score_text":
                inputs = processor(text=[edited_prompt], images=frames, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)

                logits_per_image = outputs.logits_per_image.detach().cpu().numpy()
                score = logits_per_image.mean()
                scores.append(score)
            elif args.metric == "clip_score_frame":
                inputs = processor(images=frames, return_tensors="pt")
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs).detach().cpu().numpy()
                cosine_sim_matrix = cosine_similarity(image_features)
                np.fill_diagonal(cosine_sim_matrix, 0)  # set diagonal elements to 0
                score = cosine_sim_matrix.sum() / (len(frames) * (len(frames) - 1))
                scores.append(score)
            else:
                raise NotImplementedError(args.metric)
        elif args.backend == "ms":
            if args.metric == "clip_score_text":
                frames = processor(frames)
                texts = process_text(edited_prompt)
                logits_per_image, _ = model(image=frames, text=texts)
                score = logits_per_image.mean()
                scores.append(score)
            elif args.metric == "clip_score_frame":
                frames = processor(frames)
                image_features = model.get_image_features(frames)
                cosine_sim_matrix = cosine_similarity(image_features)
                np.fill_diagonal(cosine_sim_matrix, 0)  # set diagonal elements to 0
                score = cosine_sim_matrix.sum() / (len(frames) * (len(frames) - 1))
                scores.append(score)
            else:
                raise NotImplementedError(args.metric)

    print("{}: {}".format(args.metric, sum(scores) / len(scores)))
