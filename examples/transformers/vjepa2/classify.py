# pip install -U git+https://github.com/huggingface/transformers
import time

import numpy as np
from transformers import VJEPA2VideoProcessor  # use master version=4.53.0.dev0

import mindspore as ms
import mindspore.dataset.vision as vision

from mindone.transformers.models.vjepa2 import VJEPA2ForVideoClassification

# Load model and video preprocessor
start_time = time.time()
model_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"
model = VJEPA2ForVideoClassification.from_pretrained(model_repo).set_train(False)
processor = VJEPA2VideoProcessor.from_pretrained(model_repo)
print("Loaded model and processor, time elapse: %.4fs" % (time.time() - start_time))

# To load a video, sample the number of frames according to the model.
# video_file = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
video_file = "path/to/video/file"
frame_idx = np.arange(0, model.config.frames_per_clip, 8)  # you can define more complex sampling strategy
raw_ndarray = np.fromfile(video_file, np.uint8)
video, audio = vision.DecodeVideo()(raw_ndarray)  # np array in [T, H, W, C] uint8, [C, L] fp32
video = video.transpose(0, 3, 1, 2)
video = video[frame_idx, :, :, :]  # [T, C, H, W] uint8
video = processor(video, return_tensors="pt")
video["pixel_values_videos"] = ms.Tensor(video["pixel_values_videos"].cpu().numpy())

# Preprocess and run inference
inputs = processor(video, return_tensors="pt")
print("Finished precessing video input")
for k, v in inputs.items():
    inputs[k] = ms.Tensor(v)

outputs = model(**inputs)
logits = outputs.logits

print("Top 5 predicted class names:")
top5_indices = logits.topk(5)[1][0]
top5_probs = ms.mint.softmax(logits, dim=-1).topk(5)[0][0]
for idx, prob in zip(top5_indices, top5_probs):
    text_label = model.config.id2label[idx.item()]
    print(f" - {text_label}: {prob.item():.2f}")
