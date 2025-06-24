# pip install -U git+https://github.com/huggingface/transformers
from torchcodec.decoders import VideoDecoder
import numpy as np
import time

import mindspore as ms
from mindone.transformers.models.vjepa2 import VJEPA2ForVideoClassification
from transformers import AutoVideoProcessor # use master version>4.52.4


# Load model and video preprocessor
start_time = time.time()
model_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"
model = VJEPA2ForVideoClassification.from_pretrained(model_repo).set_train(False)
processor = AutoVideoProcessor.from_pretrained(model_repo)
print("Loaded model and processor, time elapse: %.4fs"%(time.time()-start_time))

# To load a video, sample the number of frames according to the model.
# video_file = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
video_file = "path/to/video/file"
vr = VideoDecoder(video_file)
frame_idx = np.arange(0, model.config.frames_per_clip, 8) # you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # frames x channels x height x width

# Preprocess and run inference
inputs = processor(video, return_tensors="np")
print("Finished precessing video input")
print("inputs")
for k,v in inputs.items():
    inputs[k] = ms.Tensor(v)

outputs = model(**inputs)
logits = outputs.logits

print("Top 5 predicted class names:")
top5_indices = logits.topk(5).indices[0]
top5_probs = ms.mint.softmax(logits, dim=-1).topk(5).values[0]
for idx, prob in zip(top5_indices, top5_probs):
    text_label = model.config.id2label[idx.item()]
    print(f" - {text_label}: {prob:.2f}")
