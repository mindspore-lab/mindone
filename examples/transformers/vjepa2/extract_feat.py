# V-JEPA 2 is intended to represent any video (and image) to perform video classification, retrieval, or as a video encoder for VLMs.

# pip install -U git+https://github.com/huggingface/transformers
from torchcodec.decoders import VideoDecoder
import numpy as np
import time

import mindspore as ms
from mindone.transformers.models.vjepa2 import VJEPA2Model
from transformers import AutoVideoProcessor # use master version>4.52.4

start_time = time.time()
model_repo = "facebook/vjepa2-vitl-fpc64-256"
model = VJEPA2Model.from_pretrained(model_repo).set_train(False)
processor = AutoVideoProcessor.from_pretrained(model_repo)
print("Loaded model and processor, time elapse: %.4fs"%(time.time()-start_time))

# Video feature extraction #
print("*"*50)
print("Video feature extraction")

# To load a video, sample the number of frames according to the model. For this model, we use 64.
video_file = "path/to/video/file.mp4"
vr = VideoDecoder(video_file)
frame_idx = np.arange(0, 64) # choosing some frames. here, you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
video = processor(video, return_tensors="np")
video["pixel_values_videos"] = ms.Tensor(video["pixel_values_videos"])
print("input pixel_values_videos:", video["pixel_values_videos"].shape)
video_embeddings = model.get_vision_features(**video)

print("output video embeddings:", video_embeddings.shape)


# To load an image, simply copy the image to the desired number of frames.
# Image feature extraction #
print("*"*50)
print("Image feature extraction")
from transformers.image_utils import load_image

# image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
image = load_image("path/to/image/file")
pixel_values = processor(image, return_tensors="pt").to(model.device)["pixel_values_videos"]
pixel_values = pixel_values.repeat(1, 16, 1, 1, 1) # repeating image 16 times

print("input pixel_values:", pixel_values.shape)
image_embeddings = model.get_vision_features(pixel_values)    

print("output image embeddings:", image_embeddings.shape)