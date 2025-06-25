# V-JEPA 2 is intended to represent any video (and image) to perform video classification, retrieval, or as a video encoder for VLMs.

# pip install -U git+https://github.com/huggingface/transformers
import time

import numpy as np
from transformers import VJEPA2VideoProcessor  # use master version=4.53.0.dev0

import mindspore as ms
import mindspore.dataset.vision as vision

from mindone.transformers.image_utils import load_image
from mindone.transformers.models.vjepa2 import VJEPA2Model

start_time = time.time()
model_repo = "facebook/vjepa2-vitl-fpc64-256"
model = VJEPA2Model.from_pretrained(model_repo).set_train(False)
processor = VJEPA2VideoProcessor.from_pretrained(model_repo)
print("Loaded model and processor, time elapse: %.4fs" % (time.time() - start_time))

# Video feature extraction #
print("*" * 50)
print("Video feature extraction")

# To load a video, sample the number of frames according to the model. For this model, we use 64.
video_file = "path/to/video/file"
frame_idx = np.arange(0, 64)  # choosing some frames. here, you can define more complex sampling strategy
raw_ndarray = np.fromfile(video_file, np.uint8)
video, audio = vision.DecodeVideo()(raw_ndarray)  # np array in [T, H, W, C] uint8, [C, L] fp32
video = video.transpose(0, 3, 1, 2)
video = video[frame_idx, :, :, :]  # [T, C, H, W] uint8
video = processor(video, return_tensors="pt")
video["pixel_values_videos"] = ms.Tensor(video["pixel_values_videos"].cpu().numpy())

print("input pixel_values_videos:", video["pixel_values_videos"].shape)  # (1, 64, 3, 256, 256)
video_embeddings = model.get_vision_features(**video)
print("output video embeddings:", video_embeddings.shape)  # (1, 8192, 1024)


# To load an image, simply copy the image to the desired number of frames.
# Image feature extraction #
print("*" * 50)
print("Image feature extraction")

# image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
image = load_image("path/to/image/file")
print("input image size:", image.size)
pixel_values = processor(image, return_tensors="pt")["pixel_values_videos"]
pixel_values = ms.Tensor(pixel_values.cpu().numpy())
pixel_values = pixel_values.tile((1, 16, 1, 1, 1))  # repeating image 16 times

print("input pixel_values:", pixel_values.shape)  # (1, 16, 3, 256, 256) fp32
image_embeddings = model.get_vision_features(pixel_values)
print("output image embeddings:", image_embeddings.shape)  # (1, 2084, 1024)
