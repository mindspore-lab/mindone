# -*- coding: utf-8 -*-

import os
import os.path as osp

# debug use, TODO: delete later
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, ".")))

from PIL import Image
# from transformers import AutoModel, AutoImageProcessor
from emu3.tokenizer import Emu3VisionVQModel, Emu3VisionVQImageProcessor
import mindspore as ms
from mindspore import Tensor, _no_grad

@jit_class
class no_grad(_no_grad):
    """
    A context manager that suppresses gradient memory allocation in PyNative mode.
    """

    def __init__(self):
        super().__init__()
        self._pynative = ms.get_context("mode") == ms.PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)

MODEL_HUB = "BAAI/Emu3-VisionTokenizer"

model = Emu3VisionVQModel.from_pretrained(MODEL_HUB).set_train(False)
processor = Emu3VisionVQImageProcessor.from_pretrained(MODEL_HUB)
# AutoModel: "BAAI/Emu3-VisionTokenizer--image_processing_emu3visionvq.Emu3VisionVQModel
# AutoImageProcessor: "BAAI/Emu3-VisionTokenizer--image_processing_emu3visionvq.Emu3VisionVQImageProcessor"

# TODO: you need to modify the path here
# VIDEO_FRAMES_PATH = "YOUR_VIDEO_FRAMES_PATH"
VIDEO_FRAMES_PATH = "assets/video_pd"

video = os.listdir(VIDEO_FRAMES_PATH)
video.sort()
video = [Image.open(osp.join(VIDEO_FRAMES_PATH, v)) for v in video]

images = processor(video, return_tensors="np")["pixel_values"]
images = Tensor(images.unsqueeze(0)) # [1, Frames, C, H, W]

# image autoencode
image = images[:, 0]
print(image.shape)
with no_grad():
    # encode
    codes = model.encode(image)
    # decode
    recon = model.decode(codes)

recon = recon.view(-1, *recon.shape[2:])
recon_image = processor.postprocess(recon)["pixel_values"][0]
recon_image.save("recon_image.png")

# video autoencode
if images.shape[1] % model.config.temporal_downsample_factor !=0:
    images = images[:, images.shape[1] // model.config.temporal_downsample_factor * model.config.temporal_downsample_factor]
if images.shape[1] > 4:
    images = images[:, : 4 // model.config.temporal_downsample_factor * model.config.temporal_downsample_factor]
images = images.view(
    -1,
    model.config.temporal_downsample_factor,
    *images.shape[2:],
)
print(images.shape)
with no_grad():
    # encode
    codes = model.encode(images)
    # decode
    recon = model.decode(codes)

recon = recon.view(-1, *recon.shape[2:])
recon_images = processor.postprocess(recon)["pixel_values"]
for idx, im in enumerate(recon_images):
    im.save(f"recon_video_{idx}.png")
