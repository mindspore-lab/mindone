# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time

# debug use, TODO: delete later
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, ".")))

# from transformers import AutoModel, AutoImageProcessor
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image

import mindspore as ms
from mindspore import Tensor, _no_grad, jit_class, nn, ops

from mindone.utils.amp import auto_mixed_precision

# Both modes are supported
ms.set_context(mode=ms.PYNATIVE_MODE)  # PYNATIVE
# ms.set_context(mode=ms.GRAPH_MODE)      # GRAPH


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


start_time = time.time()

# TODO: you need to modify the path of MODEL_HUB here
MODEL_HUB = "BAAI/Emu3-VisionTokenizer"
MS_DTYPE = ms.bfloat16  # float16 fail to reconstruct
model = Emu3VisionVQModel.from_pretrained(MODEL_HUB, use_safetensors=True, mindspore_dtype=MS_DTYPE).set_train(False)
model = auto_mixed_precision(
    model, amp_level="O2", dtype=MS_DTYPE, custom_fp32_cells=[nn.BatchNorm3d]
)  # NOTE: nn.Conv3d used float16
processor = Emu3VisionVQImageProcessor.from_pretrained(MODEL_HUB)
# AutoModel: "BAAI/Emu3-VisionTokenizer--image_processing_emu3visionvq.Emu3VisionVQModel
# AutoImageProcessor: "BAAI/Emu3-VisionTokenizer--image_processing_emu3visionvq.Emu3VisionVQImageProcessor"
print("Load model ==> Time elapsed: %.4fs" % (time.time() - start_time))

# TODO: you need to modify the path of VIDEO_FRAMES_PATH here
VIDEO_FRAMES_PATH = "YOUR_VIDEO_FRAMES_PATH"
# VIDEO_FRAMES_PATH = "assets/video_pd"

video = os.listdir(VIDEO_FRAMES_PATH)
video.sort()
video = [Image.open(osp.join(VIDEO_FRAMES_PATH, v)) for v in video]

images = processor(video, return_tensors="np")["pixel_values"]
images = Tensor(images).unsqueeze(0)  # [1, Frames, C, H, W]

# image autoencode #


for i in range(images.shape[1]):
    image = images[:, i].to(MS_DTYPE)
    with no_grad():
        # encode
        codes = ops.stop_gradient(model.encode(image))
        # decode
        recon = ops.stop_gradient(model.decode(codes))
    if i == 0: # skip first sample
        start_time = time.time()

print(
    "Infer %d image reconstruction ==> Time elapsed: %.4fs/img"
    % (images.shape[1]-1, (time.time() - start_time) / (images.shape[1]-1))
)

recon = recon.view(-1, *recon.shape[2:])
recon_image = processor.postprocess(recon)["pixel_values"][0]
recon_image.save("recon_image.png")
print("Saved image recon_image.png")

# video autoencode #
# NOTE: number of frames must be multiple of `model.config.temporal_downsample_factor`
# This script OOM
# if images.shape[1] % model.config.temporal_downsample_factor !=0:
#     images = images[:, :images.shape[1] // model.config.temporal_downsample_factor * model.config.temporal_downsample_factor]

# only reconstruct first 4 frames (i.e., a clip):
if images.shape[1] > model.config.temporal_downsample_factor:
    images = images[:, : model.config.temporal_downsample_factor]
images = images.view(
    -1,
    model.config.temporal_downsample_factor,
    *images.shape[2:],
)
start_time = time.time()
with no_grad():
    # encode
    codes = ops.stop_gradient(model.encode(images))
    # decode
    recon = ops.stop_gradient(model.decode(codes))
print("Infer one clip of 4 frames ==> Time elapsed: %.4fs" % (time.time() - start_time))

recon = recon.view(-1, *recon.shape[2:])
recon_images = processor.postprocess(recon)["pixel_values"]
for idx, im in enumerate(recon_images):
    im.save(f"recon_video_{idx}.png")
print("Saved frames recon_video_IDX.png")
