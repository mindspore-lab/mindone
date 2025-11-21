# -*- coding: utf-8 -*-
# Adapted from https://github.com/baaivision/Emu3 to work with MindSpore.

import os
import os.path as osp
import time

# from transformers import AutoModel, AutoImageProcessor
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image

import mindspore as ms
from mindspore import Tensor, mint, ops

from mindone.diffusers.training_utils import pynative_no_grad as no_grad
from mindone.utils.amp import auto_mixed_precision

# Both modes are supported
ms.set_context(mode=ms.PYNATIVE_MODE)  # PYNATIVE
# ms.set_context(mode=ms.GRAPH_MODE)      # GRAPH

start_time = time.time()

# NOTE: you need to modify the path of MODEL_HUB here
MODEL_HUB = "BAAI/Emu3-VisionTokenizer"
MS_DTYPE = ms.bfloat16  # float16 fail to reconstruct
model = Emu3VisionVQModel.from_pretrained(MODEL_HUB, use_safetensors=True, mindspore_dtype=MS_DTYPE).set_train(False)
model = auto_mixed_precision(
    model, amp_level="O2", dtype=MS_DTYPE, custom_fp32_cells=[mint.nn.BatchNorm3d]
)  # NOTE: nn.Conv3d used float16
processor = Emu3VisionVQImageProcessor.from_pretrained(MODEL_HUB)
# Same as using AutoModel/AutoImageProcessor:
# AutoModel: "BAAI/Emu3-VisionTokenizer--image_processing_emu3visionvq.Emu3VisionVQModel
# AutoImageProcessor: "BAAI/Emu3-VisionTokenizer--image_processing_emu3visionvq.Emu3VisionVQImageProcessor"
print("Load model ==> Time elapsed: %.4fs" % (time.time() - start_time))

# NOTE: you need to modify the path of VIDEO_FRAMES_PATH here
VIDEO_FRAMES_PATH = "YOUR_VIDEO_FRAMES_PATH"

video = os.listdir(VIDEO_FRAMES_PATH)
video.sort()
video = [Image.open(osp.join(VIDEO_FRAMES_PATH, v)) for v in video]

images = processor(video, return_tensors="np")["pixel_values"]
images = Tensor(images).unsqueeze(0)  # [1, Frames, C, H, W]

# single image autoencode #
image = images[:, 0].to(MS_DTYPE)
start_time = time.time()
with no_grad():
    # encode
    codes = ops.stop_gradient(model.encode(image))
    # decode
    recon = ops.stop_gradient(model.decode(codes))

print("Infer an image reconstruction ==> Time elapsed: %.4fs" % (time.time() - start_time))
recon = recon.view(-1, *recon.shape[2:])
recon_image = processor.postprocess(recon)["pixel_values"][0]
recon_image.save("recon_image.png")
print("Saved image recon_image.png")

# video frames autoencode #
# NOTE: number of frames must be multiple of `model.config.temporal_downsample_factor`, i.e. 4
# For example, reconstruct first 4 frames (i.e., a clip):
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
