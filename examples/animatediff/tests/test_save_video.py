import glob
import os
import sys

import numpy as np
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
from animate_diff.utils.util import save_videos

img_dir = os.path.join(__dir__, "../../videocomposer/datasets/webvid5")
print(img_dir)
img_fps = glob.glob(img_dir + "/*.png")

frames = []
for img_fp in img_fps:
    img = np.asarray(Image.open(img_fp)) / 255.0
    frames.append(img)

frames = np.array([frames])
print(frames.shape)

save_videos(frames, "./tmp.gif")
