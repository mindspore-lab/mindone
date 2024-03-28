import glob
import os
import sys

import numpy as np
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.visualize.videos import export_to_gif

img_dir = os.path.join(__dir__, "../../videocomposer/datasets/webvid5")
print(img_dir)
img_fps = glob.glob(img_dir + "/*.png")

frames = [np.asarray(Image.open(img_fp)) for img_fp in img_fps]
frames = np.array([frames])
print(frames.shape)

export_to_gif(frames, "./tmp.gif", loop=0)
