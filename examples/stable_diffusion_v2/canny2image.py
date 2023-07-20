import config

import cv2
import einops
# import gradio as gr
import numpy as np
# import torch
import mindspore as ms
import mindspore.ops as ops
import random

# from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_model
from cldm.ddim_hacked import DDIMSampler


apply_canny = CannyDetector()

# create model
model = create_model('/home/mindspore/congw/project/mindone/examples/stable_diffusion_v2/models/cldm_v15.yaml')
# load_model(model, './models/control_sd15_canny.pth')