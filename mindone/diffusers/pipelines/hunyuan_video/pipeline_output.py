"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/hunyuan_video/pipeline_output.py."""

from dataclasses import dataclass

import mindspore as ms

from mindone.diffusers.utils import BaseOutput


@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    r"""
    Output class for HunyuanVideo pipelines.

    Args:
        frames (`ms.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: ms.Tensor
