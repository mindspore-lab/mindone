from typing import List, Optional

from gm.models.diffusion import DiffusionEngine
from gm.util.util import get_obj_from_str
from omegaconf import DictConfig

import mindspore as ms
from mindspore import nn, ops


def net_to_dtype(net: nn.Cell, dtype: ms.dtype, exclude_layers: Optional[List[nn.Cell]] = None):
    """
    Converts the data type of a neural network except for the layers specified in `filter_layers`.

    Args:
        net: The network to be converted.
        dtype: The data type to convert the neural network to.
        exclude_layers: A list of specific layers to exclude from the conversion. Default is None.
    """
    if net.cells():
        for cell in net.cells():
            net_to_dtype(cell, dtype, exclude_layers)
    else:
        if exclude_layers is None or type(net) not in exclude_layers:
            net.to_float(dtype)


class VideoDiffusionEngine(DiffusionEngine):
    def __init__(self, *args, en_and_decode_n_samples_a_time: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        self._force_fp16 = None
        if isinstance(self.disable_first_stage_amp, DictConfig):
            if "force_fp16" in self.disable_first_stage_amp:
                self._force_fp16 = [get_obj_from_str(item) for item in self.disable_first_stage_amp["force_fp16"]]
            self.disable_first_stage_amp = self.disable_first_stage_amp["enable"]

    def decode_first_stage(self, z):
        if self.disable_first_stage_amp:
            net_to_dtype(self.first_stage_model, ms.float32, exclude_layers=self._force_fp16)
            z = z.astype(ms.float32)

        z = 1.0 / self.scale_factor * z

        n_samples = self.en_and_decode_n_samples_a_time or z.shape[0]
        n_samples = min(z.shape[0], n_samples)
        if z.shape[0] % n_samples:
            raise ValueError("Total number of frames must be divisible by number of decoded frames at a time.")

        all_out = [
            self.first_stage_model.decode(z[n : n + n_samples], timesteps=n_samples)
            for n in range(0, z.shape[0], n_samples)
        ]
        return ops.cat(all_out, axis=0)

    def encode_first_stage(self, x):
        if self.disable_first_stage_amp:
            self.first_stage_model.to_float(ms.float32)
            x = x.astype(ms.float32)

        n_samples = self.en_and_decode_n_samples_a_time or x.shape[0]
        all_out = [self.first_stage_model.encode(x[n : n + n_samples]) for n in range(0, x.shape[0], n_samples)]

        z = ops.cat(all_out, axis=0)
        z = self.scale_factor * z
        return z
