from typing import Optional

from gm.models.diffusion import DiffusionEngine

import mindspore as ms
from mindspore import ops


class VideoDiffusionEngine(DiffusionEngine):
    def __init__(self, *args, en_and_decode_n_samples_a_time: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def decode_first_stage(self, z):
        if self.disable_first_stage_amp:
            self.first_stage_model.to_float(ms.float32)
            z = z.astype(ms.float32)

        z = 1.0 / self.scale_factor * z

        n_samples = min(z.shape[0], self.en_and_decode_n_samples_a_time)
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

        n_samples = self.en_and_decode_n_samples_a_time
        all_out = [self.first_stage_model.encode(x[n : n + n_samples]) for n in range(0, x.shape[0], n_samples)]

        z = ops.cat(all_out, axis=0)
        z = self.scale_factor * z
        return z
