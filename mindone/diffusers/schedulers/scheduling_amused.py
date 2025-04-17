import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mindspore as ms
from mindspore import mint

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.mindspore_utils import dtype_to_max
from .scheduling_utils import SchedulerMixin


def gumbel_noise(t, generator=None):
    noise = mint.zeros_like(t).uniform_(0, 1, generator=generator)
    return -mint.log((-mint.log(noise.clamp(1e-20))).clamp(1e-20))


def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    confidence = mint.log(probs.clamp(1e-20)) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = mint.sort(confidence, dim=-1)[0]
    cut_off = mint.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking


@dataclass
class AmusedSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`ms.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`ms.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: ms.Tensor
    pred_original_sample: ms.Tensor = None


class AmusedScheduler(SchedulerMixin, ConfigMixin):
    order = 1

    temperatures: ms.Tensor

    @register_to_config
    def __init__(
        self,
        mask_token_id: int,
        masking_schedule: str = "cosine",
    ):
        self.temperatures = None
        self.timesteps = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
    ):
        self.timesteps = mint.arange(num_inference_steps).flip((0,))

        if isinstance(temperature, (tuple, list)):
            self.temperatures = mint.linspace(temperature[0], temperature[1], num_inference_steps)
        else:
            self.temperatures = mint.linspace(temperature, 0.01, num_inference_steps)

    def step(
        self,
        model_output: ms.Tensor,
        timestep: ms.int64,  # type: ignore
        sample: ms.Tensor,
        starting_mask_ratio: int = 1,
        generator: Optional[ms.Generator] = None,
        return_dict: bool = False,
    ) -> Union[AmusedSchedulerOutput, Tuple]:
        two_dim_input = sample.ndim == 3 and model_output.ndim == 4

        if two_dim_input:
            batch_size, codebook_size, height, width = model_output.shape
            sample = sample.reshape(batch_size, height * width)
            model_output = model_output.reshape(batch_size, codebook_size, height * width).permute(0, 2, 1)

        unknown_map = sample == self.config.mask_token_id

        probs = model_output.softmax(axis=-1)

        probs_ = probs
        probs_ = probs_.reshape(-1, probs.shape[-1])
        pred_original_sample = mint.multinomial(probs_, 1, generator=generator)
        pred_original_sample = pred_original_sample[:, 0].view(probs.shape[:-1])
        pred_original_sample = mint.where(unknown_map, pred_original_sample, sample)

        if timestep == 0:
            prev_sample = pred_original_sample
        else:
            seq_len = sample.shape[1]
            step_idx = (self.timesteps == timestep).nonzero()
            ratio = (step_idx + 1) / len(self.timesteps)

            if self.config.masking_schedule == "cosine":
                mask_ratio = mint.cos(ratio * math.pi / 2)
            elif self.config.masking_schedule == "linear":
                mask_ratio = 1 - ratio
            else:
                raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

            mask_ratio = starting_mask_ratio * mask_ratio

            mask_len = (seq_len * mask_ratio).floor()
            # do not mask more than amount previously masked
            mask_len = mint.min(unknown_map.sum(axis=-1, keepdims=True) - 1, mask_len)
            # mask at least one
            mask_len = mint.max(ms.tensor([1]), mask_len)

            selected_probs = mint.gather(probs, -1, pred_original_sample[:, :, None])[:, :, 0]
            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = mint.where(unknown_map, selected_probs, dtype_to_max(selected_probs.dtype))

            masking = mask_by_random_topk(mask_len, selected_probs, self.temperatures[step_idx], generator)

            # Masks tokens with lower confidence.
            prev_sample = mint.where(masking, self.config.mask_token_id, pred_original_sample)

        if two_dim_input:
            prev_sample = prev_sample.reshape(batch_size, height, width)
            pred_original_sample = pred_original_sample.reshape(batch_size, height, width)

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return AmusedSchedulerOutput(prev_sample, pred_original_sample)

    def add_noise(self, sample, timesteps, generator=None):
        step_idx = (self.timesteps == timesteps).nonzero()
        ratio = (step_idx + 1) / len(self.timesteps)

        if self.config.masking_schedule == "cosine":
            mask_ratio = mint.cos(ratio * math.pi / 2)
        elif self.config.masking_schedule == "linear":
            mask_ratio = 1 - ratio
        else:
            raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

        mask_indices = mint.rand(sample.shape, generator=generator) < mask_ratio

        masked_sample = sample.copy()

        masked_sample[mask_indices] = self.config.mask_token_id

        return masked_sample
