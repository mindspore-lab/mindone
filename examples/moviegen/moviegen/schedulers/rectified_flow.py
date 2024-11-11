import logging
from math import ceil
from typing import Literal, Optional, Tuple

import numpy as np
from tqdm import tqdm

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import Tensor, mint, nn, ops
from mindspore.communication import get_rank

from ..models import LlamaModel
from ..parallel import get_model_parallel_group

logger = logging.getLogger(__name__)

__all__ = ["RFLOW", "RFlowLossWrapper"]


class LogisticNormal(nn.Cell):
    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        super().__init__()
        self.mean = loc
        self.std = scale
        self._min = Tensor(np.finfo(np.float32).tiny, dtype=ms.float32)
        self._max = Tensor(1.0 - np.finfo(np.float32).eps, dtype=ms.float32)

    def construct(self, shape: Tuple[int, ...]) -> Tensor:
        assert shape[-1] == 1
        x = mint.normal(mean=self.mean, std=self.std, size=shape)
        offset = x.shape[-1] + 1 - mint.cumsum(mint.ones(x.shape[-1]), dim=-1)
        z = self._clipped_sigmoid(x - mint.log(offset))
        z_cumprod = ops.cumprod((1 - z), dim=-1)
        y = F.pad(z, [0, 1], value=1) * F.pad(z_cumprod, [1, 0], value=1)
        return y[:, 0]

    def _clipped_sigmoid(self, x: Tensor) -> Tensor:
        x = mint.clamp(mint.sigmoid(x), min=self._min, max=self._max)
        return x


class RFLOW:
    def __init__(
        self,
        num_sampling_steps: int = 50,
        num_timesteps: int = 1000,
        sample_method: Literal["linear", "linear-quadratic"] = "linear",
    ) -> None:
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.sample_method = sample_method

    def __call__(self, model: nn.Cell, x: Tensor, ul2_emb: Tensor, metaclip_emb: Tensor, byt5_emb: Tensor) -> Tensor:
        """
        x: (N, T, C, H, W) tensor of inputs (latent representations of video)
        text_embedding: (N, L, C') tensor of the text embedding
        """
        # prepare timesteps
        if self.sample_method == "linear":
            timesteps = (1.0 - np.arange(self.num_sampling_steps) / self.num_sampling_steps) * self.num_timesteps
        else:
            first_half = ceil(self.num_sampling_steps / 2)
            second_half = self.num_sampling_steps - first_half  # in the case of an odd number of sampling steps
            linear = self.num_timesteps - np.arange(first_half)
            quadratic = (np.arange(1, second_half + 1) ** 2) / ((second_half + 1) ** 2)
            quadratic = (self.num_timesteps - (first_half - 1)) * quadratic + (first_half - 1)  # scale and shift
            quadratic = self.num_timesteps - quadratic
            timesteps = np.concatenate([linear, quadratic])

        timesteps = np.tile(timesteps[..., None], (1, x.shape[0]))
        timesteps = Tensor(timesteps, dtype=model.dtype)  # FIXME: avoid calculations on tensors outside `construct`

        for i, timestep in tqdm(enumerate(timesteps), total=self.num_sampling_steps):
            pred = model(x, timestep, ul2_emb, metaclip_emb, byt5_emb)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            x = x + pred * dt[:, None, None, None, None]

        return x


class RFlowLossWrapper(nn.Cell):
    """Wrapper for calculating the training loss"""

    def __init__(
        self,
        model: LlamaModel,
        num_timesteps: int = 1000,
        sample_method: Literal["discrete-uniform", "uniform", "logit-normal"] = "logit-normal",
        loc: float = 0.0,
        scale: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(auto_prefix=False)
        self.num_timesteps = num_timesteps
        self.eps = eps

        if sample_method == "discrete-uniform":
            self._sample_func = self._discrete_sample
        elif sample_method == "uniform":
            self._sample_func = self._uniform_sample
        elif sample_method == "logit-normal":
            self.distribution = LogisticNormal(loc=loc, scale=scale)
            self._sample_func = self._logit_normal_sample
        else:
            raise ValueError(f"Unknown sample method: {sample_method}")

        self.model = model
        self.criteria = nn.MSELoss()

        self.mp_group = get_model_parallel_group()
        if self.mp_group is not None:
            logging.info(
                f"Broadcasting all random variables from rank (0) to current rank ({get_rank(self.mp_group)}) in group `{self.mp_group}`."
            )
            self.broadcast = ops.Broadcast(0, group=self.mp_group)

    def _discrete_sample(self, size: int) -> Tensor:
        return ops.randint(0, self.num_timesteps, (size,), dtype=ms.int64)

    def _uniform_sample(self, size: int) -> Tensor:
        return mint.rand((size,), dtype=ms.float32) * self.num_timesteps

    def _logit_normal_sample(self, size: int) -> Tensor:
        return self.distribution((size, 1)) * self.num_timesteps

    def _broadcast(self, x: Tensor) -> Tensor:
        if self.mp_group is None:
            return x
        return self.broadcast((x,))[0]

    def construct(
        self, x: Tensor, ul2_emb: Tensor, metaclip_emb: Tensor, byt5_emb: Tensor, timestep: Optional[Tensor] = None
    ) -> Tensor:
        """
        Calculate the training loss for the corresponding timestep.
        x: (N, T, C, H, W) tensor of inputs (latent representations of video)
        ul2_emb: (N, L1, 4096) UL2 text embeddings
        metaclip_emb: (N, L2, 1280) MetaCLIP text embeddings
        byt5_emb: (N, L3, 1472) ByT5 text embeddings
        timestep: (N,) tensor to indicate a denoising step
        """
        x = x.to(ms.float32)

        if timestep is None:
            timestep = self._broadcast(self._sample_func(x.shape[0]))

        noise = self._broadcast(mint.normal(size=x.shape))
        x_t = self.add_noise(x, noise, timestep)

        model_output = self.model(
            x_t.to(self.model.dtype),
            timestep,
            ul2_emb.to(self.model.dtype),
            metaclip_emb.to(self.model.dtype),
            byt5_emb.to(self.model.dtype),
        ).to(ms.float32)
        v_t = x - (1 - self.eps) * noise

        # 3.1.2 Eqa (2)
        loss = self.criteria(model_output, v_t)
        return loss

    def add_noise(self, x: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """
        x: (N, T, C, H, W) tensor of ground truth
        noise: (N, T, C, H, W) tensor of white noise
        timesteps: (N,) tensor of timestamps with range [0, num_timesteps)
        """
        timesteps = 1 - timesteps.to(ms.float32) / self.num_timesteps
        timesteps = timesteps[:, None, None, None, None]

        # 3.1.2 First Eqa.
        return timesteps * x + (1 - (1 - self.eps) * timesteps) * noise
