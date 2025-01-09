import logging
from math import ceil
from typing import Literal, Optional, Tuple

import numpy as np
from tqdm import tqdm

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import mint, nn, ops
from mindspore.communication import get_rank

from ..acceleration import get_sequence_parallel_group
from ..models import LlamaModel

logger = logging.getLogger(__name__)

__all__ = ["RFLOW", "RFlowLossWrapper", "RFlowEvalLoss"]


class LogisticNormal(nn.Cell):
    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.stdnormal = ops.StandardNormal()
        self.mean = loc
        self.std = scale
        self._min = Tensor(np.finfo(np.float32).tiny, dtype=mstype.float32)
        self._max = Tensor(1.0 - np.finfo(np.float32).eps, dtype=mstype.float32)

    def construct(self, shape: Tuple[int]) -> Tensor:
        x = self.mean + self.std * self.stdnormal(shape)
        return ops.clamp(ops.sigmoid(x), self._min, self._max)


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

        self.broadcast = None
        if (sp_group := get_sequence_parallel_group()) is not None:
            logging.info(
                f"Broadcasting all random variables from rank (0) to current rank ({get_rank(sp_group)}) in group `{sp_group}`."
            )
            self.broadcast = ops.Broadcast(0, group=sp_group)

    def _discrete_sample(self, size: int) -> Tensor:
        return ops.randint(0, self.num_timesteps, (size,), dtype=mstype.int64)

    def _uniform_sample(self, size: int) -> Tensor:
        return mint.rand((size,), dtype=mstype.float32) * self.num_timesteps

    def _logit_normal_sample(self, size: int) -> Tensor:
        return self.distribution((size,)) * self.num_timesteps

    def _broadcast(self, x: Tensor) -> Tensor:
        if self.broadcast is None:
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
        x = x.to(mstype.float32)

        if timestep is None:
            timestep = self._broadcast(self._sample_func(x.shape[0]))

        noise = self._broadcast(ops.randn_like(x))
        x_t = self.add_noise(x, noise, timestep)

        model_output = self.model(
            x_t.to(self.model.dtype),
            timestep,
            ul2_emb.to(self.model.dtype),
            metaclip_emb.to(self.model.dtype),
            byt5_emb.to(self.model.dtype),
        ).to(mstype.float32)
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
        timesteps = 1 - timesteps.to(mstype.float32) / self.num_timesteps
        timesteps = timesteps[:, None, None, None, None]

        # 3.1.2 First Eqa.
        return timesteps * x + (1 - (1 - self.eps) * timesteps) * noise  # TODO: check for zero SNR


class RFlowEvalLoss(nn.Cell):
    def __init__(self, network: RFlowLossWrapper, num_sampling_steps: int = 10):
        super().__init__()
        self.network = network
        self.timesteps = Tensor(
            np.linspace(0, network.num_timesteps, num_sampling_steps + 2)[1:-1].reshape(-1, 1), dtype=mstype.float32
        )

    def construct(self, x: Tensor, ul2_emb: Tensor, metaclip_emb: Tensor, byt5_emb: Tensor, **kwargs) -> Tensor:
        loss = Tensor(0, dtype=mstype.float32)
        timesteps = mint.tile(self.timesteps, (1, x.shape[0]))
        for t in timesteps:
            loss += self.network(x, ul2_emb, metaclip_emb, byt5_emb, t)

        return loss / len(self.timesteps)
