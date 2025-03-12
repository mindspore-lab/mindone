import logging
from math import ceil
from typing import Literal, Optional, Tuple

import numpy as np
from hyvideo.utils.parallel_states import get_sequence_parallel_state, hccl_info
from tqdm import tqdm

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import mint, nn, ops
from mindspore.communication import get_rank

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
        model,
        num_timesteps: int = 1000,
        sample_method: Literal["discrete-uniform", "uniform", "logit-normal"] = "logit-normal",
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__(auto_prefix=False)
        self.num_timesteps = num_timesteps

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
        self.criteria = nn.MSELoss(reduction="mean")
        self._timesteps = Tensor(np.linspace(1, num_timesteps, num_timesteps, dtype=np.float32))

        self.broadcast = None
        if get_sequence_parallel_state() and (sp_group := hccl_info.world_size) is not None:
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
        self,
        x: Tensor,
        text_states: Tensor,
        text_mask: Tensor,
        text_states_2: Tensor,
        guidance: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the training loss for the corresponding timestep.
        x: (N, T, C, H, W) tensor of inputs (latent representations of video)
        text_states: (N, L1, 4096) LLAMA text embeddings
        text_maskL (N, L1),
        text_states_2: (N, 768) CLIP text embeddings
        guidance: (N, ), the guidance scale for distillation
        timestep: (N,) tensor to indicate a denoising step
        """
        x = x.to(mstype.float32)

        if timestep is None:
            u = self._sample_func(x.shape[0]).to(mstype.int32)
            timestep = self._timesteps[u]

        timestep = self._broadcast(timestep)

        noise = self._broadcast(ops.randn_like(x))
        x_t = self.add_noise(x, noise, timestep)

        model_output = self.model(
            x_t,
            timestep,
            text_states=text_states,
            text_mask=text_mask,
            text_states_2=text_states_2,
            guidance=guidance,
            **kwargs,
        ).to(mstype.float32)
        v_t = noise - x

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
        return timesteps * x + (1 - timesteps) * noise  # TODO: check for zero SNR


class RFlowEvalLoss(nn.Cell):
    def __init__(self, model: RFlowLossWrapper, num_sampling_steps: int = 10):
        super().__init__()
        self.model = model
        self.timesteps = Tensor(
            np.linspace(0, model.num_timesteps, num_sampling_steps + 2)[1:-1].reshape(-1, 1), dtype=mstype.float32
        )

    def construct(
        self,
        x: Tensor,
        text_states: Tensor,
        text_mask: Tensor,
        text_states_2: Tensor,
        guidance: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        loss = Tensor(0, dtype=mstype.float32)
        timesteps = mint.tile(self.timesteps, (1, x.shape[0]))
        for t in timesteps:
            loss += self.model(
                x,
                timestep=timestep,
                text_states=text_states,
                text_mask=text_mask,
                text_states_2=text_states_2,
                guidance=guidance,
                **kwargs,
            )

        return loss / len(self.timesteps)
