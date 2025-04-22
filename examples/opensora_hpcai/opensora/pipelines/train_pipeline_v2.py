import logging
from dataclasses import asdict
from typing import Optional, Union

from mindspore import PYNATIVE_MODE, Tensor, _no_grad
from mindspore import dtype as mstype
from mindspore import get_context, jit_class, mint, nn, ops, tensor
from mindspore.communication import get_rank

from ..acceleration import get_sequence_parallel_group
from ..utils.training import Condition, prepare_visual_condition_causal, prepare_visual_condition_uncausal
from .utils_v2 import time_shift

__all__ = ["DiffusionWithLoss"]

logger = logging.getLogger(__name__)


@jit_class
class no_grad(_no_grad):
    """
    A context manager that suppresses gradient memory allocation in PyNative mode.
    """

    def __init__(self):
        super().__init__()
        self._pynative = get_context("mode") == PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)


class DiffusionWithLoss(nn.Cell):
    """A training pipeline for a diffusion model

    Args:
        network (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
    """

    def __init__(
        self,
        network: nn.Cell,
        vae: Optional[nn.Cell] = None,
        is_causal_vae: bool = True,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        guidance: float = 4.0,
        sigma_min: float = 1e-5,
        condition_config: Optional[Condition] = None,
    ):
        super().__init__()
        self.network = network
        self.vae = vae
        self._prep_vc = prepare_visual_condition_causal if is_causal_vae else prepare_visual_condition_uncausal
        self._patch_size = patch_size
        self._sigma_min = sigma_min
        self._guidance = tensor(guidance, dtype=self.network.dtype)
        self._condition_config = asdict(condition_config) if condition_config is not None else None
        self.loss = mint.nn.MSELoss()

        self.broadcast = None
        if (sp_group := get_sequence_parallel_group()) is not None:
            logging.info(
                f"Broadcasting all random variables from rank (0) to current rank ({get_rank(sp_group)}) in group `{sp_group}`."
            )
            self.broadcast = ops.Broadcast(0, group=sp_group)

    def set_train(self, mode=True):
        # Set the diffusion model only to train or eval mode
        self.network.set_train(mode)

    def _broadcast(self, x: Tensor) -> Tensor:
        if self.broadcast is None:
            return x
        return self.broadcast((x,))[0]

    def _pack(self, x: Tensor) -> Tensor:
        # b c t (h ph) (w pw) -> b (t h w) (c ph pw)
        b, c, t, h, w = x.shape
        ph, pw = self._patch_size[1], self._patch_size[2]
        x = mint.reshape(x, (b, c, t, h // ph, ph, w // pw, pw))
        x = mint.permute(x, (0, 2, 3, 5, 1, 4, 6))
        return mint.reshape(x, (b, -1, c * ph * pw))

    def get_latents(self, x) -> tuple[Tensor, Union[Tensor, None]]:
        """
        x: (b c t h w)
        """
        x = x.to(self.vae.dtype)
        z = self.vae.encode(x)
        cond = None
        if self._condition_config is not None:
            cond = self._prep_vc(x, out_shape=z.shape, condition_config=self._condition_config.copy(), ae=self.vae)
            cond = self._pack(cond)
        return self._pack(z), cond

    def construct(
        self, x: Tensor, img_ids: Tensor, text_embed: Tensor, txt_ids: Tensor, y_vec: Tensor, shift_alpha: Tensor
    ) -> Tensor:
        cond = None
        if self.vae is not None:
            with no_grad():  # Pynative
                x, cond = ops.stop_gradient(self.get_latents(x))  # Graph

        loss = self.compute_loss(x, img_ids, text_embed, txt_ids, y_vec, shift_alpha, cond)

        return loss

    def compute_loss(
        self,
        x: Tensor,
        img_ids: Tensor,
        text_embed: Tensor,
        txt_ids: Tensor,
        y_vec: Tensor,
        shift_alpha: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        t = mint.sigmoid(ops.randn(x.shape[0]))
        t = time_shift(shift_alpha, t)
        t = self._broadcast(t)

        noise = self._broadcast(ops.randn_like(x))

        t_rev = 1 - t
        x_t = t_rev[:, None, None] * x + (1 - (1 - self._sigma_min) * t_rev[:, None, None]) * noise

        if self._condition_config is not None:  # for faster graph building
            cond = cond.to(self.network.dtype)
        model_pred = self.network(
            x_t.to(self.network.dtype),
            img_ids=img_ids.to(self.network.dtype),
            txt=text_embed.to(self.network.dtype),
            txt_ids=txt_ids.to(self.network.dtype),
            timesteps=t.to(self.network.dtype),
            y_vec=y_vec.to(self.network.dtype),
            cond=cond,
            guidance=mint.tile(self._guidance, (x_t.shape[0],)),
        )
        v_t = (1 - self._sigma_min) * noise - x

        return self.loss(model_pred.to(mstype.float32), v_t.to(mstype.float32))
