import logging

import mindspore as ms
from mindspore import Tensor, _no_grad
from mindspore import dtype as mstype
from mindspore import jit_class, mint, nn, ops, tensor
from mindspore.communication import get_rank

from ..acceleration import get_sequence_parallel_group
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
        self._pynative = ms.get_context("mode") == ms.PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)


class DiffusionWithLoss(nn.Cell):
    """A training pipeline for a diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        diffusion: (object): A class for Gaussian Diffusion.
        scale_factor (float): scale_factor for vae.
        condition (str): The type of conditions of model in [None, 'text', 'class'].
            If it is None, model is a un-conditional video generator.
            If it is 'text', model accepts text embeddings (B, T, N) as conditions, and generates videos.
            If it is 'class', model accepts class labels (B, ) as conditions, and generates videos.
        text_encoder (nn.Cell): A text encoding model which accepts token ids and returns text embeddings in shape (T, D).
            T is the number of tokens, and D is the embedding dimension.
        cond_stage_trainable (bool): whether to train the text encoder.
        train_with_embed (bool): whether to train with embeddings (no need vae and text encoder to extract latent features and text embeddings)
    """

    def __init__(
        self,
        network: nn.Cell,
        vae: nn.Cell = None,
        scale_factor: float = 0.18215,
        guidance: float = 4.0,
        sigma_min: float = 1e-5,
    ):
        super().__init__()
        self.network = network
        self.vae = vae
        self.scale_factor = scale_factor
        self._sigma_min = sigma_min
        self._guidance = tensor(guidance, dtype=self.network.dtype)
        self.loss = nn.MSELoss()

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

    def get_latents(self, x):
        """
        x: (b c t h w)
        """
        z = ops.stop_gradient(self.vae.encode(x))
        return z

    def construct(
        self, x: Tensor, img_ids: Tensor, text_embed: Tensor, txt_ids: Tensor, y_vec: Tensor, shift_alpha: Tensor
    ) -> Tensor:
        with no_grad():
            if self.vae is not None:
                x = self.get_latents(x)

        loss = self.compute_loss(x, img_ids, text_embed, txt_ids, y_vec, shift_alpha)

        return loss

    def compute_loss(
        self, x: Tensor, img_ids: Tensor, text_embed: Tensor, txt_ids: Tensor, y_vec: Tensor, shift_alpha: Tensor
    ) -> Tensor:
        # TODO: prepare_visual_condition

        t = mint.sigmoid(mint.randn((x.shape[0])))
        t = time_shift(shift_alpha, t)
        t = self._broadcast(t)

        noise = self._broadcast(mint.randn_like(x))

        t_rev = 1 - t
        x_t = t_rev[:, None, None] * x + (1 - (1 - self._sigma_min) * t_rev[:, None, None]) * noise

        model_pred = self.network(
            x_t.to(self.network.dtype),
            img_ids=img_ids.to(self.network.dtype),
            txt=text_embed.to(self.network.dtype),
            txt_ids=txt_ids.to(self.network.dtype),
            timesteps=t.to(self.network.dtype),
            y_vec=y_vec.to(self.network.dtype),
            guidance=self._guidance.repeat(x_t.shape[0]),
        )
        v_t = (1 - self._sigma_min) * noise - x

        return self.loss(model_pred.to(mstype.float32), v_t.to(mstype.float32))
