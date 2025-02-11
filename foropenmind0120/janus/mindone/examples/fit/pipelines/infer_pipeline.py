from typing import Any, Dict, Optional, Tuple

import numpy as np
from diffusion import create_diffusion

import mindspore as ms
from mindspore import Tensor, nn, ops

from mindone.models.modules import get_2d_sincos_pos_embed, precompute_freqs_cis_2d


class FiTInferPipeline:
    """

    Args:
        fit (nn.Cell): A `FiT` to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scale_factor (float): scale_factor for vae.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
        num_inference_steps: (int): The number of denoising steps.
    """

    def __init__(
        self,
        fit: nn.Cell,
        vae: nn.Cell,
        text_encoder: Optional[nn.Cell] = None,
        scale_factor: float = 1.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: int = 50,
        ddim_sampling: bool = True,
        model_config: Dict[str, Any] = {},
    ):
        super().__init__()
        self.model_config = model_config
        self.fit = fit
        self.vae = vae
        self.scale_factor = scale_factor
        self.guidance_rescale = guidance_rescale
        self.text_encoder = text_encoder
        self.diffusion = create_diffusion(str(num_inference_steps))
        if ddim_sampling:
            self.sampling_func = self.diffusion.ddim_sample_loop
        else:
            self.sampling_func = self.diffusion.p_sample_loop

    @ms.jit
    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents.astype(ms.float16)

    @ms.jit
    def vae_decode(self, x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    def data_prepare(self, inputs):
        x = inputs["noise"]
        y = ops.cat([inputs["y"], inputs["y_null"]], axis=0)
        x_in = ops.concat([x] * 2, axis=0)
        assert y.shape[0] == x_in.shape[0], "shape mismatch!"
        return x_in, y

    def _patchify(self, x: Tensor, p: int) -> Tensor:
        # N, C, H, W -> N, T, D
        n, c, h, w = x.shape
        nh, nw = h // p, w // p
        x = ops.reshape(x, (n, c, nh, p, nw, p))
        x = ops.transpose(x, (0, 2, 4, 3, 5, 1))
        x = ops.reshape(x, (n, nh * nw, p * p * c))
        return x

    def _unpatchify(self, x: Tensor, nh: int, nw: int, p: int, c: int) -> Tensor:
        # N, T, D -> N, C, H, W
        n, _, _ = x.shape
        x = ops.reshape(x, (n, nh, nw, p, p, c))
        x = ops.transpose(x, (0, 5, 1, 3, 2, 4))
        x = ops.reshape(x, (n, c, nh * p, nw * p))
        return x

    def _pad_latent(self, x: Tensor, p: int, max_size: int, max_length: int) -> Tensor:
        # N, C, H, W -> N, C, max_size, max_size
        n, c, _, _ = x.shape
        nh, nw = max_size // p, max_size // p

        x_fill = self._patchify(x, p)
        if x_fill.shape[1] > max_length:
            return x
        x = ops.zeros((n, max_length, p * p * c), dtype=x.dtype)
        x[:, : x_fill.shape[1]] = x_fill
        x = self._unpatchify(x, nh, nw, p, c)
        return x

    def _unpad_latent(self, x: Tensor, valid_t: int, h: int, w: int, p: int) -> Tensor:
        # N, C, max_size, max_size -> N, C, H, W
        _, c, _, _ = x.shape
        nh, nw = h // p, w // p
        x = self._patchify(x, p)
        x = x[:, :valid_t]
        x = self._unpatchify(x, nh, nw, p, c)
        return x

    def _create_pos_embed(
        self, h: int, w: int, p: int, max_length: int, embed_dim: int, method: str = "rotate"
    ) -> Tuple[Tensor, int]:
        # 1, T, D
        nh, nw = h // p, w // p
        if method == "rotate":
            pos_embed_fill = precompute_freqs_cis_2d(embed_dim, nh, nw, max_length=max_length)
        else:
            pos_embed_fill = get_2d_sincos_pos_embed(embed_dim, nh, nw)

        if pos_embed_fill.shape[0] > max_length:
            pos_embed = pos_embed_fill
        else:
            pos_embed = np.zeros((max_length, embed_dim), dtype=np.float32)
            pos_embed[: pos_embed_fill.shape[0]] = pos_embed_fill

        pos_embed = pos_embed[None, ...]
        pos_embed = Tensor(pos_embed)
        return pos_embed, pos_embed_fill.shape[0]

    def _create_mask(self, valid_t: int, max_length: int, n: int) -> Tensor:
        # 1, T
        if valid_t > max_length:
            mask = np.ones((valid_t,), dtype=np.bool_)
        else:
            mask = np.zeros((max_length,), dtype=np.bool_)
            mask[:valid_t] = True
        mask = np.tile(mask[None, ...], (n, 1))
        mask = Tensor(mask)
        return mask

    def __call__(self, inputs):
        """
        args:
            inputs: dict

        return:
            images (b H W 3)
        """
        p = self.model_config["patch_size"]
        max_size = self.model_config["max_size"]
        max_length = self.model_config["max_length"]
        embed_dim = self.model_config["embed_dim"]
        embed_method = self.model_config["embed_method"]

        z, y = self.data_prepare(inputs)
        n, _, h, w = z.shape

        z = self._pad_latent(z, p, max_size, max_length)
        pos, valid_t = self._create_pos_embed(h, w, p, max_length, embed_dim, method=embed_method)
        mask = self._create_mask(valid_t, max_length, n)

        model_kwargs = dict(y=y, pos=pos, mask=mask, cfg_scale=Tensor(self.guidance_rescale, dtype=ms.float32))
        latents = self.sampling_func(
            self.fit.construct_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
        )
        latents, _ = latents.chunk(2, axis=0)
        latents = self._unpad_latent(latents, valid_t, h, w, p)
        assert latents.dim() == 4, f"Expect to have 4-dim latents, but got {latents.shape}"

        images = self.vae_decode(latents)

        return images
