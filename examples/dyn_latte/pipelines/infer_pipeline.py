from typing import Any, Dict, Optional, Tuple

import numpy as np
from diffusion import create_diffusion

import mindspore as ms
from mindspore import Tensor, nn, ops

from mindone.models.modules import get_2d_sincos_pos_embed, precompute_freqs_cis_2d


class DynLatteInferPipeline:
    """

    Args:
        model (nn.Cell): model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scale_factor (float): scale_factor for vae.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
        num_inference_steps: (int): The number of denoising steps.
    """

    def __init__(
        self,
        model: nn.Cell,
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
        self.model = model
        self.vae = vae
        self.scale_factor = scale_factor
        self.guidance_rescale = guidance_rescale
        if self.guidance_rescale > 1.0:
            self.use_cfg = True
        else:
            self.use_cfg = False

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

    def vae_decode_video(self, x):
        """
        Args:
            x: (b f c h w), denoised latent
        Return:
            y: (b f H W 3), batch of images, normalized to [0, 1]
        """
        y = []
        for x_sample in x:
            y.append(self.vae_decode(x_sample))
        y = ops.stack(y, axis=0)  # (b f H W 3)
        return y

    def data_prepare(self, inputs):
        x = inputs["noise"]
        if self.use_cfg:
            y = ops.cat([inputs["y"], inputs["y_null"]], axis=0)
            x_in = ops.concat([x] * 2, axis=0)
            assert y.shape[0] == x_in.shape[0], "shape mismatch!"
        else:
            x_in = x
            y = inputs["y"]
        return x_in, y

    def _patchify(self, x: Tensor, p: int) -> Tensor:
        # N, F, C, H, W -> N, T, D
        n, f, c, h, w = x.shape
        nh, nw = h // p, w // p
        x = ops.reshape(x, (n, f, c, nh, p, nw, p))
        x = ops.transpose(x, (0, 1, 3, 5, 4, 6, 2))
        x = ops.reshape(x, (n, f, nh * nw, p * p * c))
        return x

    def _unpatchify(self, x: Tensor, nh: int, nw: int, p: int, c: int) -> Tensor:
        # N, F, T, D -> N, F, C, H, W
        n, _, _, _ = x.shape
        x = ops.reshape(x, (n, -1, nh, nw, p, p, c))
        x = ops.transpose(x, (0, 1, 6, 2, 4, 3, 5))
        x = ops.reshape(x, (n, -1, c, nh * p, nw * p))
        return x

    def _pad_latent(self, x: Tensor, p: int, max_size: int, max_length: int, max_frames: int) -> Tensor:
        # N, F, C, H, W -> N, max_frame, C, max_size, max_size
        n, _, c, h, w = x.shape

        x_fill = self._patchify(x, p)
        if x_fill.shape[1] > max_frames and x_fill.shape[2] > max_length:
            return x

        if h * w > max_size * max_size:
            nh, nw = h // p, w // p
        else:
            nh, nw = max_size // p, max_size // p

        fill_frames = max(x_fill.shape[1], max_frames)
        fill_length = max(x_fill.shape[2], max_length)
        x = ops.zeros((n, fill_frames, fill_length, p * p * c), dtype=x.dtype)
        x[:, : x_fill.shape[1], : x_fill.shape[2]] = x_fill
        x = self._unpatchify(x, nh, nw, p, c)
        return x

    def _unpad_latent(self, x: Tensor, valid_t: int, valid_f: int, h: int, w: int, p: int) -> Tensor:
        # N, max_frame, C, max_size, max_size -> N, F, C, H, W
        _, _, c, _, _ = x.shape
        nh, nw = h // p, w // p
        x = self._patchify(x, p)
        x = x[:, :valid_f, :valid_t]
        x = self._unpatchify(x, nh, nw, p, c)
        return x

    def _create_pos_embed(
        self, h: int, w: int, p: int, max_length: int, embed_dim: int, method: str = "rotate"
    ) -> Tuple[Tensor, int]:
        # 1, T, D
        nh, nw = h // p, w // p
        if method == "rotate":
            # we use key-norm instead of NTK
            pos_embed_fill = precompute_freqs_cis_2d(embed_dim, nh, nw)
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

    def _create_mask(self, valid_t: int, max_length: int) -> Tensor:
        # 1, T
        if valid_t > max_length:
            mask = np.ones((valid_t,), dtype=np.bool_)
        else:
            mask = np.zeros((max_length,), dtype=np.bool_)
            mask[:valid_t] = True
        mask = mask[None, ...]
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
        max_frames = self.model_config["max_frames"]
        embed_dim = self.model_config["embed_dim"]
        embed_method = self.model_config["embed_method"]

        z, y = self.data_prepare(inputs)
        _, f, _, h, w = z.shape

        z = self._pad_latent(z, p, max_size, max_length, max_frames)
        pos, valid_t = self._create_pos_embed(h, w, p, max_length, embed_dim, method=embed_method)
        mask_s = self._create_mask(valid_t, max_length)
        mask_t = self._create_mask(f, max_frames)

        if self.use_cfg:
            model_kwargs = dict(y=y, pos=pos, mask_t=mask_t, mask_s=mask_s, cfg_scale=self.guidance_rescale)
            latents = self.sampling_func(
                self.model.construct_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
            )
            latents, _ = latents.chunk(2, axis=0)
        else:
            model_kwargs = dict(y=y, pos=pos, mask_t=mask_t, mask_s=mask_s)
            latents = self.sampling_func(
                self.model.construct, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
            )

        latents = self._unpad_latent(latents, valid_t, f, h, w, p)

        images = self.vae_decode_video(latents)

        return images
