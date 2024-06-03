from typing import Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import Tensor, ops

from mindone.models.modules.pos_embed import get_2d_sincos_pos_embed

from ..schedulers.iddpm import create_diffusion

__all__ = ["InferPipeline"]


class InferPipeline:
    """An Inference pipeline for diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scale_factor (float): scale_factor for vae.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
        num_inference_steps: (int): The number of denoising steps.
        ddim_sampling: (bool): whether to use DDIM sampling. If False, will use DDPM sampling.
    """

    def __init__(
        self,
        model,
        vae,
        text_encoder=None,
        scale_factor=1.0,
        guidance_rescale=1.0,
        guidance_channels: Optional[int] = None,
        num_inference_steps=50,
        ddim_sampling=True,
        micro_batch_size=None,
    ):
        super().__init__()
        self.model = model

        self.vae = vae
        self.micro_batch_size = micro_batch_size
        self.scale_factor = scale_factor
        self.guidance_rescale = guidance_rescale
        self.guidance_channels = guidance_channels
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
    def vae_encode(self, x: Tensor) -> Tensor:
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents

    def vae_decode(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        b, c, h, w = x.shape

        if self.micro_batch_size is None:
            y = self.vae.decode(x / self.scale_factor)
        else:
            bs = self.micro_batch_size
            y_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                y_bs = self.vae.decode(x_bs / self.scale_factor)
                y_out.append(y_bs)
            y = ops.concat(y_out, axis=0)

        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    def vae_decode_video(self, x):
        """
        Args:
            x: (b c t h w), denoised latent
        Return:
            y: (b f H W 3), batch of images, normalized to [0, 1]
        """
        y = []
        for x_sample in x:
            # c t h w -> t c h w
            x_sample = x_sample.permute(1, 0, 2, 3)
            y.append(self.vae_decode(x_sample))
        y = ops.stack(y, axis=0)

        return y

    def data_prepare(self, inputs):
        x = inputs["noise"]
        mask = inputs.get("mask", None)

        if inputs["text_emb"] is None:
            text_tokens = inputs["text_tokens"]
            text_emb = self.get_condition_embeddings(text_tokens, **{"mask": mask}).to(ms.float32)
        else:
            text_emb = inputs["text_emb"]
            b, max_tokens, d = text_emb.shape

        # torch use y_embedding genearted during stdit training,
        # for token/text drop in caption embedder for condition-free guidance training. The null mask is the same as text mask.
        n = x.shape[0]
        # (n_tokens, dim_emb) -> (b n_tokens dim_emb)
        null_emb = self.model.y_embedder.y_embedding[None, :, :].repeat(n, axis=0)

        if self.use_cfg:
            y = ops.cat([text_emb, null_emb], axis=0)
            x_in = ops.concat([x] * 2, axis=0)
            assert y.shape[0] == x_in.shape[0], "shape mismatch!"
            inputs["mask"] = ops.concat([mask, mask], axis=0)
        else:
            x_in = x
            y = text_emb

        # to match stdit input format
        y = ops.expand_dims(y, axis=1)
        return x_in, y

    def get_condition_embeddings(self, text_tokens, **kwargs):
        # text conditions inputs for cross-attention
        text_emb = ops.stop_gradient(self.text_encoder(text_tokens, **kwargs))

        return text_emb

    def __call__(
        self,
        inputs: dict,
        frames_mask: Optional[Tensor] = None,
        additional_kwargs: Optional[dict] = None,
    ) -> Tuple[Union[Tensor, None], Tensor]:
        """
        args:
            inputs: dict

        return:
            images (b H W 3)
        """
        z, y = self.data_prepare(inputs)

        mask = inputs.get("mask", None)
        model_kwargs = dict(y=y)
        if mask is not None:
            model_kwargs["mask"] = mask

        if additional_kwargs is not None:
            model_kwargs.update(additional_kwargs)

        if self.use_cfg:
            model_kwargs.update({"cfg_scale": self.guidance_rescale, "cfg_channel": self.guidance_channels})
            latents = self.sampling_func(
                self.model.construct_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                frames_mask=frames_mask,
            )
            latents, _ = latents.chunk(2, axis=0)
        else:
            latents = self.sampling_func(
                self.model,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                frames_mask=frames_mask,
            )

        if self.vae is not None:
            if latents.dim() == 4:
                images = self.vae_decode(latents)
            else:
                # latents: (b c t h w)
                # out: (b T H W C)
                images = self.vae_decode_video(latents)
            return images, latents
        else:
            return None, latents


class InferPipelineFiTLike(InferPipeline):
    def __init__(
        self,
        *args,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        max_image_size: int = 512,
        embed_dim: int = 1152,
        vae_downsample_rate: float = 8.0,
        in_channels: int = 4,
        input_sq_size: int = 512,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.p = patch_size
        self.c = in_channels

        self.vae_downsample_rate = vae_downsample_rate
        self.embed_dim = embed_dim
        self.input_sq_size = input_sq_size

        self.max_size = int(max_image_size / self.vae_downsample_rate)
        self.max_length = int(self.max_size**2 / self.p[1] / self.p[2])

    def _patchify(self, x: Tensor) -> Tensor:
        # N, C, F, H, W -> N, F, T, D
        assert self.p[0] == 1
        n, _, f, h, w = x.shape
        nh, nw = h // self.p[1], w // self.p[2]
        x = ops.reshape(x, (n, self.c, f, nh, self.p[1], nw, self.p[2]))
        x = ops.transpose(x, (0, 2, 3, 5, 1, 4, 6))
        x = ops.reshape(x, (n, f, nh * nw, self.c * self.p[1] * self.p[2]))
        return x

    def _unpatchify(self, x: Tensor, nh: int, nw: int) -> Tensor:
        # N, F, T, D -> N, C, F, H, W
        assert self.p[0] == 1
        n, f, _, _ = x.shape
        x = ops.reshape(x, (n, f, nh, nw, self.c, self.p[1], self.p[2]))
        x = ops.transpose(x, (0, 4, 1, 2, 5, 3, 6))
        x = ops.reshape(x, (n, self.c, f, nh * self.p[1], nw * self.p[2]))
        return x

    def _pad_latent(self, x: Tensor) -> Tensor:
        # N, C, F, H, W -> N, C, F, max_size, max_size
        n, _, f, _, _ = x.shape
        nh, nw = self.max_size // self.p[1], self.max_size // self.p[2]

        x_fill = self._patchify(x)
        if x_fill.shape[2] > self.max_length:
            return x

        x = ops.zeros((n, f, self.max_length, self.c * self.p[1] * self.p[2]), dtype=x.dtype)
        x[:, :, : x_fill.shape[2]] = x_fill
        x = self._unpatchify(x, nh, nw)
        return x

    def _unpad_latent(self, x: Tensor, valid_t: int, h: int, w: int) -> Tensor:
        # N, C, F, max_size, max_size -> N, C, F, H, W
        nh, nw = h // self.p[1], w // self.p[2]
        x = self._patchify(x)
        x = x[:, :, :valid_t]
        x = self._unpatchify(x, nh, nw)
        return x

    def _get_dynamic_size(self, h: int, w: int) -> Tuple[int, int]:
        if h % self.p[1] != 0:
            h += self.p[1] - h % self.p[1]
        if w % self.p[2] != 0:
            w += self.p[2] - w % self.p[2]

        h = h // self.p[1]
        w = w // self.p[2]
        return h, w

    def _create_pos_embed(self, h: int, w: int) -> Tuple[Tensor, int]:
        rs = (h * w * self.vae_downsample_rate**2) ** 0.5
        ph, pw = self._get_dynamic_size(h, w)
        scale = rs / self.input_sq_size
        base_size = round((ph * pw) ** 0.5)

        # 1, T, D
        nh, nw = h // self.p[1], w // self.p[2]
        pos_embed_fill = get_2d_sincos_pos_embed(self.embed_dim, nh, nw, scale=scale, base_size=base_size)

        if pos_embed_fill.shape[0] > self.max_length:
            pos_embed = pos_embed_fill
        else:
            pos_embed = np.zeros((self.max_length, self.embed_dim), dtype=np.float32)
            pos_embed[: pos_embed_fill.shape[0]] = pos_embed_fill

        pos_embed = pos_embed[None, ...]
        pos_embed = Tensor(pos_embed, dtype=ms.float32)
        return pos_embed, pos_embed_fill.shape[0]

    def _create_mask(self, valid_t: int, n: int) -> Tensor:
        # 1, T
        if valid_t > self.max_length:
            mask = np.ones((valid_t,), dtype=np.bool_)
        else:
            mask = np.zeros((self.max_length,), dtype=np.bool_)
            mask[:valid_t] = True
        mask = np.tile(mask[None, ...], (n, 1))
        mask = Tensor(mask, dtype=ms.uint8)
        return mask

    def __call__(
        self,
        inputs: dict,
        frames_mask: Optional[Tensor] = None,
        additional_kwargs: Optional[dict] = None,
    ) -> Tuple[Union[Tensor, None], Tensor]:
        """
        args:
            inputs: dict

        return:
            images (b H W 3)
        """
        z, y = self.data_prepare(inputs)
        # b c t h w
        n, c, _, h, w = z.shape
        assert self.c == c

        z = self._pad_latent(z)
        pos_emb, valid_t = self._create_pos_embed(h, w)
        latent_mask = self._create_mask(valid_t, n)

        mask = inputs.get("mask", None)
        model_kwargs = dict(y=y)
        if mask is not None:
            model_kwargs["mask"] = mask

        if additional_kwargs is not None:
            model_kwargs.update(additional_kwargs)

        model_kwargs["pos_emb"] = pos_emb
        model_kwargs["latent_mask"] = latent_mask

        if self.use_cfg:
            model_kwargs.update({"cfg_scale": self.guidance_rescale, "cfg_channel": self.guidance_channels})
            latents = self.sampling_func(
                self.model.construct_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                frames_mask=frames_mask,
            )
            latents, _ = latents.chunk(2, axis=0)
        else:
            latents = self.sampling_func(
                self.model,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                frames_mask=frames_mask,
            )

        latents = self._unpad_latent(latents, valid_t, h, w)

        if self.vae is not None:
            if latents.dim() == 4:
                images = self.vae_decode(latents)
            else:
                # latents: (b c t h w)
                # out: (b T H W C)
                images = self.vae_decode_video(latents)
            return images, latents
        else:
            return None, latents
