from abc import ABC

from diffusion import create_diffusion

import mindspore as ms
from mindspore import ops

__all__ = ["InferPipeline"]


class InferPipeline(ABC):
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
        num_inference_steps=50,
        ddim_sampling=True,
    ):
        super().__init__()
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
        b, c, h, w = x.shape

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
        b, f, c, h, w = x.shape
        x = x.reshape((b * f, c, h, w))

        y = self.vae_decode(x)
        _, h, w, c = y.shape
        y = y.reshape((b, f, h, w, c))

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

    def __call__(self, inputs):
        """
        args:
            inputs: dict

        return:
            images (b H W 3)
        """
        z, y = self.data_prepare(inputs)
        if self.use_cfg:
            model_kwargs = dict(y=y, cfg_scale=self.guidance_rescale)
            latents = self.sampling_func(
                self.model.construct_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
            )
            latents, _ = latents.chunk(2, axis=0)
        else:
            model_kwargs = dict(y=y)
            latents = self.sampling_func(
                self.model.construct, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
            )
        if latents.dim() == 4:
            # latents: (b c h w)
            images = self.vae_decode(latents)
        else:
            # latents: (b f c h w)
            images = self.vae_decode_video(latents)
            # output (b, f, h, w, 3)
        return images
