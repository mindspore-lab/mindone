from diffusion import create_diffusion

import mindspore as ms
from mindspore import Tensor, ops


class DiTInferPipeline:
    """

    Args:
        dit (nn.Cell): A `DiT` to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scale_factor (float): scale_factor for vae.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
        num_inference_steps: (int): The number of denoising steps.
    """

    def __init__(
        self,
        dit,
        vae,
        text_encoder=None,
        scale_factor=1.0,
        guidance_rescale=0.0,
        num_inference_steps=50,
        ddim_sampling=True,
    ):
        super().__init__()
        self.dit = dit
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

    def __call__(self, inputs):
        """
        args:
            inputs: dict

        return:
            images (b H W 3)
        """
        z, y = self.data_prepare(inputs)
        model_kwargs = dict(y=y, cfg_scale=Tensor(self.guidance_rescale, dtype=ms.float32))
        latents = self.sampling_func(
            self.dit.construct_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
        )
        latents, _ = latents.chunk(2, axis=0)
        assert latents.dim() == 4, f"Expect to have 4-dim latents, but got {latents.shape}"

        images = self.vae_decode(latents)

        return images
