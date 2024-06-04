from abc import ABC, abstractmethod

from tqdm import tqdm

import mindspore as ms
from mindspore import ops


class SDInfer(ABC):
    """
    Stable Diffusion inference.

    Args:
        text_encoder (nn.Cell): Frozen text-encoder.
        unet (nn.Cell): A `UNet2DConditionModel` to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scheduler (nn.Cell): A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor (float): scale_factor for vae.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
        num_inference_steps: (int): The number of denoising steps.
    """

    def __init__(
        self,
        text_encoder,
        unet,
        vae,
        scheduler,
        scale_factor=1.0,
        guidance_rescale=0.0,
        num_inference_steps=50,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.scale_factor = scale_factor
        self.guidance_rescale = guidance_rescale
        self.num_inference_steps = ms.Tensor(num_inference_steps, ms.int32)
        self.alphas_cumprod = scheduler.alphas_cumprod

    @ms.jit
    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents.astype(ms.float16)

    @ms.jit
    def vae_decode(self, x):
        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
        return y

    @ms.jit
    def prompt_embed(self, prompt_data, negative_prompt_data):
        pos_prompt_embeds = self.text_encoder(prompt_data)
        negative_prompt_embeds = self.text_encoder(negative_prompt_data)
        prompt_embeds = ops.concat([negative_prompt_embeds, pos_prompt_embeds], axis=0)
        return prompt_embeds

    @ms.jit
    def latents_add_noise(self, image_latents, noise, ts):
        latents = self.scheduler.add_noise(image_latents, noise, self.alphas_cumprod[ts])
        return latents

    @ms.jit
    def scale_model_input(self, latents, t):
        return self.scheduler.scale_model_input(latents, t)

    @ms.jit
    def predict_noise(self, x, t_continuous, c_crossattn, guidance_scale, c_concat=None):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        t_continuous = ops.tile(t_continuous.reshape(1), (x.shape[0],))
        x_in = ops.concat([x] * 2, axis=0)
        t_in = ops.concat([t_continuous] * 2, axis=0)
        if c_concat is not None:
            c_concat = ops.concat([c_concat] * 2, axis=0)
        noise_pred = self.unet(x_in, t_in, c_concat=c_concat, c_crossattn=c_crossattn)
        noise_pred_uncond, noise_pred_text = ops.split(noise_pred, split_size_or_sections=noise_pred.shape[0] // 2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.guidance_rescale > 0:
            noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text)
        return noise_pred

    def rescale_noise_cfg(self, noise_pred, noise_pred_text):
        """
        Rescale `noise_pred` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = ops.std(noise_pred_text, axis=tuple(range(1, len(noise_pred_text.shape))), keepdims=True)
        std_cfg = ops.std(noise_pred, axis=tuple(range(1, len(noise_pred.shape))), keepdims=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_pred * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_pred = self.guidance_rescale * noise_pred_rescaled + (1 - self.guidance_rescale) * noise_pred
        return noise_pred

    @abstractmethod
    def data_prepare(self, inputs):
        pass

    def __call__(self, inputs):
        latents, c_crossattn, c_concat = self.data_prepare(inputs)
        timesteps = inputs["timesteps"]
        iterator = tqdm(timesteps, desc="Stable Diffusion Sampling", total=len(timesteps))
        for i, t in enumerate(iterator):
            ts = ms.Tensor(t, ms.int32)
            latents = self.scale_model_input(latents, ts)
            noise_pred = self.predict_noise(latents, ts, c_crossattn, inputs["scale"], c_concat)
            latents = self.scheduler(noise_pred, ts, latents, self.num_inference_steps)
        image = self.vae_decode(latents)
        return image


class SDText2Img(SDInfer):
    def __init__(
        self,
        text_encoder,
        unet,
        vae,
        scheduler,
        scale_factor=1.0,
        guidance_rescale=0.0,
        num_inference_steps=50,
    ):
        super(SDText2Img, self).__init__(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=scale_factor,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
        )

    def data_prepare(self, inputs):
        latents = inputs["noise"]
        c_crossattn = self.prompt_embed(inputs["prompt_data"], inputs["negative_prompt_data"])
        return latents, c_crossattn, None


class SDImg2Img(SDInfer):
    def __init__(
        self,
        text_encoder,
        unet,
        vae,
        scheduler,
        scale_factor=1.0,
        guidance_rescale=0.0,
        num_inference_steps=50,
    ):
        super(SDImg2Img, self).__init__(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=scale_factor,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
        )

    def data_prepare(self, inputs):
        timesteps = inputs["timesteps"]
        t0 = ms.Tensor(timesteps[0], ms.int32)
        image_latents = self.vae_encode(inputs["img"])
        latents = self.latents_add_noise(image_latents, inputs["noise"], t0)
        c_crossattn = self.prompt_embed(inputs["prompt_data"], inputs["negative_prompt_data"])
        return latents, c_crossattn, None


class SDInpaint(SDInfer):
    def __init__(
        self,
        text_encoder,
        unet,
        vae,
        scheduler,
        scale_factor=1.0,
        guidance_rescale=0.0,
        num_inference_steps=50,
    ):
        super(SDInpaint, self).__init__(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=scale_factor,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
        )

    def data_prepare(self, inputs):
        masked_image = self.vae_encode(inputs["masked_image"])
        mask = ops.ResizeNearestNeighbor(masked_image.shape[2:])(inputs["mask"])
        c_concat = ops.concat((mask, masked_image), axis=1)
        latents = inputs["noise"]
        c_crossattn = self.prompt_embed(inputs["prompt_data"], inputs["negative_prompt_data"])
        return latents, c_crossattn, c_concat


class SDControlNet(SDInfer):
    def __init__(
        self,
        text_encoder,
        unet,
        vae,
        scheduler,
        scale_factor=1.0,
        guidance_rescale=0.0,
        num_inference_steps=50,
    ):
        super(SDControlNet, self).__init__(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=scale_factor,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
        )

    def data_prepare(self, inputs):
        latents = inputs["noise"]
        c_crossattn = self.prompt_embed(inputs["prompt_data"], inputs["negative_prompt_data"])
        control = inputs["control"]
        return latents, c_crossattn, control

    @ms.jit
    def predict_noise(self, x, t_continuous, c_crossattn, guidance_scale, c_concat=None, control=None):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        t_continuous = ops.tile(t_continuous.reshape(1), (x.shape[0],))
        x_in = ops.concat([x] * 2, axis=0)
        t_in = ops.concat([t_continuous] * 2, axis=0)
        c_concat = ops.concat([c_concat] * 2, axis=0)

        noise_pred = self.unet(x_in, t_in, c_concat=c_concat, c_crossattn=c_crossattn, control=c_concat)
        noise_pred_uncond, noise_pred_text = ops.split(noise_pred, split_size_or_sections=noise_pred.shape[0] // 2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.guidance_rescale > 0:
            noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text)

        return noise_pred
