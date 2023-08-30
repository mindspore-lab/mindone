from tqdm import tqdm

import mindspore as ms
from mindspore import ops


class SD_Infer:
    """
    Stable Diffusion inference.

    Args:
        text_encoder (nn.Cell): Frozen text-encoder.
        unet (nn.Cell): A `UNet2DConditionModel` to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scheduler (nn.Cell): A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor (float): scale_factor for vae.
        guidance_scale (float): A higher guidance scale value encourages the model to generate images closely linked to the text
            prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
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
        guidance_scale=7.5,
        guidance_rescale=0.0,
        num_inference_steps=50,
    ):
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.scale_factor = scale_factor
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.do_classifier_free_guidance = guidance_scale > 1.0
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
    def latents_add_noise(self, image_latents, noise, ts):
        latents = self.scheduler.add_noise(image_latents, noise, self.alphas_cumprod[ts])
        return latents

    @ms.jit
    def predict_noise(self, x, t_continuous, condition, unconditional_condition):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        t_continuous = ops.tile(t_continuous.reshape(1), (x.shape[0],))
        if self.guidance_scale == 1.0:
            return self.unet(x, t_continuous, c_crossattn=condition)
        x_in = ops.concat([x] * 2, axis=0)
        t_in = ops.concat([t_continuous] * 2, axis=0)
        c_in = ops.concat([unconditional_condition, condition], axis=0)
        noise_pred = self.unet(x_in, t_in, c_crossattn=c_in)
        noise_pred_uncond, noise_pred_text = ops.split(noise_pred, split_size_or_sections=noise_pred.shape[0] // 2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
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

    def __call__(self, inputs):
        pass


class SD_Text2Img(SD_Infer):
    def __init__(
        self,
        text_encoder,
        unet,
        vae,
        scheduler,
        scale_factor=1.0,
        guidance_scale=7.5,
        guidance_rescale=0.0,
        num_inference_steps=50,
    ):
        super(SD_Text2Img, self).__init__(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=scale_factor,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
        )

    def __call__(self, inputs):
        prompt_embeds = self.text_encoder(inputs["prompt_data"])
        negative_prompt_embeds = self.text_encoder(inputs["negative_prompt_data"])
        latents = inputs["noise"]
        timesteps = inputs["timesteps"]
        iterator = tqdm(timesteps, desc="Stable Diffusion Sampling", total=len(timesteps))
        for i, t in enumerate(iterator):
            ts = ms.Tensor(t, ms.int32)
            noise_pred = self.predict_noise(latents, ts, prompt_embeds, negative_prompt_embeds)
            latents = self.scheduler(noise_pred, ts, latents, self.num_inference_steps)
        image = self.vae_decode(latents)
        return image


class SD_Img2Img(SD_Infer):
    def __init__(
        self,
        text_encoder,
        unet,
        vae,
        scheduler,
        scale_factor=1.0,
        guidance_scale=7.5,
        guidance_rescale=0.0,
        num_inference_steps=50,
    ):
        super(SD_Img2Img, self).__init__(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=scale_factor,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
        )

    def __call__(self, inputs):
        prompt_embeds = self.text_encoder(inputs["prompt_data"])
        negative_prompt_embeds = self.text_encoder(inputs["negative_prompt_data"])
        timesteps = inputs["timesteps"]
        t0 = ms.Tensor(timesteps[0], ms.int32)
        image_latents = self.vae_encode(inputs["img"])
        latents = self.latents_add_noise(image_latents, inputs["noise"], t0)
        iterator = tqdm(timesteps, desc="Stable Diffusion Sampling", total=len(timesteps))
        for i, t in enumerate(iterator):
            ts = ms.Tensor(t, ms.int32)
            noise_pred = self.predict_noise(latents, ts, prompt_embeds, negative_prompt_embeds)
            latents = self.scheduler(noise_pred, ts, latents, self.num_inference_steps)
        image = self.vae_decode(latents)
        return image


class SD_Inpaint(SD_Infer):
    def __init__(
        self,
        text_encoder,
        unet,
        vae,
        scheduler,
        scale_factor=1.0,
        guidance_scale=7.5,
        guidance_rescale=0.0,
        num_inference_steps=50,
    ):
        super(SD_Inpaint, self).__init__(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=scale_factor,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
        )

    @ms.jit
    def predict_noise(self, x, t_continuous, c_concat, c_crossattn):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        t_continuous = ops.tile(t_continuous.reshape(1), (x.shape[0],))
        x_in = ops.concat([x] * 2, axis=0)
        t_in = ops.concat([t_continuous] * 2, axis=0)
        c_concat = ops.concat([c_concat] * 2, axis=0)
        noise_pred = self.unet(x_in, t_in, c_concat=c_concat, c_crossattn=c_crossattn)
        noise_pred_uncond, noise_pred_text = ops.split(noise_pred, split_size_or_sections=noise_pred.shape[0] // 2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.guidance_rescale > 0:
            noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text)
        return noise_pred

    def __call__(self, inputs):
        prompt_embeds = self.text_encoder(inputs["prompt_data"])
        negative_prompt_embeds = self.text_encoder(inputs["negative_prompt_data"])
        timesteps = inputs["timesteps"]
        masked_image = self.vae_encode(inputs["masked_image"])
        mask = ops.ResizeNearestNeighbor(masked_image.shape[2:])(inputs["mask"])
        c_concat = ops.concat((mask, masked_image), axis=1)
        c_crossattn = ops.concat([negative_prompt_embeds, prompt_embeds], axis=0)
        latents = inputs["noise"]
        iterator = tqdm(timesteps, desc="Stable Diffusion Sampling", total=len(timesteps))

        for i, t in enumerate(iterator):
            ts = ms.Tensor(t, ms.int32)
            noise_pred = self.predict_noise(latents, ts, c_concat, c_crossattn)
            latents = self.scheduler(noise_pred, ts, latents, self.num_inference_steps)
        image = self.vae_decode(latents)
        return image
