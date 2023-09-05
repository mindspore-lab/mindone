import mindspore as ms
from mindspore import nn, ops


class DataPrepare(nn.Cell):
    """
    Some data prepare process. like text encode, image encode.

    Args:
        text_encoder(nn.Cell): Frozen text-encoder.
        vae(nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scheduler(nn.Cell): A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor(float): scale_factor for vae
    """

    def __init__(self, text_encoder, vae, scheduler, scale_factor=1.0):
        super(DataPrepare, self).__init__()
        self.text_encoder = text_encoder
        self.vae = vae
        self.scheduler = scheduler
        self.scale_factor = scale_factor
        self.alphas_cumprod = scheduler.alphas_cumprod

    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents.astype(ms.float16)

    def latents_add_noise(self, image_latents, noise, ts):
        latents = self.scheduler.add_noise(image_latents, noise, self.alphas_cumprod[ts])
        return latents

    def prompt_embed(self, prompt_data, negative_prompt_data):
        pos_prompt_embeds = self.text_encoder(prompt_data)
        negative_prompt_embeds = self.text_encoder(negative_prompt_data)
        prompt_embeds = ops.concat([negative_prompt_embeds, pos_prompt_embeds], axis=0)
        return prompt_embeds


class PredictNoise(nn.Cell):
    """
    Predict the noise residual.

    Args:
        unet (nn.Cell): A `UNet2DConditionModel` to denoise the encoded image latents.
        guidance_scale (float): A higher guidance scale value encourages the model to generate images closely linked to the text
            prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
    """

    def __init__(self, unet, guidance_rescale=0.0):
        super(PredictNoise, self).__init__()
        self.unet = unet
        self.guidance_rescale = guidance_rescale

    def predict_noise(self, x, t_continuous, c_crossattn, guidance_scale):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        t_continuous = ops.tile(t_continuous.reshape(1), (x.shape[0],))
        x_in = ops.concat([x] * 2, axis=0)
        t_in = ops.concat([t_continuous] * 2, axis=0)
        noise_pred = self.unet(x_in, t_in, c_crossattn=c_crossattn)
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

    def construct(self, latents, ts, c_crossattn, guidance_scale):
        return self.predict_noise(latents, ts, c_crossattn, guidance_scale)


class SchedulerPreProcess(nn.Cell):
    """
    Pre process of sampler.

    Args:
        scheduler (nn.Cell): A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(self, scheduler):
        super(SchedulerPreProcess, self).__init__()
        self.scheduler = scheduler

    def construct(self, latents, t):
        return self.scheduler.scale_model_input(latents, t)


class NoisySample(nn.Cell):
    """
    Compute the previous noisy sample x_t -> x_t-1.

    Args:
        scheduler (nn.Cell): A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(self, scheduler):
        super(NoisySample, self).__init__()
        self.scheduler = scheduler

    def construct(self, noise_pred, ts, latents, num_inference_steps):
        return self.scheduler(noise_pred, ts, latents, num_inference_steps)


class VAEDecoder(nn.Cell):
    """
    VAE Decoder

    Args:
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scale_factor (float): scale_factor for vae.
    """

    def __init__(self, vae, scale_factor=1.0):
        super(VAEDecoder, self).__init__()
        self.vae = vae
        self.scale_factor = scale_factor

    def vae_decode(self, x):
        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
        return y

    def construct(self, latents):
        return self.vae_decode(latents)


class Text2ImgDataPrepare(DataPrepare):
    """
    Some data prepare process for text2img task.
    """

    def __init__(self, text_encoder, vae, scheduler, scale_factor=1.0):
        super(Text2ImgDataPrepare, self).__init__(text_encoder, vae, scheduler, scale_factor=scale_factor)

    def construct(self, prompt_data, negative_prompt_data, noise):
        c_crossattn = self.prompt_embed(prompt_data, negative_prompt_data)
        return c_crossattn, noise


class Img2ImgDataPrepare(DataPrepare):
    """
    Some data prepare process for img2img task.
    """

    def __init__(self, text_encoder, vae, scheduler, scale_factor=1.0):
        super(Img2ImgDataPrepare, self).__init__(text_encoder, vae, scheduler, scale_factor=scale_factor)

    def construct(self, prompt_data, negative_prompt_data, img, noise, t0):
        image_latents = self.vae_encode(img)
        latents = self.latents_add_noise(image_latents, noise, t0)
        c_crossattn = self.prompt_embed(prompt_data, negative_prompt_data)
        return c_crossattn, latents


class InpaintDataPrepare(DataPrepare):
    """
    Some data prepare process for inpaint task.
    """

    def __init__(self, text_encoder, vae, scheduler, scale_factor=1.0):
        super(InpaintDataPrepare, self).__init__(text_encoder, vae, scheduler, scale_factor=scale_factor)

    def construct(self, prompt_data, negative_prompt_data, masked_image, mask, noise):
        masked_image_latents = self.vae_encode(masked_image)
        mask_reshape = ops.ResizeNearestNeighbor(masked_image_latents.shape[2:])(mask)
        c_concat = ops.concat((mask_reshape, masked_image_latents), axis=1)
        latents = noise
        c_crossattn = self.prompt_embed(prompt_data, negative_prompt_data)
        return c_crossattn, latents, c_concat


class InpaintPredictNoise(PredictNoise):
    """
    Inpainting Predict the noise residual.
    """

    def __init__(self, unet, guidance_rescale=0.0):
        super(InpaintPredictNoise, self).__init__(unet, guidance_rescale)

    def construct(self, x, t_continuous, c_crossattn, guidance_scale, c_concat):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        t_continuous = ops.tile(t_continuous.reshape(1), (x.shape[0],))
        x_in = ops.concat([x] * 2, axis=0)
        t_in = ops.concat([t_continuous] * 2, axis=0)
        c_concat = ops.concat([c_concat] * 2, axis=0)
        noise_pred = self.unet(x_in, t_in, c_concat=c_concat, c_crossattn=c_crossattn)
        noise_pred_uncond, noise_pred_text = ops.split(noise_pred, split_size_or_sections=noise_pred.shape[0] // 2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.guidance_rescale > 0:
            noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text)
        return noise_pred
