import mindspore as ms
from mindspore import nn, ops


class Embedder(nn.Cell):
    """
    Some data prepare process. like text encode, image encode.

    Args:
        text_encoder(nn.Cell): Frozen text-encoder.
        vae(nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scheduler(nn.Cell): A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor(float): scale_factor for vae
    """

    def __init__(self, text_encoder, vae, scheduler, scale_factor=1.0):
        super(Embedder, self).__init__()
        self.text_encoder = text_encoder
        self.vae = vae
        self.scheduler = scheduler
        self.scale_factor = scale_factor


class Discretization(nn.Cell):
    def __init__(self, discretization):
        super(Discretization, self).__init__()
        self.discretization = discretization

    def construct(self):
        return self.discretization()


class SchedulerPreModelInput(nn.Cell):
    def __init__(self, scheduler):
        super(SchedulerPreModelInput, self).__init__()
        self.scheduler = scheduler

    def construct(self, x, i, s_in):
        noised_input, sigma_hat_s, next_sigma, sigma_hat = self.scheduler.pre_model_input(iter_index=i, x=x, s_in=s_in)
        return noised_input, sigma_hat_s, next_sigma, sigma_hat


class SchedulerPrepareSamplingLoop(nn.Cell):
    def __init__(self, scheduler):
        super(SchedulerPrepareSamplingLoop, self).__init__()
        self.scheduler = scheduler

    def construct(self, latents):
        x, s_in = self.scheduler.prepare_sampling_loop(latents)
        return x, s_in


class Denoiser(nn.Cell):
    def __init__(self, denoiser):
        super(Denoiser, self).__init__()
        self.denoiser = denoiser

    def construct(self, sigma_hat_s, ndim):
        c_skip, c_out, c_in, c_noise = self.denoiser(sigma_hat_s, 4)
        return c_skip, c_out, c_in, c_noise


class PredictNoise(nn.Cell):
    """
    Predict the noise residual.

    Args:
        unet (nn.Cell): A `UNet2DConditionModel` to denoise the encoded image latents.
        guidance_scale (float): A higher guidance scale value encourages the model to generate images closely linked to the text
            prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
    """

    def __init__(self, unet):
        super(PredictNoise, self).__init__()
        self.unet = unet

    def construct(self, noised_input, c_noise, context, y):
        noise_pred = self.unet(x=noised_input, t=c_noise, context=context, contact=None, y=y)
        return noise_pred


class NoisySample(nn.Cell):
    """
    Compute the previous noisy sample x_t -> x_t-1.

    Args:
        scheduler (nn.Cell): A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(self, scheduler):
        super(NoisySample, self).__init__()
        self.scheduler = scheduler

    def construct(self, model_output, c_out, noised_input, c_skip, scale, x, sigma_hat, next_sigma):
        return self.scheduler(
            model_output=model_output,
            c_out=c_out,
            noised_input=noised_input,
            c_skip=c_skip,
            scale=scale,
            x=x,
            sigma_hat=sigma_hat,
            next_sigma=next_sigma,
        )


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
        y = self.vae.decode(ms.ops.div(x, self.scale_factor))
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
        return y

    def construct(self, latents):
        return self.vae_decode(latents)


class Text2ImgEmbedder(Embedder):
    """
    Some data prepare process for text2img task.
    """

    def __init__(self, text_encoder, vae, scheduler, scale_factor=1.0):
        super(Text2ImgEmbedder, self).__init__(text_encoder, vae, scheduler, scale_factor=scale_factor)

    def construct(self, clip_tokens, time_tokens, uc_clip_tokens, uc_time_tokens, noise):
        # vector, crossattn, concat
        pos_prompt_embeds = self.text_encoder(*clip_tokens.split(1), *time_tokens.split(1))
        negative_prompt_embeds = self.text_encoder(*uc_clip_tokens.split(1), *uc_time_tokens.split(1))
        vector = ops.concat((negative_prompt_embeds[0], pos_prompt_embeds[0]), 0)
        crossattn = ops.concat((negative_prompt_embeds[1], pos_prompt_embeds[1]), 0)

        vector = ops.cast(vector, ms.float32)
        crossattn = ops.cast(crossattn, ms.float32)

        return crossattn, vector, noise
