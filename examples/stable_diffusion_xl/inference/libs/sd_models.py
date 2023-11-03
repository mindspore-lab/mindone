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
        denoiser,
        scale_factor=1.0,
        guidance_rescale=0.0,
        num_inference_steps=40,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.denoiser = denoiser
        self.scale_factor = scale_factor
        self.guidance_rescale = guidance_rescale
        self.num_inference_steps = ms.Tensor(num_inference_steps, ms.int32)

    def vae_decode(self, x):
        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
        return y

    @abstractmethod
    def data_prepare(self, inputs):
        pass

    def __call__(self, inputs):
        pos_crossattn, neg_crossattn, pos_vector, neg_vector, noise = self.data_prepare(inputs)
        x, s_in = self.scheduler.prepare_sampling_loop(noise)

        for i in tqdm(range(self.num_inference_steps), desc="SDXL sampling"):
            noised_input, sigma_hat_s, next_sigma, sigma_hat = self.scheduler.pre_model_input(
                iter_index=i, x=x, s_in=s_in
            )
            c_skip, c_out, c_in, c_noise = self.denoiser(sigma_hat_s, noised_input.ndim)
            model_output = self.unet(
                noised_input * c_in,
                c_noise,
                context=ops.concat((neg_crossattn, pos_crossattn), 0),
                y=ops.concat((neg_vector, pos_vector), 0),
            )
            x = self.scheduler(
                model_output=model_output,
                c_out=c_out,
                noised_input=noised_input,
                c_skip=c_skip,
                scale=inputs["scale"],
                x=x,
                sigma_hat=sigma_hat,
                next_sigma=next_sigma,
            )
        image = self.vae_decode(x)
        return image


class SDText2Img(SDInfer):
    def __init__(
        self,
        text_encoder,
        unet,
        vae,
        scheduler,
        denoiser,
        scale_factor=1.0,
        guidance_rescale=0.0,
        num_inference_steps=40,
    ):
        super(SDText2Img, self).__init__(
            text_encoder,
            unet,
            vae,
            scheduler,
            denoiser,
            scale_factor=scale_factor,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
        )

    def data_prepare(self, inputs):
        clip_tokens = ms.Tensor(inputs["pos_clip_token"], dtype=ms.int32)
        time_tokens = ms.Tensor(inputs["pos_time_token"], dtype=ms.int32)
        uc_clip_tokens = ms.Tensor(inputs["neg_clip_token"], dtype=ms.int32)
        uc_time_tokens = ms.Tensor(inputs["neg_time_token"], dtype=ms.int32)
        noise = ms.Tensor(inputs["noise"], ms.float32)
        pos_prompt_embeds = self.text_encoder(clip_tokens.split(1) + time_tokens.split(1))
        negative_prompt_embeds = self.text_encoder(uc_clip_tokens.split(1) + uc_time_tokens.split(1))
        pos_crossattn, neg_crossattn = pos_prompt_embeds[0], negative_prompt_embeds[0]
        pos_vector, neg_vector = pos_prompt_embeds[1], negative_prompt_embeds[1]
        return pos_crossattn, neg_crossattn, pos_vector, neg_vector, noise
