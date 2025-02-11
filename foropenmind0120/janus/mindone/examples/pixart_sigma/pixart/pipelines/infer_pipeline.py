from functools import partial
from typing import List, Literal, Optional, Tuple, Union

from pixart.diffusion.dpm import DPMS, create_noise_schedule_dpms
from pixart.diffusion.iddpm import create_diffusion
from pixart.modules.pixart import PixArt
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import Tensor, mint, ops

from mindone.diffusers import AutoencoderKL
from mindone.transformers import T5EncoderModel


class PixArtInferPipeline:
    """
    Args:
        network (nn.Cell): `PixArt` network to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scale_factor (float): scale_factor for vae.
        guidance_scale (float): A higher guidance scale value for noise rescale.
        num_inference_steps: (int): The number of denoising steps.
    """

    def __init__(
        self,
        network: PixArt,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        text_tokenizer: AutoTokenizer,
        scale_factor: float = 1.0,
        guidance_scale: float = 0.0,
        num_inference_steps: int = 100,
        sampling_method: Literal["iddpm", "ddim", "dpm"] = "iddpm",
        force_freeze: bool = False,
    ):
        super().__init__()
        self.network = network
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.scale_factor = scale_factor
        self.guidance_scale = Tensor(guidance_scale, dtype=ms.float32)
        self.sampling_method = sampling_method
        self.num_inference_steps = num_inference_steps

        # Cache for diffusion objects to avoid recreaetion
        self._diffusion_cache = {}
        self._init_diffusion()

        if force_freeze:
            # freeze all components
            self.network.set_train(False)
            for param in self.network.trainable_params():
                param.requires_grad = False

            self.vae.set_train(False)
            for param in self.vae.trainable_params():
                param.requires_grad = False

            self.text_encoder.set_train(False)
            for param in self.text_encoder.trainable_params():
                param.requires_grad = False

    def _init_diffusion(self):
        """Initialize or retrieve cached diffusion object"""
        if self.sampling_method == "dpm":
            self.noise_schedule = create_noise_schedule_dpms()
            self.diffusion = partial(
                DPMS, model=self.network.construct_with_dpmsolver, noise_schedule=self.noise_schedule
            )
            self.sampling_func = None
            return

        # Create cache key for IDDPM/DDIM
        cache_key = (self.sampling_method, self.num_inference_steps)

        if cache_key not in self._diffusion_cache:
            diffusion = create_diffusion(str(self.num_inference_steps))
            sampling_func = diffusion.ddim_sample_loop if self.sampling_method == "ddim" else diffusion.p_sample_loop
            self._diffusion_cache[cache_key] = (diffusion, sampling_func)

        self.diffusion, self.sampling_func = self._diffusion_cache[cache_key]

    def set_sampling_params(self, sampling_method=None, num_inference_steps=None, guidance_scale=None):
        """Update sampling parameters and reinitialize diffusion if needed

        Args:
            sampling_method: New sampling method
            num_inference_steps: New numbers of inference step (need to recreate the sampling function in iddpm/ddim)
            guidance_scale: New guidance scale value
        """
        updated = False
        if sampling_method is not None and sampling_method != self.sampling_method:
            self.sampling_method = sampling_method
            updated = True

        if num_inference_steps is not None and num_inference_steps != self.num_inference_steps:
            self.num_inference_steps = num_inference_steps
            updated = True

        if guidance_scale is not None:
            self.guidance_scale = Tensor(guidance_scale, dtype=ms.float32)

        if updated:
            self._init_diffusion()

    @ms.jit
    def vae_decode(self, x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        y = self.vae.decode(x / self.scale_factor)[0]
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    @ms.jit
    def _text_encoding(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.text_encoder(input_ids=x, attention_mask=mask)[0]

    def get_condition_embeddings(self, text: Union[str, List[str]]) -> Tuple[Tensor, Tensor]:
        encoding = self.text_tokenizer(text, padding="max_length", truncation=True, return_tensors="np")
        input_ids, text_mask = encoding.input_ids, encoding.attention_mask
        text_emb = self._text_encoding(Tensor(input_ids), Tensor(text_mask))
        return text_emb, Tensor(text_mask).to(ms.bool_)

    def data_prepare(
        self, noise: Tensor, y: List[str], y_null: Optional[List[str]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = noise
        if y_null is None:
            y_null = "" if isinstance(y, str) else [""] * len(y)

        y, mask_y = self.get_condition_embeddings(y)
        y_null, _ = self.get_condition_embeddings(y_null)

        y = ops.concat([y, y_null], axis=0)
        mask_y = ops.tile(mask_y, (2, 1))
        x_in = ops.tile(x, (2, 1, 1, 1))
        assert y.shape[0] == x_in.shape[0], "shape mismatch!"
        return x_in, y, mask_y

    def data_prepare_dpm(
        self, noise: Tensor, y: List[str], y_null: Optional[List[str]] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = noise
        if y_null is None:
            y_null = "" if isinstance(y, str) else [""] * len(y)

        y, mask_y = self.get_condition_embeddings(y)
        y_null, _ = self.get_condition_embeddings(y_null)

        mask_y = ops.tile(mask_y, (2, 1))

        return x, y, y_null, mask_y

    def __call__(
        self, noise: Tensor, y: Union[str, List[str]], y_null: Optional[Union[str, List[str]]] = None
    ) -> Tensor:
        if self.sampling_method in ["iddpm", "ddim"]:
            z, y, mask_y = self.data_prepare(noise, y, y_null)
            model_kwargs = dict(y=y, mask_y=mask_y, cfg_scale=self.guidance_scale)

            latents = self.sampling_func(
                self.network.construct_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
            )
            latents, _ = mint.chunk(latents, 2, dim=0)
        else:
            z, y, y_null, mask_y = self.data_prepare_dpm(noise, y, y_null)
            dpm_solver = self.diffusion(
                condition=y, uncondition=y_null, cfg_scale=self.guidance_scale, model_kwargs=dict(mask_y=mask_y)
            )
            latents = dpm_solver.sample(
                z, steps=self.num_inference_steps, order=2, skip_type="time_uniform", method="multistep"
            )
        assert latents.dim() == 4, f"Expect to have 4-dim latents, but got {latents.shape}"

        images = self.vae_decode(latents.to(self.vae.dtype))

        return images
