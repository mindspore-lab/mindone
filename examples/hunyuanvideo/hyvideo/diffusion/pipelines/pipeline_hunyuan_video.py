# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import mindspore as ms
from mindspore import nn, ops, mint
import numpy as np
from dataclasses import dataclass
from packaging import version

from mindone.diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from mindone.diffusers.configuration_utils import FrozenDict
from mindone.diffusers.image_processor import VaeImageProcessor
from mindone.diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from mindone.diffusers.models import AutoencoderKL
# from mindone.diffusers.models.lora import adjust_lora_scale_text_encoder
from mindone.diffusers.schedulers import KarrasDiffusionSchedulers
USE_PEFT_BACKEND = False
from mindone.diffusers.utils import (
    # USE_PEFT_BACKEND,
    deprecate,
    logging,
    # replace_example_docstring,
    # scale_lora_layers,
    # unscale_lora_layers,
)
from mindone.diffusers.utils.mindspore_utils import randn_tensor
from mindone.diffusers.pipelines.pipeline_utils import DiffusionPipeline
from mindone.diffusers.utils import BaseOutput

from ...constants import PRECISION_TO_TYPE
from ...vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ...modules import HYVideoDiffusionTransformer

# from ...text_encoder import TextEncoder
# TODO: add text enc
TextEncoder = nn.Cell
print('Text encoder is not supported in this version.')

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[ms.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: ms.Tensor


class HunyuanVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`TextEncoder`]):
            Frozen text-encoder.
        text_encoder_2 ([`TextEncoder`]):
            Frozen text-encoder_2.
        transformer ([`HYVideoDiffusionTransformer`]):
            A `HYVideoDiffusionTransformer` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["text_encoder_2"]
    _exclude_from_cpu_offload = ["transformer"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HYVideoDiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        progress_bar_config: Dict[str, Any] = None,
        args=None,
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.args = args
        # ==========================================================================================

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode_prompt(
        self,
        prompt,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        negative_attention_mask: Optional[ms.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`ms.Tensor`, *optional*):
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`ms.Tensor`, *optional*):
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
            data_type (`str`, *optional*):
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        # TODO: support lora
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            raise NotImplementedError
            # self._lora_scale = lora_scale
            # dynamically adjust the LoRA scale
            # scale_lora_layers(text_encoder.model, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            raise NotImplementedError
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)

            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.tile((1, num_videos_per_prompt))
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.tile((1, num_videos_per_prompt, 1))
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(
                    uncond_tokens, text_encoder.tokenizer
                )

            # max_length = prompt_embeds.shape[1]
            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

            negative_prompt_outputs = text_encoder.encode(
                uncond_input, data_type=data_type
            )
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(
                    1, num_videos_per_prompt
                )
                negative_attention_mask = negative_attention_mask.view(
                    batch_size * num_videos_per_prompt, seq_len
                )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype,
            )

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, -1
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

        if text_encoder is not None:
            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                # unscale_lora_layers(text_encoder.model, lora_scale)
                raise NotImplementedError

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

    def decode_latents(self, latents, enable_tiling=True):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        if enable_tiling:
            self.vae.enable_tiling()
            image = self.vae.decode(latents)
        else:
            image = self.vae.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        if image.ndim == 4:
            image = image.permute(0, 2, 3, 1).float()
        else:
            image = image.float()
        return image

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        video_length,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        vae_ver="88-4c-sd",
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if video_length is not None:
            if "884" in vae_ver:
                if video_length != 1 and (video_length - 1) % 4 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 4 but is {video_length}."
                    )
            elif "888" in vae_ver:
                if video_length != 1 and (video_length - 1) % 8 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 8 but is {video_length}."
                    )

        if callback_steps is not None and (
            not isinstance(callback_steps, int) or callback_steps <= 0
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )


    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, dtype=dtype
            )
        else:
            assert tuple(latents.shape) == shape, f'the latent noise loaded from npy shape {tuple(latent.shape)}, no equal to the expected shape {shape}' 
            latents = latents

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self,
        w: ms.Tensor,
        embedding_dim: int = 512,
        dtype = ms.float32,
    ) -> ms.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`ms.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`ms.dtype`, *optional*, defaults to `ms.float32`):
                Data type of the generated embeddings.

        Returns:
            `ms.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        print("D--: pre-compute emb in numpy")
        half_dim = embedding_dim // 2
        emb = ops.log(ms.tensor(10000.0)) / (half_dim - 1)
        emb = ops.exp(ops.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = ops.concat([ops.sin(emb), ops.cos(emb)], axis=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = mint.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        # return self._guidance_scale > 1 and self.transformer.config.time_cond_proj_dim is None
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    # @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        negative_attention_mask: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[ms.Tensor, ms.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        prompt_mask: Optional[ms.Tensor] = None,  # used for text emb cache
        negative_prompt_mask: Optional[ms.Tensor] = None, # used for text emb cache
        prompt_embeds_2 = None, # TODO: rm after debug
        negative_prompt_embeds_2 = None, # TODO: rm after debug
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            attention_mask:

            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`numpy.Generator` or `List[numpy.Generator]`, *optional*):
                A numpy Generator to make generation deterministic.
            latents (`ms.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.

            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        # TODO: uncomment after debug
        '''
        self.check_inputs(
            prompt,
            height,
            width,
            video_length,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            vae_ver=vae_ver,
        )
        '''

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )
        if prompt_embeds is None:
            (
                prompt_embeds,
                negative_prompt_embeds,
                prompt_mask,
                negative_prompt_mask,
            ) = self.encode_prompt(
                prompt,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                attention_mask=attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_attention_mask=negative_attention_mask,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                data_type=data_type,
            )
        # if self.text_encoder_2 is not None:
        if prompt_embeds_2 is None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            ) = self.encode_prompt(
                prompt,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type=data_type,
            )

        assert prompt_embeds is not None
        assert prompt_mask is not None
        assert prompt_embeds_2 is not None

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = ops.cat([negative_prompt_embeds, prompt_embeds], axis=0)
            if prompt_mask is not None:
                prompt_mask = ops.cat([negative_prompt_mask, prompt_mask], axis=0)
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = ops.cat([negative_prompt_embeds_2, prompt_embeds_2], axis=0)


        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            timesteps,
            sigmas,
            **extra_set_timesteps_kwargs,
        )

        if "884" in vae_ver:
            video_length = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            video_length = (video_length - 1) // 8 + 1
        else:
            video_length = video_length

        # 5. Prepare latent variables
        # import pdb; pdb.set_trace()
        if self.args.latent_noise_path is not None:
            if os.path.exists(self.args.latent_noise_path):
                latents = np.load(self.args.latent_noise_path)
                latents = ms.Tensor(latents)
                print(f"D--: Loaded latent noise from {self.args.latent_noise_path}")

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            video_length,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": eta},
        )

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        # TODO: in MS, auto_mixed_precision set amp_level to auto
        autocast_enabled = (
            target_dtype != ms.float32
        ) and not self.args.disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != ms.float32
        ) and not self.args.disable_autocast

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # if is_progress_bar:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    ops.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                # TODO: check API change in ms2.5
                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (
                    ms.Tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=ms.float32,
                    ).to(target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                # predict the noise residual
                # with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                # import pdb; pdb.set_trace()
                # auto mixed precision is set with ms AMP
                noise_pred = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                    latent_model_input,  # [2, 16, 33, 24, 42]
                    t_expand,  # [2]
                    text_states=prompt_embeds,  # [2, 256, 4096]
                    text_mask=prompt_mask,  # [2, 256]
                    text_states_2=prompt_embeds_2,  # [2, 768]
                    freqs_cos=freqs_cis[0],  # [seqlen, head_dim]
                    freqs_sin=freqs_cis[1],  # [seqlen, head_dim]
                    guidance=guidance_expand,
                )
                print('D--: noise pred step done', i, t, noise_pred.shape)
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # import pdb; pdb.set_trace()
        # TODO: rm after debug
        # import pdb; pdb.set_trace()
        np.save('debug_ms/latent_output.npy', latents.asnumpy())
        # latents (b, c, t, h, w) = (1, 16, 33, 90, 160)
        if not output_type == "latent":
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if (
                hasattr(self.vae.config, "shift_factor")
                and self.vae.config.shift_factor
            ):
                latents = (
                    latents / self.vae.config.scaling_factor
                    + self.vae.config.shift_factor
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            # with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
            # import pdb; pdb.set_trace()
            if enable_tiling:
                print('D--: vae tiling enabled')
                self.vae.enable_tiling()
                image = self.vae.decode(
                    latents, # return_dict=False, generator=generator
                )
            else:
                print('D--: vae tiling disabled')
                image = self.vae.decode(
                    latents, # return_dict=False, generator=generator
                )

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)


            # TODO: why resacle the output here? [-1, 1] -> [0, 1]. which can be done in save_video_grid
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.float()

        else:
            
            image = latents

        # Offload all models
        # self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image)

