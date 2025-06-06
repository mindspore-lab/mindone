import inspect
import logging
from typing import Callable, List, Optional, Union

from opensora.acceleration.communications import AllGather
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from transformers import CLIPTokenizer, MT5Tokenizer

import mindspore as ms
from mindspore import mint, ops

from mindone.diffusers import AutoencoderKL, DDPMScheduler, FlowMatchEulerDiscreteScheduler
from mindone.diffusers.pipelines.pipeline_utils import DiffusionPipeline
from mindone.diffusers.utils import BaseOutput
from mindone.diffusers.utils.mindspore_utils import randn_tensor
from mindone.transformers import CLIPTextModelWithProjection, T5EncoderModel

# from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback #TODO:TBD


logger = logging.getLogger(__name__)


from dataclasses import dataclass

import numpy as np
import PIL
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V_v1_3


@dataclass
class OpenSoraPipelineOutput(BaseOutput):
    videos: Union[List[ms.Tensor], List[PIL.Image.Image], np.ndarray]


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = ops.std(noise_pred_text, axis=tuple(range(1, len(noise_pred_text.shape))), keepdims=True)
    std_cfg = ops.std(noise_cfg, axis=tuple(range(1, len(noise_cfg.shape))), keepdims=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
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
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
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


class OpenSoraPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = [
        "text_encoder_2",
        "tokenizer_2",
        "text_encoder",
        "tokenizer",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "prompt_embeds_2",
        "negative_prompt_embeds_2",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: MT5Tokenizer,
        transformer: OpenSoraT2V_v1_3,
        scheduler: DDPMScheduler,
        text_encoder_2: CLIPTextModelWithProjection = None,
        tokenizer_2: CLIPTokenizer = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
        )
        self.all_gather = None if not get_sequence_parallel_state() else AllGather()

    @ms.jit  # FIXME: on ms2.3, in pynative mode, text encoder's output has nan problem.
    def text_encoding_func(self, text_encoder, input_ids, attention_mask):
        return ops.stop_gradient(text_encoder(input_ids, attention_mask=attention_mask))

    def encode_prompt(
        self,
        prompt: str,
        dtype=None,
        num_samples_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        prompt_attention_mask: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask: Optional[ms.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        text_encoder_index: int = 0,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            dtype (`ms.dtype`):
                mindspore dtype
            num_samples_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            prompt_attention_mask (`ms.Tensor`, *optional*):
                Attention mask for the prompt. Required when `prompt_embeds` is passed directly.
            negative_prompt_attention_mask (`ms.Tensor`, *optional*):
                Attention mask for the negative prompt. Required when `negative_prompt_embeds` is passed directly.
            max_sequence_length (`int`, *optional*): maximum sequence length to use for the prompt.
            text_encoder_index (`int`, *optional*):
                Index of the text encoder to use. `0` for T5 and `1` for clip.
        """
        if dtype is None:
            if self.text_encoder_2 is not None:
                dtype = self.text_encoder_2.dtype
            elif self.transformer is not None:
                dtype = self.transformer.dtype
            else:
                dtype = None

        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = tokenizers[text_encoder_index]
        text_encoder = text_encoders[text_encoder_index]

        if max_sequence_length is None:
            if text_encoder_index == 0:
                max_length = 512
            if text_encoder_index == 1:
                max_length = 77
        else:
            max_length = max_sequence_length

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors=None,
            )
            text_input_ids = ms.Tensor(text_inputs.input_ids)
            untruncated_ids = ms.Tensor(tokenizer(prompt, padding="longest", return_tensors=None).input_ids)

            if untruncated_ids.shape[-1] > text_input_ids.shape[-1] or (
                untruncated_ids.shape[-1] == text_input_ids.shape[-1]
                and not ops.equal(text_input_ids, untruncated_ids).all()
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = ms.Tensor(text_inputs.attention_mask)
            prompt_embeds = self.text_encoding_func(text_encoder, text_input_ids, attention_mask=prompt_attention_mask)
            prompt_embeds = prompt_embeds[0] if isinstance(prompt_embeds, (list, tuple)) else prompt_embeds

            if text_encoder_index == 1:
                prompt_embeds = prompt_embeds.unsqueeze(1)  # b d -> b 1 d for clip

            prompt_attention_mask = prompt_attention_mask.repeat_interleave(num_samples_per_prompt, dim=0)
        else:
            prompt_attention_mask = ops.ones_like(prompt_embeds)

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, dim=1)
        prompt_embeds = prompt_embeds.view((bs_embed * num_samples_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            # elif prompt is not None and type(prompt) is not type(negative_prompt):
            #     raise TypeError(
            #         f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
            #         f" {type(prompt)}."
            #     )
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

            # max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors=None,
            )

            uncond_text_inputs = ms.Tensor(uncond_input.input_ids)
            negative_prompt_attention_mask = ms.Tensor(uncond_input.attention_mask)
            negative_prompt_embeds = self.text_encoding_func(
                text_encoder, uncond_text_inputs, attention_mask=negative_prompt_attention_mask
            )
            negative_prompt_embeds = (
                negative_prompt_embeds[0]
                if isinstance(negative_prompt_embeds, (list, tuple))
                else negative_prompt_embeds
            )

            if text_encoder_index == 1:
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1)  # b d -> b 1 d for clip
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat_interleave(
                num_samples_per_prompt, dim=0
            )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype)

            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, dim=1)
            negative_prompt_embeds = negative_prompt_embeds.view((batch_size * num_samples_per_prompt, seq_len, -1))
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        num_frames,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        prompt_embeds_2=None,
        negative_prompt_embeds_2=None,
        prompt_attention_mask_2=None,
        negative_prompt_attention_mask_2=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if (num_frames - 1) % 4 != 0:
            raise ValueError(f"`num_frames - 1` have to be divisible by 4 but is {num_frames}.")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                + f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
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
        elif prompt is None and prompt_embeds_2 is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_2`. Cannot leave both `prompt` and `prompt_embeds_2` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if prompt_embeds_2 is not None and prompt_attention_mask_2 is None:
            raise ValueError("Must provide `prompt_attention_mask_2` when specifying `prompt_embeds_2`.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if negative_prompt_embeds_2 is not None and negative_prompt_attention_mask_2 is None:
            raise ValueError(
                "Must provide `negative_prompt_attention_mask_2` when specifying `negative_prompt_embeds_2`."
            )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        if prompt_embeds_2 is not None and negative_prompt_embeds_2 is not None:
            if prompt_embeds_2.shape != negative_prompt_embeds_2.shape:
                raise ValueError(
                    "`prompt_embeds_2` and `negative_prompt_embeds_2` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds_2` {prompt_embeds_2.shape} != `negative_prompt_embeds_2`"
                    f" {negative_prompt_embeds_2.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, generator, latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            (int(num_frames) - 1) // self.vae.vae_scale_factor[0] + 1,
            int(height) // self.vae.vae_scale_factor[1],
            int(width) // self.vae.vae_scale_factor[2],
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape=shape, generator=generator, dtype=dtype)
        else:
            latents = latents

        if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_parallel_input(self, input, sp_dim):
        sp_size = hccl_info.world_size
        index = hccl_info.rank % sp_size
        assert (
            input.shape[sp_dim] % sp_size == 0
        ), f"Expect the parallel input length at dim={sp_dim} is divisble by the sp_size={sp_size}, but got {input.shape[sp_dim]}"
        input = ops.chunk(input, sp_size, sp_dim)[index]
        return input

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        timesteps: List[int] = None,
        guidance_scale: Optional[float] = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_samples_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.Generator] = None,
        latents: Optional[ms.Tensor] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        prompt_embeds_2: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds_2: Optional[ms.Tensor] = None,
        prompt_attention_mask: Optional[ms.Tensor] = None,
        prompt_attention_mask_2: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask_2: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Callable[[int, int, ms.Tensor], None]
        ] = None,  # Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_rescale: float = 0.0,
        max_sequence_length: int = 512,
    ):
        # TODO
        if hasattr(callback_on_step_end, "tensor_inputs"):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        # if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        # callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. default height and width
        num_frames = num_frames or (self.transformer.config.sample_size_t - 1) * self.vae.vae_scale_factor[0] + 1
        height = height or self.transformer.config.sample_size[0] * self.vae.vae_scale_factor[1]
        width = width or self.transformer.config.sample_size[1] * self.vae.vae_scale_factor[2]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            num_frames,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            dtype=self.transformer.dtype,
            num_samples_per_prompt=num_samples_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            text_encoder_index=0,
        )

        if self.tokenizer_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_attention_mask_2,
                negative_prompt_attention_mask_2,
            ) = self.encode_prompt(
                prompt=prompt,
                dtype=self.transformer.dtype,
                num_samples_per_prompt=num_samples_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds_2,
                negative_prompt_embeds=negative_prompt_embeds_2,
                prompt_attention_mask=prompt_attention_mask_2,
                negative_prompt_attention_mask=negative_prompt_attention_mask_2,
                max_sequence_length=77,
                text_encoder_index=1,
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_attention_mask_2 = None
            negative_prompt_attention_mask_2 = None

        # 4. Prepare timesteps
        if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        if get_sequence_parallel_state():
            world_size = hccl_info.world_size
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_samples_per_prompt,
            num_channels_latents,
            (num_frames + world_size - 1) // world_size if get_sequence_parallel_state() else num_frames,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        else:
            extra_step_kwargs = {}

        # 7 create image_rotary_emb, style embedding & time ids
        if self.do_classifier_free_guidance:
            prompt_embeds = mint.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = mint.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            if self.tokenizer_2 is not None:
                prompt_embeds_2 = mint.cat([negative_prompt_embeds_2, prompt_embeds_2], dim=0)
                prompt_attention_mask_2 = mint.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2], dim=0)

        # ==================make sp=====================================
        if get_sequence_parallel_state():
            # # b (n x) h -> b n x h
            # b, _, h = prompt_embeds.shape
            # n = world_size
            # x = prompt_embeds.shape[1] // world_size
            # prompt_embeds = prompt_embeds.reshape(b, n, x, h).contiguous()
            # rank = hccl_info.rank
            # prompt_embeds = prompt_embeds[:, rank, :, :]
            # b (n x) h -> b n x h
            prompt_embeds = self.prepare_parallel_input(prompt_embeds, sp_dim=1)
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = self.prepare_parallel_input(prompt_embeds_2, sp_dim=1)

        # ==================make sp=====================================
        # 8. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = mint.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
                current_timestep = t
                if not isinstance(current_timestep, ms.Tensor):
                    current_timestep = ms.Tensor([current_timestep], dtype=latent_model_input.dtype)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None]
                current_timestep = current_timestep.repeat_interleave(latent_model_input.shape[0], 0)

                # ==================prepare my shape=====================================
                # predict the noise residual
                if prompt_embeds.ndim == 3:
                    prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
                if prompt_attention_mask.ndim == 2:
                    prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
                if prompt_embeds_2 is not None and prompt_embeds_2.ndim == 2:
                    # prompt_embeds = prompt_embeds.unsqueeze(1)  # b d -> b 1 d #OFFICIAL VER. DONT KNOW WHY
                    # prompt_embeds_2 = prompt_embeds_2.unsqueeze(1)  #
                    raise NotImplementedError

                attention_mask = ops.ones_like(latent_model_input)[:, 0].to(ms.int32)
                if get_sequence_parallel_state():
                    attention_mask = attention_mask.repeat_interleave(world_size, dim=1)
                attention_mask = attention_mask.to(ms.bool_)
                # ==================make sp=====================================

                noise_pred = ops.stop_gradient(
                    self.transformer(
                        latent_model_input,  # (b c t h w)
                        attention_mask=attention_mask,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=current_timestep,
                        pooled_projections=prompt_embeds_2,  # UNUSED!!!!
                        return_dict=False,
                    )
                )  # b,c,t,h,w
                assert not ops.any(ops.isnan(noise_pred.float()))

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if (
                    self.do_classifier_free_guidance
                    and guidance_rescale > 0.0
                    and not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler)
                ):
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    prompt_embeds_2 = callback_outputs.pop("prompt_embeds_2", prompt_embeds_2)
                    negative_prompt_embeds_2 = callback_outputs.pop(
                        "negative_prompt_embeds_2", negative_prompt_embeds_2
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # ==================make sp=====================================
        if get_sequence_parallel_state():
            # latents_shape = list(latents.shape)  # b c t//sp h w
            # full_shape = [latents_shape[0] * world_size] + latents_shape[1:]  # # b*sp c t//sp h w
            # all_latents = ops.zeros(full_shape, dtype=latents.dtype)
            all_latents = self.all_gather(latents)
            latents_list = mint.chunk(all_latents, world_size, 0)
            latents = mint.cat(latents_list, dim=2)
        # ==================make sp=====================================

        if not output_type == "latents":
            videos = self.decode_latents(latents)
            videos = videos[:, :num_frames, :height, :width]
        else:
            videos = latents

        # Offload all models
        # self.maybe_free_model_hooks()

        if not return_dict:
            return (videos,)

        return OpenSoraPipelineOutput(videos=videos)

    # def decode_latents(self, latents):
    #     print(f'before vae decode {latents.shape}', ops.max(latents).item(), ops.min(latents).item(), ops.mean(latents).item(), ops.std(latents).item())
    #     video = self.vae.decode(latents.to(self.vae.vae.dtype)) # (b t c h w)
    #     print(f'after vae decode {latents.shape}', ops.max(video).item(), ops.min(video).item(), ops.mean(video).item(), ops.std(video).item())
    #     video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=ms.uint8).permute(0, 1, 3, 4, 2).contiguous() # b t h w c
    #     return video

    def decode_latents_per_sample(self, latents):
        # print(
        #     f"before vae decode {latents.shape}",
        #     latents.max().item(),
        #     latents.min().item(),
        #     latents.mean().item(),
        #     latents.std().item(),
        # )
        video = self.vae.decode(latents).to(ms.float32)  # (b t c h w)
        # print(
        #     f"after vae decode {video.shape}",
        #     video.max().item(),
        #     video.min().item(),
        #     video.mean().item(),
        #     video.std().item(),
        # )
        video = ops.clip_by_value((video / 2.0 + 0.5), clip_value_min=0.0, clip_value_max=1.0).permute(0, 1, 3, 4, 2)
        return video  # b t h w c

    def decode_latents(self, latents):
        per_sample_func = self.decode_latents_per_sample
        out = []
        bs = latents.shape[0]
        for i in range(bs):
            out.append(per_sample_func(latents[i : i + 1]))
        out = mint.cat(out, dim=0)
        return out  # b t h w c
