from typing import Any, Callable, Dict, List, Optional, Union

import mindone.diffusers
import mindone.transformers
import mindone.transformers.models
import numpy as np
from transformers import CLIPTokenizer, T5TokenizerFast
import mindspore as ms
from mindspore import ops, mint
from mindspore.common.api import _pynative_executor as ms_pyexecutor

import mindone
from mindone.diffusers.models import SD3Transformer2DModel
from mindone.diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from mindone.transformers.models.clip.modeling_clip import CLIPTextModelWithProjection
from mindone.transformers.models.t5.modeling_t5 import T5EncoderModel
from mindone.diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline, retrieve_timesteps 
from mindone.diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from mindone.diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from .infer.transformer_sd3_cache import sd3_transformer2d_construct, forward_blocks, forward_blocks_range
from .infer.attention_processor_cache import ToDoJointAttnProcessor, joint_transformerblock_construct
from .infer.normalization_replace import ada_groupnorm_construct, ada_layernorm_construct, ada_layernorm_continuous_construct, ada_layernormzero_construct
from .infer.modeling_t5_replace import t5_layernorm_construct, t5_attention_relative_position_bucket
from .infer.embeddings_replace import get_timestep_embedding
from .infer.modeling_clip_replace import clip_attention_construct

mindone.diffusers.models.SD3Transformer2DModel.construct=sd3_transformer2d_construct
mindone.diffusers.models.SD3Transformer2DModel.forward_blocks=forward_blocks
mindone.diffusers.models.SD3Transformer2DModel.forward_blocks_range=forward_blocks_range
mindone.diffusers.models.attention.JointTransformerBlock.construct=joint_transformerblock_construct
mindone.diffusers.models.normalization.AdaLayerNorm.construct=ada_layernorm_construct
mindone.diffusers.models.normalization.AdaLayerNormZero.construct=ada_layernormzero_construct
mindone.diffusers.models.normalization.AdaLayerNormContinuous.construct=ada_layernorm_continuous_construct
mindone.diffusers.models.normalization.AdaGroupNorm.construct=ada_groupnorm_construct
mindone.diffusers.models.embeddings.get_timestep_embedding=get_timestep_embedding
mindone.transformers.models.t5.modeling_t5.T5LayerNorm.construct=t5_layernorm_construct
mindone.transformers.models.t5.modeling_t5.T5Attention._relative_position_bucket=t5_attention_relative_position_bucket
mindone.transformers.models.clip.modeling_clip.CLIPAttention.construct=clip_attention_construct

class StableDiffusion3PipelineBoost(StableDiffusion3Pipeline):
    def __init__(
        self, 
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler, 
        vae: AutoencoderKL, 
        text_encoder: CLIPTextModelWithProjection, 
        tokenizer: CLIPTokenizer, 
        text_encoder_2: CLIPTextModelWithProjection, 
        tokenizer_2: CLIPTokenizer, 
        text_encoder_3: T5EncoderModel, 
        tokenizer_3: T5TokenizerFast
    ):
        super().__init__(
            transformer,
            scheduler, 
            vae, 
            text_encoder, 
            tokenizer, 
            text_encoder_2, 
            tokenizer_2, 
            text_encoder_3, 
            tokenizer_3
        )

        self.tgate = 20
        self.use_cache_and_tgate = False
        self.cache_params = (1, 2, 20, 10)
        self.token_merge_factor = 1.6
        self.token_merge_method = "bilinear"
        self.use_todo = False
        self.init_todo_processor = False

    def _enable_boost(self, use_cache_and_tgate: bool = True, use_todo: bool = False):
        if not self.init_todo_processor:
            self.init_todo_processor = True
            self.transformer.set_attn_processor(ToDoJointAttnProcessor())
            for block_idx, transformer_block in enumerate(self.transformer.transformer_blocks):
                transformer_block.attn.use_downsample = use_todo
                transformer_block.attn.layer_idx = block_idx
                transformer_block.attn.token_merge_factor = self.token_merge_factor
                transformer_block.attn.token_merge_method = self.token_merge_method
        if self.use_todo != use_todo: 
            for transformer_block in self.transformer.transformer_blocks:
                transformer_block.attn.use_downsample = use_todo
            self.use_todo = use_todo
        self.use_cache_and_tgate = use_cache_and_tgate \
            if self.use_cache_and_tgate != use_cache_and_tgate else self.use_cache_and_tgate

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        pooled_prompt_embeds: Optional[ms.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 77,
        use_cache_and_tgate: bool = False,
        use_todo: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`ms.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        self._enable_boost(use_cache_and_tgate=use_cache_and_tgate, use_todo=use_todo)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if self.use_cache_and_tgate:
            prompt_embeds_origin = prompt_embeds.copy()
            pooled_prompt_embeds_origin = pooled_prompt_embeds.copy()
        if self.do_classifier_free_guidance:
            prompt_embeds = ops.cat([negative_prompt_embeds, prompt_embeds], axis=0)
            pooled_prompt_embeds = ops.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], axis=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the transformer and will raise RuntimeError.
        lora_scale = self.joint_attention_kwargs.pop("scale", None) if self.joint_attention_kwargs is not None else None
        if lora_scale is not None:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self.transformer, lora_scale)

        if self.use_cache_and_tgate:
            latents_height, latents_width = latents.shape[-2:]
            patchs_num = (latents_height // self.transformer.config.patch_size) * (latents_width // self.transformer.config.patch_size)
            delta_cache = ops.zeros([2, patchs_num, 1536], dtype=ms.float16)
            delta_cache_hidden = ops.zeros([2, max_sequence_length + 77, 1536], dtype=ms.float16)

            cache_interval = self.cache_params[1]
            step_constrast = self.cache_params[3] % 2

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if not self.use_cache_and_tgate:
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = ops.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.broadcast_to((latent_model_input.shape[0],))
                    
                    ms_pyexecutor.sync()
                    ms_pyexecutor.set_async_for_graph(False)
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                    )[0]
                    ms_pyexecutor.set_async_for_graph(True)
                else:
                    if i < self.tgate:
                        latent_model_input = ops.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    else:
                        if i == self.tgate:
                            _, delta_cache = mint.chunk(delta_cache, 2)
                            _, delta_cache_hidden = mint.chunk(delta_cache_hidden, 2)
                        latent_model_input = latents
                    timestep = t.broadcast_to((latent_model_input.shape[0],))

                    ms_pyexecutor.sync()
                    ms_pyexecutor.set_async_for_graph(False)
                    if i < (self.cache_params[3] - 1):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            cache_params=self.cache_params,
                            if_skip=False,
                            use_cache=False,
                            delta_cache=delta_cache,
                            delta_cache_hidden=delta_cache_hidden,
                        )[0]
                    else:
                        noise_pred, delta_cache, delta_cache_hidden = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds if i < self.tgate else prompt_embeds_origin,
                            pooled_projections=pooled_prompt_embeds if i < self.tgate else pooled_prompt_embeds_origin,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            cache_params=self.cache_params,
                            if_skip=((i >= self.cache_params[3]) and (i % cache_interval == step_constrast)),
                            use_cache=True,
                            delta_cache=delta_cache,
                            delta_cache_hidden=delta_cache_hidden,
                        )
                    ms_pyexecutor.set_async_for_graph(True)

                # perform guidance
                if self.do_classifier_free_guidance and (not self.use_cache_and_tgate or i < self.tgate):
                    noise_pred_uncond, noise_pred_text = mint.chunk(noise_pred, 2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if lora_scale is not None:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self.transformer, lora_scale)

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.to(
                self.vae.dtype
            )  # for validation in training where vae and transformer might have different dtype

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)