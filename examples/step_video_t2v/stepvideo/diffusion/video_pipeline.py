# Copyright 2025 StepFun Inc. All Rights Reserved.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pickle
import asyncio

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter, mint
from mindspore.communication.management import get_group_size, get_rank

from mindone.diffusers.pipelines.pipeline_utils import DiffusionPipeline
from mindone.diffusers.utils import BaseOutput

from stepvideo.modules.model import StepVideoModel
from stepvideo.diffusion.scheduler import FlowMatchDiscreteScheduler
from stepvideo.utils import VideoProcessor
from stepvideo.parallel import is_distribute


def call_api_gen(url, api):
    # url =f"http://{url}:{port}/{api}-api"
    vae_url = f"http://{url}:5001/{api}-api"
    llm_url = f"http://{url}:5000/{api}-api"

    import aiohttp
    async def _fn(samples, *args, **kwargs):
        if api=='vae':
            data = {
                    "samples": samples,
                }
            url = vae_url
        elif api == 'caption':
            data = {
                    "prompts": samples,
                }
            url = llm_url
        else:
            raise Exception(f"Not supported api: {api}...")
        
        async with aiohttp.ClientSession() as sess:
            data_bytes = pickle.dumps(data)
            async with sess.get(url, data=data_bytes, timeout=12000) as response:
                result = bytearray()
                while not response.content.at_eof():
                    chunk = await response.content.read(1024)
                    result += chunk
                response_data = pickle.loads(result)
        return response_data
        
    return _fn


@dataclass
class StepVideoPipelineOutput(BaseOutput):
    video: Union[Tensor, np.ndarray]
    

class StepVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using StepVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running, etc.).

    Args:
        transformer ([`StepVideoModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae_url:
            remote vae server's url.
        caption_url:
            remote caption (stepllm and clip) server's url.
    """

    def __init__(
        self,
        transformer: StepVideoModel,
        scheduler: FlowMatchDiscreteScheduler,
        vae_url: str = '127.0.0.1',
        caption_url: str = '127.0.0.1',
        save_path: str = './results',
        name_suffix: str = '',
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
        )
        
        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 8
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 16
        self.video_processor = VideoProcessor(save_path, name_suffix)
        
        self.vae_url = vae_url
        self.caption_url = caption_url
        self.setup_api(self.vae_url, self.caption_url)
        
    def setup_api(self, vae_url, caption_url):
        self.vae_url = vae_url
        self.caption_url = caption_url
        self.caption = call_api_gen(caption_url, 'caption')
        self.vae = call_api_gen(vae_url, 'vae')
        return self
    
    def encode_prompt(
        self,
        prompt: str,
        neg_magic: str = '',
        pos_magic: str = '',
    ):
        prompts = [prompt+pos_magic]
        bs = len(prompts)
        prompts += [neg_magic]*bs
        
        data = asyncio.run(self.caption(prompts))
        prompt_embeds, prompt_attention_mask, clip_embedding = Tensor(data['y']), Tensor(data['y_mask']), Tensor(data['clip_embedding'])

        return prompt_embeds, clip_embedding, prompt_attention_mask

    def decode_vae(self, samples: Tensor):
        samples = asyncio.run(self.vae(samples.asnumpy()))
        return samples

    def check_inputs(self, num_frames, width, height):
        num_frames = max(num_frames//17*17, 1)
        width = max(width//16*16, 16)
        height = max(height//16*16, 16)
        return num_frames, width, height

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: 64,
        height: int = 544,
        width: int = 992,
        num_frames: int = 204,
        dtype: Optional[int] = None,
        # TODO: add numpy generator
        latents: Optional[Tensor] = None,
    ) -> Tensor:
        if latents is not None:
            return latents.to(dtype=dtype)

        num_frames, width, height = self.check_inputs(num_frames, width, height)
        shape = (
            batch_size,
            max(num_frames//17*3, 1),
            num_channels_latents,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )   # b,f,c,h,w

        latents = mint.randn(shape, dtype=dtype)

        if is_distribute():
            latents = ops.AllGather()(latents[None])[0]
            print("synchronize latent between cards suceess.")

        return latents


    # @inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 544,
        width: int = 992,
        num_frames: int = 204,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        time_shift: float = 13.0,
        neg_magic: str = "",
        pos_magic: str = "",
        num_videos_per_prompt: Optional[int] = 1,
        # TODO: add numpy generator
        latents: Optional[Tensor] = None,
        output_type: Optional[str] = "mp4",
        output_file_name: Optional[str] = "",
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `544`):
                The height in pixels of the generated image.
            width (`int`, defaults to `992`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `204`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `9.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. 
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            output_file_name(`str`, *optional*`):
                The output mp4 file name.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`StepVideoPipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~StepVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`StepVideoPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        # 1. Check inputs. Raise error if not correct

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, prompt_embeds_2, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            neg_magic=neg_magic,
            pos_magic=pos_magic,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        prompt_embeds_2 = prompt_embeds_2.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            time_shift=time_shift
        )

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            ms.bfloat16,
            latents,
        )

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = mint.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(transformer_dtype)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                
                timestep = mint.broadcast_to(t, (latent_model_input.shape[0],)).to(latent_model_input.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    encoder_hidden_states_2=prompt_embeds_2,
                    return_dict=False,
                )
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents
                )
                
                progress_bar.update()

        # if not torch.distributed.is_initialized() or int(torch.distributed.get_rank())==0:
        if not is_distribute() or get_group_size() == 1 or get_rank() == 0:
            if not output_type == "latent":
                video = self.decode_vae(latents)  # np.ndarray, fp32
                
                # # zhy_test, numpy video save
                # _np_path = "./results/video.npy"
                # np.save(_np_path, video)
                # print(f"save video as numpy on {_np_path=}")

                # save video
                self.video_processor.postprocess_video(video, output_file_name=output_file_name, output_type=output_type)
            else:
                video = latents

            # Offload all models
            # self.maybe_free_model_hooks()

            if not return_dict:
                return (video, )

            return StepVideoPipelineOutput(video=video)
        