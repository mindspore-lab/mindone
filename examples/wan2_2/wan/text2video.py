# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from typing import Any, Callable, Optional, Tuple, Union

from tqdm import tqdm

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.distributed as dist
import mindspore.nn as nn

from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .distributed.zero import free_model, shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:
    def __init__(
        self,
        config: Any,
        checkpoint_dir: str,
        rank: int = 0,
        t5_zero3: bool = False,
        dit_zero3: bool = False,
        use_sp: bool = False,
        t5_cpu: bool = False,
        init_on_cpu: bool = False,
        convert_model_dtype: bool = False,
    ) -> None:
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_zero3 (`bool`, *optional*, defaults to False):
                Enable ZeRO3 sharding for T5 model
            dit_zero3 (`bool`, *optional*, defaults to False):
                Enable ZeRO3 sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_zero3.
            init_on_cpu (`bool`, *optional*, defaults to False):
                Enable initializing Transformer Model on CPU. Only works without ZeRO3 or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
        """
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_zero3 or dit_zero3 or use_sp:
            self.init_on_cpu = False

        shard_fn = shard_model
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_zero3 else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        with nn.no_init_parameters():
            self.low_noise_model = WanModel.from_pretrained(checkpoint_dir, subfolder=config.low_noise_checkpoint)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_zero3=dit_zero3,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype,
        )

        with nn.no_init_parameters():
            self.high_noise_model = WanModel.from_pretrained(checkpoint_dir, subfolder=config.high_noise_checkpoint)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_zero3=dit_zero3,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype,
        )
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self._previous_model_name = None

    def _configure_model(
        self,
        model: WanModel,
        use_sp: bool,
        dit_zero3: bool,
        shard_fn: Callable[[nn.Cell], nn.Cell],
        convert_model_dtype: bool,
    ) -> nn.Cell:
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (mindspore.nn.Cell):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_zero3 (`bool`):
                Enable ZeRO3 sharding for DiT model.
            shard_fn (callable):
                The function to apply ZeRO3 sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without ZeRO3.

        Returns:
            mindspore.nn.Cell:
                The configured model.
        """
        model.set_train(False)
        for param in model.trainable_params():
            param.requires_grad = False

        if use_sp:
            for block in model.blocks:
                block.self_attn.construct = types.MethodType(sp_attn_forward, block.self_attn)
            model.construct = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_zero3:
            model = shard_fn(model)

        if convert_model_dtype:
            model.to(self.param_dtype)

        return model

    def _prepare_model_for_timestep(self, t: ms.Tensor, boundary: int, offload_model: bool) -> nn.Cell:
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (ms.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            mindspore.nn.Cell:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = "high_noise_model"
            offload_model_name = "low_noise_model"
        else:
            required_model_name = "low_noise_model"
            offload_model_name = "high_noise_model"

        if self._previous_model_name is None:
            self._previous_model_name = required_model_name

        if offload_model and offload_model_name == self._previous_model_name:
            free_model(self, self._previous_model_name)
            self._previous_model_name = required_model_name

        return getattr(self, required_model_name)

    def generate(
        self,
        input_prompt: str,
        size: Tuple[int, int] = (1280, 720),
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 50,
        guide_scale: Union[float, Tuple[float, float]] = 5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = False,
    ) -> Optional[ms.Tensor]:
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to False):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            mindspore.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )

        seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (self.patch_size[1] * self.patch_size[2])
                * target_shape[1]
                / self.sp_size
            )
            * self.sp_size
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = ms.Generator()
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            context = self.text_encoder([input_prompt])
            context_null = self.text_encoder([n_prompt])
            if offload_model:
                free_model(self, "text_encoder")
        else:
            context = self.text_encoder([input_prompt])
            context_null = self.text_encoder([n_prompt])

        noise = [
            mint.randn(
                (target_shape[0], target_shape[1], target_shape[2], target_shape[3]), dtype=ms.float32, generator=seed_g
            )
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, "no_sync", noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, "no_sync", noop_no_sync)

        # evaluation mode
        with (
            # torch.amp.autocast("cuda", dtype=self.param_dtype),
            no_sync_low_noise(),
            no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
                )
                sample_scheduler.set_timesteps(sampling_steps, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(sample_scheduler, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {"context": context, "seq_len": seq_len}
            arg_null = {"context": context_null, "seq_len": seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = mint.stack(timestep)

                model = self._prepare_model_for_timestep(t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

                noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
                )[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                free_model(self, self._previous_model_name)

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            ms.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
