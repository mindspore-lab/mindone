# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
import os
import random
import sys
from functools import partial

from tqdm import tqdm

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.distributed as dist
from mindspore.communication import GlobalComm, get_group_size
from mindspore.nn.utils import no_init_parameters

from mindone.trainers.zero import prepare_network

from .acceleration.parallel_states import create_parallel_group
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:
    def __init__(
        self,
        config,
        checkpoint_dir,
        rank=0,
        t5_zero3=False,
        dit_zero3=False,
        use_usp=False,
        t5_cpu=False,
    ):
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
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_zero3.
        """
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if use_usp:
            self.sp_size = get_group_size(GlobalComm.WORLD_COMM_GROUP)
            create_parallel_group(self.sp_size)
        else:
            self.sp_size = 1

        shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_zero3 else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint), dtype=self.param_dtype)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        with no_init_parameters():
            self.model = WanModel.from_pretrained(checkpoint_dir, mindspore_dtype=self.param_dtype)
        self.model.init_parameters_data()
        self.model.set_train(False)
        for param in self.model.trainable_params():
            param.requires_grad = False

        # TODO: GlobalComm.INITED -> mint.is_initialzed
        if GlobalComm.INITED:
            dist.barrier()
        if dit_zero3:
            self.model = shard_fn(self.model)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(
        self,
        input_prompt,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=False,
    ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
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
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        noise = [
            mint.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=self.param_dtype,
                generator=seed_g,
            )
        ]

        # evaluation mode
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

            noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]

            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        if offload_model:
            raise NotImplementedError()
        if self.rank == 0:
            # TODO: handle this
            # np.save("latent.npy", x0[0].to(ms.float32).asnumpy())
            videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            raise NotImplementedError()
        # TODO: GlobalComm.INITED -> mint.is_initialzed
        if GlobalComm.INITED:
            dist.barrier()

        return videos[0] if self.rank == 0 else None
