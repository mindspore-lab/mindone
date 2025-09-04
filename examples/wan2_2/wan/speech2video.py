# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from decord import VideoReader
from PIL import Image
from tqdm import tqdm

import mindspore as ms
import mindspore.dataset.vision as vision
import mindspore.mint as mint
import mindspore.mint.distributed as dist
import mindspore.nn as nn

from .distributed.util import get_world_size
from .distributed.zero import free_model, shard_model
from .modules.s2v.audio_encoder import AudioEncoder
from .modules.s2v.model_s2v import WanModel_S2V, sp_attn_forward_s2v
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanS2V:
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
        Initializes the image-to-video generation model components.

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
                No usage now. For compatibility only.
            init_on_cpu (`bool`, *optional*, defaults to False):
                No usage now. For compatibility only.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
        """
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
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

        self.vae = Wan2_1_VAE(vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        with nn.no_init_parameters():
            self.noise_model = WanModel_S2V.from_pretrained(checkpoint_dir)
        self.noise_model = self._configure_model(
            model=self.noise_model,
            use_sp=use_sp,
            dit_zero3=dit_zero3,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype,
        )

        with nn.no_init_parameters():
            self.audio_encoder = AudioEncoder(model_id=os.path.join(checkpoint_dir, "wav2vec2-large-xlsr-53-english"))
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.motion_frames = config.transformer.motion_frames
        self.drop_first_motion = config.drop_first_motion
        self.fps = config.sample_fps
        self.audio_sample_m = 0

    def _configure_model(
        self,
        model: WanModel_S2V,
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

        Returns:
            mindspore.nn.Cell:
                The configured model.
        """
        model.set_train(False)
        for param in model.trainable_params():
            param.requires_grad = False

        if use_sp:
            for block in model.blocks:
                block.self_attn.construct = types.MethodType(sp_attn_forward_s2v, block.self_attn)
            model.use_context_parallel = True

        if dist.is_initialized():
            dist.barrier()

        if dit_zero3:
            model = shard_fn(model)

        if convert_model_dtype:
            model.to(self.param_dtype)

        return model

    def get_size_less_than_area(
        self, height: int, width: int, target_area: int = 1024 * 704, divisor: int = 64
    ) -> Tuple[int, int]:
        if height * width <= target_area:
            # If the original image area is already less than or equal to the target,
            # no resizing is needed—just padding. Still need to ensure that the padded area doesn't exceed the target.
            max_upper_area = target_area
            min_scale = 0.1
            max_scale = 1.0
        else:
            # Resize to fit within the target area and then pad to multiples of `divisor`
            max_upper_area = target_area  # Maximum allowed total pixel count after padding
            d = divisor - 1
            b = d * (height + width)
            a = height * width
            c = d**2 - max_upper_area

            # Calculate scale boundaries using quadratic equation
            min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (2 * a)  # Scale when maximum padding is applied
            max_scale = math.sqrt(max_upper_area / (height * width))  # Scale without any padding

        # We want to choose the largest possible scale such that the final padded area does not exceed max_upper_area
        # Use binary search-like iteration to find this scale
        find_it = False
        for i in range(100):
            scale = max_scale - (max_scale - min_scale) * i / 100
            new_height, new_width = int(height * scale), int(width * scale)

            # Pad to make dimensions divisible by 64
            pad_height = (64 - new_height % 64) % 64
            pad_width = (64 - new_width % 64) % 64

            padded_height, padded_width = new_height + pad_height, new_width + pad_width

            if padded_height * padded_width <= max_upper_area:
                find_it = True
                break

        if find_it:
            return padded_height, padded_width
        else:
            # Fallback: calculate target dimensions based on aspect ratio and divisor alignment
            aspect_ratio = width / height
            target_width = int((target_area * aspect_ratio) ** 0.5 // divisor * divisor)
            target_height = int((target_area / aspect_ratio) ** 0.5 // divisor * divisor)

            # Ensure the result is not larger than the original resolution
            if target_width >= width or target_height >= height:
                target_width = int(width // divisor * divisor)
                target_height = int(height // divisor * divisor)

            return target_height, target_width

    def prepare_default_cond_input(
        self,
        map_shape: List[int] = [3, 12, 64, 64],
        motion_frames: int = 5,
        lat_motion_frames: int = 2,
        enable_mano: bool = False,
        enable_kp: bool = False,
        enable_pose: bool = False,
    ) -> Optional[ms.Tensor]:
        default_value = [1.0, -1.0, -1.0]
        cond_enable = [enable_mano, enable_kp, enable_pose]
        cond = []
        for d, c in zip(default_value, cond_enable):
            if c:
                map_value = mint.ones(map_shape, dtype=self.param_dtype) * d
                cond_lat = mint.cat([map_value[:, :, 0:1].repeat(1, 1, motion_frames, 1, 1), map_value], dim=2)
                cond_lat = mint.stack(self.vae.encode(cond_lat.to(self.param_dtype)))[:, :, lat_motion_frames:].to(
                    self.param_dtype
                )

                cond.append(cond_lat)
        if len(cond) >= 1:
            cond = mint.cat(cond, dim=1)
        else:
            cond = None
        return cond

    def encode_audio(self, audio_path: str, infer_frames: int) -> Tuple[ms.Tensor, int]:
        z = self.audio_encoder.extract_audio_feat(audio_path, return_all_layers=True)
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m
        )
        audio_embed_bucket = audio_embed_bucket.to(self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat

    def read_last_n_frames(
        self, video_path: str, n_frames: int, target_fps: int = 16, reverse: bool = False
    ) -> np.ndarray:
        """
        Read the last `n_frames` from a video at the specified frame rate.

        Parameters:
            video_path (str): Path to the video file.
            n_frames (int): Number of frames to read.
            target_fps (int, optional): Target sampling frame rate. Defaults to 16.
            reverse (bool, optional): Whether to read frames in reverse order.
                                    If True, reads the first `n_frames` instead of the last ones.

        Returns:
            np.ndarray: A NumPy array of shape [n_frames, H, W, 3], representing the sampled video frames.
        """
        vr = VideoReader(video_path)
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)

        interval = max(1, round(original_fps / target_fps))

        required_span = (n_frames - 1) * interval

        start_frame = max(0, total_frames - required_span - 1) if not reverse else 0

        sampled_indices = []
        for i in range(n_frames):
            indice = start_frame + i * interval
            if indice >= total_frames:
                break
            else:
                sampled_indices.append(indice)

        return vr.get_batch(sampled_indices).asnumpy()

    def load_pose_cond(
        self, pose_video: str, num_repeat: int, infer_frames: int, size: Tuple[int, int]
    ) -> List[ms.Tensor]:
        HEIGHT, WIDTH = size
        if pose_video is not None:
            pose_seq = self.read_last_n_frames(
                pose_video, n_frames=infer_frames * num_repeat, target_fps=self.fps, reverse=True
            )

            resize_opreat = vision.Resize(min(HEIGHT, WIDTH))
            crop_opreat = vision.CenterCrop((HEIGHT, WIDTH))

            cond_tensor = ms.from_numpy(pose_seq)
            cond_tensor = cond_tensor.permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
            cond_tensor = crop_opreat(resize_opreat(cond_tensor)).permute(1, 0, 2, 3).unsqueeze(0)

            padding_frame_num = num_repeat * infer_frames - cond_tensor.shape[2]
            cond_tensor = mint.cat([cond_tensor, -mint.ones([1, 3, padding_frame_num, HEIGHT, WIDTH])], dim=2)

            cond_tensors = mint.chunk(cond_tensor, num_repeat, dim=2)
        else:
            cond_tensors = [-mint.ones([1, 3, infer_frames, HEIGHT, WIDTH])]

        COND = []
        for r in range(len(cond_tensors)):
            cond = cond_tensors[r]
            cond = mint.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_lat = mint.stack(self.vae.encode(cond.to(dtype=self.param_dtype)))[:, :, 1:]  # for mem save
            COND.append(cond_lat)
        return COND

    def get_gen_size(self, size: int, max_area: int, ref_image_path: str, pre_video_path: str) -> Tuple[int, int]:
        if size is not None:
            HEIGHT, WIDTH = size
        else:
            if pre_video_path:
                ref_image = self.read_last_n_frames(pre_video_path, n_frames=1)[0]
            else:
                ref_image = np.array(Image.open(ref_image_path).convert("RGB"))
            HEIGHT, WIDTH = ref_image.shape[:2]
        HEIGHT, WIDTH = self.get_size_less_than_area(HEIGHT, WIDTH, target_area=max_area)
        return (HEIGHT, WIDTH)

    def generate(
        self,
        input_prompt: str,
        ref_image_path: str,
        audio_path: str,
        num_repeat: int = 1,
        pose_video: Optional[str] = None,
        max_area: int = 720 * 1280,
        infer_frames: int = 80,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 40,
        guide_scale: float = 5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = False,
        init_first_frame: bool = False,
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            ref_image_path ('str'):
                Input image path
            audio_path ('str'):
                Audio for video driven
            num_repeat ('int'):
                Number of clips to generate; will be automatically adjusted based on the audio length
            pose_video ('str'):
                If provided, uses a sequence of poses to drive the generated video
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            infer_frames (`int`, *optional*, defaults to 80):
                How many frames to generate per clips. The number should be 4n
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to False):
                If True, offloads models to CPU during generation to save VRAM
            init_first_frame (`bool`, *optional*, defaults to False):
                Whether to use the reference image as the first frame (i.e., standard image-to-video generation)

        Returns:
            mindspore.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        size = self.get_gen_size(size=None, max_area=max_area, ref_image_path=ref_image_path, pre_video_path=None)
        HEIGHT, WIDTH = size
        channel = 3

        resize_opreat = vision.Resize(min(HEIGHT, WIDTH))
        crop_opreat = vision.CenterCrop((HEIGHT, WIDTH))
        tensor_trans = vision.ToTensor()

        ref_image = None
        motion_latents = None

        if ref_image is None:
            ref_image = np.array(Image.open(ref_image_path).convert("RGB"))
        if motion_latents is None:
            motion_latents = mint.zeros([1, channel, self.motion_frames, HEIGHT, WIDTH], dtype=self.param_dtype)

        # extract audio emb
        audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)
        if num_repeat is None or num_repeat > nr:
            num_repeat = nr

        lat_motion_frames = (self.motion_frames + 3) // 4
        model_pic = crop_opreat(resize_opreat(Image.fromarray(ref_image)))

        ref_pixel_values = tensor_trans(model_pic)
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(0) * 2 - 1.0  # b c 1 h w
        ref_pixel_values = ref_pixel_values.to(dtype=self.vae.dtype)
        ref_latents = mint.stack(self.vae.encode(ref_pixel_values))

        # encode the motion latents
        videos_last_frames = motion_latents
        drop_first_motion = self.drop_first_motion
        if init_first_frame:
            drop_first_motion = False
            motion_latents[:, :, -6:] = ref_pixel_values
        motion_latents = mint.stack(self.vae.encode(motion_latents))

        # get pose cond input if need
        COND = self.load_pose_cond(pose_video=pose_video, num_repeat=num_repeat, infer_frames=infer_frames, size=size)

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        context = self.text_encoder([input_prompt])
        context_null = self.text_encoder([n_prompt])
        if offload_model:
            free_model(self, "text_encoder")

        out = []
        # evaluation mode
        with (
            # torch.amp.autocast('cuda', dtype=self.param_dtype),
        ):
            for r in range(num_repeat):
                seed_g = ms.Generator()
                seed_g.manual_seed(seed + r)

                lat_target_frames = (infer_frames + 3 + self.motion_frames) // 4 - lat_motion_frames
                target_shape = [lat_target_frames, HEIGHT // 8, WIDTH // 8]
                noise = [
                    mint.randn(
                        16, target_shape[0], target_shape[1], target_shape[2], dtype=self.param_dtype, generator=seed_g
                    )
                ]
                max_seq_len = np.prod(target_shape) // 4

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

                latents = noise
                left_idx = r * infer_frames
                right_idx = r * infer_frames + infer_frames
                cond_latents = COND[r] if pose_video else COND[0] * 0
                cond_latents = cond_latents.to(dtype=self.param_dtype)
                audio_input = audio_emb[..., left_idx:right_idx]
                input_motion_latents = motion_latents.clone()

                arg_c = {
                    "context": context[0:1],
                    "seq_len": max_seq_len,
                    "cond_states": cond_latents,
                    "motion_latents": input_motion_latents,
                    "ref_latents": ref_latents,
                    "audio_input": audio_input,
                    "motion_frames": [self.motion_frames, lat_motion_frames],
                    "drop_motion_frames": drop_first_motion and r == 0,
                }
                if guide_scale > 1:
                    arg_null = {
                        "context": context_null[0:1],
                        "seq_len": max_seq_len,
                        "cond_states": cond_latents,
                        "motion_latents": input_motion_latents,
                        "ref_latents": ref_latents,
                        "audio_input": 0.0 * audio_input,
                        "motion_frames": [self.motion_frames, lat_motion_frames],
                        "drop_motion_frames": drop_first_motion and r == 0,
                    }

                for _, t in enumerate(tqdm(timesteps)):
                    latent_model_input = latents[0:1]
                    timestep = [t]

                    timestep = mint.stack(timestep)

                    noise_pred_cond = self.noise_model(latent_model_input, t=timestep, **arg_c)

                    if guide_scale > 1:
                        noise_pred_uncond = self.noise_model(latent_model_input, t=timestep, **arg_null)
                        noise_pred = [u + guide_scale * (c - u) for c, u in zip(noise_pred_cond, noise_pred_uncond)]
                    else:
                        noise_pred = noise_pred_cond

                    temp_x0 = sample_scheduler.step(
                        noise_pred[0].unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
                    )[0]
                    latents[0] = temp_x0.squeeze(0)

                if offload_model:
                    free_model(self, "noise_model")

                latents = mint.stack(latents)
                if not (drop_first_motion and r == 0):
                    decode_latents = mint.cat([motion_latents, latents], dim=2)
                else:
                    decode_latents = mint.cat([ref_latents, latents], dim=2)
                image = mint.stack(self.vae.decode(decode_latents))
                image = image[:, :, -(infer_frames):]
                if drop_first_motion and r == 0:
                    image = image[:, :, 3:]

                overlap_frames_num = min(self.motion_frames, image.shape[2])
                videos_last_frames = mint.cat(
                    [videos_last_frames[:, :, overlap_frames_num:], image[:, :, -overlap_frames_num:]], dim=2
                )
                videos_last_frames = videos_last_frames.to(dtype=motion_latents.dtype)
                motion_latents = mint.stack(self.vae.encode(videos_last_frames))
                out.append(image)

        videos = mint.cat(out, dim=2)
        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            ms.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
