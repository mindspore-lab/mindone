# Adapted from https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V to work with MindSpore.
import random
import time
from pathlib import Path

import numpy as np
from hyvideo.constants import NEGATIVE_PROMPT, NEGATIVE_PROMPT_I2V, PRECISION_TO_TYPE, PROMPT_TEMPLATE
from hyvideo.diffusion.pipelines import HunyuanVideoPipeline
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.modules import load_model
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.text_encoder import TextEncoder
from hyvideo.utils.data_utils import align_to, generate_crop_size_list, get_closest_ratio
from hyvideo.utils.lora_utils import load_lora_for_pipeline
from hyvideo.vae import load_vae
from loguru import logger
from PIL import Image

import mindspore as ms
from mindspore import amp, mint
from mindspore.dataset import transforms, vision

from mindone.utils.amp import auto_mixed_precision


def parallelize_transformer(pipe):
    raise NotImplementedError


class Inference(object):
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        logger=None,
        parallel_args=None,
    ):
        self.vae = vae
        self.vae_kwargs = vae_kwargs

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        self.model = model
        self.pipeline = pipeline
        self.use_cpu_offload = use_cpu_offload

        self.args = args
        self.logger = logger
        self.parallel_args = parallel_args

    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, **kwargs):
        """
        Initialize the Inference pipeline.

        Args:
            pretrained_model_path (str or pathlib.Path): The model path, including t2v(t2i), text encoder and vae checkpoints.
            args (argparse.Namespace): The arguments for the pipeline.
            device (int): The device for inference. Default is 0.
        """
        # ========================================================================
        logger.info(f"Got text-to-video model root path: {pretrained_model_path}")
        parallel_args = None

        # =========================== Build main model ===========================
        logger.info("Building model...")
        factor_kwargs = {
            "dtype": PRECISION_TO_TYPE[args.precision],
            "attn_mode": args.attn_mode,
            "use_conv2d_patchify": args.use_conv2d_patchify,
            "i2v_condition_type": args.i2v_condition_type,
        }

        dtype = factor_kwargs["dtype"]
        rank_id = kwargs.get("rank_id", 0)
        if args.i2v_mode and args.i2v_condition_type == "latent_concat":
            in_channels = args.latent_channels * 2 + 1
            image_embed_interleave = 2
        elif args.i2v_mode and args.i2v_condition_type == "token_replace":
            in_channels = args.latent_channels
            image_embed_interleave = 4
        else:
            in_channels = args.latent_channels
            image_embed_interleave = 1
        out_channels = args.latent_channels

        if args.embedded_cfg_scale:
            factor_kwargs["guidance_embed"] = True

        model = load_model(
            name=args.model,
            zero_stage=args.zero_stage,
            text_states_dim=args.text_states_dim,
            text_states_dim_2=args.text_states_dim_2,
            in_channels=in_channels,
            out_channels=out_channels,
            factor_kwargs=factor_kwargs,
        )
        if args.use_fp8:
            raise NotImplementedError("fp8 is not supported yet.")

        if args.enable_ms_amp and dtype != ms.float32:
            logger.warning(f"Use MS auto mixed precision, amp_level: {args.amp_level}")
            if args.amp_level == "auto":
                amp.auto_mixed_precision(model, amp_level=args.amp_level, dtype=dtype)
            else:
                from hyvideo.modules.embed_layers import SinusoidalEmbedding
                from hyvideo.modules.norm_layers import LayerNorm, RMSNorm

                whitelist_ops = [
                    LayerNorm,
                    RMSNorm,
                    SinusoidalEmbedding,
                ]
                logger.info("custom fp32 cell for dit: ", whitelist_ops)
                model = auto_mixed_precision(
                    model, amp_level=args.amp_level, dtype=dtype, custom_fp32_cells=whitelist_ops
                )

        model = Inference.load_state_dict(args, model, pretrained_model_path, rank_id=rank_id)
        model.set_train(False)

        # ============================= Build extra models ========================
        # VAE
        vae, _, s_ratio, t_ratio = load_vae(
            args.vae,
            precision=args.vae_precision,
            logger=logger,
        )
        vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

        # Text encoder
        if args.i2v_mode:
            args.text_encoder = "llm-i2v"
            args.tokenizer = "llm-i2v"
            args.prompt_template = "dit-llm-encode-i2v"
            args.prompt_template_video = "dit-llm-encode-video-i2v"

        if args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
        elif args.prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = args.text_len + crop_start

        # prompt_template
        prompt_template = PROMPT_TEMPLATE[args.prompt_template] if args.prompt_template is not None else None

        # prompt_template_video
        prompt_template_video = (
            PROMPT_TEMPLATE[args.prompt_template_video] if args.prompt_template_video is not None else None
        )

        if args.text_embed_path is not None:
            text_encoder = None
            text_encoder_2 = None
        else:
            text_encoder = TextEncoder(
                text_encoder_type=args.text_encoder,
                max_length=max_length,
                text_encoder_precision=args.text_encoder_precision,
                tokenizer_type=args.tokenizer,
                i2v_mode=args.i2v_mode,
                prompt_template=prompt_template,
                prompt_template_video=prompt_template_video,
                hidden_state_skip_layer=args.hidden_state_skip_layer,
                apply_final_norm=args.apply_final_norm,
                logger=logger,
                image_embed_interleave=image_embed_interleave,
                enable_ms_amp=False,
            )
            if args.text_encoder_2 is not None:
                text_encoder_2 = TextEncoder(
                    text_encoder_type=args.text_encoder_2,
                    max_length=args.text_len_2,
                    text_encoder_precision=args.text_encoder_precision_2,
                    tokenizer_type=args.tokenizer_2,
                    logger=logger,
                    enable_ms_amp=True,
                )

        return cls(
            args=args,
            vae=vae,
            vae_kwargs=vae_kwargs,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            model=model,
            use_cpu_offload=args.use_cpu_offload,
            logger=logger,
            parallel_args=parallel_args,
        )

    @staticmethod
    def load_state_dict(args, model, pretrained_model_path, rank_id=0):
        load_key = args.load_key
        if args.i2v_mode:
            dit_weight = Path(args.i2v_dit_weight)
            task_name = "t2i"
        else:
            dit_weight = Path(args.dit_weight)
            task_name = "t2v"

        if dit_weight is None:
            model_dir = pretrained_model_path / f"{task_name}_{args.model_resolution}"
            files = list(model_dir.glob("*.pt"))
            if len(files) == 0:
                raise ValueError(f"No model weights found in {model_dir}")
            if str(files[0]).startswith("pytorch_model_"):
                model_path = dit_weight / f"pytorch_model_{load_key}.pt"
            elif any(str(f).endswith("_model_states.pt") for f in files):
                files = [f for f in files if str(f).endswith("_model_states.pt")]
                model_path = files[0]
                if len(files) > 1:
                    logger.warning(f"Multiple model weights found in {dit_weight}, using {model_path}")
            else:
                raise ValueError(
                    f"Invalid model path: {dit_weight} with unrecognized weight format: "
                    f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                    f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                    f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                    f"specific weight file, please provide the full path to the file."
                )
        else:
            if dit_weight.is_dir():
                if args.zero_stage is None:
                    files = list(dit_weight.glob("*.pt"))
                    if len(files) == 0:
                        raise ValueError(f"No model weights found in {dit_weight}")
                    if str(files[0]).startswith("pytorch_model_"):
                        model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                    elif any(str(f).endswith("_model_states.pt") for f in files):
                        files = [f for f in files if str(f).endswith("_model_states.pt")]
                        model_path = files[0]
                        if len(files) > 1:
                            logger.warning(f"Multiple model weights found in {dit_weight}, using {model_path}")
                    else:
                        raise ValueError(
                            f"Invalid model path: {dit_weight} with unrecognized weight format: "
                            f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                            f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                            f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                            f"specific weight file, please provide the full path to the file."
                        )
                else:
                    if args.zero_stage != 3:
                        raise ValueError(
                            f"The {dit_weight} should a file path instead of a directory for zero-stage {args.zero_stage}"
                            f"Please check the weightfile and try again."
                        )
                    else:
                        subfolders = list(dit_weight.glob("rank_*"))
                        if len(subfolders) == 0:
                            raise ValueError(f"No sharded weights found in {dit_weight}")
                        subfolder = dit_weight / f"rank_{rank_id}" / "ckpt"
                        assert subfolder.exists(), f"Subfolder {subfolder} not found"
                        files = list(subfolder.glob("*.ckpt"))
                        if len(files) == 0:
                            raise ValueError(f"No model weights found in {subfolder}")
                        model_path = sorted(files)[-1]
                        if len(files) > 1:
                            logger.warning(f"Multiple model weights found in {subfolder}, using {model_path}")

            elif dit_weight.is_file():
                model_path = dit_weight
            else:
                if "rank_*" in str(dit_weight):
                    # a wildcard pattern
                    model_path = Path(str(dit_weight).replace("rank_*", f"rank_{rank_id}"))
                else:
                    raise ValueError(f"Invalid model path: {dit_weight}")

        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")
        logger.info(f"Loading checkpoint {model_path}...")

        model.load_from_checkpoint(str(model_path))

        return model

    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        return size


class HunyuanVideoSampler(Inference):
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        logger=None,
        parallel_args=None,
    ):
        super().__init__(
            args,
            vae,
            vae_kwargs,
            text_encoder,
            model,
            text_encoder_2=text_encoder_2,
            pipeline=pipeline,
            use_cpu_offload=use_cpu_offload,
            logger=logger,
            parallel_args=parallel_args,
        )

        self.pipeline = self.load_diffusion_pipeline(
            args=args,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            model=self.model,
        )
        if args.i2v_mode:
            self.default_negative_prompt = NEGATIVE_PROMPT_I2V
            if args.use_lora:
                self.pipeline = load_lora_for_pipeline(
                    self.pipeline,
                    args.lora_path,
                    LORA_PREFIX_TRANSFORMER="Hunyuan_video_I2V_lora",
                    alpha=args.lora_scale,
                )
                logger.info(f"load lora {args.lora_path} into pipeline, lora scale is {args.lora_scale}.")
        else:
            self.default_negative_prompt = NEGATIVE_PROMPT
        self.default_negative_prompt = NEGATIVE_PROMPT

    def load_diffusion_pipeline(
        self,
        args,
        vae,
        text_encoder,
        text_encoder_2,
        model,
        scheduler=None,
        progress_bar_config=None,
        data_type="video",
    ):
        """Load the denoising scheduler for inference."""
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift,
                    reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")

        pipeline = HunyuanVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            progress_bar_config=progress_bar_config,
            args=args,
        )
        if self.use_cpu_offload:
            # pipeline.enable_sequential_cpu_offload()
            raise NotImplementedError

        return pipeline

    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        # 884
        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        if isinstance(self.model.patch_size, int):
            assert all(s % self.model.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            assert all(s % self.model.patch_size[idx] == 0 for idx, s in enumerate(latents_size)), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.model.hidden_size // self.model.heads_num
        rope_dim_list = self.model.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.args.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    # @torch.no_grad()
    def predict(
        self,
        prompt,
        height=192,
        width=336,
        video_length=129,
        seed=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6,
        flow_shift=5.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        text_embed_path=None,
        output_type="pil",
        i2v_mode=False,
        i2v_resolution="720p",
        i2v_image_path=None,
        i2v_condition_type=None,
        i2v_stability=True,
        **kwargs,
    ):
        """
        Predict the image/video from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                height (int): The height of the output video. Default is 192.
                width (int): The width of the output video. Default is 336.
                video_length (int): The frame number of the output video. Default is 129.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                infer_steps (int): The number of inference steps. Default is 50.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                flow_shift (float): The flow shift for the generation. Default is 5.0.
                embedded_guidance_scale (float): embedded guidance scale for the generation. Default is None.
                batch_size (int): batch size for inference. Default is 1.
                num_images_per_prompt (int): The number of images per prompt. Default is 1.
                i2v_mode (bool): Whether to open i2v mode. Default is False.
                i2v_resolution (str): Resolution for i2v inference. Default is 720p.
                i2v_image_path (str): Image path for i2v inference. Default is None.
        """
        out_dict = dict()

        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if isinstance(seed, ms.Tensor):
            seed = seed.asnumpy().tolist()
        if seed is None:
            seeds = [
                # NOTE: original is random.randint(0, 1_000_000)
                random.randint(0, 100)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [seed + i for _ in range(batch_size) for i in range(num_videos_per_prompt)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) + j for i in range(batch_size) for j in range(num_videos_per_prompt)]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
        # TODO: can enable it to check align with torch
        # generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        generator = [np.random.Generator(np.random.PCG64(seed=seed)) for seed in seeds]
        out_dict["seeds"] = seeds

        # ========================================================================
        # Arguments: target_width, target_height, target_video_length
        # ========================================================================
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(f"`video_length-1` must be a multiple of 4, got {video_length}")

        logger.info(f"Input (height, width, video_length) = ({height}, {width}, {video_length})")

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if prompt is None:
            if text_embed_path is None:
                raise ValueError("Either `prompt` or `text_embed_path` must be specified.")
            prompt = ""

        if not isinstance(prompt, str):
            raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        # negative prompt
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")
        negative_prompt = [negative_prompt.strip()]

        if text_embed_path is not None:
            # read embedding from folder
            data = np.load(text_embed_path)
            prompt_embeds = data["prompt_embeds"]
            prompt_mask = data["prompt_mask"]
            prompt_embeds_2 = data["prompt_embeds_2"]

            if len(prompt_embeds.shape) == 2:
                # (seq_len, embed_dim) -> (1, seq_len, embed_dim)
                prompt_embeds = np.expand_dims(prompt_embeds, axis=0)
                prompt_mask = np.expand_dims(prompt_mask, axis=0)
                prompt_embeds_2 = np.expand_dims(prompt_embeds_2, axis=0)

            prompt_embeds = ms.Tensor(prompt_embeds)
            prompt_mask = ms.Tensor(prompt_mask, dtype=ms.bool_)
            prompt_embeds_2 = ms.Tensor(prompt_embeds_2)

            if self.args.cfg_scale > 1.0:
                negative_prompt_embeds = data["negative_prompt_embeds"]
                negative_prompt_mask = data["negative_prompt_mask"]
                negative_prompt_embeds_2 = data["negative_prompt_embeds_2"]
                negative_prompt_embeds = ms.Tensor(negative_prompt_embeds)
                negative_prompt_mask = ms.Tensor(negative_prompt_mask, dtype=ms.bool_)
                negative_prompt_embeds_2 = ms.Tensor(negative_prompt_embeds_2)
            else:
                negative_prompt_embeds = None
                negative_prompt_mask = None
                negative_prompt_embeds_2 = None
        else:
            prompt_embeds = None
            prompt_mask = None
            prompt_embeds_2 = None
            negative_prompt_embeds = None
            negative_prompt_mask = None
            negative_prompt_embeds_2 = None

        # ========================================================================
        # Scheduler
        # ========================================================================
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift, reverse=self.args.flow_reverse, solver=self.args.flow_solver
        )
        self.pipeline.scheduler = scheduler
        # ---------------------------------
        # Reference condition
        # ---------------------------------
        img_latents = None
        semantic_images = None
        if i2v_mode:
            if i2v_resolution == "720p":
                bucket_hw_base_size = 960
            elif i2v_resolution == "540p":
                bucket_hw_base_size = 720
            elif i2v_resolution == "360p":
                bucket_hw_base_size = 480
            else:
                raise ValueError(f"i2v_resolution: {i2v_resolution} must be in [360p, 540p, 720p]")

            semantic_images = [Image.open(i2v_image_path).convert("RGB")]
            origin_size = semantic_images[0].size

            crop_size_list = generate_crop_size_list(bucket_hw_base_size, 32)
            aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
            closest_size, closest_ratio = get_closest_ratio(
                origin_size[1], origin_size[0], aspect_ratios, crop_size_list
            )

            ref_image_transform = transforms.Compose(
                [
                    vision.Resize(closest_size, interpolation=vision.Inter.BILINEAR),
                    vision.CenterCrop(closest_size),
                    vision.ToTensor(),
                    vision.Normalize([0.5], [0.5], is_hwc=False),
                ]
            )
            semantic_image_pixel_values = [
                ms.tensor(ref_image_transform(semantic_image)[0]) for semantic_image in semantic_images
            ]
            semantic_image_pixel_values = mint.cat(semantic_image_pixel_values).unsqueeze(0).unsqueeze(2)

            img_latents = self.pipeline.vae.encode(semantic_image_pixel_values, sample_posterior=False)  # B, C, F, H, W
            img_latents = img_latents * self.pipeline.vae.config.scaling_factor

            target_height, target_width = closest_size
        # ========================================================================
        # Build Rope freqs
        # ========================================================================
        # TODO: part of RopE can be pre-compute
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(target_video_length, target_height, target_width)
        n_tokens = freqs_cos.shape[0]

        # ========================================================================
        # Print infer args
        # ========================================================================
        debug_str = f"""
                        height: {target_height}
                         width: {target_width}
                  video_length: {target_video_length}
                        prompt: {prompt}
               text_embed_path: {text_embed_path}
                    neg_prompt: {negative_prompt}
                          seed: {seed}
                   infer_steps: {infer_steps}
         num_videos_per_prompt: {num_videos_per_prompt}
                guidance_scale: {guidance_scale}
                      n_tokens: {n_tokens}
                    flow_shift: {flow_shift}
       embedded_guidance_scale: {embedded_guidance_scale}
                 i2v_stability: {i2v_stability}"""
        logger.debug(debug_str)

        # ========================================================================
        # Pipeline inference
        # ========================================================================
        start_time = time.time()
        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type=output_type,
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=True,
            vae_ver=self.args.vae,
            enable_tiling=self.args.vae_tiling,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_mask=negative_prompt_mask,
            prompt_embeds_2=prompt_embeds_2,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            i2v_mode=i2v_mode,
            i2v_condition_type=i2v_condition_type,
            i2v_stability=i2v_stability,
            img_latents=img_latents,
            semantic_images=semantic_images,
        )
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict
