"""
AnimateDiff inference pipeline
"""
import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import mindspore as ms

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from ad.pipelines.infer_engine import AnimateDiffText2Video
from ad.utils.cond_data import transform_conditional_images
from ad.utils.load_models import build_model_from_config, load_adapter_lora, load_controlnet, load_motion_modules

from mindone.utils.config import instantiate_from_config, str2bool
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.ms_mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.ms_mode,
        device_target=args.device_target,
        device_id=device_id,
    )

    return device_id


def main(args):
    # set work dir and save dir
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{Path(args.config).stem}-{time_str}"
    set_logger(name="", output_dir=save_dir)

    # 0. parse and merge config
    # 1) sd config, 2) db ckpt path, 3) lora ckpt path, 4) mm ckpt path, 5) unet additional args, 6) noise schedule args
    config = OmegaConf.load(args.config)

    # support multiple tasks
    for task_name in config.keys():
        ad_config = config[task_name]

        dreambooth_path = ad_config.get("dreambooth_path", "")
        adapter_lora_path = ad_config.get("adapter_lora_path", "")
        controlnet_path = ad_config.get("controlnet_path", "")
        # lora_model_path = ad_config.get("lora_model_path", "")
        # style_lora_alpha = lora_scale = ad_config.get("lora_alpha", 0.8)

        motion_module_paths = ad_config.get("motion_module", "")
        motion_module_path = motion_module_paths[0]  # TODO: support testing multiple ckpts
        if args.motion_module_path != "":
            motion_module_path = args.motion_module_path

        if not hasattr(ad_config, "motion_module_lora_configs"):
            ad_config.motion_module_lora_configs = [dict(path="", alpha=1.0)]
        if args.motion_lora_path != "":
            ad_config.motion_module_lora_configs[0].path = args.motion_lora_path
        if args.motion_lora_alpha is not None:
            ad_config.motion_module_lora_configs[0].alpha = args.motion_lora_alpha
        motion_lora_config = ad_config.motion_module_lora_configs[0]

        seeds, steps, guidance_scale = ad_config.get("seed", 0), ad_config.steps, ad_config.guidance_scale
        prompts = ad_config.prompt
        n_prompts = list(ad_config.n_prompt) * len(prompts) if len(ad_config.n_prompt) == 1 else ad_config.n_prompt
        if args.prompt != "":
            prompts[0] = args.prompt
        seeds = [seeds] * len(prompts) if isinstance(seeds, int) else seeds
        assert len(n_prompts) == len(
            prompts
        ), f"Expect that the negative prompts length is the same as the positive prompts length, but got {len(n_prompts)} and {len(prompts)}"
        assert len(seeds) >= len(
            prompts
        ), f"Expect int seed or seed list of length no smamller than number of positive prompts, but got {len(seeds)} and {len(prompts)}"

        sd_config = OmegaConf.load(ad_config.get("sd_config", args.sd_config))
        sd_model_path = args.pretrained_model_path

        # mm_zero_initialize = sd_config.model.params.unet_config.params.motion_module_kwargs.get("zero_initialize", True)

        if dreambooth_path != "":
            if os.path.exists(dreambooth_path):
                sd_model_path = dreambooth_path  # DB params naming rule is the same sd ldm
            else:
                logger.warning(f"dreambooth path {dreambooth_path} not exist.")
        # use_lora = True if lora_model_path != "" else False
        if adapter_lora_path != "":
            if not os.path.exists(adapter_lora_path):
                logger.warning(f"domain adapter path {adapter_lora_path} not exist. Will not apply adapter lora model.")
            adapter_lora_scale = ad_config.get("domain_lora_scale", 1.0)
        controlnet_images, controlnet_image_index = None, None
        if controlnet_path != "":
            if not os.path.exists(controlnet_path):
                logger.warning(f"sparse control encoder path {controlnet_path} not exist. Will not use controlnet.")
            controlnet_images = ad_config.get("controlnet_images", "")
            assert controlnet_images != "", "controlnet_images must be provided when using controlnet"
            controlnet_image_index = list(ad_config.get("controlnet_image_indexs", [0]))
            assert len(controlnet_image_index) != 0, "controlnet_image_index must be provided when using controlnet"
            controlnet_image_index = ms.Tensor(controlnet_image_index, dtype=ms.int64)

        # TODO: merge unet addition kwargs to sd_confg
        inference_config = OmegaConf.load(ad_config.get("inference_config", args.inference_config))
        # unet_additional_kwargs = inference_config.unet_additional_kwargs
        noise_scheduler_kwargs = inference_config.noise_scheduler_kwargs
        use_motion_module = sd_config.model.params.unet_config.params.use_motion_module
        use_controlnet = controlnet_path != "" and os.path.exists(controlnet_path)
        use_adapter_lora = adapter_lora_path != "" and os.path.exists(adapter_lora_path)

        sd_config.model.params.first_stage_config.params.use_fp16 = args.vae_fp16

        # 1. init env
        init_env(args)
        set_random_seed(42)

        # 2. build model and load weights
        # 1)  create vae, text encoder, and unet and load sd weights
        sd_model = build_model_from_config(
            sd_config,
            ckpt=sd_model_path,
            use_motion_module=use_motion_module,  # indicate unet 2d->3d param name changes
        )
        # load lora to sd if applicable
        if use_adapter_lora:
            sd_model = load_adapter_lora(sd_model, adapter_lora_path, adapter_lora_scale)
        # load sparse control encoder to sd if applicable
        if use_controlnet:
            sd_model = load_controlnet(sd_model, controlnet_path)

        text_encoder = sd_model.cond_stage_model
        unet = sd_model.model
        vae = sd_model.first_stage_model

        # 2) load motion module weights
        if args.all_in_one_ckpt != "":
            # allow loading the whole single file
            logger.info("Loading all in one ckpt from {}".format(args.all_in_one_ckpt))
            param_dict = ms.load_checkpoint(args.all_in_one_ckpt)
            param_not_load, ckpt_not_load = ms.load_param_into_net(sd_model, param_dict)
            assert (
                len(param_not_load) == len(ckpt_not_load) == 0
            ), "Exist ckpt params not loaded: {} (total: {})\nor net params not loaded: {} (total: {})".format(
                ckpt_not_load, len(ckpt_not_load), param_not_load, len(param_not_load)
            )
        elif use_motion_module:
            unet = load_motion_modules(unet, motion_module_path, motion_lora_config=motion_lora_config)

            # set mixed precision for mm
            # mm_amp_level = args.mm_amp_level
            # unet.diffusion_model = unet.diffusion_model.set_mm_amp_level(mm_amp_level)

        # 3) initiate controlnet and load controlnet ckpt
        ad_config.W = ad_config.get("W", args.W)
        ad_config.H = ad_config.get("H", args.H)
        ad_config.L = ad_config.get("L", args.L)

        if use_controlnet:
            image_paths = controlnet_images
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            assert (
                len(image_paths) <= ad_config.L
            ), f"The length of provided image paths must be equal to the number of frames L={ad_config.L}"

            controlnet_images = transform_conditional_images(
                image_paths, ad_config.H, ad_config.W, random_crop=True, normalize=False, save_dir=save_dir
            )  # （b f c h w)
            controlnet_images = ms.Tensor(controlnet_images).transpose((0, 2, 1, 3, 4))  # （b f c h w) -> (b c f h w)

        # ddim sampler
        # TODO: merge noise_scheduler_kwargs and ddim.yaml
        sampler_config = OmegaConf.load("configs/inference/scheduler/ddim.yaml")  # base template
        sampler_config.params.beta_start = noise_scheduler_kwargs.beta_start  # overwrite
        sampler_config.params.beta_end = noise_scheduler_kwargs.beta_end
        sampler_config.params.beta_schedule = noise_scheduler_kwargs.beta_schedule

        logger.info(f"noise beta scheduler: {sampler_config.params.beta_schedule}")

        scheduler = instantiate_from_config(sampler_config)

        # 3. build inference pipeline
        pipeline = AnimateDiffText2Video(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=sd_model.scale_factor,
            num_inference_steps=steps,
        )

        if use_controlnet and unet.diffusion_model.controlnet.use_simplified_condition_embedding:
            b, c, f, h, w = controlnet_images.shape
            controlnet_images = controlnet_images.transpose((0, 2, 1, 3, 4)).reshape(
                (b * f, c, h, w)
            )  # (b c f h w) -> (b*f c h w)
            controlnet_images = pipeline.vae_encode(controlnet_images * 2.0 - 1.0)
            _, c, h, w = controlnet_images.shape
            controlnet_images = controlnet_images.reshape((b, f, c, h, w)).transpose(
                (0, 2, 1, 3, 4)
            )  # （b f c h w) -> (b c f h w)

        # 4. run sampling for multiple samples
        num_prompts = len(prompts)
        bs = 1  # batch size
        sample_idx = 0
        for i in range(num_prompts):
            ms.set_seed(seeds[i])
            prompt = prompts[i]
            n_prompt = n_prompts[i]

            # creat inputs
            inputs = {}
            inputs["prompt"] = prompt
            inputs["prompt_data"] = sd_model.tokenize([prompt] * bs)
            inputs["negative_prompt"] = n_prompt
            inputs["negative_prompt_data"] = sd_model.tokenize([n_prompt] * bs)
            # inputs["timesteps"] = timesteps
            inputs["scale"] = ms.Tensor(guidance_scale, ms.float16)

            # latent noisy frames: b c f h w
            shape = (bs, 4, ad_config.L, ad_config.H // 8, ad_config.W // 8)
            if args.init_latent_path is not None:
                logger.info(f"Loading latent noise from {args.init_latent_path}")
                noise = np.load(args.init_latent_path)
                assert noise.shape == shape, "Shape of loaded init latent noise is not correct."
            else:
                noise = np.random.randn(*shape)

            inputs["noise"] = ms.Tensor(noise, ms.float16)

            # control images
            inputs["controlnet_images"] = controlnet_images
            inputs["controlnet_image_index"] = controlnet_image_index

            logger.info(f"Sampling prompt: {prompts[i]}")
            start_time = time.time()

            # infer
            x_samples = pipeline(inputs)  # (b f H W 3)
            x_samples = x_samples.asnumpy()
            # print("D--: pipeline output ", x_samples.shape)

            end_time = time.time()

            # save result
            os.makedirs(save_dir, exist_ok=True)
            prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_fp = f"{save_dir}/{sample_idx}-{prompt}.gif"
            save_videos(x_samples, save_fp, loop=0)

            # save_videos_grid(sample, f"{save_dir}/sample/{sample_idx}-{prompt}.gif")
            logger.info(f"save to {save_fp}")
            sample_idx += 1

            logger.info("Time cost: {:.3f}s".format(end_time - start_time))

    logger.info(f"Done! All generated images are saved in: {save_dir}" f"\nEnjoy.")
    OmegaConf.save(config, f"{save_dir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/prompts/v2/1-ToonYou.yaml")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="models/stable_diffusion/sd_v1.5-d0ab7146.ckpt",
    )
    parser.add_argument("--all_in_one_ckpt", type=str, default="", help="if not empty, load SD+mm from this file")
    parser.add_argument("--inference_config", type=str, default="configs/inference/inference-v2.yaml")
    parser.add_argument(
        "--vae_fp16",
        type=str2bool,
        default=None,
        help="whether use fp16 on vae. If None, will use the precision defined in `sd_config`. Should keep it same as vae precision set in training."
        "For inference with checkpoints coverted from torch, should set it True.",
    )
    parser.add_argument(
        "--init_latent_path",
        type=str,
        default=None,
        help="path to initial latent noise (npy file). If not None, seed will not make effect and the initial latent noise will be used for sampling.",
    )
    parser.add_argument(
        "--motion_module_path",
        type=str,
        default="",
        help="if not empty, overwrite the path in configs/prompts/{version}/{task}.yaml",
    )
    parser.add_argument(
        "--motion_lora_path",
        type=str,
        default="",
        help="if not empty, overwrite the path in configs/prompts/{version}/{task}.yaml",
    )
    parser.add_argument(
        "--motion_lora_alpha",
        type=int,
        default=None,
        help="if not empty, overwrite the path in configs/prompts/{version}/{task}.yaml",
    )
    # Use ldm config method instead of diffusers and transformers
    parser.add_argument("--sd_config", type=str, default="configs/stable_diffusion/v1-inference-mmv2.yaml")

    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="text prompt. If not empty, it will overwrite the prompt defined in `config` yaml",
    )

    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )

    args = parser.parse_args()

    print(args)

    main(args)
