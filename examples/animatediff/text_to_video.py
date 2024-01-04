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
from ad.utils.load_models import load_model_from_config, merge_motion_lora_to_unet

from mindone.utils.config import instantiate_from_config
from mindone.utils.logger import set_logger
from mindone.utils.params import load_param_into_net_with_filter
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.ms_mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.ms_mode,
        device_target=args.target_device,
        device_id=device_id,
        # max_device_memory="30GB", # adapt for 910b
    )
    if args.target_device == "Ascend":
        # TODO: test needed? on 910b ms2.2-1028
        # ms.set_context(ascend_config={"precision_mode": "allow_fp32_to_fp16"})  # Only effective on Ascend 901B
        pass

    return device_id


def main(args):
    # set work dir and save dir
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{Path(args.config).stem}-{time_str}"
    set_logger(name="", output_dir=save_dir)

    # 0. parse and merge config
    # 1) sd config, 2) db ckpt path, 3) lora ckpt path, 4) mm ckpt path, 5) unet additional args, 6) noise schedule args
    config = OmegaConf.load(args.config)
    task_name = list(config.keys())[0]  # TODO: support multiple tasks
    ad_config = config[task_name]

    dreambooth_path = ad_config.get("dreambooth_path", "")
    # lora_model_path = ad_config.get("lora_model_path", "")
    # style_lora_alpha = lora_scale = ad_config.get("lora_alpha", 0.8)

    motion_module_paths = ad_config.get("motion_module", "")
    motion_module_path = motion_module_paths[0]  # TODO: support testing multiple ckpts
    motion_lora_config = ad_config.get("motion_module_lora_configs", [None])[0]

    seeds, steps, guidance_scale = ad_config.get("seed", 0), ad_config.steps, ad_config.guidance_scale
    prompts = ad_config.prompt
    n_prompts = ad_config.n_prompt
    seeds = [seeds] * len(prompts) if isinstance(seeds, int) else seeds

    sd_config = OmegaConf.load(args.sd_config)
    sd_model_path = args.pretrained_model_path

    # mm_zero_initialize = sd_config.model.params.unet_config.params.motion_module_kwargs.get("zero_initialize", True)

    if dreambooth_path != "":
        if os.path.exists(dreambooth_path):
            sd_model_path = dreambooth_path  # DB params naming rule is the same sd ldm
        else:
            logger.warning(f"dreambooth path {dreambooth_path} not exist.")
    # use_lora = True if lora_model_path != "" else False

    # TODO: merge unet addition kwargs to sd_confg
    inference_config = OmegaConf.load(ad_config.get("inference_config", args.inference_config))
    # unet_additional_kwargs = inference_config.unet_additional_kwargs
    noise_scheduler_kwargs = inference_config.noise_scheduler_kwargs
    use_motion_module = sd_config.model.params.unet_config.params.use_motion_module

    # 1. init env
    init_env(args)
    set_random_seed(42)

    # 2. build model components for ldm
    # 1)  create vae, text encoder, and unet and load weights
    sd_model = load_model_from_config(
        sd_config,
        ckpt=sd_model_path,
        use_motion_module=use_motion_module,  # indicate unet 2d->3d param name changes
    )

    text_encoder = sd_model.cond_stage_model
    unet = sd_model.model
    vae = sd_model.first_stage_model

    # load motion module weights if use mm
    add_ldm_prefix = True
    ldm_prefix = "model.diffusion_model."
    if use_motion_module:
        logger.info("Loading motion module from {}".format(motion_module_path))
        mm_state_dict = ms.load_checkpoint(motion_module_path)

        # add prefix (used in the whole sd model) to param if needed
        mm_pnames = list(mm_state_dict.keys())
        for pname in mm_pnames:
            if add_ldm_prefix:
                if not pname.startswith(ldm_prefix):
                    new_pname = ldm_prefix + pname
                    mm_state_dict[new_pname] = mm_state_dict.pop(pname)

        params_not_load, ckpt_not_load = load_param_into_net_with_filter(
            unet,
            mm_state_dict,
            filter=mm_state_dict.keys(),
        )
        if len(ckpt_not_load) > 0:
            logger.warning("The following params in mm ckpt are not loaded into net: {}".format(ckpt_not_load))
        assert len(ckpt_not_load) == 0, "All params in motion module must be loaded"

        # motion lora
        if motion_lora_config is not None:
            _mlora_path, alpha = motion_lora_config["path"], motion_lora_config["alpha"]
            logger.info("Loading motion lora from {}".format(_mlora_path))
            unet = merge_motion_lora_to_unet(unet, _mlora_path, alpha)

    # 2) ddim sampler
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
        noise = np.random.randn(bs, 4, args.L, args.H // 8, args.W // 8)
        inputs["noise"] = ms.Tensor(noise, ms.float16)

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
    parser.add_argument("--inference_config", type=str, default="configs/inference/inference-v2.yaml")
    # Use ldm config method instead of diffusers and transformers
    parser.add_argument("--sd_config", type=str, default="configs/stable_diffusion/v1-inference-unet3d.yaml")

    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    # MS new args
    parser.add_argument("--target_device", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )

    args = parser.parse_args()

    print(args)

    main(args)
