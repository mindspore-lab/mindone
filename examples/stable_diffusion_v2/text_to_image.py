'''
Text to image generation
'''
import logging
import os
import time
import sys
import argparse
from PIL import Image
from omegaconf import OmegaConf

import numpy as np
import mindspore as ms

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workspace)
from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.modules.lora import inject_trainable_lora
from ldm.modules.logger import set_logger
from ldm.util import str2bool, is_old_ms_version

logger = logging.getLogger("text_to_image")

def seed_everything(seed):
    if seed:
        ms.set_seed(seed)
        np.random.seed(seed)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, use_lora=False, lora_rank=4, lora_fp16=True, lora_only_ckpt=None, verbose=False):
    logger.info(f"Loading model from {ckpt}")
    model = instantiate_from_config(config.model)

    def _load_model(_model, ckpt_fp, verbose=True):
        if os.path.exists(ckpt_fp):
            param_dict = ms.load_checkpoint(ckpt_fp)
            if param_dict:
                if is_old_ms_version():
                    param_not_load = ms.load_param_into_net(_model, param_dict)
                else:
                    param_not_load, ckpt_not_load = ms.load_param_into_net(_model, param_dict)
                if verbose:
                    logger.info("Net params not loaded:", [p for p in param_not_load if not p.startswith('adam')])
                #logger.info("ckpt not load:", [p for p in ckpt_not_load if not p.startswith('adam')])
        else:
            logger.warning(f"!!!Warning!!!: {ckpt_fp} doesn't exist")

    if use_lora:
        logger.info('Loading LoRA model.')
        load_lora_only = True if lora_only_ckpt is not None else False
        if not load_lora_only:
            injected_attns, injected_trainable_params = inject_trainable_lora(
                                                            model,
                                                            rank=lora_rank,
                                                            use_fp16=(model.model.diffusion_model.dtype==ms.float16),
                                                                )
            _load_model(model, ckpt)

        else:
            # load the main pratrained model
            _load_model(model, ckpt)
            logger.info('Pretrained SD loaded')
            # inject lora params
            injected_attns, injected_trainable_params = inject_trainable_lora(
                                                            model,
                                                            rank=lora_rank,
                                                            use_fp16=(model.model.diffusion_model.dtype==ms.float16),
                                                                )
            # load finetuned lora params
            _load_model(model, lora_only_ckpt, verbose=False)
            logger.info('LoRA params loaded.')

        #assert len(injected_attns)==32, 'Expecting 32 injected attention modules, but got {len(injected_attns)}'

    else:
        _load_model(model, ckpt)

    model.set_train(False)
    for param in model.trainable_params():
        param.requires_grad = False

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        nargs="?",
        default="",
        help="path to a file containing prompt list (each line in the file correspods to a prompt to render)."
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        nargs="?",
        default="2.0",
        help="Stable diffusion version, 1.x or 2.0. 1.x support Chinese prompts. 2.0 support English prompts."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="A cute wolf in winter forest",
        help="the prompt to render"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        nargs="?",
        default="output",
        help="dir to write results to"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to config which constructs model. If None, select by version",
    )
    parser.add_argument('--use_lora', default=False, type=str2bool, help='whether the checkpoint used for inference is finetuned from LoRA')
    parser.add_argument('--lora_rank', default=4, type=int, help='lora rank. The bigger, the larger the LoRA model will be, but usually gives better generation quality.')
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--lora_ckpt_path",
        type=str,
        default=None,
        help="path to lora only checkpoint. Set it if use_lora is not None",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default='logging.INFO',
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    opt = parser.parse_args()
    # overwrite env var by parsed arg
    if opt.version:
        os.environ['SD_VERSION'] = opt.version
    if opt.ckpt_path is None:
        opt.ckpt_path = "models/wukong-huahua-ms.ckpt" if opt.version.startswith('1.') else "models/stablediffusionv2_512.ckpt"
    if opt.config is None:
        opt.config = "configs/v1-inference-chinese.yaml" if opt.version.startswith('1.') else "configs/v2-inference.yaml"
    if opt.scale is None:
        opt.scale = 7.5 if opt.version.startswith('1.') else 9.0


    work_dir = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"WORK DIR:{work_dir}")

    os.makedirs(opt.output_path, exist_ok=True)
    outpath = opt.output_path

    # set logger
    set_logger(
            name="",
            output_dir=outpath,
            rank=0,
            log_level=logging.INFO, #eval(opt.log_level),
        )

    
    # read prompts
    batch_size = opt.n_samples
    if not opt.data_path:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        logger.info(f"Reading prompts from {opt.data_path}")
        with open(opt.data_path, "r") as f:
            prompts = f.read().splitlines()
            num_prompts = len(prompts)
            # TODO: try to put different prompts in a batch
            data = [batch_size * [prompt] for prompt in prompts]
    
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # set ms context
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(
        mode=ms.context.GRAPH_MODE,
        device_target="Ascend",
        device_id=device_id,
        max_device_memory="30GB"
    )

    seed_everything(opt.seed)
    
    # create model
    if not os.path.isabs(opt.config):
        opt.config = os.path.join(work_dir, opt.config)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(
                        config,
                        ckpt=opt.ckpt_path,
                        use_lora=opt.use_lora,
                        lora_rank=opt.lora_rank,
                        lora_only_ckpt=opt.lora_ckpt_path,
                        )

    
    # create sampler
    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
        sname = 'dpm_solver' 
    else:
        sampler = PLMSSampler(model)
        sname = 'plms' 

    # log
    key_info = '\n' + '=' * 40 + '\n'
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {0}",
            f"Distributed mode: False",
            f"Number of input prompts: {len(data)}",
            f"Number of trials for each prompt: {opt.n_iter}",
            f"Number of samples in each trial: {opt.n_samples}",
            f"Model: StableDiffusion v{opt.version}",
            f"Checkpont path: {opt.ckpt_path}",
            f"Lora checkpoint path: {opt.lora_ckpt_path if opt.use_lora else None}",
            f"Use fp16: {model.model.diffusion_model.dtype==ms.float16}",
            f"Sampler: {sname}",
            f"Sampling steps: {opt.ddim_steps}",
        ]
    )
    key_info += "\n" + "=" * 40
    logger.info(key_info) 

    # infer
    start_code = None
    if opt.fixed_code:
        stdnormal = ms.ops.StandardNormal()
        start_code = stdnormal((opt.n_samples, 4, opt.H // 8, opt.W // 8))

    all_samples = list()
    for i, prompts in enumerate(data):
        logger.info("[{}/{}] Generating {} images for prompts:\n{}".format(i, len(data), batch_size, prompts[0]))
        for n in range(opt.n_iter):
            start_time = time.time()

            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c = model.get_learned_conditioning(prompts)
            shape = [4, opt.H // 8, opt.W // 8]
            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                            conditioning=c,
                                            batch_size=opt.n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=opt.scale,
                                            unconditional_conditioning=uc,
                                            eta=opt.ddim_eta,
                                            x_T=start_code
                                            )
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = ms.ops.clip_by_value((x_samples_ddim + 1.0) / 2.0,
                                                  clip_value_min=0.0, clip_value_max=1.0)
            x_samples_ddim_numpy = x_samples_ddim.asnumpy()

            if not opt.skip_save:
                for x_sample in x_samples_ddim_numpy:
                    x_sample = 255. * x_sample.transpose(1, 2, 0)
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1

            if not opt.skip_grid:
                all_samples.append(x_samples_ddim_numpy)

            end_time = time.time()
            logger.info(f"the infer time of a batch is {end_time-start_time}")

    logger.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
      f" \nEnjoy.")

if __name__ == "__main__":
    main()
