import argparse
import os
import sys
import logging
import datetime
import glob
from typing import Union, List, Tuple

import numpy as np
import yaml
import random
from PIL import Image
import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision import Resize, CenterCrop, ToTensor, Normalize
from mindspore.communication.management import get_group_size, get_rank, init

sys.path.append("../stable_diffusion_v2/")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from utils import read_captions_from_csv, read_captions_from_txt, _check_cfgs_in_parser
from ldm.models.diffusion.ddim import DDIMSampler

from mindone.utils.logger import set_logger
from mindone.utils.misc import to_abspath
from mindone.utils.seed import set_random_seed
# from mindone.visualize.videos import save_videos
from mindone.utils.config import instantiate_from_config, str2bool

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    enable_dvm: bool = False,
    debug: bool = False,
):
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)
    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if debug and mode == ms.GRAPH_MODE:  # force PyNative mode when debugging
        logger.warning("Debug mode is on, switching execution mode to PyNative.")
        mode = ms.PYNATIVE_MODE

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()

        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            pynative_synchronize=debug,
        )

    if enable_dvm:
        # FIXME: the graph_kernel_flags settting is a temp solution to fix dvm loss convergence in ms2.3-rc2. Refine it for future ms version.
        ms.set_context(enable_graph_kernel=True, graph_kernel_flags="--disable_cluster_ops=Pow,Select")

    return rank_id, device_num

# def data_parallel_split(x, device_id, device_num):
# # def data_parallel_split(data: Union[List, Tuple], device_id, device_num):
#     n = len(x)
#     shard_size = n // device_num
#     if device_id is None:
#         device_id = 0
#     base_data_idx = device_id * shard_size

#     if device_num in [None, 1]:
#         shard = x
#     if device_id == device_num - 1:
#         shard = x[device_id * shard_size :]
#     else:
#         shard = x[device_id * shard_size : (device_id + 1) * shard_size]

#     return shard, base_data_idx


def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

def load_data_prompts(data_dir, video_size=(256,256), video_frames=16, interp=False):
    transform = transforms.Compose([
        Resize(min(video_size)),
        CenterCrop(video_size),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), is_hwc=False)])
    ## load prompts
    prompt_file = get_filelist(data_dir, ['txt'])
    assert len(prompt_file) > 0, "Error: found NO prompt file!"
    ###### default prompt
    default_idx = 0
    default_idx = min(default_idx, len(prompt_file)-1)
    if len(prompt_file) > 1:
        print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
    ## only use the first one (sorted by name) if multiple exist
    
    ## load video
    file_list = get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
    # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
    data_list = []
    filename_list = []
    prompt_list = load_prompts(prompt_file[default_idx])
    n_samples = len(prompt_list)
    for idx in range(n_samples):
        if interp:
            image1 = Image.open(file_list[2*idx]).convert('RGB')
            image_tensor1 = ms.Tensor(transform(image1)[0]).unsqueeze(1) # [c,1,h,w]
            image2 = Image.open(file_list[2*idx+1]).convert('RGB')
            image_tensor2 = ms.Tensor(transform(image2)[0]).unsqueeze(1) # [c,1,h,w]
            frame_tensor1 = ops.repeat_interleave(image_tensor1, repeats=video_frames//2, axis=1)
            frame_tensor2 = ops.repeat_interleave(image_tensor2, repeats=video_frames//2, axis=1)
            # frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            # frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor = ops.cat([frame_tensor1, frame_tensor2], axis=1)
            # import pdb;pdb.set_trace()
            _, filename = os.path.split(file_list[idx*2])
        else:
            image = Image.open(file_list[idx]).convert('RGB')
            image_tensor = ms.Tensor(transform(image)[0]).unsqueeze(1)  # [c,1,h,w]
            frame_tensor = ops.repeat_interleave(image_tensor, repeats=video_frames, axis=1)
            # frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
            # import pdb;pdb.set_trace()
            _, filename = os.path.split(file_list[idx])

        data_list.append(frame_tensor)
        filename_list.append(filename)
        
    return filename_list, data_list, prompt_list


def main(args):
    if args.append_timestr:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir = f"{args.output_path}/{time_str}"
    else:
        save_dir = f"{args.output_path}"

    os.makedirs(save_dir, exist_ok=True)
    if args.save_latent:
        latent_dir = os.path.join(args.output_path, "denoised_latents")
        os.makedirs(latent_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    rank_id, device_num = init_env(
        args.mode,
        args.seed,
        args.use_parallel,
        device_target=args.device_target,
        enable_dvm=args.enable_dvm,
        debug=args.debug,
    )

    # 1.1 get captions prompt_path
    # if args.prompt_path.endswith(".csv"):
    #     captions = read_captions_from_csv(args.prompt_path)
    # elif args.prompt_path.endswith(".txt"):
    #     captions = read_captions_from_txt(args.prompt_path)
    filename_list, data_list, prompt_list = load_data_prompts(
                                                args.prompt_dir, 
                                                video_size=(args.height, args.width), 
                                                video_frames=args.num_frames, 
                                                interp=args.interp
                                                )

    # TODO: data parallel split, support parallel inference 
    # captions, base_data_idx = data_parallel_split(captions, rank_id, device_num)
    # print(f"Num captions for rank {rank_id}: {len(captions)}")


    # 2. model initiate and weight loading
    
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=3, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--num_frames", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument(
        "--append_timestr",
        type=str2bool,
        default=True,
        help="If true, an subfolder named with timestamp under output_path will be created to save the sampling results",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="samples",
        help="output dir to save the generated videos",
    )
    parser.add_argument(
        "--save_latent",
        type=str2bool,
        default=True,
        help="Save denoised video latent. If True, the denoised latents will be saved in $output_path/denoised_latents",
    )
    
    # MS new args
    parser.add_argument("--enable_dvm", default=False, type=str2bool, help="enable dvm mode")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--debug", type=str2bool, default=False, help="Execute inference in debug mode.")
    
    ## currently not support looping video and generative frame interpolation
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")    
    default_args = parser.parse_args()

    __dir__ = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(__dir__, ".."))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = to_abspath(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    # convert to absolute path, necessary for modelarts
    # args.ckpt_path = to_abspath(abs_path, args.ckpt_path)
    # args.vae_checkpoint = to_abspath(abs_path, args.vae_checkpoint)
    # args.prompt_path = to_abspath(abs_path, args.prompt_path)
    # args.output_path = to_abspath(abs_path, args.output_path)
    # args.text_embed_folder = to_abspath(abs_path, args.text_embed_folder)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
