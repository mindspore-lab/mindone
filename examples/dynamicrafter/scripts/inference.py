import argparse
import os
import sys
import logging
import datetime
import glob
from typing import Union, List, Tuple

from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
import yaml
import random
from PIL import Image
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision import Resize, CenterCrop, ToTensor, Normalize
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore import JitConfig

sys.path.append("../stable_diffusion_v2/")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from utils import read_captions_from_csv, read_captions_from_txt, _check_cfgs_in_parser
# from ldm.models.diffusion.ddim import DDIMSampler
from lvdm.models.samplers.ddim import DDIMSampler

from lvdm.modules.networks.util import rearrange_in_gn5d_bs, rearrange_out_gn5d

from mindone.utils.logger import set_logger
from mindone.utils.misc import to_abspath
from mindone.utils.seed import set_random_seed
from mindone.utils.config import instantiate_from_config, str2bool
from mindone.visualize.videos import save_videos
from mindone.utils.amp import auto_mixed_precision

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
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


def _transform_before_save(video):
    video = ops.transpose(video, (0, 2, 3, 4, 1))
    video = video.asnumpy()  # check the dtype
    video = np.clip(video, -1, 1)
    video = (video + 1.0) / 2.0
    return video

def save_results_seperate(prompt, samples, filename, fakedir, fps=10, loop=False):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        path = os.path.join(savedirs[idx], f'{filename.split(".")[0]}_sample{idx}.mp4')
        video_transform = _transform_before_save(video)
        # import pdb;pdb.set_trace()
        save_videos(video_transform, path)
        """
        # b,c,t,h,w
        # video = video.detach().cpu()
        if loop: # remove the last frame
            video = video[:,:,:-1,...]
        video = ops.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i,...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(ms.uint8).permute(1, 2, 3, 0) #thwc
            path = os.path.join(savedirs[idx].replace('samples', 'samples_separate'), f'{filename.split(".")[0]}_sample{i}.mp4')
            torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})
        # save_videos(frames: np.ndarray, path: str, fps: Union[int, float] = 8, loop=0, concat=False):
        """

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange_out_gn5d(videos)
    # x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange_in_gn5d_bs(z, b=b)
    # z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(model,
                           prompts,
                           videos,
                           noise_shape,
                           n_samples=1,
                           ddim_steps=50,
                           ddim_eta=1.,
                            unconditional_guidance_scale=1.0,
                            cfg_img=None,
                            fs=None,
                            text_input=False,
                            multiple_cond_cfg=False,
                            loop=False,
                            interp=False,
                            timestep_spacing='uniform',
                            guidance_rescale=0.0,
                            **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    # fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)
    fs = ms.Tensor([fs] * batch_size, dtype=ms.int64)

    if not text_input:
        prompts = [""]*batch_size

    img = videos[:,:,0] #bchw
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [ops.cat([cond_emb,img_emb], axis=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b c t h w
        if loop or interp:
            img_cat_cond = ops.zeros_like(z)
            img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
            img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
        else:
            img_cat_cond = z[:,:,:1,:,:]
            img_cat_cond = ops.repeat_interleave(img_cat_cond, repeats=z.shape[2], axis=2)
            # img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond] # b c 1 h w
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = ops.zeros_like(cond_emb)
        uc_img_emb = model.embedder(ops.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [ops.cat([uc_emb,uc_img_emb], axis=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [ops.cat([uc_emb,img_emb], axis=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:

            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=cond_mask,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )
        # import pdb;pdb.set_trace()
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        # import pdb;pdb.set_trace()
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = ops.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def main(args):
    if args.append_timestr:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir = f"{args.savedir}/{time_str}"
    else:
        save_dir = f"{args.savedir}"

    os.makedirs(save_dir, exist_ok=True)
    # if args.save_latent:
    #     latent_dir = os.path.join(args.output_path, "denoised_latents")
    #     os.makedirs(latent_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    rank_id, device_num = init_env(
        args.mode,
        args.seed,
        args.use_parallel,
        device_target=args.device_target,
        debug=args.debug,
    )

    # 1.1 get captions prompt_path
    # if args.prompt_path.endswith(".csv"):
    #     captions = read_captions_from_csv(args.prompt_path)
    # elif args.prompt_path.endswith(".txt"):
    #     captions = read_captions_from_txt(args.prompt_path)

    assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    filename_list, data_list, prompt_list = load_data_prompts(
                                                args.prompt_dir, 
                                                video_size=(args.height, args.width), 
                                                video_frames=args.video_length, 
                                                interp=args.interp
                                                )

    # TODO: data parallel split, support parallel inference 
    # captions, base_data_idx = data_parallel_split(captions, rank_id, device_num)
    # print(f"Num captions for rank {rank_id}: {len(captions)}")


    # 2. model initiate and weight loading
    # TODO: model mixed precision setting, torch autocast
        ## model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    # model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model.perframe_ae = args.perframe_ae
    # import pdb;pdb.set_trace()

    # assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    
    if args.ckpt_path:
        logger.info(f"Loading ckpt {args.ckpt_path} into model")
        assert os.path.exists(args.ckpt_path), f"{args.ckpt_path} not found."
        params = ms.load_checkpoint(args.ckpt_path)
        m, u = ms.load_param_into_net(model, params)
        logger.info(f"{len(m)} net params not load: {m}")
        logger.info(f"{len(u)} ckpt params not load: {u}")
        # model.load_from_checkpoint(args.ckpt_path)
    else:
        logger.warning(f"Model uses random initialization!")

    model.set_train(False)

    # Backend mode setting:
    if args.mode == 0:  # jit_config only takes effect in Graph mode
        ms.set_context(jit_config={"jit_level": args.jit_level})

    # mixed precision setting
    WHITELIST_OPS = [nn.GroupNorm, nn.LayerNorm]
    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}
    if args.dtype in ["fp16", "bf16"]:
        model = auto_mixed_precision(
            model, amp_level=args.amp_level, dtype=dtype_map[args.dtype], custom_fp32_cells=WHITELIST_OPS
        )


    """get ms params
    ms_params = [p for p in model.parameters_and_names()]
    with open(f"tools/ms_param_{args.width}.txt", "w") as f:
        for k, v in ms_params:
            # k = k.replace(".gamma", ".weight").replace(".beta", ".bias")
            f.write(k + ":" + str(v.shape) + ":" + str(v.dtype).lower() + "\n")
    print(f"Num of params of ms weight: {len(ms_params)}")       
    import pdb;pdb.set_trace()
    """
    # run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]
    
    # fakedir = os.path.join(args.savedir, "samples")
    # fakedir_separate = os.path.join(args.savedir, "samples_separate")

    # # os.makedirs(fakedir, exist_ok=True)
    # os.makedirs(fakedir_separate, exist_ok=True)

    # Print key info
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"MindSpore jit_level: {args.jit_level if args.mode == 0 else 'NOT in effect in PYNATIVE mode'}",
            f"amp_level: {args.amp_level}",
            f"amp_level with dtype: {args.dtype}",
            f"enable_flash_attention: {model.model.diffusion_model.enable_flash_attention}",
            f"DDIM steps: {args.ddim_steps}",
            f"Height*Width: {args.height}*{args.width}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    for idx, indice in tqdm(enumerate(range(0, len(prompt_list), args.bs)), desc='Sample Batch'):
                prompts = prompt_list[indice:indice+args.bs]
                videos = data_list[indice:indice+args.bs]
                filenames = filename_list[indice:indice+args.bs]
                if isinstance(videos, list):
                    videos = ops.stack(videos, axis=0)
                else:
                    videos = videos.unsqueeze(0)

                batch_samples = image_guided_synthesis(model,
                                                       prompts,
                                                       videos,
                                                       noise_shape,
                                                       args.n_samples,
                                                       args.ddim_steps,
                                                       args.ddim_eta,
                                                       args.unconditional_guidance_scale,
                                                       args.cfg_img,
                                                       args.frame_stride,
                                                       args.text_input,
                                                       args.multiple_cond_cfg,
                                                       args.loop,
                                                       args.interp,
                                                       args.timestep_spacing,
                                                       args.guidance_rescale,
                                                       )
                # import pdb;pdb.set_trace()
                ## save each example individually
                for n, samples in enumerate(batch_samples):
                    ## samples : [n_samples,c,t,h,w]
                    prompt = prompts[n]
                    filename = filenames[n]
                    # save_results(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
                    save_results_seperate(prompt, samples, filename, save_dir, fps=8, loop=args.loop)

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
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
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
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for model. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--amp_level",
        default="O2",
        type=str,
        help="mindspore amp level, O1: most fp32, only layers in whitelist compute in fp16 (dense, conv, etc); \
            O2: most fp16, only layers in blacklist compute in fp32 (batch norm etc)",
    )

    # parser.add_argument(
    #     "--output_path",
    #     type=str,
    #     default="samples",
    #     help="output dir to save the generated videos",
    # )
    # parser.add_argument(
    #     "--save_latent",
    #     type=str2bool,
    #     default=True,
    #     help="Save denoised video latent. If True, the denoised latents will be saved in $output_path/denoised_latents",
    # )
    
    # MS new args
    parser.add_argument("--jit_level", default="O0", type=str, help="model jit config level, refer to MindSpore doc")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--debug", type=str2bool, default=False, help="Execute inference in debug mode.")
    
    ## currently not support looping video and generative frame interpolation
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")    
    # default_args = parser.parse_args()

    # __dir__ = os.path.dirname(os.path.abspath(__file__))
    # abs_path = os.path.abspath(os.path.join(__dir__, ".."))
    # if default_args.config:
    #     logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
    #     default_args.config = to_abspath(abs_path, default_args.config)
    #     with open(default_args.config, "r") as f:
    #         cfg = yaml.safe_load(f)
    #         _check_cfgs_in_parser(cfg, parser)
    #         parser.set_defaults(**cfg)
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
