'''
Extract depth map from input image, then generate new images conditioning on the depth map and the input prompt.

Examples:
    # Depth map will be extracted from input image. 
    $ python depth_to_image.py --prompt "two tiger" --image 000000039769.jpg
    # Depth map is given.
    $ python depth_to_image.py --prompt "two tiger" --depth_map 000000039769_depth.png

TODO:
    auto download midas weights to models/depth_estimator/.
    auto download sd-2-depth ckpt to models/.
    parallel running on multiple initial images 
'''

import argparse
import datetime
import logging
import math
import os
import shutil
import sys
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import cv2

import mindspore as ms
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops

workspace = os.path.dirname(os.path.abspath(__file__))
print("workspace:", workspace, flush=True)
sys.path.append(workspace)
from ldm.models.diffusion.plms import PLMSSampler

from ldm.modules.logger import set_logger
from ldm.modules.train.tools import set_random_seed
from ldm.util import instantiate_from_config

from conditions.midas_ms import midas_v3

logger = logging.getLogger("depth_to_image")


def get_depth_estimator(model_type='midas_v3_dpt_large_384', estimator_ckpt_path='models/depth_estimator/midas_v3_dpt_large_ms.ckpt'):
    if model_type == 'midas_v3_dpt_large_384':
        depth_model = midas_v3(pretrained=True, ckpt_path=estimator_ckpt_path)
    else:
        raise NotImplementedError
    
    return depth_model


def estimate_depth(images, depth_estimator):
    '''
    Use MiDas as depth estimator.
    Args:
        images: rgb image as PIL object, shape [h, w, 3], value: 0-255 
            or, list of PIL images  [n, h, w, 3]
            
    return: 
        depth map as numpy array, shpae [384, 384]
            or [n, 384, 384]
    '''
    if not isinstance(images, list):
        images = [images]

    # 1. preproess
    # hyper-params ref: https://huggingface.co/stabilityai/stable-diffusion-2-depth/blob/main/feature_extractor/preprocessor_config.json
    h = w = 384 # input image size for depth estimator 
    rescale=True
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # 1.1 resize to 384
    images = [img.resize((w, h),  resample=Image.BICUBIC) for img in images ]# resample=2 => BICUBIC
    images = [np.array(img, dtype=np.float32) for img in images]
    images = np.array(images, dtype=np.float32)  # [bs, h, w, 3]

    # 1.2 rescale to [0, 1]
    if rescale:
        images = images / 255.0
    # 1.3 normalize to [-1, 1]
    images = (images - mean) / std
    # 1.4 format tensor batch [bs, 3, h, w]
    images = np.transpose(images, (0, 3, 1, 2))
    images = Tensor(images, dtype=mstype.float32)
    assert len(images.shape)==4 and images.shape[1]==3, f"Expecting model input shape: [bs, 3, H, W], but got {image.shape}"
    
    # 2. infer 
    logger.info("Running depth estimation on input image...")
    st = time.time() 
    depth_maps = depth_estimator(images).asnumpy() # [bs, 1, h, w] 
    depth_maps = np.squeeze(depth_maps) # [bs, h, w] or [h, w]
    #print("Time cost: ", time.time() - st)
    logger.debug(f"depth est output: {}, {}, {}".format(depth_maps.shape, depth_maps.min(), depth_maps.max()))
    
    return depth_maps

def save_img(img_np, fn='tmp.png', norm=False, gray=False):
    from matplotlib import pyplot as plt
    #f, (ax1, ax2) = plt.subplots(1, 2)
    cmap = 'gray' if gray else 'viridis'
    if norm:
        new = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.float32)
        new = (img_np - img_np.min()) / (img_np.max() - img_np.min()) 
        #plt.imshow(new)
        plt.imsave(fn, new, cmap=cmap)
        return new
    else:
        #plt.imshow(img_np)
        plt.imsave(fn, img_np, cmap=cmap)
        return img_np


def prepare_latent_z(init_image=None, num_samples=4, h=512, w=512, vae_scale_factor=8, model=None, sampler=None,
    strength=1., sample_steps=50, seed=None):
    #timestep=50, ):
    '''
    prepare latent code z_T for diffusion
    if init_image is None, latent z is pure noise. Otherwsie, latient z is latent init image added with noise, where the noise is added according to forward diffusion timestep t.
    Args:
	init_image: None or PIL image (shape: h w 3, value: 0-255) or str of image path
        h, w: the target image size

    '''
    assert 0 <= strength <= 1., "strength must be in [0, 1]" 

    def vae_image_process(img, h=512, w=512):
        if isinstance(img, str):
            img = Image.open(img)
        img = img.convert("RGB")
        if img.size != (w, h):
            img = img.resize((w, h), resample=Image.LANCZOS) # default is LANCZOS in original vae preprocess, for best quality.
        img = np.array(img, dtype=np.float32)
        img /= 255.
        img = img * 2 - 1  # [-1, 1]
        img = ms.Tensor([img])
        return img

    if (init_image is not None) and strength < 1. :
        # latent image + noise as start latent code
        # get latent image, vae  input format: shape (h, w, 3), value: [-1, 1]
        init_image = vae_image_process(init_image, h=h, w=w)
        # TODO: do it outside for main loop to avoid repeat running?
        print("Running VAE encoder to extract latent image...")
        img_latent, _ = model.get_input(init_image, None) 
        img_latent = ops.tile(img_latent, (num_samples, 1, 1, 1))
        
        # add noise
        start_sample_step = sample_steps - int(sample_steps * strength)
        sampler.make_schedule(sample_steps, verbose=False) # TODO: affect global?
        timestep = sampler.ddim_timesteps[::-1][start_sample_step][None, ...]  # forward diffusion timestep

        start_code = model.q_sample(img_latent, timestep, ms.numpy.randn(img_latent.shape))
    else:
        # guassian noise as start latent code  
        prng = np.random.RandomState(seed)
        start_code = prng.randn(num_samples, 4, h // vae_scale_factor, w // vae_scale_factor)
        start_code = Tensor(start_code, dtype=mstype.float32)  # z_T
        start_sample_step = 0

    return start_code, start_sample_step


def prepare_conditions(depth, txt, num_samples=1, height=512, width=512, vae_scale_factor=8, save_depth_map=False):
    '''
    depth map to latent inputs 

    depth: PIL image, shape [H, W], value range: [0, 255]
            or np.ndarray, in range [0, 1.0]
    returns: batch : 
       depth: [bs, 1, H, W] 
    '''
    #print("D--, input depth map", depth.size)

    # resize  
    h_z = height // vae_scale_factor 
    w_z  = width // vae_scale_factor 
    if isinstance(depth, np.ndarray):
        depth = cv2.resize(depth, dsize=(w_z, h_z), interpolation=cv2.INTER_CUBIC) # same as torch bicubic 
    else:
        depth = depth.resize((w_z, h_z), resample=Image.BICUBIC) # NOTE: the order id width, height.
        depth = np.array(depth, dtype=np.float32) 
    
    if save_depth_map:
        save_img(depth, "tmp_depth_map_resized.png", norm=True, gray=True)

    # rescale to [-1, 1] 
    depth_min = np.amin(depth) #(depth_map, axis=[1, 2, 3], keepdim=True)
    depth_max = np.amax(depth)  #depth_map, axis=[1, 2, 3], keepdim=True)
    depth = 2.0 * (depth - depth_min) / (depth_max - depth_min) - 1.0
    
    # repeat to [bs, 1, h_z, w_z]
    depth = np.expand_dims(depth, axis=[0, 1])
    depth = depth.repeat(num_samples, axis=0)
    assert len(depth.shape)==4 and depth.shape[1]==1, f'expect shape [n, 1, h, w], but got {depth.shape}'

    depth = Tensor(depth, dtype=mstype.float32)

    batch = {
        "txt": num_samples * [txt],
        "depth": depth,
    }

    return batch


def depth_to_image(sampler, depth, prompt, seed, scale, sample_steps, num_samples=1, w=512, h=512, init_image=None, strength=0.8):
    model = sampler.model
    
    start_code, start_sample_step = prepare_latent_z(init_image=init_image, num_samples=num_samples, h=h, w=w, model=model, sampler=sampler, strength=strength, sample_steps=sample_steps, seed=seed)

    batch = prepare_conditions(depth, txt=prompt, num_samples=num_samples, height=h, width=w) 

    tokenized_prompts = model.tokenize(batch["txt"]) 
    c = model.get_learned_conditioning(tokenized_prompts)
    # TODO: make batch after computing prompt embedding

    bchw = [num_samples, 4, h // 8, w // 8]

    c_cat = batch['depth'] # (bs, 1, h//8, w//8)

    # hybrid conditions, work with DiffusionWrapper.construct
    cond = {"c_concat": c_cat, "c_crossattn": c}

    # unconditional guidance
    uc_tokenized_prompts = model.tokenize(num_samples * [""])
    uc_cross = model.get_learned_conditioning(uc_tokenized_prompts)
    uc_full = {"c_concat": c_cat, "c_crossattn": uc_cross}

    shape = [model.channels, h // 8, w // 8]
    samples_cfg, intermediates = sampler.sample(
        sample_steps,
        num_samples,
        shape,
        cond,  # hybrid condition
        verbose=False,
        eta=0.0, # ddim_eta
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc_full,
        x_T=start_code,  # z_T
        T0=start_sample_step,
    )

    x_samples = model.decode_first_stage(samples_cfg)

    result = ops.clip_by_value((x_samples + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
    result = result.asnumpy().transpose(0, 2, 3, 1)
    result = result * 255
    result = [Image.fromarray(img.astype(np.uint8)) for img in result]

    return result


def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def main(args):
    # set logger
    set_logger(
        name="",
        output_dir=args.save_path,
        rank=0,
        log_level=eval(args.log_level),
    )

    # init
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(
        mode=args.ms_mode,
        # mode=ms.context.GRAPH_MODE,
        device_target="Ascend",
        device_id=device_id,
        max_device_memory="30GB",
    )

    if args.save_graph:
        save_graphs_path = "graph"
        shutil.rmtree(save_graphs_path)
        ms.context.set_context(save_graphs=True, save_graphs_path=save_graphs_path)

    set_random_seed(args.seed)

    if not os.path.isabs(args.config):
        args.config = os.path.join(workspace, args.config)
    config = OmegaConf.load(f"{args.config}")
    
    def _check_image_size(image, tar_img_size, do_grid_resize=True, grid_size=64):
        # grid_size = 64 # vae_scale_factor * (2**num_downsample_times)
        if tar_img_size is None:
            print("Input image size: (h,w)=", image.size[1], image.size[0])
            tar_w, tar_h = image.size # pil size order is diff from cv2/np shape
        else:
            tar_w, tar_h = tar_img_size
        if do_grid_resize:
            tar_w = int(math.ceil(tar_w / grid_size) * grid_size) 
            tar_h = int(math.ceil(tar_h / grid_size) * grid_size) 
        print("Target image size (h, w) = ", tar_h, tar_w)

        assert (tar_w % 8 == 0) and (tar_h % 8 == 0), "image size should be a multiple of 8. Please resize to requirement." 

        return tar_w, tar_h

    # process inputs
    num_samples = args.num_samples
    prompt = args.prompt
    if args.depth_map is None:
        assert args.image is not None, 'Either depth_map or image must be provided'
        image = Image.open(args.image).convert("RGB")
        tar_w, tar_h = _check_image_size(image, args.img_size)

        depth_estimator = get_depth_estimator() # TODO: init before for loop 
        depth_map = estimate_depth(image, depth_estimator)
        dm_np = save_img(depth_map, "tmp_depth_map.png", norm=True, gray=True)

        init_image = Image.open(args.image) # TODO: reuse opened image 
        vis_images = [init_image, Image.fromarray((dm_np*255).astype(np.uint8)).resize((tar_w, tar_h))]
    else:
        depth_map = Image.open(args.depth_map).convert("L")
        init_image = None
        vis_images = [depth_map] 
        tar_w, tar_h = _check_image_size(depth_map, args.img_size)

    # build models
    model = load_model_from_config(config, args.ckpt_path)

    # build sampler
    sname = args.sampler.lower()
    if sname == "plms":
        sampler = PLMSSampler(model)
    # elif sname == 'dpm_solver_pp':
    #    sampler = DPMSolverSampler(model, "dpmsolver++", prediction_type='noise')
    else:
        raise TypeError(f"unsupported sampler type: {sname}")

    # log
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
            f"Model: StableDiffusion v-{args.version}",
            f"Precision: {model.model.diffusion_model.dtype}",
            f"Pretrained ckpt path: {args.ckpt_path}",
            f"Sampler: {sname}",
            f"Sampling steps: {args.sample_steps}",
            f"Uncondition guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)
    logger.info("Running text-guided image inpainting...")
    
    # sampling
    for _ in range(math.ceil(num_samples / args.batch_size)):
        output = depth_to_image(
            sampler=sampler,
            depth=depth_map,
            prompt=prompt,
            seed=args.seed,
            scale=args.guidance_scale,
            sample_steps=args.sample_steps,
            num_samples=args.batch_size,
            h=tar_h,
            w=tar_w,
            init_image=init_image,
            strength=args.strength,
        )
        vis_images.extend(output)

    # save output
    im_save = image_grid(vis_images, 1, num_samples + 2)
    ct = datetime.datetime.now().strftime("%Y_%d_%b_%H_%M_%S_")
    img_name = ct + prompt.replace(" ", "_") + f"_strg{args.strength}.png"
    os.makedirs(args.save_path, exist_ok=True)
    im_save.save(os.path.join(args.save_path, img_name))

    logger.info(f"Done! All generated images are saved in: {args.save_path}" f"\nEnjoy.")


def load_model_from_config(config, ckpt, verbose=False):
    logger.info(f"Loading model from {ckpt}")
    model = instantiate_from_config(config.model)
    if os.path.exists(ckpt):
        param_dict = ms.load_checkpoint(ckpt)
        if param_dict:
            param_not_load, _ = ms.load_param_into_net(model, param_dict)
            logger.info("Net params not loaded: {}".format(param_not_load))
    else:
        logger.info(f"!!!Warning!!!: {ckpt} doesn't exist")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="path to input image. If None, depth_map should be provided.")
    parser.add_argument("--depth_map", type=str, default=None, help="path to depth map. If None, depth_map will be extracted from input image")
    parser.add_argument("--save_path", type=str, default="output/depth", help="path to save image")
    parser.add_argument("--prompt", type=str, required=True, help="")
    parser.add_argument("--config", type=str, default=None, help="")
    parser.add_argument("--ckpt_path", type=str, default=None, help="")
    parser.add_argument("--aug", type=str, default="resize", help="augment type")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--num_samples", type=int, default=4, help="num of total samples")
    parser.add_argument("--img_size", type=int, default=None, help="if None, target image size the same as input image size. Otherwise, img_size is an integer for target h and w")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size of model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="")
    parser.add_argument("--sample_steps", type=int, default=50, help="")
    parser.add_argument(
        "--sampler", type=str, default="plms", help="support plms, ddim, dpm_solver, dpm_solver_pp, uni_pc"
    )
    parser.add_argument("--save_graph", action="store_true", help="")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="indicates extent to tranform the image. A value of `1` essentially ignore the image."
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        nargs="?",
        default="2.0",  # "1.5_wk",
        help="Stable diffusion version, wukong or 2.0",
    )

    args = parser.parse_args()

    # overwrite env var by parsed arg
    if args.version:
        os.environ["SD_VERSION"] = args.version
    if args.ckpt_path is None:
        args.ckpt_path = "models/sd_v2_depth.ckpt"
    if args.config is None:
        args.config = "configs/v2-depth-inference.yaml"

    if args.guidance_scale is None:
        args.guidance_scale = 9.0 if args.version.startswith("2.") else 7.5

    main(args)
