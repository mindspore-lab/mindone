import argparse
import json
import logging
import os
import time

import numpy as np
from megfile import smart_open, smart_path_join
from omegaconf import OmegaConf
from openlrm.datasets.cam_utils import build_camera_principle, build_camera_standard, camera_normalization_objaverse
from openlrm.datasets.objaverse import ObjaverseDataset
from openlrm.utils import seed_everything, str2bool
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import mindspore as ms
from mindspore import mint

from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def calc_ssim(img1, img2):
    # [0,1]
    # ssim is the only metric extremely sensitive to gray being compared to b/w
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2, data_range=1.0)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i], data_range=1.0))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), data_range=1.0)
    else:
        raise ValueError("Wrong input image dimensions.")


def load_data(root_dirs, uid, render_image_res, num_all_views=32, normalize_camera=True, normed_dist_to_center="auto"):
    root_dir = os.path.join(root_dirs, uid)
    pose_dir = os.path.join(root_dir, "pose")
    rgba_dir = os.path.join(root_dir, "rgba")
    intrinsics_path = os.path.join(root_dir, "intrinsics.npy")

    # load intrinsics
    intrinsics = np.load(smart_open(intrinsics_path, "rb"))
    intrinsics = ms.Tensor(intrinsics, dtype=ms.float32)

    # sample views (incl. source view and side views)
    sample_views = range(num_all_views)
    poses, rgbs, bg_colors = [], [], []
    source_image = None
    for idx, view in enumerate(sample_views):
        pose_path = smart_path_join(pose_dir, f"{view:03d}.npy")
        rgba_path = smart_path_join(rgba_dir, f"{view:03d}.png")
        pose = ObjaverseDataset._load_pose(pose_path)
        bg_color = [1.0, 1.0, 1.0]

        # load image
        rgb = ObjaverseDataset._load_rgba_image(
            rgba_path, bg_color=1.0, resize=render_image_res, crop_pos=None, crop_size=None
        )

        poses.append(pose)
        rgbs.append(rgb)
        bg_colors.append(bg_color)
    source_image = rgbs[0]  # first image as input cond image [1, C, H, W]
    poses = mint.stack(poses, dim=0)
    rgbs = mint.cat(rgbs, dim=0)  # [M, C, H, W]

    if normalize_camera:
        poses = camera_normalization_objaverse(normed_dist_to_center, poses)

    # build source and target camera features
    source_camera = build_camera_principle(poses[:1], intrinsics.unsqueeze(0))  # [1, 12+4]
    render_camera = build_camera_standard(poses, intrinsics.tile((poses.shape[0], 1, 1)))  # [M, 16+9]

    # image value in [0, 1]
    source_image = mint.clamp(source_image, 0.0, 1.0)  # [C, H, W]
    all_images = mint.clamp(rgbs, 0.0, 1.0)  # [side+1, C, H, W]

    return {
        "source_camera": source_camera,  # [1, 12+4]
        "render_camera": render_camera.unsqueeze(0),  # [1, M, 16+9]
        "source_image": source_image,  # [1, C, H, W]
        "target_images": all_images,  # [M, C, H, W]
    }


def evaluate(args):
    save_dir = args.output_path

    rank_id = 0
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
        pynative_synchronize=args.debug,
        jit_config={"jit_level": args.jit_level},
        device_id=int(os.getenv("DEVICE_ID")),
    )
    seed_everything(args.seed)
    logger = set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # valdata preparation
    cfg_path = os.path.join(args.model_path, "cfg.yaml")
    cfg = OmegaConf.load(cfg_path)
    # load data uids
    meta_path = cfg.dataset.subsets[0].meta_path.val
    with open(meta_path, "r") as f:
        uids = json.load(f)
    root_dirs = cfg.dataset.subsets[0].root_dirs[0]

    # init model & load ckpt
    from openlrm.models import ModelLRMWithLossEval

    model = ModelLRMWithLossEval(cfg)
    model.set_train(False)

    # load pretrained checkpoint
    logger.info(f"Loading ckpt in {args.model_path}.")
    assert os.path.isdir(args.model_path) and (
        os.path.isfile(os.path.join(args.model_path, "ckpt", "train_resume.ckpt"))
        or (args.ckpt_name is not None and (os.path.isfile(os.path.join(args.model_path, "ckpt", args.ckpt_name))))
    )

    if args.ckpt_name is not None:
        model_file = os.path.join(args.model_path, "ckpt", args.ckpt_name)
    else:
        model_file = os.path.join(args.model_path, "ckpt", "train_resume.ckpt")
    print(f"Loading weights from local pretrained directory: {model_file}")
    state_dict = ms.load_checkpoint(model_file)
    # Check loading keys:
    model_state_dict = {k: v for k, v in model.parameters_and_names()}
    state_dict_tmp = {}
    for k, v in state_dict.items():
        if ("norm" in k) and ("mlp" not in k):  # for LayerNorm but not ModLN's mlp
            k = k.replace(".weight", ".gamma").replace(".bias", ".beta")
        if "adam_" not in k:  # not to load optimizer
            state_dict_tmp[k] = v
    state_dict = state_dict_tmp
    loaded_keys = list(state_dict.keys())
    expexted_keys = list(model_state_dict.keys())
    original_loaded_keys = loaded_keys
    missing_keys = list(set(expexted_keys) - set(loaded_keys))
    unexpected_keys = list(set(loaded_keys) - set(expexted_keys))
    mismatched_keys = []
    for checkpoint_key in original_loaded_keys:
        if (
            checkpoint_key in model_state_dict
            and checkpoint_key in state_dict
            and state_dict[checkpoint_key].shape != model_state_dict[checkpoint_key].shape
        ):
            mismatched_keys.append(
                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[checkpoint_key].shape)
            )

    print(
        f"Loading LRMModel...\nmissing_keys: {missing_keys}, \nunexpected_keys: {unexpected_keys}, \nmismatched_keys: {mismatched_keys}"
    )
    print(f"state_dict.dtype {state_dict[loaded_keys[0]].dtype}")  # float32
    # Instantiate the model
    param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict, strict_load=False)
    print(f"Loaded checkpoint: param_not_load {param_not_load}, ckpt_not_load {ckpt_not_load}")

    if args.ckpt_name is not None:
        epoch_num = int(args.ckpt_name.strip()[9:-5])
    else:
        epoch_num = state_dict["epoch_num"].item()
    logger.info(f"Loaded checkpoint at Epoch #{epoch_num}")
    image_path = os.path.join(save_dir, f"e{epoch_num}")
    if not args.debug:
        print("Save images in", image_path)
        os.makedirs(image_path, exist_ok=True)

    # Inference
    batches_time = []
    valset_len = len(uids)
    cnt = 0
    mean_psnr = 0.0
    mean_ssim = 0.0
    for index in range(valset_len):
        data_batch = load_data(root_dirs, uids[index], render_image_res=cfg.dataset.source_image_res)

        # infer
        start_time = time.time()

        render_images = model.forward_nocalloss(
            source_camera=data_batch["source_camera"],
            render_camera=data_batch["render_camera"],
            source_image=data_batch["source_image"],
            render_size=cfg.dataset.source_image_res,
        )[
            0
        ]  # only batch=1  [M, C, H, W]

        batch_time = time.time() - start_time
        batches_time.append(batch_time)
        logger.info("Batch time cost: %.3fs.", batch_time)

        target_images = data_batch["target_images"]  # [M, C, H, W]

        for view_id, (target, pred) in enumerate(zip(target_images, render_images)):
            target = target.asnumpy()
            pred = pred.asnumpy()
            mean_psnr += psnr(target, pred)
            mean_ssim += calc_ssim(target, pred)

            # save mviews outputs
            im = Image.fromarray(np.clip(pred.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            im.save(os.path.join(image_path, f"{uids[index]}_{view_id:03d}.jpg"))

            # if save target image:
            # im = Image.fromarray(np.clip(target.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            # im.save(os.path.join(image_path, f"{uids[index]}_{view_id:03d}_target.jpg"))

        cnt += target_images.shape[0]

    if len(batches_time) > 1:
        del batches_time[0]  # for measuring speed, eliminate the very first batch that is particularly slow
    mean_time = sum(batches_time) / len(batches_time)
    logger.info("Mean Batch time: %.3fs.", mean_time)

    mean_psnr /= cnt
    mean_ssim /= cnt

    logger.info("Evaluated %d views" % cnt)
    logger.info("Avg. PSNR: %f, SSIM: %f" % (mean_psnr, mean_ssim))
    logger.info("=============================")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="model path containing cfg.yaml, ckpt folder")
    parser.add_argument("--ckpt_name", type=str, default=None, help="indicate one ckpt, e.g. openlrm-e10000.ckpt")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When debugging, set it true, to avoid saving too many ckpts and burn out the storage.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="output dir to save the generated videos",
    )
    parser.add_argument(
        "--append_timestr",
        type=str2bool,
        default=True,
        help="If true, an subfolder named with timestamp under output_path will be created to save the sampling results",
    )
    parser.add_argument("--datacfg", default="configs/instant-nerf-large-train.yaml")
    parser.add_argument("--modelcfg", default="configs/instant-nerf-large-train.yaml")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--seed", type=int, default=42, help="Inference seed")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports [“O0”, “O1”, “O2”]."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="infer batch size")
    parser.add_argument(
        "--dtype",
        default="fp32",  # if amp level O0/1, must pass fp32
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what computation data type to use. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
