import logging

logger = logging.getLogger("canny2image")
# logger.setLevel(logging.ERROR)

import datetime
import os
import sys

import cv2
import numpy as np
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_model
from conditions.canny.canny_detector import CannyDetector
from conditions.segmentation.segment_detector import SegmentDetector
from conditions.utils import HWC3, resize_image
from ldm.modules.logger import set_logger
from PIL import Image

import mindspore as ms
import mindspore.ops as ops

MODE = {
    "canny": "canny",
    "segmentation": "segmentation",
}

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workspace)


def main(args):
    ms.set_seed(args.seed)
    # set logger
    set_logger(
        name="",
        output_dir=args.output_path,
        rank=0,
        log_level=eval(args.log_level),
    )
    work_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"WORK DIR:{work_dir}")
    outpath = os.path.join(
        work_dir, args.output_path + args.task_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(outpath, exist_ok=True)
    logger.info(f"Output:{outpath}")

    # set ms context
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend", device_id=device_id)
    # ms.set_context(mode=ms.context.GRAPH_MODE, device_target='Ascend', device_id=6)

    # create model
    if os.path.exists(args.model_config):
        model = create_model(args.model_config)
        model.set_train(False)
    else:
        logger.error(f"model config file {args.model_config} not exists")

    if os.path.exists(args.model_ckpt):
        load_model(model, args.model_ckpt)
    else:
        logger.error(f"model checkpoint file {args.model_ckpt} not exists")

    if os.path.exists(args.input_image):
        image_path = args.input_image
        image = cv2.imread(image_path)
        input_image = np.array(image, dtype=np.uint8)
    else:
        logger.error(f"input image file {args.input_image} not exists")

    sampler = DDIMSampler(model)
    image_resolution = args.image_resolution  # 256~768

    num_samples = args.n_samples  # 1~12
    strength = args.strength  # 1~2
    guess_mode = args.guess_mode
    low_threshold = args.low_threshold  # 100# 1~255
    high_threshold = args.high_threshold  # 200 # 1~255
    ddim_steps = args.sampling_steps  # 1~100
    scale = args.scale  # 0.1~30
    eta = args.ddim_eta
    # a_prompt = 'best quality, extremely detailed'
    a_prompt = args.a_prompt
    n_prompt = args.n_prompt
    prompt = "" if args.prompt is None else args.prompt

    img = resize_image(HWC3(input_image), image_resolution)
    H, W, C = img.shape

    # log
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "".join(
        [
            f"image_resolution: {image_resolution}\n",
            f"num_samples: {num_samples}\n",
            f"low_threshold: {low_threshold}\n",
            f"high_threshold: {high_threshold}\n",
            f"strength: {strength}\n",
            f"guess_mode: {guess_mode}\n",
            f"ddim_steps: {ddim_steps}\n",
            f"scale: {scale}\n",
            f"seed: {args.seed}\n",
            f"eta: {eta}\n",
            f"a_prompt: {a_prompt}\n",
            f"n_prompt: {n_prompt}\n",
            f"prompt: {prompt}\n",
            f"mode: {args.mode}\n",
            f"input_image: {image_path}\n",
            f"output_path: {outpath}\n",
            f"model_config: {args.model_config}\n",
            f"model_ckpt: {args.model_ckpt}\n",
            f"log_level: {args.log_level}\n",
            f"task_name: {args.task_name}\n",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    if args.mode == MODE["canny"]:
        apply_canny = CannyDetector()
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)
    elif args.mode == MODE["segmentation"]:
        if os.path.exists(args.condition_ckpt_path):
            apply_segment = SegmentDetector(ckpt_path=args.condition_ckpt_path)
        else:
            logger.warning(
                f"!!!Warning!!!: Condition Detector checkpoint path {args.condition_ckpt_path} doesn't exist"
            )
        detected_map = apply_segment(img)
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        raise NotImplementedError(f"mode {args.mode} not supported")

    Image.fromarray(detected_map).save(
        os.path.join(outpath, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_detected_map.png")
    )

    control = ms.Tensor(detected_map.copy()).float() / 255.0
    control = control.permute(2, 0, 1)
    control = ops.stack([control for _ in range(num_samples)], axis=0)

    c_crossattn = model.get_learned_conditioning(model.tokenize([prompt + ", " + a_prompt] * num_samples))
    cond = {"c_concat": [control], "c_crossattn": [c_crossattn]}

    un_cond_c_crossattn = model.get_learned_conditioning(model.tokenize([n_prompt] * num_samples))
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [un_cond_c_crossattn]}

    shape = (4, H // 8, W // 8)
    model.control_scales = (
        [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
    )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

    logger.info("Start inference")
    samples, intermediates = sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
    )

    def decode_and_save_result(args, samples, detected_map, outpath, filename):
        x_samples = model.decode_first_stage(samples)
        x_samples = (ops.transpose(x_samples, (0, 2, 3, 1)) * 127.5 + 127.5).asnumpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]
        if args.mode == MODE["canny"]:
            results = [255 - detected_map] + results
        elif args.mode == MODE["segmentation"]:
            results = [detected_map] + results
        else:
            raise NotImplementedError(f"mode {args.mode} not supported")

        for i, result in enumerate(results):
            img = Image.fromarray(result)
            tmp_filename = f"{filename}_index{i}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
            img.save(os.path.join(outpath, tmp_filename))
            logger.info(f"Save result with filename {tmp_filename} done.")
            # print(result)

    decode_and_save_result(args, samples, detected_map, outpath, "results")

    logger.info(f"Save result to {outpath} done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--output_path", type=str, nargs="?", default="output/", help="dir to write results to")
    parser.add_argument("--input_image", type=str, help="path to input image")
    parser.add_argument(
        "--task_name", type=str, default="canny2image", help="task name as folder name, used to save results"
    )
    parser.add_argument("--model_config", type=str, required=True, help="model config file (.yaml)")
    parser.add_argument("--model_ckpt", type=str, required=True, help="model checkpoint file path")
    parser.add_argument("--n_samples", type=int, default=4, choices=range(1, 13), help="num samples")
    parser.add_argument("--image_resolution", type=int, default=512, choices=range(256, 769), help="image resolution")

    parser.add_argument("--strength", type=float, default=1, help="strength")
    parser.add_argument("--guess_mode", type=bool, default=False, help="guess mode")
    parser.add_argument(
        "--sampling_steps", type=int, default=20, choices=range(1, 101), help="number of ddim sampling steps"
    )
    parser.add_argument("--scale", type=float, default=9.0, help="scale")
    parser.add_argument(
        "--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed")

    parser.add_argument("--a_prompt", type=str, default="best quality", help="added prompt")
    parser.add_argument(
        "--n_prompt",
        type=str,
        default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        help="negative prompt",
    )
    parser.add_argument("--prompt", type=str, default=None, help="prompt")

    parser.add_argument(
        "--mode",
        type=str,
        default="canny",
        choices=list(MODE.keys()),
        help="control net task mode, only support canny now",
    )
    # args for canny
    parser.add_argument("--low_threshold", type=int, default=100, choices=range(1, 256), help="low threshold for canny")
    parser.add_argument(
        "--high_threshold", type=int, default=200, choices=range(1, 256), help="high threshold for canny"
    )
    # args for model-based condition ckpt path
    parser.add_argument(
        "--condition_ckpt_path", type=str, default="", help="checkpoint path for contition control model"
    )

    args = parser.parse_args()

    main(args)
