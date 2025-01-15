import os
import sys

import mindspore as ms

# TODO: remove in future when mindone is ready for install
mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))

import logging
import time

from opensora.npu_config import npu_config
from opensora.utils.sample_utils import get_args, prepare_pipeline, run_model_and_save_samples

from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = get_args()
    save_dir = args.save_img_path
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init environment
    rank_id, device_num = npu_config.set_npu_env(args)
    npu_config.print_ops_dtype_info()

    # 2. build models and pipeline
    if args.num_frames != 1 and args.enhance_video is not None:  # TODO
        from opensora.sample.VEnhancer.enhance_a_video import VEnhancer

        enhance_video_model = VEnhancer(model_path=args.enhance_video, version="v2", device=args.device)
    else:
        enhance_video_model = None

    pipeline = prepare_pipeline(args)  # build I2V/T2V pipeline

    if args.caption_refiner is not None:
        from opensora.sample.caption_refiner import OpenSoraCaptionRefiner

        caption_refiner_model = OpenSoraCaptionRefiner(args.caption_refiner, dtype=ms.float16)
    else:
        caption_refiner_model = None

    # 3. inference
    start_time = time.time()
    run_model_and_save_samples(
        args, pipeline, rank_id, device_num, save_dir, caption_refiner_model, enhance_video_model
    )
    end_time = time.time()
    time_cost = end_time - start_time
    logger.info(f"Inference time cost: {time_cost:0.3f}s")
    logger.info(f"Inference speed: {len(args.text_prompt) / time_cost:0.3f} samples/s")
    logger.info(f"{'latents' if args.save_latents else 'videos' } saved to {save_dir}")
