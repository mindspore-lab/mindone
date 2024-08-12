import os
import shutil
from time import strftime
from addict import Dict
from utils.arg_parser import parse_args_and_config

import mindspore as ms
from mindspore import context
from mindspore.amp import auto_mixed_precision

from utils.preprocess import CropAndExtract
from datasets.dataset_audio2coeff import AudioCoeffDataset
from datasets.dataset_facerender import FaceRenderDataset
from models.audio2coeff import Audio2Coeff
from models.facerender.animate import AnimateFromCoeff


def main(args, config):
    context.set_context(
        mode=config.system.mode, device_target="Ascend", device_id=int(args.device_id)
    )

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style

    # init model
    preprocess_model = CropAndExtract(config.preprocess)
    audio_to_coeff = Audio2Coeff(config)
    animate_from_coeff = AnimateFromCoeff(config.facerender)

    amp_level = config.system.get("amp_level", "O0")
    auto_mixed_precision(audio_to_coeff.audio2exp_model, amp_level)
    auto_mixed_precision(audio_to_coeff.audio2pose_model, amp_level)
    auto_mixed_precision(animate_from_coeff.generator, amp_level)

    # audio2coeff
    ds_audiocoeff = AudioCoeffDataset(
        args=args,
        preprocessor=preprocess_model,
        save_dir=save_dir,
    )

    data_audiocoeff = ds_audiocoeff.process_data(pic_path, audio_path)

    ref_pose_coeff_path = data_audiocoeff["ref_pose_coeff_path"]
    crop_pic_path = data_audiocoeff["crop_pic_path"]
    first_coeff_path = data_audiocoeff["first_coeff_path"]
    crop_info = data_audiocoeff["crop_info"]

    coeff_path = audio_to_coeff.generate(
        data_audiocoeff, save_dir, pose_style, ref_pose_coeff_path
    )

    # coeff2video
    ds_facerender = FaceRenderDataset(
        args=args,
        config=config,
        batch_size=args.batch_size,
        expression_scale=args.expression_scale,
        still_mode=args.still,
        preprocess=args.preprocess,
        size=args.size,
    )

    data_facerender = ds_facerender.process_data(
        crop_pic_path, args.driven_audio, first_coeff_path, coeff_path
    )
    result = animate_from_coeff.generate(
        data_facerender,
        save_dir,
        pic_path,
        crop_info,
        enhancer=args.enhancer,
        background_enhancer=args.background_enhancer,
        preprocess=args.preprocess,
        img_size=args.size,
    )

    shutil.move(result, save_dir + ".mp4")
    print("The generated video is named:", save_dir + ".mp4")

    if not args.verbose:
        shutil.rmtree(save_dir)


if __name__ == "__main__":
    args, cfg = parse_args_and_config()
    config = Dict(cfg)
    main(args, config)
