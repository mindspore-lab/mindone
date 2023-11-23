import os
import sys
import shutil
from time import strftime
from addict import Dict
from utils.arg_parser import parse_args_and_config

args, cfg = parse_args_and_config()

import mindspore as ms
from mindspore import context
from mindspore.amp import auto_mixed_precision

from utils.preprocess import CropAndExtract
from datasets.generate_batch import get_data
from datasets.generate_facerender_batch import get_facerender_data
from models.audio2coeff import Audio2Coeff
from models.facerender.animate import AnimateFromCoeff


def main(args, config):
    context.set_context(mode=config.system.mode,
                        device_target="Ascend",
                        device_id=int(args.device_id)
                        )

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    # init model
    preprocess_model = CropAndExtract(config.preprocess)
    audio_to_coeff = Audio2Coeff(config)
    animate_from_coeff = AnimateFromCoeff(config.facerender)

    amp_level = config.system.get("amp_level", "O0")
    auto_mixed_precision(audio_to_coeff.audio2exp_model, amp_level)
    auto_mixed_precision(audio_to_coeff.audio2pose_model, amp_level)
    auto_mixed_precision(animate_from_coeff.generator, amp_level)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,
                                                                           source_image_flag=True, pic_size=args.size)

    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(
            os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(
            ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path = None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(
                os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ = preprocess_model.generate(
                ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path = None

    # audio2ceoff
    batch = get_data(first_coeff_path, audio_path,
                     ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(
        batch, save_dir, pose_style, ref_pose_coeff_path)

    # coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                               batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                               expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)

    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info,
                                         enhancer=args.enhancer,
                                         background_enhancer=args.background_enhancer,
                                         preprocess=args.preprocess,
                                         img_size=args.size
                                         )

    shutil.move(result, save_dir+'.mp4')
    print('The generated video is named:', save_dir+'.mp4')

    if not args.verbose:
        shutil.rmtree(save_dir)


if __name__ == '__main__':

    config = Dict(cfg)
    main(args, config)
