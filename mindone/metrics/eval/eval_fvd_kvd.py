import argparse
import os

from fvd.fvd_kvd import Distance, get_video_paths

from mindspore import context


def eval_fvd_kvd(args):
    real_videos = get_video_paths(args.real_video_dir)
    gen_videos = get_video_paths(args.gen_video_dir)
    print("Num real videos: ", len(real_videos))
    print("Num generated videos: ", len(gen_videos))

    device_id = int(os.getenv("DEVICE_ID", 0))
    context.set_context(
        mode=0,
        device_id=device_id,
    )
    distance = Distance(ckpt_path=args.ckpt)
    gen_feature, gt_feature = distance.comput_mode_feature(gen_videos, real_videos)
    print("Start calculating FVD")
    fvd = distance.comput_fvd(gen_feature, gt_feature)
    print(f"FVD is {fvd} ")
    print("---------------------------")
    print("Start calculating KVD")
    kvd = distance.comput_kvd(gen_feature, gt_feature)
    print(f"KVD is {kvd} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_video_dir", type=str, help="Directory to real videos")
    parser.add_argument("--gen_video_dir", type=str, help="Directory to generated videos")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Pretrain model ckpt",
    )
    args = parser.parse_args()

    eval_fvd_kvd(args)
