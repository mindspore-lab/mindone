import argparse

from mindspore import context

from mindone.metrics import Frechet_Kernel_Video_Distance
from mindone.metrics.utils import get_video_path

VIDEO_EXTENSIONS = {".mp4", ".gif"}


parser = argparse.ArgumentParser()
parser.add_argument("--real_video_dir", type=str, help="Directory to real videos")
parser.add_argument("--gen_video_dir", type=str, help="Directory to generated videos")
parser.add_argument("--num_frames", default=32, type=int, help="frames for videos")
args = parser.parse_args()

real_videos = get_video_path(args.real_video_dir)
gen_videos = get_video_path(args.gen_video_dir)
print("Num real videos: ", len(real_videos))
print("Num generated videos: ", len(gen_videos))

context.set_context(mode=0)
distance = Frechet_Kernel_Video_Distance(num_frames=args.num_frames)
print("Start calculating model feature")
gen_feature, gt_feature = distance.comput_mode_feature(gen_videos, real_videos)
print("Start calculating FVD")
fvd = distance.comput_fvd(gen_feature, gt_feature)
print(f"FVD is {fvd} ")
print("---------------------------")
print("Start calculating KVD")
kvd = distance.comput_kvd(gen_feature, gt_feature)
print(f"KVD is {kvd} ")
