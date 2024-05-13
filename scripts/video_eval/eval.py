import argparse

from mindspore import context

from mindone.metrics import ClipScore, Frechet_Kernel_Video_Distance

VIDEO_EXTENSIONS = {".mp4"}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="openai/clip-vit-base-patch32/",
    help="the name of a (Open/)CLIP model as shown in HuggingFace." "Default: openai/clip-vit-base-patch32/",
)
parser.add_argument("--gen_video_dir", type=str, help="path to data folder." "Default: None")
parser.add_argument("--gen_csv_path", type=str, default=None, help="path to video caption path." "Default: None")
parser.add_argument("--gt_video_dir", type=str, help="path to data folder." "Default: None")
parser.add_argument("--gt_csv_path", type=str, default=None, help="path to video caption path." "Default: None")
parser.add_argument("--sample_n_frames", default=64, type=int, help="sample frames for videos")
parser.add_argument("--sample_stride", default=1, type=int, help="sample frames for videos")
parser.add_argument(
    "--metric", type=str, default="clip_score_text", choices=["clip_score_text", "clip_score_frame", "FVD", "KVD"]
)
args = parser.parse_args()

context.set_context(mode=0)

if args.metric in ["clip_score_text", "clip_score_frame"]:
    clip_score = ClipScore(model_name=args.model_name)
    print(f"start calculating {args.metric}")
    score = clip_score.calucation_score(args.gen_video_dir, args.gen_csv_path, metric=args.metric)
    print("{}: {}".format(args.metric, score))
elif args.metric in ["fvd", "kvd"]:
    distance = Frechet_Kernel_Video_Distance(sample_n_frames=args.sample_n_frames, sample_stride=args.sample_stride)
    print("start calculating model feature")
    gen_feature, gt_feature = distance.comput_mode_feature(
        args.gen_video_dir, args.gt_video_dir, gen_csv_path=args.gen_csv_path, gt_csv_path=args.gt_csv_path
    )
    print("start calculating FVD")
    fvd = distance.comput_fvd(gen_feature, gt_feature)
    print(f"FVD is {fvd}")
    print("start calculating KVD")
    kvd = distance.comput_kvd(gen_feature, gt_feature)
    print(f"kvd is {kvd}")
else:
    raise NotImplementedError(args.metric)
