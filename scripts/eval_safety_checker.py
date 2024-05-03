import argparse

from mindone.metrics import SafetyChecker

parser = argparse.ArgumentParser()
parser.add_argument(
    "--safety_version",
    type=int,
    default=2,
    help="the version of stable diffusion to use for its safety checker. Option: 1, 2" "Default: 2",
)
parser.add_argument(
    "--model_name",
    default="openai/clip-vit-large-patch14",
    type=str,
    help="the name of a (Open/)CLIP model as shown in HuggingFace." " Default: openai/clip-vit-large-patch14",
)
parser.add_argument(
    "--image_path_or_dir",
    default=None,
    type=str,
    help="input data for predict, it support image data path or directory." " Default: None",
)
parser.add_argument(
    "--video_path_or_dir",
    default=None,
    type=str,
    help="input data for predict, it support video data path or directory." " Default: None",
)
parser.add_argument(
    "--settings_path",
    default="mindone/metrics/config/safety_settings_v2.yaml",
    type=str,
    help="YAML file for a list of NSFW concepts as safety settings"
    " Default: mindone/metrics/config/safety_settings_v2.yaml",
)
parser.add_argument(
    "--threshold",
    default=0.2,
    type=float,
    help="a 0-1 scalar-valued threshold above which we believe an image is NSFW" " Default: 0.2",
)

args = parser.parse_args()
checker = SafetyChecker(**vars(args))

if args.image_path_or_dir:
    _, has_nsfw_concepts = checker.eval_images(args.image_path_or_dir)
elif args.video_path_or_dir:
    has_nsfw_concepts = checker.eval_videos(args.video_path_or_dir)
else:
    raise ValueError("Please input image path or video path")

print(has_nsfw_concepts)
