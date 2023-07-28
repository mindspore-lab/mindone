import argparse

from tools._common import L2_norm_ops, load_images
from tools._common.clip import CLIPImageProcessor, parse

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net


class SafetyChecker2:
    def __init__(
        self,
        backend="ms",
        config="tools/_common/clip/configs/clip_vit_l_14.yaml",
        ckpt_path=None,
        model_name="openai/clip-vit-large-patch14",
        check_certificate=False,
        **kwargs,
    ):
        if backend == "pt":
            import torch
            from transformers import CLIPModel, CLIPProcessor

            # equivalent to no-check-certificate flag in wget
            if not check_certificate:
                import os

                os.environ["CURL_CA_BUNDLE"] = ""

            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)

            from nsfw_model_pt import NSFWModelPT

            nsfw_model = NSFWModelPT()
            nsfw_model.load_state_dict(torch.load("tools/safety_checker/l14_nsfw.pth"))

        elif backend == "ms":
            from nsfw_model import NSFWModel

            nsfw_model = NSFWModel()
            param = load_checkpoint("tools/safety_checker/l14_nsfw.ckpt")

            param_not_load = load_param_into_net(nsfw_model, param)
            if ms.__version__[0] == "2":
                assert len(param_not_load[0]) == 0 and len(param_not_load[1]) == 0
            else:
                assert len(param_not_load) == 0

            # parse config file
            from tools._common.clip import CLIPModel

            config = parse(config, ckpt_path)

            model = CLIPModel(config)
            processor = CLIPImageProcessor()

        else:
            raise ValueError(f"Unknown backend: {backend}. Valid backend: [ms, pt]")

        print(f"Backend: {backend}")

        self.model = model
        self.nsfw_model = nsfw_model
        self.processor = processor
        self.backend = backend

    def eval(self, paths):
        images, paths = load_images(paths)
        print(f"{len(images)} images are loaded")

        if self.backend == "ms":
            images = self.processor(images)
        else:
            images = self.processor(images=images, return_tensors='pt').pixel_values

        return self.__call__(images)

    def __call__(self, images):
        image_features = self.model.get_image_features(images)
        if self.backend == "ms":
            norm = L2_norm_ops(image_features)
        else:
            norm = image_features.norm(p=2, dim=-1, keepdim=True)
        image_features = image_features / norm

        scores = self.nsfw_model(image_features)
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="tools/_common/clip/configs/clip_vit_l_14.yaml",
        type=str,
        help="YAML config files for ms backend" " Default: tools/_common/clip/configs/clip_vit_l_14.yaml",
    )
    parser.add_argument(
        "--model_name",
        default="openai/clip-vit-large-patch14",
        type=str,
        help="the name of a (Open/)CLIP model as shown in HuggingFace for pt backend."
        " Default: openai/clip-vit-large-patch14",
    )
    parser.add_argument(
        "--image_path_or_dir",
        default=None,
        type=str,
        help="input data for predict, it support real data path or data directory." " Default: None",
    )
    parser.add_argument("--ckpt_path", default=None, type=str, help="load model checkpoint." " Default: None")
    parser.add_argument(
        "--backend",
        default="ms",
        type=str,
        help="backend to do CLIP model inference for CLIP score compute. Option: ms, pt." " Default: ms",
    )
    parser.add_argument(
        "--check_certificate",
        action="store_true",
        help="set this flag to check for certificate for downloads (checks)",
    )
    args = parser.parse_args()
    checker = SafetyChecker2(**vars(args))

    assert args.image_path_or_dir is not None
    scores = checker.eval(args.image_path_or_dir)

    print(scores)
