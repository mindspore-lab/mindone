import argparse
import os
import sys

# add current working dir to path to prevent ModuleNotFoundError
sys.path.insert(0, os.getcwd())

from tools._common import L2_norm_ops, load_images
from tools._common.clip import CLIPImageProcessor, parse
from tools.safety_checker.safety_checker import SafetyChecker

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net


class SafetyChecker2(SafetyChecker):
    def __init__(
        self,
        backend="ms",
        config="tools/_common/clip/configs/clip_vit_l_14.yaml",
        ckpt_path=None,
        model_name="openai/clip-vit-large-patch14",
        check_certificate=False,
        threshold=0.2,
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
            self.image_size = config.vision_config.image_size
            self.dtype = ms.float32 if config.dtype == "float32" else ms.float16
            model = CLIPModel(config)
            processor = CLIPImageProcessor()

        else:
            raise ValueError(f"Unknown backend: {backend}. Valid backend: [ms, pt]")

        print(f"Backend: {backend}")

        self.model = model
        self.nsfw_model = nsfw_model
        self.processor = processor
        self.backend = backend
        self.threshold = threshold

    def eval(self, paths):
        images, paths = load_images(paths)
        print(f"{len(images)} images are loaded")

        if self.backend == "ms":
            images = self.processor(images)
        else:
            images = self.processor(images=images, return_tensors="pt").pixel_values

        return self.__call__(images)

    def __call__(self, images):
        original_images = images

        if self.backend == "ms" and (images.shape[-1] != self.image_size or images.shape[-2] != self.image_size):
            import numpy as np
            from PIL import Image

            from mindspore import ops

            images_ = []
            for i in range(images.shape[0]):
                im = Image.fromarray((255.0 * images[i].transpose((1, 2, 0))).astype(np.uint8).asnumpy())
                im = im.resize((self.image_size, self.image_size))
                im = ms.Tensor(np.asarray(im), self.dtype)
                images_.append(im)
            images = ops.stack(images_).transpose((0, 3, 1, 2))

        image_features = self.model.get_image_features(images)
        if self.backend == "ms":
            norm = L2_norm_ops(image_features)
        else:
            norm = image_features.norm(p=2, dim=-1, keepdim=True)
        image_features = image_features / norm

        scores = self.nsfw_model(image_features)

        has_nsfw_concepts = [score if score > self.threshold else 0 for score in scores]

        if self.backend == "pt":
            import torch
        for idx, has_nsfw_concepts in enumerate(has_nsfw_concepts):
            if has_nsfw_concepts:
                if self.backend == "pt":
                    original_images[idx] = torch.zeros_like(original_images[idx])
                else:
                    original_images[idx] = ops.zeros(original_images[idx].shape)

        if any(has_nsfw_concepts):
            print(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        return original_images, has_nsfw_concepts


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
        "--threshold",
        default=0.2,
        type=float,
        help="a 0-1 scalar-valued threshold above which we believe an image is NSFW" " Default: 0.2",
    )
    parser.add_argument(
        "--check_certificate",
        action="store_true",
        help="set this flag to check for certificate for downloads (checks)",
    )
    args = parser.parse_args()
    checker = SafetyChecker2(**vars(args))

    assert args.image_path_or_dir is not None
    _, has_nsfw_concepts = checker.eval(args.image_path_or_dir)

    print(has_nsfw_concepts)
