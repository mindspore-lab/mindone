"""
Implementation of two versions of safety checker in stable
diffusion 1/2, respectively
"""
import argparse
import os
import sys

import numpy as np
import yaml
from PIL import Image

# # equivalent to no-check-certificate flag in wget
# os.environ["CURL_CA_BUNDLE"] = ""
# add current working dir to path to prevent ModuleNotFoundError
sys.path.insert(0, os.getcwd())

from tools._common import L2_norm_ops, load_images
from tools._common.clip import CLIPImageProcessor, CLIPModel, CLIPTokenizer, parse
from tools.safety_checker.nsfw_model import NSFWModel
from tools.safety_checker.utils import locate_model

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net, ops

try:
    import torch
    from tools.safety_checker.nsfw_model_pt import NSFWModelPT

    is_torch_available = True
except ImportError:
    is_torch_available = False

try:
    from transformers import CLIPModel as CLIPModelPT
    from transformers import CLIPProcessor

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class SafetyChecker:
    def __init__(
        self,
        safety_version=2,
        backend="ms",
        config="tools/_common/clip/configs/clip_vit_l_14.yaml",
        ckpt_path=None,
        tokenizer_path="ldm/models/clip/bpe_simple_vocab_16e6.txt.gz",
        model_name="openai/clip-vit-large-patch14",
        settings_path="tools/safety_checker/safety_settings_v2.yaml",
        threshold=0.2,
        **kwargs,
    ):
        assert safety_version in [1, 2]

        if backend == "pt":
            assert is_torch_available is True, "torch is not installed, please install torch."
            assert is_transformers_available is True, "transformers is not installed, please install transformers."

            model = CLIPModelPT.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)

            def process_text(text):
                return processor(text=text, return_tensors="pt", padding=True).input_ids

            if safety_version == 2:
                nsfw_model = NSFWModelPT()
                model_path = locate_model(backend="pt")
                assert model_path is not None
                nsfw_model.load_state_dict(torch.load(model_path))

        elif backend == "ms":
            # parse config file
            config = parse(config, ckpt_path)
            self.image_size = config.vision_config.image_size
            self.dtype = ms.float32 if config.dtype == "float32" else ms.float16
            model = CLIPModel(config)
            processor = CLIPImageProcessor()
            tokenizer = CLIPTokenizer(tokenizer_path, pad_token="!")

            def process_text(text):
                return ms.Tensor(tokenizer(text, padding="max_length", max_length=77)["input_ids"])

            if safety_version == 2:
                nsfw_model = NSFWModel()
                model_path = locate_model(backend="ms")
                assert model_path is not None
                param = load_checkpoint(model_path)

                param_not_load = load_param_into_net(nsfw_model, param)
                if ms.__version__[0] == "2":
                    assert len(param_not_load[0]) == 0 and len(param_not_load[1]) == 0
                else:
                    assert len(param_not_load) == 0
        else:
            raise ValueError(f"Unknown backend: {backend}. Valid backend: [ms, pt]")

        print(f"Backend: {backend}")

        if safety_version == 1:
            with open(settings_path) as f:
                settings = yaml.safe_load(f)

            self.nsfw_threshs = settings["nsfw"]["concepts"]
            self.special_threshs = settings["special"]["concepts"]
            self.nsfw_concepts = list(self.nsfw_threshs.keys())
            self.special_concepts = list(self.special_threshs.keys())

            self.nsfw_features = model.get_text_features(process_text(self.nsfw_concepts))
            self.special_features = model.get_text_features(process_text(self.special_concepts))
        else:
            self.nsfw_model = nsfw_model
            self.threshold = threshold

        self.model = model
        self.processor = processor
        self.backend = backend
        self.safety_version = safety_version

    def eval_safety(self, special_sim, nsfw_sim):
        num_nsfw = len(self.nsfw_threshs)
        num_special = len(self.special_threshs)
        num_images = special_sim.shape[0]

        results = []
        for i in range(num_images):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}
            adjustment = 0.0

            for j in range(num_special):
                sim = special_sim[i][j]
                thresh = self.special_threshs[self.special_concepts[j]]
                result_img["special_scores"][j] = round(float(sim - thresh + adjustment), 4)
                if result_img["special_scores"][j] > 0:
                    result_img["special_care"].append((j, result_img["special_scores"][j]))
                    adjustment = 0.01

            for j in range(num_nsfw):
                sim = nsfw_sim[i][j]
                thresh = self.nsfw_threshs[self.nsfw_concepts[j]]
                result_img["concept_scores"][j] = round(float(sim - thresh + adjustment), 4)
                if result_img["concept_scores"][j] > 0:
                    result_img["bad_concepts"].append(j)

            results.append(result_img)
        return results

    def cosine_distance(self, v, w):
        if self.backend == "ms":
            v /= L2_norm_ops(v)
            w /= L2_norm_ops(w)
            return ops.matmul(v, w.T)
        else:
            v /= v.norm(p=2, dim=-1, keepdim=True)
            w /= w.norm(p=2, dim=-1, keepdim=True)
            return torch.mm(v, w.T)

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
            images_ = []
            for i in range(images.shape[0]):
                im = Image.fromarray((255.0 * images[i].transpose((1, 2, 0))).astype(np.uint8).asnumpy())
                im = im.resize((self.image_size, self.image_size))
                im = ms.Tensor(np.asarray(im), self.dtype)
                images_.append(im)
            images = ops.stack(images_).transpose((0, 3, 1, 2))

        image_features = self.model.get_image_features(images)

        if self.safety_version == 1:
            nsfw_sim = self.cosine_distance(image_features, self.nsfw_features)
            special_sim = self.cosine_distance(image_features, self.special_features)
            scores = self.eval_safety(special_sim, nsfw_sim)
            has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in scores]
        else:
            if self.backend == "ms":
                norm = L2_norm_ops(image_features)
            else:
                norm = image_features.norm(p=2, dim=-1, keepdim=True)
            image_features = image_features / norm
            scores = self.nsfw_model(image_features)
            has_nsfw_concepts = [score if score > self.threshold else 0 for score in scores]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
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
        "--safety_version",
        type=int,
        default=2,
        help="the version of stable diffusion to use for its safety checker. Option: 1, 2" "Default: 2",
    )
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
        "--tokenizer_path",
        default="ldm/models/clip/bpe_simple_vocab_16e6.txt.gz",
        type=str,
        help="load tokenizer checkpoint." " Default: ldm/models/clip/bpe_simple_vocab_16e6.txt.gz",
    )
    parser.add_argument(
        "--settings_path",
        default="tools/safety_checker/safety_settings_v2.yaml",
        type=str,
        help="YAML file for a list of NSFW concepts as safety settings"
        " Default: tools/safety_checker/safety_settings_v2.yaml",
    )
    parser.add_argument(
        "--threshold",
        default=0.2,
        type=float,
        help="a 0-1 scalar-valued threshold above which we believe an image is NSFW" " Default: 0.2",
    )

    args = parser.parse_args()
    checker = SafetyChecker(**vars(args))

    assert args.image_path_or_dir is not None
    _, has_nsfw_concepts = checker.eval(args.image_path_or_dir)

    print(has_nsfw_concepts)
