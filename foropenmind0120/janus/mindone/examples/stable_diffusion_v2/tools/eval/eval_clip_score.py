import argparse
import json
import os
import sys
from functools import partial

# add current working dir to path to prevent ModuleNotFoundError
sys.path.insert(0, os.getcwd())

from ldm.util import is_old_ms_version
from PIL import Image
from tools._common.clip import CLIPImageProcessor, CLIPModel, CLIPTokenizer, parse

import mindspore
from mindspore import ops

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="tools/_common/clip/configs/clip_vit_b_16.yaml",
        type=str,
        help="YAML config files for ms backend" " Default: tools/_common/clip/configs/clip_vit_b_16.yaml",
    )
    parser.add_argument(
        "--model_name",
        default="openai/clip-vit-base-patch16",
        type=str,
        help="the name of a (Open/)CLIP model as shown in HuggingFace for pt backend."
        " Default: openai/clip-vit-base-patch16",
    )
    parser.add_argument(
        "--image_path_or_dir",
        default=None,
        type=str,
        help="input data for predict, it support real data path or data directory." " Default: None",
    )
    parser.add_argument(
        "--prompt_or_path",
        default=None,
        type=str,
        help="prompt corresponding to the image from image path." " Default: None",
    )
    parser.add_argument(
        "--backend",
        default="ms",
        type=str,
        help="backend to do CLIP model inference for CLIP score compute. Option: ms, pt." " Default: ms",
    )
    parser.add_argument("--ckpt_path", default=None, type=str, help="load model checkpoint." " Default: None")
    parser.add_argument(
        "--tokenizer_path",
        default="ldm/models/clip/bpe_simple_vocab_16e6.txt.gz",
        type=str,
        help="load model checkpoint." " Default: ldm/models/clip/bpe_simple_vocab_16e6.txt.gz",
    )
    parser.add_argument(
        "--save_result",
        default=True,
        type=str,
        help="save results or not, if set to True then save to result_path." " Default: True",
    )
    parser.add_argument(
        "--result_path",
        default="results.json",
        type=str,
        help="the path for saving results if save_result is set to True." " Default: results.json",
    )
    parser.add_argument("--quiet", action="store_true", help="set this flag to avoid printing scores")

    args = parser.parse_args()

    # load images
    assert args.image_path_or_dir is not None
    images = []
    if os.path.isdir(args.image_path_or_dir) and os.path.exists(args.image_path_or_dir):
        image_path_or_dir = [
            os.path.join(root, file)
            for root, _, file_list in os.walk(os.path.join(args.image_path_or_dir))
            for file in file_list
            if file.endswith(".jpg")
            or file.endswith(".png")
            or file.endswith(".jpeg")
            or file.endswith(".JPEG")
            or file.endswith("bmp")
        ]
        image_path_or_dir.sort()
        images = [Image.open(p) for p in image_path_or_dir]
        args.image_path_or_dir = image_path_or_dir
    else:
        images = [Image.open(args.image_path_or_dir)]
        args.image_path_or_dir = [args.image_path_or_dir]
    images = [image.resize((224, 224)) for image in images]
    # load prompts
    assert args.prompt_or_path is not None
    texts = []
    if os.path.exists(args.prompt_or_path):
        with open(args.prompt_or_path) as f:
            texts = [p.strip() for p in f.readlines()]
    else:
        texts = [args.prompt_or_path]
    args.prompt_or_path = texts
    assert len(images) % len(texts) == 0
    imgs_per_prompt = len(images) // len(texts)
    if imgs_per_prompt == 1:
        print(f"{len(images)} image-text pairs are loaded")
    else:
        print(f"{len(images)} images and {len(texts)} texts are loaded; Evaluate {imgs_per_prompt} images per prompt")

    print(f"Backend: {args.backend}")
    if args.backend == "pt":
        from clip_score import compute_torchmetric_clip

        if imgs_per_prompt == 1:
            score = compute_torchmetric_clip(images, texts, args.model_name)
        else:
            scores = []
            for i in range(imgs_per_prompt):
                inputs = [images[i::imgs_per_prompt], texts]
                score = compute_torchmetric_clip(*inputs, args.model_name)
                scores.append(score)
            score = sum(scores) / len(scores)

    elif args.backend == "ms":
        image_processor = CLIPImageProcessor()
        text_processor = CLIPTokenizer(args.tokenizer_path, pad_token="!")

        def process_text(prompt):
            return mindspore.Tensor(text_processor(prompt, padding="max_length", max_length=77)["input_ids"]).reshape(
                1, -1
            )

        images = [image_processor(image) for image in images]
        texts = [process_text(text) for text in texts]

        # parse config file
        config = parse(args.config, args.ckpt_path)
        model = CLIPModel(config)
        # get L2 norm operator
        if not is_old_ms_version("2.0.0-alpha"):
            L2_norm_ops = partial(ops.norm, ord=2, dim=1, keepdim=True)
        else:
            L2_norm_ops = partial(ops.norm, p=2, axis=1, keep_dims=True)
        results = []
        for i, text in enumerate(texts):
            text_feature = model.get_text_features(text)
            text_feature = text_feature / L2_norm_ops(text_feature)
            for j in range(imgs_per_prompt):
                image_index = imgs_per_prompt * i + j
                image = images[image_index]
                image_feature = model.get_image_features(image)
                image_feature = image_feature / L2_norm_ops(image_feature)
                res = float(ops.matmul(image_feature, text_feature.T)[0][0] * 100)
                results.append(res)
                if not args.quiet:
                    print(args.image_path_or_dir[image_index], args.prompt_or_path[i], "->", round(res, 4))
            if not args.quiet:
                print("-" * 20)
        score = sum(results) / len(results)

        # save results
        if args.save_result:
            with open(args.result_path, "w") as f:
                for i, text in enumerate(texts):
                    for j in range(imgs_per_prompt):
                        index = imgs_per_prompt * i + j
                        line = {
                            "prompt": args.prompt_or_path[i],
                            "image_path": os.path.abspath(args.image_path_or_dir[index]),
                            "clip_score": results[index],
                        }
                        f.write(json.dumps(line) + "\n")
    else:
        raise ValueError(f"Unknown backend: {args.backend}. Valid backend: [ms, pt]")

    print("Mean score =", score)
