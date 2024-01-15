import argparse
import os
from functools import partial

from PIL import Image
from tools._common.clip import CLIPImageProcessor, CLIPModel, CLIPTokenizer, parse

import mindspore
from mindspore import ops


def load_images(image_path_or_dir):
    images = []
    if os.path.isdir(image_path_or_dir) and os.path.exists(image_path_or_dir):
        image_path_or_dir = [
            os.path.join(root, file)
            for root, _, file_list in os.walk(os.path.join(image_path_or_dir))
            for file in file_list
            if file.endswith(".jpg")
            or file.endswith(".png")
            or file.endswith(".jpeg")
            or file.endswith(".JPEG")
            or file.endswith("bmp")
        ]
        image_path_or_dir.sort()
        images = [Image.open(p) for p in image_path_or_dir]
    else:
        images = [Image.open(image_path_or_dir)]
        image_path_or_dir = [image_path_or_dir]
    images = [image.resize((224, 224)) for image in images]
    return images, image_path_or_dir


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
        "--real_image_path_or_dir",
        default=None,
        type=str,
        help="input data for predict, it support real data path or data directory." " Default: None",
    )
    parser.add_argument(
        "--gen_image_path_or_dir",
        default=None,
        type=str,
        help="input data for predict, it support real data path or data directory." " Default: None",
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
    parser.add_argument("--quiet", action="store_true", help="set this flag to avoid printing scores")
    parser.add_argument(
        "--check_certificate",
        action="store_true",
        help="set this flag to check for certificate for downloads (checks)",
    )
    args = parser.parse_args()

    # load real and generated images
    assert args.real_image_path_or_dir is not None
    real_images, args.real_image_path_or_dir = load_images(args.real_image_path_or_dir)
    assert args.gen_image_path_or_dir is not None
    gen_images, args.gen_image_path_or_dir = load_images(args.gen_image_path_or_dir)
    print("Num real images: ", len(real_images))
    print("Num generated images: ", len(gen_images))

    print(f"Backend: {args.backend}")
    if args.backend == "pt":
        raise ValueError("Not support pt backend for now.")

    elif args.backend == "ms":
        image_processor = CLIPImageProcessor()
        text_processor = CLIPTokenizer(args.tokenizer_path, pad_token="!")

        real_images = [image_processor(image) for image in real_images]
        gen_images = [image_processor(image) for image in gen_images]
        # get L2 norm operator
        if mindspore.__version__.startswith("2"):
            L2_norm_ops = partial(ops.norm, ord=2, dim=1, keepdim=True)
        else:
            L2_norm_ops = partial(ops.norm, p=2, axis=1, keep_dims=True)
        # parse config file
        config = parse(args.config, args.ckpt_path)
        model = CLIPModel(config)
        real_image_features = []
        for i_real_image in range(len(real_images)):
            real_image = real_images[i_real_image]
            real_image_feature = model.get_image_features(real_image)
            real_image_feature = real_image_feature / L2_norm_ops(real_image_feature)
            real_image_features.append(real_image_feature)
            if not args.quiet:
                print(f"Extracting real image features: {i_real_image+1}/{len(real_images)}", end="\r")
        real_image_features = ops.stack(real_image_features, axis=0).squeeze(1)
        gen_image_features = []
        for i_gen_image in range(len(gen_images)):
            gen_image = gen_images[i_gen_image]
            gen_image_feature = model.get_image_features(gen_image)
            gen_image_feature = gen_image_feature / L2_norm_ops(gen_image_feature)
            gen_image_features.append(gen_image_feature)
            if not args.quiet:
                print(f"Extracting generated image features: {i_gen_image+1}/{len(gen_images)}", end="\r")
        gen_image_features = ops.stack(gen_image_features, axis=0).squeeze(1)
        results = ops.matmul(real_image_features, gen_image_features.permute((1, 0))) * 100
        score = ops.sum(results) / results.numel()
    else:
        raise ValueError(f"Unknown backend: {args.backend}. Valid backend: [ms, pt]")

    print("Mean score =", score)
