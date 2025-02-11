import argparse
import os
import time

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from test_accuracy_data import PROMPTS


def clip_score(model_clip, tokenizer, preprocess, prompt, img_files, device):
    imgs = []
    texts = []
    for img_file in img_files:
        img = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
        imgs.append(img)
        text = tokenizer([prompt]).to(device)
        texts.append(text)

    img = torch.cat(imgs)
    text = torch.cat(texts)

    with torch.no_grad():
        text_ft = model_clip.encode_text(text).float()
        img_ft = model_clip.encode_image(img).float()
        score = F.cosine_similarity(img_ft, text_ft).squeeze()

    return score.cpu()


def get_model(args):
    model_name = "ViT-H-14"

    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    start = time.time()
    print("Load clip model...")
    model_clip, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=args.model_path,
        device=device,
    )
    model_clip.eval()
    print(f">done. elapsed time: {(time.time()-start):.3f}")

    tokenizer = open_clip.get_tokenizer(model_name)

    return model_clip, preprocess, tokenizer, device


def calculate_clip_score(model_clip, preprocess, tokenizer, device, data_path):
    all_scores = 0
    for i, prompt in enumerate(PROMPTS):
        print(f"prompt: [{prompt}]")
        images_score = clip_score(
            model_clip=model_clip,
            tokenizer=tokenizer,
            preprocess=preprocess,
            prompt=prompt,
            img_files=[os.path.join(data_path, f"output_{i}.png")],
            device=device,
        )

        print(f"score: {images_score:.5f}")
        all_scores += images_score

    avg_score = (all_scores / len(PROMPTS)).cpu()
    print(f">> average score: {avg_score:.5f}")
    return avg_score


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="open-clip pretrained weight path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="compute device",
    )

    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    model_clip, preprocess, tokenizer, device = get_model(args)
    clip_score = {}
    for _, sub_dirs, _ in os.walk("./data"):
        for dir in sub_dirs:
            clip_score[dir] = calculate_clip_score(
                model_clip,
                preprocess,
                tokenizer,
                device,
                data_path=os.path.join("./data", dir),
            )

    assert abs(clip_score["cache_gate"] - clip_score["base"]) < 5e-3
    assert abs(clip_score["cache_gate_todo"] - clip_score["base"]) < 5e-3


if __name__ == "__main__":
    main()
